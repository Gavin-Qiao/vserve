"""Start/stop vLLM via systemd."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vserve.config import (
    cfg,
    find_systemd_unit_path,
    read_profile_yaml,
    unit_content_matches_backend,
    unit_uses_environment_file,
    validate_systemd_service_name,
    write_active_manifest,
)


def _assert_vllm_unit_safe_for_privileged_action() -> None:
    service_name = cfg().service_name
    validate_systemd_service_name(service_name)
    unit = find_systemd_unit_path(service_name)
    if unit is None:
        return
    try:
        content = unit.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Cannot verify {service_name}.service before privileged systemctl action: {exc}") from None
    if not unit_content_matches_backend(
        content,
        backend_name="vllm",
        root=cfg().vllm_root,
        expected_paths=[cfg().active_yaml],
    ):
        raise RuntimeError(f"{service_name}.service does not look like a vserve vLLM unit")


def _systemctl(action: str, timeout: int = 30, *, non_interactive: bool = False) -> tuple[bool, str, str]:
    command = ["systemctl", action, cfg().service_name]
    if action in {"start", "stop", "restart", "reload"}:
        try:
            _assert_vllm_unit_safe_for_privileged_action()
        except Exception as exc:
            return False, "", str(exc)
        if non_interactive:
            command.insert(0, "-n")
        command.insert(0, "sudo")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"systemctl {action} timed out after {timeout}s"
    except Exception as exc:
        return False, "", str(exc)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def _update_active_symlink(config_path: Path) -> None:
    active = cfg().active_yaml
    active.parent.mkdir(parents=True, exist_ok=True)
    try:
        active.unlink(missing_ok=True)
        active.symlink_to(config_path.resolve())
    except OSError as exc:
        raise RuntimeError(f"failed to update active config link {active}: {exc}") from None


def _snapshot_active_config() -> dict:
    active = cfg().active_yaml
    if active.is_symlink():
        return {"kind": "symlink", "target": active.resolve(strict=False)}
    try:
        if active.exists():
            return {
                "kind": "file",
                "content": active.read_bytes(),
                "mode": active.stat().st_mode,
            }
    except OSError:
        return {"kind": "missing"}
    return {"kind": "missing"}


def _restore_active_config(snapshot: dict) -> None:
    active = cfg().active_yaml
    active.parent.mkdir(parents=True, exist_ok=True)
    try:
        active.unlink(missing_ok=True)
        if snapshot.get("kind") == "symlink":
            active.symlink_to(Path(snapshot["target"]))
        elif snapshot.get("kind") == "file":
            active.write_bytes(snapshot["content"])
            active.chmod(snapshot["mode"] & 0o777)
    except OSError as exc:
        raise RuntimeError(f"failed to restore active config link {active}: {exc}") from None


def _write_vllm_manifest(config_path: Path, *, status: str, error: str | None = None) -> None:
    gpu_index = int(getattr(cfg(), "gpu_index", 0) or 0)
    manifest = {
        "backend": "vllm",
        "service_name": cfg().service_name,
        "config_path": str(config_path.resolve()),
        "gpu_index": gpu_index,
        "cuda_visible_devices": str(gpu_index),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        manifest["error"] = error
    write_active_manifest(manifest, cfg().run_dir / "active-manifest.json")


def _upsert_env_file(extra: dict[str, str] | None = None) -> Path:
    c = cfg()
    env_path = c.vllm_root / "configs" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, str] = {
        "CUDA_HOME": str(c.cuda_home),
        "TMPDIR": str(c.vllm_root / "tmp"),
        "VLLM_RPC_BASE_PATH": str(c.vllm_root / "tmp"),
        "CUDA_VISIBLE_DEVICES": str(int(getattr(c, "gpu_index", 0) or 0)),
    }
    if extra:
        values.update(extra)
    lines: list[str] = []
    seen: set[str] = set()
    if env_path.exists():
        try:
            lines = env_path.read_text().splitlines()
        except OSError:
            lines = []
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in values:
            updated.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            updated.append(line)
    for key, value in values.items():
        if key not in seen:
            updated.append(f"{key}={value}")
    env_path.write_text("\n".join(updated) + "\n")
    return env_path


def _resolve_quant_envs(config_path: Path) -> dict[str, str]:
    """Read the active YAML config; return any quant-method-specific env vars.

    Used by ``start_vllm`` to write `VLLM_USE_FLASHINFER_MOE_FP4=1` etc. into
    the env file when the launched model uses NVFP4 / ModelOpt-NVFP4 / MXFP4.
    Silent fallthrough on read errors — env vars are advisory.
    """
    try:
        data = read_profile_yaml(config_path) or {}
    except Exception:
        return {}
    quant = data.get("quantization") if isinstance(data, dict) else None
    if not isinstance(quant, str):
        return {}
    from vserve.models import QUANT_ENV_VARS
    # Map vLLM `--quantization` strings back to env-table keys.
    qkey = quant.lower()
    if qkey in QUANT_ENV_VARS:
        return dict(QUANT_ENV_VARS[qkey])
    return {}


def _service_uses_env_file(env_path: Path) -> bool:
    unit = find_systemd_unit_path(cfg().service_name)
    if unit is None:
        return True
    try:
        content = unit.read_text()
    except OSError:
        return True
    return unit_uses_environment_file(content, env_path)


def is_vllm_running() -> bool:
    ok, output, err = _systemctl("is-active", timeout=5)
    status = output.strip().lower()
    if ok and status == "active":
        return True
    if status in {"inactive", "failed"}:
        return False
    if status in {"activating", "deactivating", "reloading"}:
        raise RuntimeError(f"systemctl is-active {cfg().service_name} is transitional: {status}")
    if "could not be found" in err.lower():
        return False
    if err:
        raise RuntimeError(f"systemctl is-active {cfg().service_name} failed: {err}")
    return False


def start_vllm(config_path: Path, *, non_interactive: bool = False) -> None:
    snapshot = _snapshot_active_config()
    # Quant-specific env vars (Q): write VLLM_USE_FLASHINFER_MOE_FP4 etc.
    # alongside the base CUDA/TMP/RPC values so the systemd service picks
    # them up from the EnvironmentFile.
    env_path = _upsert_env_file(_resolve_quant_envs(config_path))
    if not _service_uses_env_file(env_path):
        raise RuntimeError(
            f"{cfg().service_name}.service must reference EnvironmentFile={env_path} "
            "so CUDA_VISIBLE_DEVICES can enforce the configured GPU index."
        )
    _update_active_symlink(config_path)
    _write_vllm_manifest(config_path, status="starting")
    ok, _out, err = _systemctl("start", non_interactive=non_interactive)
    if not ok:
        _restore_active_config(snapshot)
        _write_vllm_manifest(config_path, status="failed", error=err)
        raise RuntimeError(f"systemctl start failed: {err}")


def stop_vllm(*, non_interactive: bool = False) -> None:
    ok, _out, err = _systemctl("stop", non_interactive=non_interactive)
    if not ok:
        raise RuntimeError(f"systemctl stop failed: {err}")
