"""`vserve doctor` — system-readiness checks.

Extracted from `cli.py` in 0.6.3 per audit
`docs/audits/2026-05-20-cli-sprawl.md` (the inline doctor command was
513 lines including a 364-line nested helper). The body now lives here;
the `@app.command()` registration stays in `cli.py` as a thin wrapper.

Public entry point: :func:`run_doctor` — takes ``console`` and the two
flags from the CLI, returns ``(summary_dict, checks_list)`` so the CLI
wrapper can decide on the exit code.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from rich.console import Console


def doctor_summary_label(warn_count: int, fail_count: int) -> str:
    """One-line summary suitable for the human-readable footer."""
    if fail_count == 0 and warn_count == 0:
        return "All clear"
    if fail_count == 0:
        return f"{warn_count} warning(s) found"
    if warn_count == 0:
        return f"{fail_count} issue(s) found"
    return f"{fail_count} issue(s) and {warn_count} warning(s) found"


def run_doctor(
    console: Console,
    *,
    strict: bool,
    json_output: bool,
    safe_path_exists,
    safe_resolve_path,
    all_models_fn,
    read_limits_for_fn,
) -> None:
    """Run all readiness checks. Prints either rich-formatted output (default)
    or one big JSON blob (``--json``). Raises ``typer.Exit(1)`` when
    ``strict`` is set and any check fails.

    Takes the cli-side helpers as parameters to avoid cli ↔ cli_doctor
    circular imports.
    """
    from vserve.backends import _BACKENDS, running_backend as _running_backend
    from vserve.config import (
        LOGS_DIR,
        VLLM_BIN,
        VLLM_ROOT,
        active_yaml_path,
        cfg as _cfg,
        find_systemd_unit_path,
        try_read_profile_yaml,
        unit_uses_environment_file,
    )

    ok_count = 0
    warn_count = 0
    fail_count = 0
    checks: list[dict[str, str]] = []

    def _emit(*args: Any, **kwargs: Any) -> None:
        if not json_output:
            console.print(*args, **kwargs)

    def _ok(msg: str) -> None:
        nonlocal ok_count
        checks.append({"status": "ok", "message": msg, "fix": ""})
        _emit(f"  [green]OK[/green]    {msg}")
        ok_count += 1

    def _fail(msg: str, fix: str = "") -> None:
        nonlocal fail_count
        checks.append({"status": "fail", "message": msg, "fix": fix})
        _emit(f"  [red]FAIL[/red]  {msg}")
        if fix:
            _emit(f"          Fix: {fix}")
        fail_count += 1

    def _warn(msg: str, fix: str = "") -> None:
        nonlocal warn_count
        checks.append({"status": "warn", "message": msg, "fix": fix})
        _emit(f"  [yellow]WARN[/yellow]  {msg}")
        if fix:
            _emit(f"          Fix: {fix}")
        warn_count += 1

    def _fail_or_warn(required: bool, msg: str, fix: str = "") -> None:
        if required:
            _fail(msg, fix)
        else:
            _warn(msg, fix)

    _emit("\n[bold]vserve doctor[/bold]\n")

    _c = _cfg()
    vllm_runtime_present = VLLM_BIN.exists()
    vllm_required = vllm_runtime_present or _c.llamacpp_root is None

    # -- Environment --
    _emit("  [bold]Environment[/bold]")

    # nvcc
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5,
                           env={**os.environ, "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", "")})
        if r.returncode == 0:
            nvcc_lines = [ln for ln in r.stdout.splitlines() if "release" in ln]
            _ok(f"nvcc {nvcc_lines[0].split('release')[-1].strip().rstrip(',') if nvcc_lines else 'found'}")
        else:
            _fail("nvcc not working", "Install CUDA toolkit or check /usr/local/cuda/bin/nvcc")
    except Exception:
        _fail("nvcc not found", "Install: sudo apt install nvidia-cuda-toolkit  OR  https://developer.nvidia.com/cuda-downloads")

    # vLLM
    if vllm_required:
        try:
            r = subprocess.run([str(VLLM_BIN), "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ver = r.stdout.strip() or r.stderr.strip() or "found"
                _ok(f"vLLM {ver}")
            else:
                details = r.stderr.strip() or r.stdout.strip()
                _fail(f"vLLM not working at {VLLM_BIN}", details or "Check the vLLM installation and environment")
        except Exception:
            _fail(f"vLLM not found at {VLLM_BIN}")
    else:
        _warn("vLLM not configured (skipped — llama.cpp-only setup)")

    # GPU
    try:
        from vserve.gpu import get_gpu_info
        gpu = get_gpu_info()
        mem_used = 0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5)
            mem_used = int(out.decode().strip().split("\n")[0])
        except Exception:
            pass
        _ok(f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, {mem_used} MiB used)")
    except Exception:
        _fail("GPU not accessible", "Install NVIDIA drivers: https://www.nvidia.com/drivers  then check: nvidia-smi")

    # Host hardening: driver-package hold + log rotation. Both guard failure
    # modes that took the fleet down from outside the inference stack (an
    # unattended driver bump broke NVML; an un-rotated log grew to 232 MB).
    try:
        from vserve.host_health import check_log_rotation, check_nvidia_driver_held

        for result in (check_nvidia_driver_held(), check_log_rotation(VLLM_ROOT / "logs")):
            if result["ok"]:
                _ok(result["message"])
            else:
                _warn(result["message"], result.get("fix", ""))
    except Exception:
        _warn("Host hardening checks (driver hold / log rotation) could not run")

    # -- Backends --
    _emit("\n  [bold]Backends[/bold]")
    for b in _BACKENDS:
        for desc, check_fn in b.doctor_checks():
            try:
                if check_fn():
                    _ok(desc)
                else:
                    _warn(desc)
            except Exception:
                _warn(f"{desc} (check error)")

    # -- Per-backend checks --
    # ── vLLM ──
    _emit("\n  [bold]vLLM[/bold]")

    if not vllm_required:
        _warn("vLLM checks skipped (llama.cpp-only setup)")
    else:
        try:
            from vserve.runtime import check_vllm_compatibility, collect_vllm_runtime_info

            runtime_info = collect_vllm_runtime_info(_c)
            runtime_check = check_vllm_compatibility(runtime_info)
            if runtime_check.supported:
                _ok(f"Runtime supported ({runtime_info.vllm_version}, {runtime_check.range})")
            else:
                _fail("Unsupported vLLM runtime", "; ".join(runtime_check.errors))
            for warning in runtime_check.warnings:
                _warn(warning)
        except Exception as exc:
            _warn("Could not complete vLLM runtime compatibility check", str(exc))

    try:
        r = subprocess.run(["id", _c.service_user], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok(f"Service user '{_c.service_user}' exists")
        else:
            _fail_or_warn(vllm_required, f"Service user '{_c.service_user}' not found")
    except Exception:
        _fail_or_warn(vllm_required, "Cannot check service user")

    svc_path = find_systemd_unit_path(_c.service_name)
    if svc_path is not None:
        try:
            svc_content = svc_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f"Could not read systemd unit: {svc_path.name}", str(exc))
        else:
            unit_failed = False
            if "ProtectSystem=strict" in svc_content:
                _fail_or_warn(
                    vllm_required,
                    "systemd unit has ProtectSystem=strict",
                    "Remove it — breaks nvcc JIT compilation",
                )
                unit_failed = vllm_required
            if "TimeoutStartSec" not in svc_content:
                _warn("No TimeoutStartSec in service — default 90s may be too short for JIT",
                      "Add TimeoutStartSec=600")
            env_path = VLLM_ROOT / "configs" / ".env"
            if not unit_uses_environment_file(svc_content, env_path):
                _fail_or_warn(
                    vllm_required,
                    "systemd unit does not load vserve configs/.env",
                    f"Add EnvironmentFile={env_path} so CUDA_VISIBLE_DEVICES is enforced",
                )
                unit_failed = vllm_required
            if not unit_failed:
                _ok("systemd unit configured correctly")
    else:
        _fail_or_warn(
            vllm_required,
            f"No systemd unit found for {_c.service_name}.service",
            "Create one: https://docs.vllm.ai  or see vserve docs/troubleshooting.md",
        )

    env_path = VLLM_ROOT / "configs" / ".env"
    if env_path.exists():
        try:
            env_content = env_path.read_text()
            missing = [v for v in ["CUDA_HOME", "TMPDIR", "VLLM_RPC_BASE_PATH", "CUDA_VISIBLE_DEVICES"] if v not in env_content]
            if missing:
                _warn(f".env missing: {', '.join(missing)}")
            else:
                _ok(".env has required variables")
        except PermissionError:
            _ok(".env exists (not readable — OK, contains secrets)")
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f".env unreadable: {env_path}", str(exc))
    else:
        _fail_or_warn(vllm_required, f"No .env at {env_path}")

    vllm_models = VLLM_ROOT / "models"
    if vllm_models.exists():
        try:
            vllm_mc = sum(1 for p in vllm_models.glob("*/*/config.json"))
        except OSError as exc:
            _warn(f"Models dir unreadable: {vllm_models}", str(exc))
        else:
            _ok(f"Models dir: {vllm_models} ({vllm_mc} models)")
    else:
        _warn(f"Models dir {vllm_models} does not exist")

    cache_checks = [
        (VLLM_ROOT / ".cache" / "flashinfer", "FlashInfer JIT"),
        (VLLM_ROOT / ".cache" / "vllm" / "torch_compile_cache", "torch.compile"),
    ]
    for cdir, label in cache_checks:
        if not cdir.exists():
            _warn(f"{label} cache missing — first start will JIT compile (2-10 min)")
            continue
        try:
            files = []
            for f in cdir.rglob("*"):
                try:
                    if f.is_file():
                        files.append(f)
                except OSError:
                    continue
        except OSError as exc:
            _warn(f"{label} cache unreadable", str(exc))
            continue
        if not files:
            _warn(f"{label} cache dir exists but is empty — first start may be slow")
            continue
        size_bytes = 0
        for f in files:
            try:
                size_bytes += f.stat().st_size
            except OSError:
                continue
        _ok(f"{label} cache ({size_bytes / (1024 * 1024):.0f} MB)")

    active = active_yaml_path()
    active_is_symlink = active.is_symlink()
    if active_is_symlink:
        target = safe_resolve_path(active)
        if target is None:
            _warn("active.yaml symlink is unreadable or recursive")
        elif safe_path_exists(target):
            cfg_data = try_read_profile_yaml(active)
            if cfg_data is None:
                _warn(f"active.yaml unreadable: {target.name}")
            else:
                model_path = cfg_data.get("model", "")
                if isinstance(model_path, (str, os.PathLike)) and model_path and Path(model_path).exists():
                    _ok(f"active.yaml → {target.name}")
                else:
                    _fail_or_warn(vllm_required, f"active.yaml model path missing: {model_path}")
        else:
            _fail_or_warn(vllm_required, f"active.yaml → broken symlink: {target}")
    elif safe_path_exists(active):
        _ok("active.yaml exists (not a symlink)")
    else:
        _ok("No active.yaml (clean — will be created on vserve run)")

    tmp_dir = VLLM_ROOT / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        _ok(f"TMPDIR at {tmp_dir}")
        try:
            sockets = list(tmp_dir.rglob("*"))
            stale = [s for s in sockets if s.is_socket()]
        except OSError as exc:
            _warn(f"Could not inspect TMPDIR sockets: {tmp_dir}", str(exc))
        else:
            if len(stale) > 10:
                _warn(f"{len(stale)} stale sockets in {tmp_dir}", f"sudo find {tmp_dir} -type s -delete")
    else:
        _fail_or_warn(
            vllm_required,
            f"TMPDIR {tmp_dir} does not exist",
            f"sudo mkdir -p {tmp_dir} && sudo chown vllm:llm {tmp_dir}",
        )

    # ── llama.cpp ──
    _emit("\n  [bold]llama.cpp[/bold]")

    lc_root = _c.llamacpp_root
    if lc_root is None:
        _warn("llama.cpp not configured (run vserve init to detect)")
    else:
        _ok(f"Root: {lc_root}")

        lc_bin = lc_root / "bin" / "llama-server"
        if lc_bin.exists():
            try:
                r = subprocess.run([str(lc_bin), "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    ver_line = ""
                    for ln in (r.stdout + r.stderr).splitlines():
                        if "version" in ln.lower() or ln.startswith("b"):
                            ver_line = ln.strip()
                            break
                    _ok(f"llama-server {ver_line}" if ver_line else f"llama-server at {lc_bin}")
                else:
                    details = r.stderr.strip() or r.stdout.strip()
                    _fail(f"llama-server at {lc_bin} is not working",
                          details or "Build or reinstall llama.cpp")
            except Exception:
                _fail(f"llama-server at {lc_bin} is not working")
        elif __import__("shutil").which("llama-server"):
            _ok("llama-server found on PATH")
        else:
            _fail("llama-server not found",
                  "Build: cmake -B build -DGGML_CUDA=ON && cmake --build build -t llama-server")

        lc_user = _c.llamacpp_service_user
        try:
            r = subprocess.run(["id", lc_user], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                _ok(f"Service user '{lc_user}' exists")
            else:
                _fail(f"Service user '{lc_user}' not found",
                      f"sudo useradd -r -s /usr/sbin/nologin -g llm {lc_user}")
        except Exception:
            _fail("Cannot check llama-cpp service user")

        lc_svc = _c.llamacpp_service_name
        lc_svc_path = find_systemd_unit_path(lc_svc)
        if lc_svc_path is not None:
            try:
                lc_svc_content = lc_svc_path.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                _warn(f"Could not read {lc_svc}.service", str(exc))
            else:
                issues = []
                active_sh = lc_root / "configs" / "active.sh"
                active_sh_has_cuda = False
                try:
                    active_sh_has_cuda = active_sh.exists() and "CUDA_VISIBLE_DEVICES" in active_sh.read_text()
                except (OSError, UnicodeDecodeError):
                    active_sh_has_cuda = False
                if "CUDA_VISIBLE_DEVICES" not in lc_svc_content and not active_sh_has_cuda:
                    issues.append("missing CUDA_VISIBLE_DEVICES (may use wrong GPU)")
                if "TimeoutStartSec" not in lc_svc_content:
                    issues.append("no TimeoutStartSec (large models need time)")
                if issues:
                    _warn(f"{lc_svc}.service: {'; '.join(issues)}")
                else:
                    _ok(f"{lc_svc}.service unit configured correctly")
        else:
            _fail(f"No systemd unit found for {lc_svc}.service",
                  f"Create /etc/systemd/system/{lc_svc}.service with ExecStart={lc_root}/configs/active.sh")

        lc_models = lc_root / "models"
        if lc_models.exists():
            try:
                from vserve.model_files import iter_recursive_files_with_suffix
                model_count = len(iter_recursive_files_with_suffix(lc_models, ".gguf"))
            except OSError as exc:
                _warn(f"Models dir unreadable: {lc_models}", str(exc))
            else:
                _ok(f"Models dir: {lc_models} ({model_count} GGUF files)")
        else:
            _warn(f"Models dir {lc_models} does not exist",
                  f"sudo mkdir -p {lc_models} && sudo chown {lc_user}:llm {lc_models}")

        lc_configs = lc_root / "configs"
        if lc_configs.exists():
            active_sh = lc_configs / "active.sh"
            active_json = lc_configs / "active.json"
            if active_sh.exists():
                _ok("active.sh launch script present")
            else:
                _ok(f"Configs dir: {lc_configs} (no active config yet)")
            if active_json.exists():
                try:
                    lc_cfg = json.loads(active_json.read_text())
                    lc_model = lc_cfg.get("model", "")
                    if isinstance(lc_model, (str, os.PathLike)) and lc_model and Path(lc_model).exists():
                        _ok(f"active.json → {Path(lc_model).name}")
                    elif lc_model:
                        _fail(f"active.json model path missing: {lc_model}")
                except Exception:
                    _warn("active.json exists but unreadable")
        else:
            _warn(f"Configs dir {lc_configs} does not exist",
                  f"sudo mkdir -p {lc_configs} && sudo chown {lc_user}:llm {lc_configs}")

        try:
            import gguf  # type: ignore[import-not-found, import-untyped]  # noqa: F401
            _ok("gguf package installed")
        except ImportError:
            _warn("gguf package not installed — needed for GGUF metadata reading",
                  "pip install 'vserve\\[llamacpp]'")

    # -- Shared --
    _emit("\n  [bold]Shared[/bold]")

    def _port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _health_ok(backend, port: int) -> bool:
        try:
            from urllib.request import urlopen
            with urlopen(backend.health_url(port), timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _active_backend_port(backend) -> int:
        try:
            if backend.name == "llamacpp":
                active_config_path_fn = getattr(backend, "_active_config_path", None)
                active_json = active_config_path_fn().with_suffix(".json") if callable(active_config_path_fn) else None
                if active_json is not None and active_json.exists():
                    data = json.loads(active_json.read_text())
                    if isinstance(data, dict):
                        return int(data.get("port", _c.port))
            active_cfg = try_read_profile_yaml(active_yaml_path()) or {}
            return int(active_cfg.get("port", _c.port))
        except Exception:
            return int(_c.port)

    _port = _c.port
    port_in_use = _port_open(_port)
    active_backend = _running_backend()
    if active_backend is not None:
        active_port = _active_backend_port(active_backend)
        active_port_open = _port_open(active_port)
        active_health_ok = _health_ok(active_backend, active_port)
        if active_port_open or active_health_ok:
            _ok(f"{active_backend.display_name} serving on port {active_port}")
            if active_port != _port:
                _warn(f"Configured default port is {_port}, active backend uses {active_port}")
        else:
            _fail(
                f"{active_backend.display_name} appears active but port {active_port} is not open",
                f"Check: sudo journalctl -u {active_backend.service_name} --no-pager -n 50",
            )
    elif port_in_use:
        _fail(f"Port {_port} in use but no backend running — something else is bound",
              f"Check: sudo lsof -i :{_port}  then stop the other service or change port in ~/.config/vserve/config.yaml")
    else:
        _ok(f"Port {_port} available")

    log_file = LOGS_DIR / "vllm.log"
    if log_file.exists():
        try:
            size_mb = log_file.stat().st_size / (1024 * 1024)
        except OSError as exc:
            _warn(f"Could not read log metadata: {log_file}", str(exc))
        else:
            if size_mb > 100:
                _warn(f"vllm.log is {size_mb:.0f} MB", "Consider truncating or adding log rotation")
            else:
                _ok(f"vllm.log ({size_mb:.0f} MB)")

    import grp
    try:
        tty_members = grp.getgrnam("tty").gr_mem
        me = os.environ.get("USER", "")
        if me in tty_members:
            _ok("tty group (terminal messaging between users)")
        else:
            _warn(f"'{me}' not in tty group — vserve can't DM other users",
                  f"sudo usermod -aG tty {me}  (then re-login)")
    except KeyError:
        _warn("tty group not found")

    all_models = all_models_fn()
    probed = sum(1 for m in all_models if read_limits_for_fn(m.provider, m.model_name))
    _ok(f"{len(all_models)} models downloaded, {probed} probed")

    summary = {"ok": ok_count, "warn": warn_count, "fail": fail_count}
    if json_output:
        typer.echo(json.dumps({"summary": summary, "checks": checks}, sort_keys=True))
    else:
        console.print(
            f"\n  [bold]{doctor_summary_label(warn_count, fail_count)}[/bold]  "
            f"({ok_count} ok, {warn_count} warn, {fail_count} fail)\n",
        )
    if strict and fail_count:
        raise typer.Exit(1)
