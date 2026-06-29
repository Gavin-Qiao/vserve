"""Paths, YAML/JSON read/write, and configuration discovery."""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_log = logging.getLogger(__name__)

SYSTEMD_UNIT_DIRS = (
    Path("/etc/systemd/system"),
    Path("/run/systemd/system"),
    Path("/lib/systemd/system"),
    Path("/usr/local/lib/systemd/system"),
    Path("/usr/lib/systemd/system"),
)
_SYSTEMD_SERVICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:-]*$")


def validate_systemd_service_name(service_name: str) -> str:
    """Reject names that cannot be a single systemd service unit."""
    if (
        not service_name
        or service_name in {".", ".."}
        or "/" in service_name
        or "\\" in service_name
        or any(ch.isspace() for ch in service_name)
        or not _SYSTEMD_SERVICE_RE.match(service_name)
    ):
        raise ValueError(f"Invalid systemd service name: {service_name!r}")
    return service_name


def find_systemd_unit_path(service_name: str) -> Path | None:
    """Return the first existing systemd unit path for a service."""
    validate_systemd_service_name(service_name)
    unit_name = f"{service_name}.service"
    try:
        result = subprocess.run(
            ["systemctl", "show", unit_name, "--property=FragmentPath", "--property=LoadState"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            fragment_path = ""
            for line in result.stdout.splitlines():
                if line.startswith("FragmentPath="):
                    fragment_path = line.removeprefix("FragmentPath=").strip()
                    break
            if fragment_path:
                path = Path(fragment_path)
                if path.exists():
                    return path
    except Exception:
        pass
    for unit_dir in SYSTEMD_UNIT_DIRS:
        unit_path = unit_dir / unit_name
        try:
            if unit_path.exists():
                return unit_path
        except OSError:
            continue
    return None


def systemd_environment_files(content: str) -> list[Path]:
    """Parse EnvironmentFile entries from systemd unit content."""
    paths: list[Path] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "EnvironmentFile":
            continue
        try:
            tokens = shlex.split(value, comments=False, posix=True)
        except ValueError:
            tokens = value.split()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token.startswith("-"):
                token = token[1:]
            if token:
                paths.append(Path(token))
    return paths


def unit_uses_environment_file(content: str, env_path: Path) -> bool:
    """Return true only when a unit references the exact vserve env file."""
    expected = env_path.resolve(strict=False)
    for candidate in systemd_environment_files(content):
        try:
            if candidate.resolve(strict=False) == expected:
                return True
        except OSError:
            if candidate == env_path:
                return True
    return False


def unit_content_matches_backend(
    content: str,
    *,
    backend_name: str,
    root: Path,
    expected_paths: list[Path] | None = None,
) -> bool:
    """Best-effort ownership check before privileged systemctl actions."""
    needles = []
    root_text = str(root)
    if root_text not in {"", "/"}:
        needles.append(root_text)
    needles.extend(str(path) for path in expected_paths or [] if str(path) not in {"", "/"})
    if backend_name == "vllm":
        needles.append("vllm")
    elif backend_name == "llamacpp":
        needles.extend(["llama-server", "llama-cpp", "llamacpp"])
    return any(needle and needle in content for needle in needles)


def _atomic_write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            fd = -1
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if mode is not None:
            path.chmod(mode)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json_mapping(path: Path, *, warn: bool, label: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        if warn:
            _log.warning("Failed to parse %s JSON %s: %s", label, path, exc)
        return None
    if not isinstance(data, dict):
        if warn:
            _log.warning("Ignoring invalid %s JSON %s: expected object", label, path)
        return None
    return data


def _read_yaml_mapping(path: Path, *, warn: bool, label: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        if warn:
            _log.warning("Failed to parse %s YAML %s: %s", label, path, exc)
        return None
    if not isinstance(data, dict):
        if warn:
            _log.warning("Ignoring invalid %s YAML %s: expected mapping", label, path)
        return None
    return data

CONFIG_FILE = Path.home() / ".config" / "vserve" / "config.yaml"
LIMITS_SCHEMA_VERSION = 5
ACTIVE_MANIFEST_SCHEMA_VERSION = 1

# Hardcoded defaults — used only when discovery fails and no config file exists.
_DEFAULT_VLLM_ROOT = "/opt/vllm"
_DEFAULT_SERVICE_NAME = "vllm"
_DEFAULT_SERVICE_USER = "vllm"
_DEFAULT_PORT = 8888
_DEFAULT_CUDA_HOME = "/usr/local/cuda"


@dataclass
class BackendConfig:
    name: str
    root: Path | None
    service_name: str
    service_user: str
    port: int | None = None


@dataclass
class VserveConfig:
    vllm_root: Path
    models_dir: Path
    configs_dir: Path
    benchmarks_dir: Path
    logs_dir: Path
    run_dir: Path
    vllm_bin: Path
    vllm_python: Path
    cuda_home: Path
    service_name: str
    service_user: str
    port: int
    gpu_index: int = 0
    gpu_memory_utilization: float | None = None
    gpu_overhead_gb: float | None = None
    llamacpp_root: Path | None = None
    llamacpp_service_name: str = "llama-cpp"
    llamacpp_service_user: str = "llama-cpp"
    # Separate systemd service for block-diffusion (dLLM) models that need a
    # runtime newer than the pinned-stable one. None = feature off (dLLMs use
    # the default vLLM service). See backends.vllm.resolve_vllm_service_name.
    dllm_service_name: str | None = None
    backends: dict[str, BackendConfig] = field(default_factory=dict)

    @property
    def active_yaml(self) -> Path:
        return self.vllm_root / "configs" / "active.yaml"


def _discover_vllm_root() -> Path | None:
    """Try to find the vLLM root by locating the vllm binary or checking common paths."""
    # Method 1: find vllm on PATH
    vllm_path = shutil.which("vllm")
    if vllm_path:
        real = Path(vllm_path).resolve()
        # e.g. /opt/vllm/venv/bin/vllm → /opt/vllm
        if real.parent.name == "bin" and real.parent.parent.name in ("venv", ".venv"):
            return real.parent.parent.parent

    # Method 2: check common locations
    for candidate in [
        Path("/opt/vllm"),
        Path.home() / "vllm",
        Path.home() / ".vllm",
    ]:
        if (candidate / "venv" / "bin" / "vllm").exists() or \
           (candidate / ".venv" / "bin" / "vllm").exists():
            return candidate

    return None


def _discover_cuda_home() -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return Path(_DEFAULT_CUDA_HOME)


def _discover_service() -> tuple[str, str]:
    """Find vLLM systemd service name and user."""
    try:
        result = subprocess.run(
            ["systemctl", "list-unit-files", "--type=service", "--no-pager"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            unit = parts[0]
            name = unit.removesuffix(".service")
            lowered = name.lower()
            if lowered == "vllm" or lowered.endswith("-vllm") or lowered.endswith("_vllm"):
                # Try to read User= from the unit
                show = subprocess.run(
                    ["systemctl", "show", name, "--property=User"],
                    capture_output=True, text=True, timeout=5,
                )
                user = show.stdout.strip().removeprefix("User=") or _DEFAULT_SERVICE_USER
                return name, user
    except Exception:
        pass
    return _DEFAULT_SERVICE_NAME, _DEFAULT_SERVICE_USER


def _discover_port(vllm_root: Path) -> int:
    """Try to read port from active config."""
    active = vllm_root / "configs" / "active.yaml"
    if active.exists():
        try:
            with open(active) as f:
                data = yaml.safe_load(f) or {}
            return int(str(data.get("port", _DEFAULT_PORT)))
        except Exception:
            pass
    return _DEFAULT_PORT


def _vllm_venv_bin(vllm_root: Path) -> Path:
    for name in ("venv", ".venv"):
        candidate = vllm_root / name / "bin"
        if (candidate / "vllm").exists() or (candidate / "python").exists():
            return candidate
    return vllm_root / "venv" / "bin"


def _build_config(vllm_root: Path, cuda_home: Path, service_name: str,
                  service_user: str, port: int, *,
                  gpu_index: int = 0,
                  gpu_memory_utilization: float | None = None,
                  gpu_overhead_gb: float | None = None,
                  llamacpp_root: Path | None = None,
                  llamacpp_service_name: str = "llama-cpp",
                  llamacpp_service_user: str = "llama-cpp",
                  dllm_service_name: str | None = None) -> VserveConfig:
    vllm_venv_bin = _vllm_venv_bin(vllm_root)
    backends = {
        "vllm": BackendConfig(
            name="vllm",
            root=vllm_root,
            service_name=service_name,
            service_user=service_user,
            port=port,
        ),
        "llamacpp": BackendConfig(
            name="llamacpp",
            root=llamacpp_root,
            service_name=llamacpp_service_name,
            service_user=llamacpp_service_user,
        ),
    }
    return VserveConfig(
        vllm_root=vllm_root,
        models_dir=vllm_root / "models",
        configs_dir=vllm_root / "configs" / "models",
        benchmarks_dir=vllm_root / "benchmarks",
        logs_dir=vllm_root / "logs",
        run_dir=vllm_root / "run",
        vllm_bin=vllm_venv_bin / "vllm",
        vllm_python=vllm_venv_bin / "python",
        cuda_home=cuda_home,
        service_name=service_name,
        service_user=service_user,
        port=port,
        gpu_index=gpu_index,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_overhead_gb=gpu_overhead_gb,
        llamacpp_root=llamacpp_root,
        llamacpp_service_name=llamacpp_service_name,
        llamacpp_service_user=llamacpp_service_user,
        dllm_service_name=dllm_service_name,
        backends=backends,
    )


def _coerce_config_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _coerce_config_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def load_config() -> VserveConfig:
    """Load config: file > auto-discovery > defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            _log.warning("Failed to parse config YAML %s: %s", CONFIG_FILE, exc)
        else:
            if isinstance(raw, dict):
                backend_raw = raw.get("backends")
                backends: dict[str, object] = backend_raw if isinstance(backend_raw, dict) else {}
                vllm_raw_obj = backends.get("vllm")
                llamacpp_raw_obj = backends.get("llamacpp")
                vllm_raw: dict[str, object] = vllm_raw_obj if isinstance(vllm_raw_obj, dict) else {}
                llamacpp_raw: dict[str, object] = llamacpp_raw_obj if isinstance(llamacpp_raw_obj, dict) else {}
                gpu_raw = raw.get("gpu")
                gpu: dict[str, object] = gpu_raw if isinstance(gpu_raw, dict) else {}

                root = Path(str(vllm_raw.get("root", raw.get("vllm_root", _DEFAULT_VLLM_ROOT))))
                gpu_util_raw = gpu.get("memory_utilization", raw.get("gpu_memory_utilization"))
                gpu_overhead_raw = gpu.get("overhead_gb", raw.get("gpu_overhead_gb"))
                gpu_index_raw = gpu.get("index", raw.get("gpu_index", 0))
                llamacpp_root_raw = llamacpp_raw.get("root", raw.get("llamacpp_root"))
                dllm_svc_raw = vllm_raw.get("dllm_service_name", raw.get("dllm_service_name"))
                return _build_config(
                    vllm_root=root,
                    cuda_home=Path(str(raw.get("cuda_home", _DEFAULT_CUDA_HOME))),
                    service_name=str(vllm_raw.get("service_name", raw.get("service_name", _DEFAULT_SERVICE_NAME))),
                    service_user=str(vllm_raw.get("service_user", raw.get("service_user", _DEFAULT_SERVICE_USER))),
                    port=_coerce_config_int(vllm_raw.get("port", raw.get("port")), _DEFAULT_PORT),
                    gpu_index=_coerce_config_int(gpu_index_raw, 0),
                    gpu_memory_utilization=_coerce_config_float(gpu_util_raw),
                    gpu_overhead_gb=_coerce_config_float(gpu_overhead_raw),
                    llamacpp_root=Path(str(llamacpp_root_raw)) if llamacpp_root_raw else None,
                    llamacpp_service_name=str(llamacpp_raw.get("service_name", raw.get("llamacpp_service_name", "llama-cpp"))),
                    llamacpp_service_user=str(llamacpp_raw.get("service_user", raw.get("llamacpp_service_user", "llama-cpp"))),
                    dllm_service_name=str(dllm_svc_raw) if dllm_svc_raw else None,
                )
            _log.warning("Ignoring invalid config format in %s: expected mapping", CONFIG_FILE)

    # Auto-discover
    root = _discover_vllm_root() or Path(_DEFAULT_VLLM_ROOT)
    cuda_home = _discover_cuda_home()
    service_name, service_user = _discover_service()
    port = _discover_port(root)
    return _build_config(root, cuda_home, service_name, service_user, port)


def save_config(cfg: VserveConfig) -> Path:
    """Write config to ~/.config/vserve/config.yaml."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "schema_version": 2,
        "cuda_home": str(cfg.cuda_home),
        "backends": {
            "vllm": {
                "root": str(cfg.vllm_root),
                "service_name": cfg.service_name,
                "service_user": cfg.service_user,
                "port": cfg.port,
            },
        },
    }
    if cfg.dllm_service_name:
        data["backends"]["vllm"]["dllm_service_name"] = cfg.dllm_service_name
    gpu: dict[str, float | int] = {"index": cfg.gpu_index}
    if cfg.gpu_memory_utilization is not None:
        gpu["memory_utilization"] = cfg.gpu_memory_utilization
    if cfg.gpu_overhead_gb is not None:
        gpu["overhead_gb"] = cfg.gpu_overhead_gb
    data["gpu"] = gpu
    if cfg.llamacpp_root is not None:
        data["backends"]["llamacpp"] = {
            "root": str(cfg.llamacpp_root),
            "service_name": cfg.llamacpp_service_name,
            "service_user": cfg.llamacpp_service_user,
        }
    content = "# vserve configuration — generated by vserve init\n"
    content += yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    _atomic_write_text(CONFIG_FILE, content)
    return CONFIG_FILE


# --- Lazy singleton ---

_cfg: VserveConfig | None = None


def cfg() -> VserveConfig:
    """Return the global config, loading on first access."""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def reset_config() -> None:
    """Reset the singleton — for testing."""
    global _cfg
    _cfg = None


# --- Backward-compatible module-level constants ---
# These call cfg() lazily so imports don't trigger discovery at import time.


# Backward-compatible module-level constants via __getattr__.
# `from vserve.config import MODELS_DIR` resolves lazily on first access,
# returning a real Path from cfg(). Zero changes needed in callers.
_COMPAT = {
    "MODELS_DIR": "models_dir",
    "CONFIGS_DIR": "configs_dir",
    "BENCHMARKS_DIR": "benchmarks_dir",
    "LOGS_DIR": "logs_dir",
    "VLLM_BIN": "vllm_bin",
    "VLLM_PYTHON": "vllm_python",
    "VLLM_ROOT": "vllm_root",
}


def __getattr__(name: str):  # type: ignore[override]
    if name in _COMPAT:
        return getattr(cfg(), _COMPAT[name])
    raise AttributeError(f"module 'vserve.config' has no attribute {name!r}")


def limits_path(provider: str, model_name: str) -> Path:
    return cfg().configs_dir / f"{provider}--{model_name}.limits.json"


def profile_path(provider: str, model_name: str, profile: str) -> Path:
    return cfg().configs_dir / f"{provider}--{model_name}.{profile}.yaml"


def active_yaml_path() -> Path:
    return cfg().active_yaml


def active_manifest_path() -> Path:
    return cfg().run_dir / "active-manifest.json"


def read_active_manifest(path: Path | None = None) -> dict | None:
    return _read_json_mapping(path or active_manifest_path(), warn=True, label="active manifest")


def write_active_manifest(data: dict, path: Path | None = None) -> None:
    manifest = dict(data)
    manifest.setdefault("schema_version", ACTIVE_MANIFEST_SCHEMA_VERSION)
    _atomic_write_text(path or active_manifest_path(), json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_limits(path: Path) -> dict | None:
    data = _read_json_mapping(path, warn=True, label="limits")
    if data is None:
        return None
    limits = data.get("limits")
    if not isinstance(limits, dict):
        _log.warning("Ignoring invalid limits JSON %s: missing limits object", path)
        return None
    normalized = dict(data)
    normalized_limits: dict[str, object] = {}
    for ctx, per_dtype in limits.items():
        try:
            ctx_key = str(int(str(ctx)))
        except (TypeError, ValueError):
            _log.warning("Ignoring invalid limits JSON %s: context keys must be integers", path)
            return None
        if isinstance(per_dtype, dict):
            inner: dict[str, int | None] = {}
            for dtype, value in per_dtype.items():
                if not isinstance(dtype, str):
                    _log.warning("Ignoring invalid limits JSON %s: dtype keys must be strings", path)
                    return None
                if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                    _log.warning("Ignoring invalid limits JSON %s: limits values must be integers or null", path)
                    return None
                inner[dtype] = value
            normalized_limits[ctx_key] = inner
            continue
        if per_dtype is not None and (isinstance(per_dtype, bool) or not isinstance(per_dtype, int)):
            _log.warning("Ignoring invalid limits JSON %s: flat limits values must be integers or null", path)
            return None
        normalized_limits[ctx_key] = per_dtype
    normalized["limits"] = normalized_limits
    return normalized


def write_limits(path: Path, data: dict) -> None:
    out = dict(data)
    out.setdefault("schema_version", LIMITS_SCHEMA_VERSION)
    _atomic_write_text(path, json.dumps(out, indent=2) + "\n")


def read_limits_for(provider: str, model_name: str) -> dict | None:
    """One-call helper: ``read_limits(limits_path(provider, model_name))``.

    0.6.3: collapses the 5+ inline call sites flagged by audit
    `docs/audits/2026-05-20-cli-sprawl.md`.
    """
    return read_limits(limits_path(provider, model_name))


def limits_cache_matches(data: dict | None, *, backend: str, fingerprint: dict) -> bool:
    if not data:
        return False
    if data.get("schema_version") != LIMITS_SCHEMA_VERSION:
        return False
    if data.get("backend") != backend:
        return False
    cached_fingerprint = data.get("fingerprint")
    if not isinstance(cached_fingerprint, dict):
        return False
    return cached_fingerprint == fingerprint


def read_profile_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = _read_yaml_mapping(path, warn=False, label="profile")
    if data is None:
        raise ValueError(f"Invalid profile YAML: {path}")
    return data


def try_read_profile_yaml(path: Path) -> dict | None:
    """Read profile YAML for diagnostics; return None on parse failure."""
    try:
        return read_profile_yaml(path)
    except Exception as exc:
        _log.warning("Failed to parse profile YAML %s: %s", path, exc)
        return None


def write_profile_yaml(path: Path, data: dict, comment: str = "") -> None:
    parts: list[str] = []
    if comment:
        for line in comment.splitlines():
            parts.append(f"# {line}\n")
    parts.append(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
    _atomic_write_text(path, "".join(parts), mode=0o644)
