"""Runtime metadata, compatibility checks, and upgrade helpers."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from vserve.model_files import is_weight_file_name, iter_recursive_files_with_suffix

SUPPORTED_VLLM_RANGE = ">=0.20,<0.23"
# 0.22 is supported, but the pinned "stable" runtime stays 0.21 — it serves
# every bundled model unflagged, whereas Gemma-4 NVFP4 needs extra multimodal
# caps on 0.22 (the video-profiling OOM; vserve auto-caps land in 0.6.4).
PINNED_STABLE_VLLM = "0.21.0"
DETECTOR_SCHEMA_VERSION = 2
_VLLM_MIN = Version("0.20")
_VLLM_MAX = Version("0.23")

# vLLM 0.22 renamed --chat-template-kwargs to --default-chat-template-kwargs
# and deprecated the FlashInfer MoE env vars (removal targeted 0.23). Every
# version gate in vserve compares against this single boundary.
VLLM_FLAG_MIGRATION_VERSION = Version("0.22")

RUNTIME_CACHE_DIR = Path.home() / ".cache" / "vserve" / "runtime"
VLLM_RUNTIME_CACHE_FILE = RUNTIME_CACHE_DIR / "vllm.json"
_RUNTIME_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeInfo:
    """Facts collected from an external inference runtime."""

    backend: str
    executable: Path | None
    python: Path | None
    vllm_version: str | None = None
    torch_version: str | None = None
    torch_cuda: str | None = None
    transformers_version: str | None = None
    huggingface_hub_version: str | None = None
    pip_check_ok: bool | None = None
    pip_check_output: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)

    def fingerprint(self) -> dict[str, str | None]:
        """Return runtime fields that should invalidate tuning caches when they drift."""
        return {
            "vllm_version": self.vllm_version,
            "torch_version": self.torch_version,
            "torch_cuda": self.torch_cuda,
            "transformers_version": self.transformers_version,
        }


@dataclass(frozen=True)
class CompatibilityCheck:
    """Result of checking an installed runtime against vserve's support policy."""

    backend: str
    range: str
    supported: bool
    messages: list[str]
    warnings: list[str]
    errors: list[str]


_METADATA_SCRIPT = (
    "import importlib.metadata as m, json\n"
    "def version(name):\n"
    "    try:\n"
    "        return m.version(name)\n"
    "    except Exception:\n"
    "        return None\n"
    "torch_version = version('torch')\n"
    "torch_cuda = None\n"
    "try:\n"
    "    import torch\n"
    "    torch_cuda = getattr(torch.version, 'cuda', None)\n"
    "except Exception:\n"
    "    pass\n"
    "print(json.dumps({\n"
    "    'vllm': version('vllm'),\n"
    "    'torch': torch_version,\n"
    "    'torch_cuda': torch_cuda,\n"
    "    'transformers': version('transformers'),\n"
    "    'huggingface_hub': version('huggingface-hub'),\n"
    "}))\n"
)


def _parse_version_text(text: str) -> str | None:
    match = re.search(r"(?i)vllm\s+([0-9][^\s]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"([0-9]+(?:\.[0-9]+)+(?:[a-z0-9.+-]+)?)", text)
    return match.group(1) if match else None


def _vllm_runtime_cache_key(vllm_bin: Path, vllm_python: Path) -> str | None:
    """Cache key derived from venv contents — any pip install changes it.

    Uses the unresolved vllm_python path because resolving the symlink chain
    (vllm_python → python3 → /usr/bin/python3.12) lands outside the venv. The
    venv layout we care about is always `<venv>/bin/python` paired with
    `<venv>/lib/pythonX.Y/site-packages`.
    """
    paths: list[Path] = [vllm_bin, vllm_python]
    try:
        venv_lib = vllm_python.parent.parent / "lib"
        if venv_lib.exists():
            for child in venv_lib.iterdir():
                sp = child / "site-packages"
                if sp.exists():
                    paths.append(sp)
                    break
    except OSError:
        pass
    parts: list[str] = []
    for path in paths:
        try:
            parts.append(f"{path}:{path.stat().st_mtime_ns}")
        except OSError:
            return None
    return "|".join(parts)


def _read_vllm_runtime_cache(cache_path: Path, *, expected_key: str) -> RuntimeInfo | None:
    """Return a cached RuntimeInfo if the cache file matches expected_key."""
    try:
        text = cache_path.read_text()
    except OSError:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != _RUNTIME_CACHE_SCHEMA_VERSION:
        return None
    if data.get("cache_key") != expected_key:
        return None
    executable = data.get("executable")
    python = data.get("python")
    errors_raw = data.get("errors") or []
    errors = tuple(str(e) for e in errors_raw) if isinstance(errors_raw, (list, tuple)) else ()
    return RuntimeInfo(
        backend="vllm",
        executable=Path(executable) if isinstance(executable, str) else None,
        python=Path(python) if isinstance(python, str) else None,
        vllm_version=data.get("vllm_version"),
        torch_version=data.get("torch_version"),
        torch_cuda=data.get("torch_cuda"),
        transformers_version=data.get("transformers_version"),
        huggingface_hub_version=data.get("huggingface_hub_version"),
        # pip_check intentionally not cached — fresh probe is required when needed
        pip_check_ok=None,
        pip_check_output="",
        errors=errors,
    )


def _write_vllm_runtime_cache(cache_path: Path, *, cache_key: str, info: RuntimeInfo) -> None:
    """Best-effort write of the runtime cache. Never raises."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    payload = {
        "schema_version": _RUNTIME_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "executable": str(info.executable) if info.executable else None,
        "python": str(info.python) if info.python else None,
        "vllm_version": info.vllm_version,
        "torch_version": info.torch_version,
        "torch_cuda": info.torch_cuda,
        "transformers_version": info.transformers_version,
        "huggingface_hub_version": info.huggingface_hub_version,
        "errors": list(info.errors),
    }
    try:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, cache_path)
    except OSError:
        pass


def invalidate_vllm_runtime_cache(cache_path: Path | None = None) -> None:
    """Remove the cached vLLM runtime info. Safe if the file is missing."""
    path = cache_path or VLLM_RUNTIME_CACHE_FILE
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def installed_vllm_version(config: Any | None = None) -> Version | None:
    """Best-effort parsed version of the installed vLLM runtime.

    Fast path only (cached probe, no pip check). Returns None when the
    runtime is missing or the version doesn't parse — callers MUST treat
    None as "behave like pre-0.22" so a broken probe can only ever cause
    deprecation warnings, never a launch failure.
    """
    info = collect_vllm_runtime_info(config, prefer_cache=True, with_pip_check=False)
    if not info.vllm_version:
        return None
    try:
        return Version(info.vllm_version)
    except InvalidVersion:
        return None


def collect_vllm_runtime_info(
    config: Any | None = None,
    *,
    prefer_cache: bool = False,
    with_pip_check: bool = True,
    cache_path: Path | None = None,
) -> RuntimeInfo:
    """Collect vLLM, Python package, and (optionally) dependency-health facts.

    On the hot path (`vserve run`), pass `prefer_cache=True, with_pip_check=False`
    — this serves a cached RuntimeInfo when the venv mtimes haven't changed and
    skips the slow `pip check` step entirely. `vserve runtime check vllm` and
    `vserve doctor` always run the full probe with `pip check`.
    """
    if config is None:
        from vserve.config import cfg

        config = cfg()

    vllm_bin = Path(config.vllm_bin)
    vllm_python = Path(config.vllm_python)
    cache_file = cache_path or VLLM_RUNTIME_CACHE_FILE

    if prefer_cache:
        cache_key = _vllm_runtime_cache_key(vllm_bin, vllm_python)
        if cache_key is not None:
            cached = _read_vllm_runtime_cache(cache_file, expected_key=cache_key)
            if cached is not None:
                return cached

    errors: list[str] = []
    cli_version: str | None = None
    metadata: dict[str, Any] = {}
    pip_check_ok: bool | None = None
    pip_check_output = ""

    try:
        result = subprocess.run(
            [str(vllm_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip() or result.stderr.strip()
        if result.returncode == 0:
            cli_version = _parse_version_text(output)
        else:
            errors.append(output or f"{vllm_bin} --version failed")
    except Exception as exc:
        errors.append(f"{vllm_bin} --version failed: {exc}")

    try:
        result = subprocess.run(
            [str(vllm_python), "-c", _METADATA_SCRIPT],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            metadata = json.loads(result.stdout.strip() or "{}")
        else:
            details = result.stderr.strip() or result.stdout.strip()
            errors.append(details or f"{vllm_python} metadata probe failed")
    except Exception as exc:
        errors.append(f"{vllm_python} metadata probe failed: {exc}")

    if with_pip_check:
        try:
            result = subprocess.run(
                [str(vllm_python), "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            pip_check_ok = result.returncode == 0
            pip_check_output = result.stdout.strip() or result.stderr.strip()
        except Exception as exc:
            pip_check_ok = False
            pip_check_output = str(exc)

    info = RuntimeInfo(
        backend="vllm",
        executable=vllm_bin,
        python=vllm_python,
        vllm_version=metadata.get("vllm") or cli_version,
        torch_version=metadata.get("torch"),
        torch_cuda=metadata.get("torch_cuda"),
        transformers_version=metadata.get("transformers"),
        huggingface_hub_version=metadata.get("huggingface_hub"),
        pip_check_ok=pip_check_ok,
        pip_check_output=pip_check_output,
        errors=tuple(errors),
    )

    # Refresh cache when we just probed live, even on the with_pip_check path —
    # the cached fields (versions, executable) don't include pip_check state.
    if not info.errors and info.vllm_version:
        cache_key = _vllm_runtime_cache_key(vllm_bin, vllm_python)
        if cache_key is not None:
            _write_vllm_runtime_cache(cache_file, cache_key=cache_key, info=info)

    return info


def check_vllm_compatibility(info: RuntimeInfo) -> CompatibilityCheck:
    """Check vLLM runtime against vserve's pinned support policy."""
    messages: list[str] = []
    warnings: list[str] = []
    raw_errors = getattr(info, "errors", ())
    errors = list(raw_errors) if isinstance(raw_errors, (list, tuple)) else []

    if not info.vllm_version:
        errors.append("Could not determine vLLM version.")
    else:
        try:
            parsed = Version(info.vllm_version)
        except InvalidVersion:
            errors.append(f"Invalid vLLM version: {info.vllm_version}")
        else:
            if parsed.is_prerelease or parsed.is_devrelease:
                errors.append(f"vLLM pre-release/dev build is unsupported: {info.vllm_version}")
            if not (_VLLM_MIN <= parsed < _VLLM_MAX):
                errors.append(f"vLLM {info.vllm_version} is outside supported range {SUPPORTED_VLLM_RANGE}.")
            if not errors:
                messages.append(f"vLLM {info.vllm_version} is within supported range {SUPPORTED_VLLM_RANGE}.")

    if info.pip_check_ok is False:
        errors.append(f"pip check failed: {info.pip_check_output or 'dependency conflicts found'}")
    elif info.pip_check_ok is None:
        warnings.append("pip check was not run.")

    if not info.torch_version:
        warnings.append("Could not determine torch version.")
    if not info.transformers_version:
        warnings.append("Could not determine transformers version.")

    return CompatibilityCheck(
        backend="vllm",
        range=SUPPORTED_VLLM_RANGE,
        supported=not errors,
        messages=messages,
        warnings=warnings,
        errors=errors,
    )


def upgrade_vllm_stable(config: Any | None = None, *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Force-reinstall the supported stable vLLM runtime into the configured venv."""
    if config is None:
        from vserve.config import cfg

        config = cfg()
    vllm_python = Path(config.vllm_python)
    if not vllm_python.exists():
        raise RuntimeError(f"vLLM Python not found at {vllm_python}")
    result = subprocess.run(
        [
            str(vllm_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            f"vllm=={PINNED_STABLE_VLLM}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or f"pip install vllm=={PINNED_STABLE_VLLM} failed")
    invalidate_vllm_runtime_cache()
    return result


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenizer_template_hash(model_path: Path) -> str | None:
    path = model_path / "tokenizer_config.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or "chat_template" not in data:
        return None
    template = data["chat_template"]
    if isinstance(template, str):
        return _sha256_text(template)
    try:
        normalized = json.dumps(template, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return None
    return _sha256_text(normalized)


def _model_file_identity(model_path: Path) -> list[dict[str, int | str]]:
    identity: list[dict[str, int | str]] = []
    if not model_path.exists():
        return identity
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_path).as_posix()
        if not (is_weight_file_name(rel) or rel.lower().endswith(".index.json")):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        identity.append({
            "path": rel,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return identity


def _gguf_metadata_hash(model_path: Path) -> str | None:
    gguf_files = iter_recursive_files_with_suffix(model_path, ".gguf")
    if not gguf_files:
        return None
    digest = hashlib.sha256()
    for path in gguf_files:
        digest.update(path.relative_to(model_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            with open(path, "rb") as f:
                digest.update(f.read(1024 * 1024))
        except OSError:
            continue
    return digest.hexdigest()


def build_tuning_fingerprint(
    *,
    model_info: Any,
    gpu: Any,
    backend: str,
    gpu_mem_util: float,
    runtime_info: Any | None = None,
) -> dict[str, Any]:
    """Return the inputs that make a cached tuning table valid."""
    fingerprint: dict[str, Any] = {
        "backend": backend,
        "model_path": str(model_info.path),
        "quant_method": model_info.quant_method,
        "architecture": model_info.architecture,
        "is_moe": model_info.is_moe,
        "max_position_embeddings": model_info.max_position_embeddings,
        "num_kv_heads": model_info.num_kv_heads,
        "head_dim": model_info.head_dim,
        "num_layers": model_info.num_layers,
        "model_size_gb": model_info.model_size_gb,
        "gpu_name": getattr(gpu, "name", None),
        "gpu_index": getattr(gpu, "index", None),
        "gpu_driver": getattr(gpu, "driver", None),
        "gpu_cuda": getattr(gpu, "cuda", None),
        "vram_total_gb": getattr(gpu, "vram_total_gb", None),
        "gpu_memory_utilization": gpu_mem_util,
        "detector_schema_version": DETECTOR_SCHEMA_VERSION,
        "tokenizer_template_hash": _tokenizer_template_hash(Path(model_info.path)),
        "gguf_metadata_hash": _gguf_metadata_hash(Path(model_info.path)),
        "model_file_identity": _model_file_identity(Path(model_info.path)),
        "vllm_version": None,
        "torch_version": None,
        "torch_cuda": None,
        "transformers_version": None,
        "runtime_executable": None,
        "llama_server_version": None,
    }
    if runtime_info is not None:
        if isinstance(runtime_info, dict):
            fingerprint["runtime_executable"] = runtime_info.get("executable")
            fingerprint["llama_server_version"] = runtime_info.get("llama_server_version")
        else:
            runtime_fingerprint = runtime_info.fingerprint()
            if isinstance(runtime_fingerprint, dict):
                fingerprint.update(runtime_fingerprint)
    return fingerprint
