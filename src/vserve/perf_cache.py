"""Performance cache: persisted (model, GPU, build) → measured tok/s.

The picker uses this to display real measured throughput per (context,
kv_dtype, slots) cell instead of a math estimate. A formula gets you
±20-30% on good days and 5-8× wrong on expert-spill / build-regression
cases — exactly the cases users most need to know about. We measure
instead.

Workflow:
- After ``vserve run`` reaches health-OK, a short streaming probe records
  decode_tps + ttft + e2e for the active (model, GPU, build, config).
- The picker matrix annotates cells with cache hits.
- Invalidation: cache key includes a ``build_id`` (vLLM version /
  llama.cpp commit) and ``driver``; mismatches surface "stale" and are
  hidden from the picker until refreshed.

Stored as one JSON file per cache entry under
``~/.cache/vserve-perf/<key>.json``. Atomic write + tmp-and-rename to
survive concurrent writes from multiple ``vserve run`` sessions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PerfEntry:
    """One measured datapoint for a specific (model, GPU, backend, build, config)."""

    model_path: str
    backend: str
    gpu_uuid: str
    build_id: str
    driver: str
    config_hash: str
    context: int
    kv_dtype: str
    slots: int
    decode_tps_p50: float | None
    decode_tps_p99: float | None = None
    ttft_ms_p50: float | None = None
    e2e_ms_p99: float | None = None
    measured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_count: int = 0
    served_name: str | None = None

    def cache_key(self) -> str:
        return cache_key(
            model_path=self.model_path, gpu_uuid=self.gpu_uuid,
            backend=self.backend, build_id=self.build_id,
            config_hash=self.config_hash,
        )


def _default_cache_dir() -> Path:
    """Per-user cache dir. XDG_CACHE_HOME if set, else ~/.cache."""
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        root = Path(base)
    else:
        root = Path.home() / ".cache"
    return root / "vserve-perf"


def cache_dir() -> Path:
    """Public cache directory accessor; creates it on first call."""
    d = _default_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_hash_from_cfg(cfg: dict, backend: str) -> str:
    """Stable hash for the parts of the launch config that affect throughput.

    Skips fields like ``port`` / ``host`` / sampler temperature that don't
    move tok/s. Includes context, KV dtype/k+v, parallel slots, batch sizes,
    quant, attention backend.
    """
    relevant: dict[str, object] = {}
    keys: tuple[str, ...]
    if backend == "vllm":
        keys = (
            "max-model-len", "max-num-seqs", "kv-cache-dtype",
            "max-num-batched-tokens", "quantization", "block-size",
            "attention-config", "speculative-config", "compilation-config",
        )
    else:  # llamacpp
        keys = (
            "ctx_size", "ctx_per_slot", "n_gpu_layers", "parallel",
            "cache_type_k", "cache_type_v", "batch_size", "ubatch_size",
            "n_cpu_moe", "override_tensors",
            "spec_draft", "swa_full",
        )
    for k in keys:
        if k in cfg:
            relevant[k] = cfg[k]
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_key(
    *,
    model_path: str,
    gpu_uuid: str,
    backend: str,
    build_id: str,
    config_hash: str,
) -> str:
    """Hex digest used as the JSON filename."""
    payload = "|".join((model_path, gpu_uuid, backend, build_id, config_hash))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def write_entry(entry: PerfEntry, *, directory: Path | None = None) -> Path:
    """Atomically write a PerfEntry to the cache. Returns the file path.

    Atomic write: tmp file + rename. Survives concurrent writes from
    multiple ``vserve run`` sessions targeting the same key.
    """
    d = directory if directory is not None else cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    key = entry.cache_key()
    path = d / f"{key}.json"
    payload = json.dumps(asdict(entry), indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=d, prefix=f".{key}.", suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    return path


def read_entry(path: Path) -> PerfEntry | None:
    """Read a single PerfEntry JSON, returning None on any error."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return PerfEntry(**data)
    except TypeError:
        return None


def lookup_for_picker(
    *,
    model_path: str,
    gpu_uuid: str,
    backend: str,
    build_id: str,
    directory: Path | None = None,
) -> list[PerfEntry]:
    """Return every cached entry matching (model, GPU, backend, build).

    Caller renders the cells; entries with mismatched build_id are not
    returned (we never show stale).
    """
    d = directory if directory is not None else cache_dir()
    if not d.exists():
        return []
    out: list[PerfEntry] = []
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        return []
    for f in files:
        entry = read_entry(f)
        if entry is None:
            continue
        if (
            entry.model_path == model_path
            and entry.gpu_uuid == gpu_uuid
            and entry.backend == backend
            and entry.build_id == build_id
        ):
            out.append(entry)
    return out


def lookup_one(
    *,
    model_path: str,
    gpu_uuid: str,
    backend: str,
    build_id: str,
    config_hash: str,
    directory: Path | None = None,
) -> PerfEntry | None:
    """Return the most recent measurement for an exact (model, GPU, backend,
    build, config) tuple. Returns None when no match."""
    key = cache_key(
        model_path=model_path, gpu_uuid=gpu_uuid,
        backend=backend, build_id=build_id, config_hash=config_hash,
    )
    d = directory if directory is not None else cache_dir()
    path = d / f"{key}.json"
    if not path.exists():
        return None
    return read_entry(path)


def gpu_uuid_or_index(gpu_info) -> str:
    """Best-effort stable GPU identifier. Uses UUID if NVML exposes one,
    else the integer index. Used to scope cache entries — if the user
    moves to a different GPU model, old measurements are not reused."""
    for attr in ("uuid", "gpu_uuid"):
        v = getattr(gpu_info, attr, None)
        if isinstance(v, str) and v:
            return v
    return f"idx-{getattr(gpu_info, 'index', 0)}-{getattr(gpu_info, 'name', 'unknown')}"


def vllm_build_id(runtime_info) -> str:
    """Stable build identifier for vLLM. Combines version + commit prefix
    when available; falls back to version-only or "unknown"."""
    if runtime_info is None:
        return "unknown"
    version = getattr(runtime_info, "version", None) or ""
    commit = getattr(runtime_info, "commit", None) or ""
    if version and commit:
        return f"vllm-{version}-{commit[:7]}"
    if version:
        return f"vllm-{version}"
    return "vllm-unknown"


def llamacpp_build_id(build_info) -> str:
    """Stable build identifier for llama.cpp. Uses build_number + commit
    when both present; commit-only otherwise."""
    if build_info is None:
        return "llamacpp-unknown"
    n = getattr(build_info, "build_number", None)
    c = getattr(build_info, "commit", None)
    if n and c:
        return f"llamacpp-b{n}-{c[:7]}"
    if n:
        return f"llamacpp-b{n}"
    if c:
        return f"llamacpp-{c[:7]}"
    return "llamacpp-unknown"
