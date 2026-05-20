"""Backend Protocol — the interface every inference engine must implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


@dataclass(frozen=True)
class RuntimeIdentity:
    backend: str
    executable: Path | str | None
    version: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityResult:
    backend: str
    supported: bool
    messages: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    supported_range: str | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    tools: bool = False
    reasoning: bool = False
    embedding: bool = False
    tool_parser: str | None = None
    reasoning_parser: str | None = None


@dataclass(frozen=True)
class ServiceStatus:
    backend: str
    running: bool | None
    service_name: str
    health_url: str | None = None
    error: str | None = None


BackendConfig = dict[str, Any]
TuningResult = dict[str, Any]
ActiveManifest = Mapping[str, Any]
CacheFingerprint = Mapping[str, Any]
ProfileData = Mapping[str, Any]


class Choices(TypedDict, total=False):
    """Canonical keys for the ``choices`` dict passed to
    :py:meth:`Backend.build_config`.

    0.6.3: codifies the contract the audit
    (`docs/audits/2026-05-20-backend-consistency.md` finding #2) flagged
    as drifting between backends. Existing call sites still pass plain
    dicts; this TypedDict is a documentation + type-check hook for
    future call sites to use ``Choices(context=..., slots=...)``.

    Per-backend specifics:

    * **Both backends**: ``context``, ``port``, ``tools``, ``tool_parser``,
      ``reasoning_parser``, ``thinking``, ``chat_template_kwargs``,
      ``embedding``, ``pooling``.
    * **vLLM only** (silently dropped by llama.cpp):
      ``kv_dtype`` (use ``kv_cache_k`` / ``kv_cache_v`` on llama.cpp),
      ``slots`` (canonical name; llama.cpp uses ``parallel`` as an
      alias for the same concept),
      ``batched_tokens``, ``gpu_mem_util``, ``trust_remote_code``,
      ``performance_mode``, ``optimization_level``, ``block_size``,
      ``kv_cache_memory_bytes``, ``enable_prefix_caching``,
      ``attention_backend``, ``gpu_compute_cap``.
    * **llama.cpp only** (silently dropped by vLLM):
      ``parallel`` (alias for ``slots``), ``n_gpu_layers``,
      ``kv_cache_k``, ``kv_cache_v``, ``batch_size``, ``ubatch_size``,
      ``override_tensors``, ``mmap``, ``cache_reuse``, ``cram_mb``,
      ``slot_save_path``, ``swa_full``, ``n_cpu_moe``,
      ``reasoning_budget``, ``mmproj``, ``reasoning_format``,
      ``spec_draft``.
    """

    # Both backends
    context: int
    port: int
    tools: bool
    tool_parser: str | None
    reasoning_parser: str | None
    thinking: bool | None
    chat_template_kwargs: dict[str, Any]
    embedding: bool
    pooling: str | None

    # vLLM-side (slots is canonical; llama.cpp aliases as `parallel`)
    slots: int
    kv_dtype: str
    batched_tokens: int | None
    gpu_mem_util: float
    trust_remote_code: bool
    performance_mode: str | None
    optimization_level: str | None
    block_size: int | None
    kv_cache_memory_bytes: int | None
    enable_prefix_caching: bool
    attention_backend: str | None
    gpu_compute_cap: int | None

    # llama.cpp-side
    parallel: int
    n_gpu_layers: int
    kv_cache_k: str
    kv_cache_v: str
    batch_size: int | None
    ubatch_size: int | None
    override_tensors: list[str]
    mmap: bool
    cache_reuse: int | None
    cram_mb: int | None
    slot_save_path: str | None
    swa_full: bool
    n_cpu_moe: int | None
    reasoning_budget: int | None
    mmproj: str | None
    reasoning_format: str | None
    spec_draft: dict[str, Any] | None


@runtime_checkable
class Backend(Protocol):
    """What every inference backend must provide."""

    name: str
    display_name: str

    @property
    def service_name(self) -> str:
        """Systemd unit name for this backend."""
        ...

    @property
    def service_user(self) -> str:
        """System user that owns this backend service."""
        ...

    @property
    def root_dir(self) -> Path:
        """Root directory for this backend (e.g. /opt/vllm, /opt/llama-cpp)."""
        ...

    def can_serve(self, model: ModelInfo) -> bool:
        """Can this backend serve this model? (format check)"""
        ...

    def find_entrypoint(self) -> Path | str | None:
        """Locate the server binary or launch command. None = not installed."""
        ...

    def runtime_info(self) -> RuntimeIdentity:
        """Collect version and dependency facts for this backend runtime.

        0.6.3: tightened from ``RuntimeIdentity | Any`` to ``RuntimeIdentity``
        once llamacpp.runtime_info was migrated off the ad-hoc dict shape.
        """
        ...

    def compatibility(self) -> CompatibilityResult:
        """Return this backend's runtime compatibility check.

        0.6.3: tightened from ``CompatibilityResult | Any`` to
        ``CompatibilityResult`` once llamacpp.compatibility was migrated
        off the ad-hoc dict shape.
        """
        ...

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float) -> TuningResult:
        """Calculate optimal serving parameters. Returns limits dict."""
        ...

    def build_config(self, model: ModelInfo, choices: dict) -> BackendConfig:
        """Build serving config from interactive wizard choices."""
        ...

    def quant_flag(self, method: str | None) -> str:
        """Return CLI quantization flag string. Empty if N/A."""
        ...

    def start(self, config_path: Path, *, non_interactive: bool = False) -> None:
        """Start the inference server with the given config."""
        ...

    def stop(self, *, non_interactive: bool = False) -> None:
        """Stop the inference server."""
        ...

    def is_running(self) -> bool:
        """Check if the inference server is currently active."""
        ...

    def health_url(self, port: int) -> str:
        """Return the health check URL for this backend."""
        ...

    def active_manifest_path(self) -> Path:
        """Return the backend-owned manifest path for active runtime state."""
        ...

    def detect_tools(self, model_path: Path) -> Mapping[str, Any]:
        """Detect tool calling capabilities. Keys vary by backend."""
        ...

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        """Return (description, check_fn) pairs for diagnostics."""
        ...
