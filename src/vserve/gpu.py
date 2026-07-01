"""GPU info via nvidia-smi."""

import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    driver: str
    cuda: str
    # Compute capability as a packed integer (e.g. 120 = sm120 / Blackwell
    # RTX, 100 = sm100 / Blackwell DC, 90 = sm90 / Hopper). None when nvidia-smi
    # didn't surface it.
    compute_cap: int | None = None

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def vram_used_gb(self) -> float:
        return self.vram_used_mb / 1024


def _run_nvidia_smi() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,utilization.gpu,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


def _get_cuda_version() -> str:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    for line in result.stdout.splitlines():
        if "CUDA Version:" in line:
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                return parts[1].strip().split()[0]
    return "unknown"


def get_gpu_info(config: object | None = None) -> GpuInfo:
    if config is None:
        try:
            from vserve.config import cfg

            config = cfg()
        except Exception:
            config = None
    gpu_index = int(getattr(config, "gpu_index", 0) or 0)
    raw = _run_nvidia_smi()
    rows = [line for line in raw.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    if gpu_index < 0 or gpu_index >= len(rows):
        raise ValueError(f"Configured GPU index {gpu_index} is unavailable; found {len(rows)} GPU(s)")
    parts = [p.strip() for p in rows[gpu_index].split(",")]
    if len(parts) < 5:
        raise RuntimeError(f"Could not parse nvidia-smi GPU row: {rows[gpu_index]}")
    name = parts[0]
    vram_total = int(parts[1])
    vram_used = int(parts[3])
    driver = parts[4]
    cuda = _get_cuda_version()
    # compute_cap is "8.6", "9.0", "12.0" etc.; convert to packed int
    # (12.0 → 120, 9.0 → 90). Missing/garbled values silently fall through.
    compute_cap: int | None = None
    if len(parts) >= 6:
        raw_cc = parts[5].strip()
        try:
            major_str, _, minor_str = raw_cc.partition(".")
            major = int(major_str)
            minor = int(minor_str or "0")
            compute_cap = major * 10 + minor
        except (TypeError, ValueError):
            compute_cap = None
    return GpuInfo(
        index=gpu_index,
        name=name,
        vram_total_mb=vram_total,
        vram_used_mb=vram_used,
        vram_free_mb=vram_total - vram_used,
        driver=driver,
        cuda=cuda,
        compute_cap=compute_cap,
    )


# Base VRAM (GiB) reserved before KV cache for the CUDA context, sampler, and a
# floor of activation / CUDA-graph memory. Model-class extras (MoE, multimodal) are
# added on top by resolve_gpu_memory_utilization via models.runtime_headroom_gb.
DEFAULT_GPU_OVERHEAD_GB = 3.5


def compute_gpu_memory_utilization(
    vram_total_gb: float, overhead_gb: float = DEFAULT_GPU_OVERHEAD_GB
) -> float:
    """Reserve overhead for CUDA context, activations, CUDA graphs, and sampler."""
    if vram_total_gb <= 0:
        raise ValueError(f"VRAM total must be positive to compute GPU memory utilization, got {vram_total_gb:.3f} GB")
    return (vram_total_gb - overhead_gb) / vram_total_gb


def resolve_gpu_memory_utilization(
    vram_total_gb: float,
    *,
    requested: float | None = None,
    config: object | None = None,
    model: object | None = None,
) -> float:
    """Resolve the effective GPU memory policy used by add, tune, and run.

    When ``model`` is given and the util is auto-derived (no ``requested`` and no
    configured util), MoE / multimodal models get extra headroom on top of the base
    overhead so the auto util doesn't OOM under concurrent load or encoder profiling.
    """
    def _validate(value: float, label: str) -> float:
        if not 0.5 <= value <= 0.99:
            raise ValueError(f"{label} must resolve to a GPU memory utilization between 0.5 and 0.99, got {value:.3f}")
        return value

    if requested is not None:
        return _validate(float(requested), "--gpu-util")
    configured_util = getattr(config, "gpu_memory_utilization", None)
    if configured_util is not None:
        return _validate(float(configured_util), "gpu.memory_utilization")
    configured_overhead = getattr(config, "gpu_overhead_gb", None)
    overhead_gb = (
        float(configured_overhead)
        if configured_overhead is not None
        else DEFAULT_GPU_OVERHEAD_GB
    )
    # Model-class headroom (MoE / multimodal) on top of the base overhead: the auto
    # util must leave room for transient activation / CUDA-graph / autotuner memory
    # or it OOMs under concurrent load. An explicit --gpu-util or a configured
    # gpu.memory_utilization (both handled above) intentionally skip this.
    from vserve.models import runtime_headroom_gb
    overhead_gb += runtime_headroom_gb(model)
    return _validate(
        compute_gpu_memory_utilization(vram_total_gb, overhead_gb),
        "gpu.overhead_gb",
    )


def _configured_gpu_index() -> int:
    try:
        from vserve.config import cfg

        return int(getattr(cfg(), "gpu_index", 0) or 0)
    except Exception:
        return 0


def get_fan_speed() -> int:
    """Read current fan speed via NVML."""
    import pynvml  # type: ignore[import-untyped]
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(_configured_gpu_index())
        return pynvml.nvmlDeviceGetFanSpeed_v2(h, 0)
    finally:
        pynvml.nvmlShutdown()


def get_fan_count() -> int:
    """Return the number of fans exposed by NVML for the primary GPU."""
    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(_configured_gpu_index())
        return pynvml.nvmlDeviceGetNumFans(h)
    finally:
        pynvml.nvmlShutdown()


def set_fan_speed(percent: int) -> None:
    """Set GPU fan to a fixed percentage (30-100) via NVML. Requires root."""
    if not 30 <= percent <= 100:
        raise ValueError(f"Fan speed must be 30-100, got {percent}")

    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(_configured_gpu_index())
        for i in range(pynvml.nvmlDeviceGetNumFans(h)):
            pynvml.nvmlDeviceSetFanSpeed_v2(h, i, percent)
    finally:
        pynvml.nvmlShutdown()


def restore_fan_auto() -> None:
    """Restore automatic (vBIOS-controlled) fan control via NVML. Requires root."""
    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(_configured_gpu_index())
        restore_default = getattr(pynvml, "nvmlDeviceSetDefaultFanSpeed_v2", None)
        for i in range(pynvml.nvmlDeviceGetNumFans(h)):
            if restore_default is not None:
                try:
                    restore_default(h, i)
                    continue
                except Exception:
                    pass
            pynvml.nvmlDeviceSetFanControlPolicy(h, i, 0)
    finally:
        pynvml.nvmlShutdown()
