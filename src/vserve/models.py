"""Model scanning, detection, and fuzzy matching."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from vserve.model_files import iter_top_level_files_with_suffix

_log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    path: Path
    provider: str
    model_name: str
    architecture: str
    model_type: str
    quant_method: str | None
    max_position_embeddings: int
    is_moe: bool
    model_size_gb: float
    num_kv_heads: int | None = None
    num_layers: int | None = None
    head_dim: int | None = None
    is_gguf: bool = False
    quant_tier: str | None = None  # Unsloth Dynamic 2.0 tier (e.g. Q4_K_XL, IQ4_XS, MXFP4_MOE)
    mmproj_path: Path | None = None  # llama.cpp vision/audio projector (mmproj-*.gguf)
    global_head_dim: int | None = None  # Gemma-4 global-attn layer head_dim (when != head_dim)
    is_multimodal: bool = False  # has a vision/audio tower (needs encoder-profiling VRAM headroom)

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model_name}"

    @property
    def is_embedding(self) -> bool:
        name = self.model_name.lower()
        arch = self.architecture.lower()
        return (
            "embed" in name or "embed" in arch
            or "e5" in name.split("-")
            or "bge" in name.split("-")
            or "rerank" in name
        )

    @property
    def is_unsloth_ud(self) -> bool:
        """True when this is an Unsloth Dynamic 2.0 (UD-) quantized GGUF.

        Unsloth's UD- quants (Q4_K_XL, Q5_K_XL, IQ4_XS variants in the
        unsloth/* repos) are imatrix-calibrated against calibration_v5,
        documented at ~99.9% KL divergence vs FP16. Detection is filename-
        based — the UD- prefix marks the calibrated build.
        """
        if not self.is_gguf or self.provider.lower() != "unsloth":
            return False
        # Match any GGUF file in the model dir whose name contains a
        # `UD-` prefix on the quant tag (e.g. `model-UD-Q4_K_XL.gguf`).
        try:
            for path in self.path.iterdir():
                if path.suffix.lower() == ".gguf" and "-UD-" in path.name:
                    return True
        except OSError:
            return False
        return False


QUANT_FLAGS = {
    "gptq":             "--quantization gptq_marlin",
    "awq":              "--quantization awq_marlin",
    "fp8":              "--quantization fp8",
    "compressed-tensors": "",
    "none":             "",
    # Item Q: vLLM 0.21+ quant variants — NVFP4 (Blackwell), MXFP4 (MoE),
    # bitsandbytes (single-card 4-bit), gguf (vLLM can serve GGUF directly).
    "nvfp4":            "--quantization nvfp4",
    "modelopt":         "--quantization modelopt",  # ModelOpt checkpoint NVFP4
    "mxfp4":            "--quantization mxfp4",
    "mxfp4_moe":        "--quantization mxfp4",
    "bitsandbytes":     "--quantization bitsandbytes",
    "gguf":             "--quantization gguf",
}


# Item Q: environment-variable plumbing per quant method. Emitted into the
# systemd EnvironmentFile when the model uses one of these methods.
# Sources: vLLM 0.21 release notes (FlashInfer MoE FP4 backend), R3 (Unsloth
# Dynamic FP8 / NVFP4 artifacts).
# vLLM >=0.22 deprecates these env vars (still deprecated-not-removed as of
# 0.23 — warn-and-redirect) in favor of the hardware-aware --moe-backend
# flag; serve.py suppresses them on a known >=0.22 runtime (see
# _resolve_quant_envs) and `vserve run --moe-backend` is the explicit
# replacement knob.
QUANT_ENV_VARS: dict[str, dict[str, str]] = {
    "nvfp4": {
        "VLLM_USE_FLASHINFER_MOE_FP4": "1",
        "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
    },
    "modelopt": {
        "VLLM_USE_FLASHINFER_MOE_FP4": "1",
        "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
    },
    # MXFP4 uses the Humming backend by default (vLLM 0.21/0.22; the 0.22
    # flag-world equivalent is --moe-backend humming) — no env vars
    # required, but reserve the entry for future per-backend tuning.
    "mxfp4": {},
    "mxfp4_moe": {},
}


# Extra VRAM (GiB) to hold back from the KV pool, on top of the base CUDA/context
# overhead, for transient runtime memory the static KV math can't see: per-step
# activations, the CUDA-graph capture pool, and kernel autotuner (FlashInfer fp8 /
# NVFP4) workspace. These spike under *concurrent* load (MoE expert dispatch) and at
# multimodal encoder profiling — the two ways an over-filled GPU OOMs at serve time.
# Calibrated on RTX PRO 5000 (48 GB, sm120): MoE-only reserve lands util ~0.895 and
# MoE+vision ~0.843 — matching the empirically-stable 0.90 / 0.85 configs.
_MOE_RUNTIME_HEADROOM_GB = 1.5
_MULTIMODAL_RUNTIME_HEADROOM_GB = 2.5


def runtime_headroom_gb(model: object | None = None) -> float:
    """Model-class VRAM headroom (GiB) to reserve beyond the base overhead.

    Returned value is *added* to the GPU overhead before deriving the auto
    gpu-memory-utilization, so MoE / multimodal models get the transient-memory
    slack that a single fixed reserve doesn't. Zero for plain dense text models
    (and when ``model`` is None, preserving the prior behaviour).
    """
    if model is None:
        return 0.0
    reserve = 0.0
    if getattr(model, "is_moe", False):
        reserve += _MOE_RUNTIME_HEADROOM_GB
    if getattr(model, "is_multimodal", False):
        reserve += _MULTIMODAL_RUNTIME_HEADROOM_GB
    return reserve


def recommend_quant_for_arch(sm: int, is_moe: bool, available_quants: set[str] | None = None) -> str | None:
    """Recommend the canonical quant variant for a (compute_cap, MoE) pair.

    Compute capability mapping:
      sm120 — Blackwell RTX consumer (PRO 5000) → NVFP4 dense, MXFP4 MoE
      sm100 — Blackwell datacenter (B200) → NVFP4 dense, MXFP4 MoE
      sm90  — Hopper (H100) → FP8 dense / FP8 MoE
      sm89  — Ada (4090/A6000-Ada) → AWQ-Marlin
      <89   — older → GGUF or fp16

    ``available_quants`` filters by what the model ships; pass None to
    return the "ideal" recommendation regardless of availability.
    """
    if sm >= 100:
        primary = "mxfp4" if is_moe else "nvfp4"
        if available_quants is None or primary in available_quants:
            return primary
        # Fall through to FP8 if NVFP4 not packaged for this model.
        if available_quants is not None and "fp8" in available_quants:
            return "fp8"
    if sm >= 90:
        if available_quants is None or "fp8" in available_quants:
            return "fp8"
        if available_quants is not None and "awq" in available_quants:
            return "awq"
    if sm >= 89:
        if available_quants is None or "awq" in available_quants:
            return "awq"
    if available_quants is None or "gguf" in available_quants:
        return "gguf"
    return None


# Unsloth Dynamic 2.0 quant tiers (https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).
# bits_per_weight is approximate effective bits including imatrix overhead.
# min_vram_gb_per_b is a rough VRAM-per-billion-param recommendation.
# quality_rank is a relative ordering (higher = closer to BF16 reference output).
UNSLOTH_QUANT_TIERS: dict[str, dict] = {
    "TQ1_0":     {"bits_per_weight": 1.6, "min_vram_gb_per_b": 0.20, "quality_rank": 1},
    "Q2_K_XL":   {"bits_per_weight": 2.8, "min_vram_gb_per_b": 0.35, "quality_rank": 2},
    "Q3_K_XL":   {"bits_per_weight": 3.4, "min_vram_gb_per_b": 0.45, "quality_rank": 3},
    "IQ4_XS":    {"bits_per_weight": 4.2, "min_vram_gb_per_b": 0.55, "quality_rank": 4},
    "IQ4_NL":    {"bits_per_weight": 4.5, "min_vram_gb_per_b": 0.58, "quality_rank": 5},
    "Q4_K_M":    {"bits_per_weight": 4.5, "min_vram_gb_per_b": 0.60, "quality_rank": 5},
    "Q4_K_XL":   {"bits_per_weight": 4.7, "min_vram_gb_per_b": 0.62, "quality_rank": 6},
    "Q5_K_M":    {"bits_per_weight": 5.5, "min_vram_gb_per_b": 0.70, "quality_rank": 7},
    "Q5_K_XL":   {"bits_per_weight": 5.7, "min_vram_gb_per_b": 0.72, "quality_rank": 8},
    "Q6_K":      {"bits_per_weight": 6.5, "min_vram_gb_per_b": 0.82, "quality_rank": 9},
    "Q6_K_XL":   {"bits_per_weight": 6.7, "min_vram_gb_per_b": 0.84, "quality_rank": 9},
    "Q8_0":      {"bits_per_weight": 8.5, "min_vram_gb_per_b": 1.05, "quality_rank": 10},
    "Q8_K_XL":   {"bits_per_weight": 8.5, "min_vram_gb_per_b": 1.10, "quality_rank": 10},
    "MXFP4_MOE": {"bits_per_weight": 4.0, "min_vram_gb_per_b": 0.50, "quality_rank": 6, "moe_only": True},
    "BF16":      {"bits_per_weight": 16.0, "min_vram_gb_per_b": 2.10, "quality_rank": 12},
    "F16":       {"bits_per_weight": 16.0, "min_vram_gb_per_b": 2.10, "quality_rank": 12},
}

_UD_TIER_PATTERN = re.compile(
    r"-UD-(?P<tier>TQ1_0|Q2_K_XL|Q3_K_XL|IQ4_XS|IQ4_NL|Q4_K_M|Q4_K_XL|Q5_K_M|Q5_K_XL|"
    r"Q6_K_XL|Q6_K|Q8_K_XL|Q8_0|MXFP4_MOE|BF16|F16)",
    re.IGNORECASE,
)


def parse_unsloth_quant_tier(filename: str) -> str | None:
    """Extract the Unsloth Dynamic 2.0 quant tier from a GGUF filename.

    Returns the canonical tier name (e.g. ``Q4_K_XL``, ``IQ4_XS``, ``MXFP4_MOE``)
    when the filename contains a ``-UD-<tier>`` segment, else None.
    """
    m = _UD_TIER_PATTERN.search(filename)
    return m.group("tier").upper() if m else None


def find_mmproj(model_dir: Path) -> Path | None:
    """Locate the multimodal projector GGUF (mmproj-*.gguf) in a model dir.

    Gemma-4 / Llava-style multimodal models ship a separate projector GGUF.
    Without ``--mmproj`` llama.cpp silently serves them as text-only. Quantized
    mmproj variants reportedly produce garbage; we prefer BF16 when available.
    """
    try:
        bf16_candidates = sorted(model_dir.glob("mmproj*BF16*.gguf"))
        if bf16_candidates:
            return bf16_candidates[0]
        all_candidates = sorted(model_dir.glob("mmproj*.gguf"))
        return all_candidates[0] if all_candidates else None
    except OSError:
        return None


def _validate_index_shards(model_dir: Path) -> None:
    for index_path in sorted(model_dir.glob("*.index*.json")):
        if not ("safetensors.index" in index_path.name or "bin.index" in index_path.name):
            continue
        try:
            data = json.loads(index_path.read_text())
        except Exception as exc:
            raise ValueError(f"invalid weight index {index_path.name}: {exc}") from None
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight index {index_path.name}: missing weight_map")
        missing = sorted({
            str(shard)
            for shard in weight_map.values()
            if not (index_path.parent / str(shard)).exists()
        })
        if missing:
            raise ValueError(f"{model_dir} missing shard(s) referenced by {index_path.name}: {', '.join(missing)}")


def quant_flag(method: str | None) -> str:
    if method is None:
        return ""
    return QUANT_FLAGS.get(method, "")


def detect_model(model_dir: Path) -> ModelInfo:
    config_path = model_dir / "config.json"
    gguf_files = iter_top_level_files_with_suffix(model_dir, ".gguf")

    has_safetensors = bool(iter_top_level_files_with_suffix(model_dir, ".safetensors"))
    has_bin = bool(iter_top_level_files_with_suffix(model_dir, ".bin"))
    if gguf_files and not has_safetensors and not has_bin:
        # GGUF model — may or may not have config.json alongside
        size_bytes = sum(f.stat().st_size for f in gguf_files)
        size_gb = round(size_bytes / (1024**3), 1)
        # Pick the first tier found across the model's GGUF files.
        tier: str | None = None
        for gguf_file in gguf_files:
            tier = parse_unsloth_quant_tier(gguf_file.name)
            if tier:
                break
        # MXFP4_MOE tier always implies MoE even if legacy metadata says dense.
        gguf_is_moe = tier == "MXFP4_MOE"
        mmproj = find_mmproj(model_dir)
        return ModelInfo(
            path=model_dir,
            provider=model_dir.parent.name,
            model_name=model_dir.name,
            architecture="gguf",
            model_type="gguf",
            quant_method=("mxfp4_moe" if tier == "MXFP4_MOE" else None),
            max_position_embeddings=0,  # read from GGUF metadata at tune time
            is_moe=gguf_is_moe,
            model_size_gb=size_gb,
            is_gguf=True,
            quant_tier=tier,
            mmproj_path=mmproj,
        )

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)
    _validate_index_shards(model_dir)

    text_config = config.get("text_config", {})

    arch = (config.get("architectures") or ["unknown"])[0]
    model_type = config.get("model_type", "unknown")
    qmethod = (config.get("quantization_config") or {}).get("quant_method")
    max_pos = text_config.get(
        "max_position_embeddings",
        config.get("max_position_embeddings", 32768),
    )
    num_experts = text_config.get(
        "num_experts",
        config.get("num_experts",
                    text_config.get("num_local_experts",
                                    config.get("num_local_experts"))),
    )
    is_moe = num_experts is not None and num_experts > 0

    # Multimodal = ships a vision/audio tower. vLLM runs a real encoder forward on
    # dummy items at startup (profiling), and the encoder weights + that transient
    # need VRAM the KV math doesn't reserve — so these need extra headroom (or
    # --language-model-only). Detect from the standard HF multimodal config keys.
    is_multimodal = (
        config.get("vision_config") is not None
        or config.get("audio_config") is not None
        or any(
            k in config
            for k in ("image_token_id", "image_token_index", "video_token_id", "audio_token_id")
        )
    )

    # Architecture fields for KV cache calculation
    num_kv_heads = text_config.get(
        "num_key_value_heads",
        config.get("num_key_value_heads"),
    )
    num_layers = text_config.get(
        "num_hidden_layers",
        config.get("num_hidden_layers"),
    )
    hidden_size = text_config.get("hidden_size", config.get("hidden_size"))
    num_attn_heads = text_config.get(
        "num_attention_heads",
        config.get("num_attention_heads"),
    )
    if num_kv_heads is None and num_attn_heads is not None:
        num_kv_heads = num_attn_heads
    head_dim_val = text_config.get("head_dim", config.get("head_dim"))
    # Gemma-4-style hybrid head_dim: sliding-window layers use head_dim while
    # global-attn layers use global_head_dim (typically 2x). The interleave
    # follows sliding_window_pattern (default 6 = 5 local + 1 global). Use the
    # weighted average for KV-byte budgeting; downstream PagedAttention block-
    # rounds per layer so this is the right average.
    global_head_dim_val = text_config.get("global_head_dim", config.get("global_head_dim"))
    if (
        isinstance(head_dim_val, int) and not isinstance(head_dim_val, bool)
        and isinstance(global_head_dim_val, int) and not isinstance(global_head_dim_val, bool)
        and global_head_dim_val != head_dim_val
    ):
        pattern_len_raw = text_config.get(
            "sliding_window_pattern",
            config.get("sliding_window_pattern", 6),
        )
        try:
            pattern_len = int(pattern_len_raw) if pattern_len_raw else 6
        except (TypeError, ValueError):
            pattern_len = 6
        if pattern_len < 1:
            pattern_len = 6
        head_dim_val = ((pattern_len - 1) * head_dim_val + global_head_dim_val) // pattern_len
    if head_dim_val is None and hidden_size and num_attn_heads:
        head_dim_val = hidden_size // num_attn_heads

    size_bytes = sum(f.stat().st_size for f in iter_top_level_files_with_suffix(model_dir, ".safetensors"))
    size_bytes += sum(f.stat().st_size for f in iter_top_level_files_with_suffix(model_dir, ".bin"))
    size_gb = round(size_bytes / (1024**3), 1)

    return ModelInfo(
        path=model_dir,
        provider=model_dir.parent.name,
        model_name=model_dir.name,
        architecture=arch,
        model_type=model_type,
        quant_method=qmethod,
        max_position_embeddings=max_pos,
        is_moe=is_moe,
        is_multimodal=is_multimodal,
        model_size_gb=size_gb,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        head_dim=head_dim_val,
        global_head_dim=(
            global_head_dim_val
            if isinstance(global_head_dim_val, int) and not isinstance(global_head_dim_val, bool)
            else None
        ),
    )


def scan_models(models_root: Path) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    if not models_root.exists():
        return models
    try:
        provider_dirs = sorted(models_root.iterdir())
    except OSError as exc:
        _log.warning("Skipping models root %s: %s", models_root, exc)
        return models
    for provider_dir in provider_dirs:
        if provider_dir.is_symlink() or not provider_dir.is_dir():
            continue
        try:
            model_dirs = sorted(provider_dir.iterdir())
        except OSError as exc:
            _log.warning("Skipping provider directory %s: %s", provider_dir, exc)
            continue
        for model_dir in model_dirs:
            if model_dir.is_symlink() or not model_dir.is_dir():
                continue
            if (model_dir / ".vserve-ignore").exists():
                continue
            try:
                has_config = (model_dir / "config.json").exists()
                has_gguf = bool(iter_top_level_files_with_suffix(model_dir, ".gguf"))
                if not has_config and not has_gguf:
                    continue
                models.append(detect_model(model_dir))
            except Exception as exc:
                _log.warning("Skipping model directory %s: %s", model_dir, exc)
                continue
    return models


def fuzzy_match(query: str, models: list[ModelInfo]) -> list[ModelInfo]:
    query = query.strip()
    for m in models:
        if query == str(m.path):
            return [m]

    for m in models:
        if query == m.full_name:
            return [m]

    query_lower = query.lower()
    contiguous = [
        m
        for m in models
        if query_lower in m.model_name.lower()
        or query_lower in m.full_name.lower()
    ]
    if contiguous:
        return contiguous

    tokens = [t for t in re.split(r"[^a-z0-9]+", query_lower) if t]
    if not tokens:
        return []

    ranked: list[tuple[int, str, ModelInfo]] = []
    for m in models:
        features = [
            m.provider,
            m.model_name,
            m.full_name,
            m.architecture,
            m.model_type,
            m.quant_method or "",
            "gguf" if m.is_gguf else "safetensors",
            "embedding" if m.is_embedding else "",
        ]
        haystack = " ".join(features).lower()
        if not all(token in haystack for token in tokens):
            continue
        score = 0
        name_lower = m.model_name.lower()
        full_lower = m.full_name.lower()
        for token in tokens:
            if token == (m.quant_method or "").lower():
                score += 8
            if token in name_lower:
                score += 4
            if token in full_lower:
                score += 2
            if token == m.provider.lower():
                score += 1
        ranked.append((-score, m.full_name.lower(), m))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [m for _score, _name, m in ranked]
