"""Per-architecture default sampling parameters.

Sources:
- Gemma 3 / 4: Google + Unsloth https://unsloth.ai/docs/models/gemma-4
- Qwen3 family: Unsloth https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
- Qwen3-Coder: Unsloth https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally
- DeepSeek V3.1: Unsloth https://unsloth.ai/docs/models/tutorials/deepseek-v3.1-how-to-run-locally
- Kimi K2: Unsloth https://unsloth.ai/docs/models/tutorials/kimi-k2-thinking-how-to-run-locally
- Llama 4: Unsloth https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune

Lowering temperature on Qwen3-Thinking variants is documented to trap the
model in loops; greedy is explicitly banned on thinking variants by Unsloth.
"""

from __future__ import annotations

from dataclasses import dataclass

# `_GGUF_ARCH_TO_HF_ARCH` moved to vserve.arch_registry in 0.6.3 — the
# audit (`docs/audits/2026-05-20-registries-coherence.md`) consolidated
# all arch-keyed tables into one canonical home. Re-exported here so
# external callers keep working without churn.
from vserve.arch_registry import _GGUF_ARCH_TO_HF_ARCH


@dataclass(frozen=True)
class SamplingDefaults:
    """Sampler parameters recommended by the model's family / fine-tune lineage."""

    temp: float
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None


# Map architecture string (matches HF config.json `architectures[0]`) to
# the recommended sampler. Keep architecture-derived; downstream callers
# decide whether to emit / advertise / allow opt-out.
SAMPLING_DEFAULTS: dict[str, SamplingDefaults] = {
    # Gemma 3 / 4 (all variants — instruct, base, dense, MoE, multimodal)
    "Gemma3ForCausalLM":              SamplingDefaults(1.0, 0.95, 64, 0.01),
    "Gemma3ForConditionalGeneration": SamplingDefaults(1.0, 0.95, 64, 0.01),
    "Gemma4ForCausalLM":              SamplingDefaults(1.0, 0.95, 64, 0.01),
    "Gemma4ForConditionalGeneration": SamplingDefaults(1.0, 0.95, 64, 0.01),
    # Qwen 3 family (non-thinking → temp 0.7; thinking → temp 0.6 / 1.0 per variant)
    "Qwen3ForCausalLM":               SamplingDefaults(0.7, 0.8, 20, 0.01),
    "Qwen3CoderForCausalLM":          SamplingDefaults(0.7, 0.8, 20, 0.01, repeat_penalty=1.05),
    "Qwen3XmlForCausalLM":            SamplingDefaults(0.7, 0.8, 20, 0.01),
    "Qwen3MoeForCausalLM":            SamplingDefaults(0.7, 0.8, 20, 0.01),
    "Qwen3A3BForCausalLM":            SamplingDefaults(0.7, 0.8, 20, 0.01),
    # Qwen 3.5 — thinking-default, low temp, presence penalty
    "Qwen35ForCausalLM":              SamplingDefaults(0.6, 0.95, 20, 0.0, presence_penalty=1.0),
    # Qwen 3.6 — general (high-temp thinking)
    "Qwen36ForCausalLM":              SamplingDefaults(1.0, 0.95, 20, 0.0, presence_penalty=1.5),
    "Qwen36MoeForCausalLM":           SamplingDefaults(1.0, 0.95, 20, 0.0, presence_penalty=1.5),
    # DeepSeek V3 / V3.1 / V3.2 / V4
    "DeepseekV3ForCausalLM":          SamplingDefaults(0.6, 0.95, min_p=0.01),
    "DeepseekV31ForCausalLM":         SamplingDefaults(0.6, 0.95, min_p=0.01),
    "DeepseekV32ForCausalLM":         SamplingDefaults(0.6, 0.95, min_p=0.01),
    "DeepseekV4ForCausalLM":          SamplingDefaults(0.6, 0.95, min_p=0.01),
    # Llama 4
    "Llama4ForCausalLM":              SamplingDefaults(0.6, 0.9, min_p=0.01),
    "Llama4MoeForCausalLM":           SamplingDefaults(0.6, 0.9, min_p=0.01),
    # Kimi K2 (instruct = low temp, thinking = high temp)
    "KimiK2ForCausalLM":              SamplingDefaults(0.6, min_p=0.01),
    "KimiK2ThinkingForCausalLM":      SamplingDefaults(1.0, min_p=0.01),
}


def get_sampling_defaults(architecture: str | None) -> SamplingDefaults | None:
    """Return recipe defaults for the given architecture or None when unknown.

    Callers should treat None as "do not emit sampler flags" — the engine's
    default sampler applies. Per-request sampler overrides via the OpenAI
    API always supersede these defaults at inference time.
    """
    if not architecture:
        return None
    return SAMPLING_DEFAULTS.get(architecture)


def get_sampling_defaults_from_gguf_arch(gguf_arch: str | None) -> SamplingDefaults | None:
    """Map a GGUF ``general.architecture`` string (lowercase short name) to
    the recipe defaults. Returns None when unmapped."""
    if not gguf_arch:
        return None
    hf_arch = _GGUF_ARCH_TO_HF_ARCH.get(gguf_arch.lower())
    return SAMPLING_DEFAULTS.get(hf_arch) if hf_arch else None


def render_recipe_summary(defaults: SamplingDefaults) -> str:
    """Format a one-line summary for the CLI startup banner.

    Example: ``temp=1.0 top_p=0.95 top_k=64 min_p=0.01``.
    """
    parts = [f"temp={defaults.temp}"]
    if defaults.top_p is not None:
        parts.append(f"top_p={defaults.top_p}")
    if defaults.top_k is not None:
        parts.append(f"top_k={defaults.top_k}")
    if defaults.min_p is not None:
        parts.append(f"min_p={defaults.min_p}")
    if defaults.presence_penalty is not None:
        parts.append(f"presence_penalty={defaults.presence_penalty}")
    if defaults.repeat_penalty is not None:
        parts.append(f"repeat_penalty={defaults.repeat_penalty}")
    return " ".join(parts)
