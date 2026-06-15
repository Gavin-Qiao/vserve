"""Speculative-decoding recipe selection (item M).

Three independent shapes:
- ``ngram``: vLLM-only prompt-lookup based prediction; zero extra-model cost.
- ``draft``: pair a small (≤1.5B) same-family model as the speculator.
- ``mtp``: Multi-Token Prediction variant GGUF (Unsloth Qwen3.6 family).

Blocklists:
- A3B-style MoE: spec-decode is net-negative on RTX 3090 per llamacpp#19493
  benchmarks (0/19 configs positive; mean −3 to −12% decode tokens/sec).
- vLLM Gemma-4 + MTP + multi-tool streaming → first call's args corrupted
  (vllm#41967). Refuse MTP when arch=Gemma4 AND tools enabled.
- vLLM any spec method + quantized KV → DFlash spec-decode broken (vllm#41559).
  Auto-emit cudagraph_mode: NONE (item AA handles this in vLLM build_config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpecConfig:
    method: str           # "ngram" | "draft" | "mtp"
    draft_model_path: Path | None = None
    n_max: int = 5        # max speculated tokens per step
    n_min: int = 1        # min speculated tokens per step
    p_min: float = 0.6    # acceptance threshold


# Architecture → recommended spec method. ``mtp`` is only chosen if a
# matching MTP-suffixed GGUF / safetensor variant is available.
SPEC_METHOD_BY_ARCH: dict[str, str] = {
    "Qwen36ForCausalLM":  "mtp",
    "Qwen36MoeForCausalLM": "mtp",
    # Canonical arch that real Qwen3.5/3.6 checkpoints actually report (both
    # Qwen3.5 and Qwen3.6 register as Qwen3_5MoeForConditionalGeneration). The
    # synthetic Qwen36* keys above only match GGUF short-names, so without
    # these the safetensors/vLLM path got no MTP recommendation — the same
    # gap b94a823 closed for the tool/reasoning-parser tables. Still gated by
    # find_mtp_variant(), so checkpoints without an MTP variant fall through
    # to draft/ngram and are unaffected.
    "Qwen3_5ForConditionalGeneration":    "mtp",
    "Qwen3_5MoeForConditionalGeneration": "mtp",
}

# Architectures where spec decoding is net-negative on common consumer GPUs.
# Source: llamacpp#19493 (A3B-style MoE expert-saturation analysis).
SPEC_BLOCKLIST: frozenset[str] = frozenset({
    "Qwen3A3BForCausalLM",
    "Qwen3MoeForCausalLM",
    "GptOssForCausalLM",  # was misspelled as GptOssMoeForCausalLM — never matched any model
    "MixtralForCausalLM",
    "DeepseekV2ForCausalLM",
})


@dataclass
class DraftCandidate:
    path: Path
    architecture: str
    size_b: float
    tokenizer_model: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    add_bos: bool | None = None
    add_eos: bool | None = None


def vocab_compatible(target_arch: str, target_bos: int | None, target_eos: int | None,
                     draft: DraftCandidate) -> bool:
    """Spec-decode requires identical tokenizers. Compare BOS/EOS pair and
    architecture family. Architectures must match on the family root (e.g.
    Qwen3* with Qwen3*), tokenizer must match BOS+EOS exactly.

    0.6.3 bug fix 2: previously this used ``arch[:5]`` for family matching,
    which collided ``Gemma3`` and ``Gemma4`` (5-letter shared prefix). It
    now uses :func:`vserve.arch_registry.family_of` so Gemma3 vs Gemma4
    (different tokenizers, incompatible vocab for spec-decode) are
    correctly distinguished.
    """
    from vserve.arch_registry import family_of

    if not target_arch or not draft.architecture:
        return False
    target_family = family_of(target_arch)
    draft_family = family_of(draft.architecture)
    if target_family is None or draft_family is None:
        # Unknown arch — be conservative and refuse spec-decode.
        return False
    if target_family != draft_family:
        return False
    if target_bos is None or target_eos is None:
        return False
    if draft.bos_token_id is None or draft.eos_token_id is None:
        return False
    return target_bos == draft.bos_token_id and target_eos == draft.eos_token_id


def find_mtp_variant(model_path: Path) -> Path | None:
    """Look for a sibling MTP-suffixed GGUF/safetensors variant of the model.

    Unsloth ships ``-MTP-GGUF`` variants alongside the base for Qwen3.6.
    Returns the path when present, None otherwise.
    """
    parent = model_path.parent
    try:
        siblings = [d for d in parent.iterdir() if d.is_dir() and d != model_path]
    except OSError:
        return None
    for sibling in siblings:
        name = sibling.name.lower()
        if "mtp" in name:
            return sibling
    return None


def pick_spec_config(
    *,
    architecture: str | None,
    backend: str,
    model_path: Path,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    tools_enabled: bool = False,
    available_drafts: Iterable[DraftCandidate] = (),
    force: bool = False,
) -> SpecConfig | None:
    """Pick a spec-decode method for the (architecture, backend) pair.

    Returns None when spec-decode is disabled by the blocklist (override via
    ``force=True``). Returns an ``SpecConfig`` with the chosen method.

    Walks:
      1. Blocklist → None unless force
      2. MTP variant (Qwen3.6) → method=mtp
      3. Same-family ≤1.5B draft model → method=draft
      4. vLLM-only ngram fallback (zero extra-model cost)
      5. llama.cpp without a draft → None
    """
    if not architecture:
        return None
    if architecture in SPEC_BLOCKLIST and not force:
        return None
    # vllm#41967: Gemma-4 + MTP + multi-tool streaming corrupts the first
    # call's args. Refuse MTP when tools are enabled for Gemma-4.
    is_gemma4 = architecture.startswith("Gemma4")

    recommended = SPEC_METHOD_BY_ARCH.get(architecture)
    if recommended == "mtp" and not (is_gemma4 and tools_enabled):
        variant = find_mtp_variant(model_path)
        if variant is not None:
            return SpecConfig(method="mtp", draft_model_path=variant, n_max=5)

    # Try a same-family ≤1.5B draft model.
    for draft in available_drafts:
        if draft.size_b > 1.5:
            continue
        if vocab_compatible(architecture, bos_token_id, eos_token_id, draft):
            return SpecConfig(method="draft", draft_model_path=draft.path, n_max=3, p_min=0.6)

    # ngram is vLLM-only (llama.cpp doesn't have an equivalent zero-cost mode).
    if backend == "vllm":
        return SpecConfig(method="ngram", n_max=5, n_min=1)
    return None
