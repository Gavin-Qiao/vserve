"""Speculative-decoding recipe selection (item M).

Three independent shapes:
- ``ngram``: prompt-lookup prediction; zero extra-model cost. Enabled on vLLM.
  llama.cpp also has it (``--spec-type ngram-*``) as of the pinned runtime, but
  vserve keeps it off there — benchmarked net-negative for batched serving
  (docs/research/2026-06-19-llamacpp-throughput-goedel.md).
- ``draft``: pair a small (≤1.5B) same-family model as the speculator.
- ``mtp``: Multi-Token Prediction. Two flavors:
  * in-checkpoint (vLLM >= 0.24 unified ``method: mtp``): the checkpoint
    itself carries the MTP draft layers — Qwen3.5/3.6 ``mtp_num_hidden_layers``,
    DeepSeek/GLM/Qwen3-Next-style ``num_nextn_predict_layers``, MiniMax
    ``num_mtp_modules``. No draft model path; vLLM loads the layers from the
    target checkpoint (see :func:`native_mtp_layers`).
  * sibling variant: a separate ``-MTP``-suffixed checkpoint next to the model
    dir used as the draft (Unsloth Qwen3.6 MTP-GGUF family; the only MTP
    flavor llama.cpp supports).

Blocklists:
- A3B-style MoE: spec-decode is net-negative on RTX 3090 per llamacpp#19493
  benchmarks (0/19 configs positive; mean −3 to −12% decode tokens/sec).
- vLLM Gemma-4 + MTP + multi-tool streaming → first call's args corrupted
  (vllm#41967). Refuse MTP when arch=Gemma4 AND tools enabled.
- vLLM any spec method + quantized KV → DFlash spec-decode broken (vllm#41559).
  Auto-emit cudagraph_mode: NONE (item AA handles this in vLLM build_config).
"""

from __future__ import annotations

import json
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


# Default MTP speculation depth. num_speculative_tokens=1 is the worst
# non-zero setting (-19% decode on Qwen3.6-27B FP8; min useful depth is 2)
# and k=3 wins on the acceptance-vs-depth curve — see
# docs/research/2026-05-20-spec-decode-acceptance.md §5/§6 + action item 1.
MTP_NUM_SPECULATIVE_TOKENS = 3

# ngram (prompt-lookup) default depth — vLLM's own ngram default range.
NGRAM_NUM_SPECULATIVE_TOKENS = 5

# config.json keys that mark in-checkpoint MTP draft layers, mirroring the
# detection in vLLM 0.24's SpeculativeConfig (which rewrites these into
# ``n_predict`` per model family).
_NATIVE_MTP_LAYER_KEYS: tuple[str, ...] = (
    "mtp_num_hidden_layers",     # Qwen3.5/3.6 (top-level or text_config)
    "num_nextn_predict_layers",  # DeepSeek V3/V4, GLM-4.x MoE, Qwen3-Next,
                                 # Nemotron-H, ERNIE, LongCat, HY-V3, Exaone-MoE
    "num_mtp_modules",           # MiniMax M3
)


def native_mtp_layers(model_path: Path) -> int | None:
    """Number of in-checkpoint MTP draft layers (``n_predict``), or None.

    Reads config.json at the top level and inside ``text_config`` (VLM-style
    checkpoints such as Qwen3.5/3.6 keep the MTP keys on the text sub-config).
    Only a positive integer counts — 0 means the checkpoint shipped without
    its MTP head.
    """
    try:
        cfg = json.loads((model_path / "config.json").read_text())
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    subconfigs = [cfg]
    text_config = cfg.get("text_config")
    if isinstance(text_config, dict):
        subconfigs.append(text_config)
    for sub in subconfigs:
        for key in _NATIVE_MTP_LAYER_KEYS:
            n = sub.get(key)
            if isinstance(n, int) and not isinstance(n, bool) and n > 0:
                return n
    return None


def _default_mtp_tokens(n_predict: int | None) -> int:
    """Speculation depth for MTP given the checkpoint's draft-layer count.

    n_predict=1 (the common case — Qwen3.5/3.6 ship one MTP layer) reuses the
    module 3× per the acceptance research; deeper native stacks run at their
    own depth so no reuse (and no divisibility constraint) is involved.
    """
    if n_predict is None or n_predict <= 1:
        return MTP_NUM_SPECULATIVE_TOKENS
    return n_predict


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
    # Qwen3.5/3.6 MoE-A3B: measured 2026-07-10 on RTX PRO 5000 (sm120),
    # vLLM 0.24, NVFP4 — in-checkpoint MTP k=3 hit 60% acceptance yet decoded
    # at −52% (c1) / −53% (c8) vs no-spec: the (k+1)-token verify step
    # multiplies MoE expert weight traffic and the bf16 (quant-excluded)
    # draft layer adds sequential forwards. Expert saturation dominates even
    # on Blackwell. Explicit `--mtp` / force=True still allow it.
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen36MoeForCausalLM",  # GGUF short-name twin of the same A3B family
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


def resolve_mtp_request(
    *,
    backend: str,
    model_path: Path,
    num_tokens: int | None = None,
) -> SpecConfig:
    """Resolve an explicit MTP request (``vserve run --mtp``) into a SpecConfig.

    Unlike :func:`pick_spec_config` (the recommendation engine), an explicit
    request skips the blocklist — the user asked for MTP, so they get MTP or
    a ``ValueError`` explaining exactly why it can't work here. It never
    silently substitutes another spec method.

    vLLM: prefer in-checkpoint MTP layers (draft_model_path=None → the
    ``speculative-config`` block carries no ``model`` key and vLLM loads the
    draft layers from the target checkpoint); fall back to a sibling ``-MTP``
    variant checkpoint. llama.cpp: sibling MTP-variant GGUF only — GGUF
    conversions don't carry the safetensors MTP head.
    """
    if num_tokens is not None and num_tokens < 1:
        raise ValueError("MTP speculation depth (--mtp-tokens) must be >= 1.")
    if backend == "vllm":
        n_predict = native_mtp_layers(model_path)
        if n_predict is not None:
            tokens = num_tokens if num_tokens is not None else _default_mtp_tokens(n_predict)
            if tokens > n_predict and tokens % n_predict != 0:
                # vLLM reuses the MTP module for depths beyond n_predict and
                # requires divisibility — fail here instead of at engine boot.
                raise ValueError(
                    f"--mtp-tokens {tokens} must be a multiple of the checkpoint's "
                    f"{n_predict} MTP layer(s) (vLLM MTP module reuse)."
                )
            return SpecConfig(method="mtp", draft_model_path=None, n_max=tokens)
        variant = find_mtp_variant(model_path)
        if variant is not None:
            return SpecConfig(
                method="mtp",
                draft_model_path=variant,
                n_max=num_tokens if num_tokens is not None else MTP_NUM_SPECULATIVE_TOKENS,
            )
        raise ValueError(
            "This checkpoint has no MTP weights: config.json has none of "
            f"{'/'.join(_NATIVE_MTP_LAYER_KEYS)} and no MTP-variant sibling "
            "checkpoint exists next to the model directory."
        )
    variant = find_mtp_variant(model_path)
    if variant is None:
        raise ValueError(
            "MTP on llama.cpp needs an MTP-variant GGUF sibling next to the "
            "model directory (e.g. Unsloth *-MTP-GGUF); none was found."
        )
    return SpecConfig(
        method="mtp",
        draft_model_path=variant,
        n_max=num_tokens if num_tokens is not None else MTP_NUM_SPECULATIVE_TOKENS,
    )


def resolve_spec_request(
    *,
    method: str,
    backend: str,
    model_path: Path,
    architecture: str | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    tools_enabled: bool = False,
    available_drafts: Iterable[DraftCandidate] = (),
    num_tokens: int | None = None,
) -> SpecConfig | None:
    """Resolve an explicit ``vserve run --spec METHOD`` request.

    ``method`` is one of ``auto | ngram | mtp | draft`` (``off`` never reaches
    here — the CLI short-circuits it). Explicit methods return exactly that
    method or raise ValueError with the precise reason; ``auto`` delegates to
    :func:`pick_spec_config` (the recommendation engine, blocklist included)
    and may return None, meaning "no net-positive method for this
    model/backend" — which is a valid answer, not an error.
    """
    if method == "mtp":
        return resolve_mtp_request(
            backend=backend, model_path=model_path, num_tokens=num_tokens,
        )
    if method == "ngram":
        if backend != "vllm":
            raise ValueError(
                "ngram (prompt-lookup) spec-decode is vLLM-only in vserve — "
                "benchmarked net-negative for batched llama.cpp serving "
                "(docs/research/2026-06-19-llamacpp-throughput-goedel.md)."
            )
        return SpecConfig(method="ngram", n_max=NGRAM_NUM_SPECULATIVE_TOKENS, n_min=1)
    if method == "draft":
        for draft in available_drafts:
            if draft.size_b > 1.5:
                continue
            if vocab_compatible(architecture or "", bos_token_id, eos_token_id, draft):
                return SpecConfig(
                    method="draft", draft_model_path=draft.path, n_max=3, p_min=0.6,
                )
        raise ValueError(
            "No compatible draft model found locally: spec-decode needs a "
            "same-family model <= 1.5B with an identical tokenizer "
            "(matching BOS/EOS ids)."
        )
    if method == "auto":
        return pick_spec_config(
            architecture=architecture,
            backend=backend,
            model_path=model_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tools_enabled=tools_enabled,
            available_drafts=available_drafts,
        )
    raise ValueError(f"Unknown spec method {method!r} — use auto, off, ngram, mtp, or draft.")


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
      2. MTP — in-checkpoint layers (vLLM) or sibling MTP variant → method=mtp
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
        # In-checkpoint MTP first (vLLM-only; llama.cpp can't load the
        # safetensors MTP head). Before 0.6.8 this path was gated on
        # find_mtp_variant() alone, so Qwen3.5/3.6 safetensors checkpoints —
        # whose MTP layers live INSIDE the checkpoint (mtp_num_hidden_layers),
        # with no sibling dir — never got their MTP recommendation.
        if backend == "vllm":
            n_predict = native_mtp_layers(model_path)
            if n_predict is not None:
                return SpecConfig(
                    method="mtp", draft_model_path=None,
                    n_max=_default_mtp_tokens(n_predict),
                )
        variant = find_mtp_variant(model_path)
        if variant is not None:
            return SpecConfig(
                method="mtp", draft_model_path=variant,
                n_max=MTP_NUM_SPECULATIVE_TOKENS,
            )

    # Try a same-family ≤1.5B draft model.
    for draft in available_drafts:
        if draft.size_b > 1.5:
            continue
        if vocab_compatible(architecture, bos_token_id, eos_token_id, draft):
            return SpecConfig(method="draft", draft_model_path=draft.path, n_max=3, p_min=0.6)

    # ngram (prompt-lookup) is the zero-extra-model fallback. llama.cpp DOES
    # support it as of the pinned runtime (--spec-type ngram-mod/ngram-simple/…),
    # but it's benchmarked net-negative for batched llama.cpp serving on this
    # fleet — Goedel-Prover-V2-32B (Qwen2.5 dense) at np=5 lost ~9% decode tok/s
    # (ngram-mod, default n-match=24, ~26% accept) and barely activated even
    # single-stream — so it stays disabled there, mirroring the A3B-MoE
    # blocklist. See docs/research/2026-06-19-llamacpp-throughput-goedel.md.
    if backend == "vllm":
        return SpecConfig(method="ngram", n_max=5, n_min=1)
    return None
