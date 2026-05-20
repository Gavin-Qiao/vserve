"""Single canonical home for architecture-keyed registries.

Before 0.6.3 these tables lived scattered across `backends/vllm.py`,
`recipes/sampling.py`, etc. — and they drifted. The audit (see
`docs/audits/2026-05-20-registries-coherence.md`) caught four real bugs
including a Gemma3 / Gemma4 collision in a `arch[:5]` prefix-slice that
this module's :func:`family_of` was created to replace.

Add new entries here — every dependent module imports from this file.

Architecture-name convention: keys are vLLM-style **full** arch names
(e.g. ``Qwen3MoeForCausalLM``, NOT ``Qwen3Moe``). Family names are
lowercase canonical roots (e.g. ``"qwen3"``).
"""

from __future__ import annotations


# ── tool-call parser table ────────────────────────────────────────────────
# Source: vLLM 0.21 `tool_parsers/__init__.py`. Names must match the keys
# registered by `ToolParserManager`.

_ARCH_TO_TOOL_PARSER: dict[str, str] = {
    # Gemma 3 / 4 family
    "Gemma3ForCausalLM":              "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM":              "gemma4",
    "Gemma4ForConditionalGeneration": "gemma4",
    # Llama 3.x / 4 — 3.1/3.2/3.3 use the JSON-style parser, 4 uses pythonic
    "LlamaForCausalLM":               "llama3_json",
    "Llama4ForCausalLM":              "llama4_pythonic",
    "Llama4MoeForCausalLM":           "llama4_pythonic",
    # Qwen3 family — Hermes-format on base, qwen3_coder for coder variants
    "Qwen3ForCausalLM":               "hermes",
    "Qwen35ForCausalLM":              "hermes",
    "Qwen36ForCausalLM":              "hermes",
    "Qwen3MoeForCausalLM":            "hermes",
    "Qwen3A3BForCausalLM":            "hermes",
    "Qwen36MoeForCausalLM":           "hermes",
    "Qwen3CoderForCausalLM":          "qwen3_coder",
    "Qwen3XmlForCausalLM":            "qwen3_xml",
    # DeepSeek V-series — per-version parser
    "DeepseekV3ForCausalLM":          "deepseek_v3",
    "DeepseekV31ForCausalLM":         "deepseek_v31",
    "DeepseekV32ForCausalLM":         "deepseek_v32",
    "DeepseekV4ForCausalLM":          "deepseek_v4",
    # Moonshot Kimi K2 (instruct + thinking)
    "KimiK2ForCausalLM":              "kimi_k2",
    "KimiK2ThinkingForCausalLM":      "kimi_k2",
    # GLM 4 family (Z.ai)
    "Glm4MoeForCausalLM":             "glm45",
    "Glm47MoeForCausalLM":            "glm47",
    # IBM Granite (text-only Granite-3 + Granite-4 with mixed schema)
    "GraniteForCausalLM":             "granite",
    "Granite4ForCausalLM":            "granite4",
    # Cohere Command R/R+ (Command-4 parser supersedes Command-3)
    "CohereForCausalLM":              "cohere_command4",
    # Baidu ERNIE 4.5
    "Ernie4ForCausalLM":              "ernie45",
    # AI21 Jamba
    "JambaForCausalLM":               "jamba",
    # Salesforce xLAM
    "XlamForCausalLM":                "xlam",
    # Liquid LFM 2 / 2.5
    "Lfm2ForCausalLM":                "lfm2",
    "Lfm25ForCausalLM":               "lfm25",
    # Mistral
    "MistralForCausalLM":             "mistral",
    "MistralThinkingForCausalLM":     "mistral",
    # GPT-OSS (OpenAI Harmony)
    "GptOssForCausalLM":              "openai",
    # InternLM (2.x uses same parser)
    "InternLMForCausalLM":            "internlm",
    "InternLM2ForCausalLM":           "internlm",
}


# ── reasoning-parser table ────────────────────────────────────────────────
# Reasoning parsers split the thinking trace from the answer in OpenAI-format
# responses (so clients see `message.reasoning_content` distinct from
# `message.content`). Names must match `ReasoningParserManager` keys.
# Source: https://docs.vllm.ai/en/latest/features/reasoning_outputs.html

_ARCH_TO_REASONING_PARSER: dict[str, str] = {
    "Gemma3ForCausalLM":              "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM":              "gemma4",
    "Gemma4ForConditionalGeneration": "gemma4",
    "DeepseekV3ForCausalLM":          "deepseek_r1",
    "DeepseekV31ForCausalLM":         "deepseek_r1",
    "DeepseekV32ForCausalLM":         "deepseek_r1",
    "DeepseekV4ForCausalLM":          "deepseek_r1",
    "Qwen3ForCausalLM":               "qwen3",
    "Qwen35ForCausalLM":              "qwen3",
    "Qwen36ForCausalLM":              "qwen3",
    "Qwen3MoeForCausalLM":            "qwen3",
    "Qwen36MoeForCausalLM":           "qwen3",
    "Qwen3A3BForCausalLM":            "qwen3",
    "KimiK2ThinkingForCausalLM":      "deepseek_r1",  # uses <think> markers
    "MistralForCausalLM":             "mistral",
    "MistralThinkingForCausalLM":     "mistral",
    "GptOssForCausalLM":              "openai_gptoss",
}


# ── forced attention backend ──────────────────────────────────────────────
# Architectures with non-default attention layouts. The default vLLM
# backend-auto-pick can mis-route MLA / heterogeneous-head_dim archs.
# Compute-cap routing (FLASHMLA → TOKENSPEED_MLA on sm≥100) lives in the
# caller (`backends/vllm._forced_attention_backend`).

_ARCH_FORCES_BACKEND: dict[str, str] = {
    "DeepseekV2ForCausalLM":          "FLASHMLA",
    "DeepseekV3ForCausalLM":          "FLASHMLA",
    "DeepseekV31ForCausalLM":         "FLASHMLA",
    "DeepseekV32ForCausalLM":         "FLASHMLA",
    "DeepseekV4ForCausalLM":          "FLASHMLA",
    "KimiK2ForCausalLM":              "FLASHMLA",
    "KimiK2ThinkingForCausalLM":      "FLASHMLA",
    "LongcatFlashForCausalLM":        "FLASHMLA",
    # GPT-OSS on SM120 (Blackwell RTX) forces TRITON_ATTN because FlashInfer
    # doesn't support attention sinks on that compute capability (vllm#40153).
    "GptOssForCausalLM":              "TRITON_ATTN",
    # Gemma-4 already forces TRITON_ATTN via heterogeneous-head_dim path;
    # keep explicit here so this table is the single source of truth.
    "Gemma4ForCausalLM":              "TRITON_ATTN",
    "Gemma4ForConditionalGeneration": "TRITON_ATTN",
}


# ── GGUF short-name → HF arch-name mapping ────────────────────────────────
# GGUF stores `general.architecture` as a lowercase short name.

_GGUF_ARCH_TO_HF_ARCH: dict[str, str] = {
    "gemma3":     "Gemma3ForCausalLM",
    "gemma4":     "Gemma4ForCausalLM",
    "qwen3":      "Qwen3ForCausalLM",
    "qwen3coder": "Qwen3CoderForCausalLM",
    "qwen3moe":   "Qwen3MoeForCausalLM",
    "qwen3a3b":   "Qwen3A3BForCausalLM",
    "qwen35":     "Qwen35ForCausalLM",
    "qwen36":     "Qwen36ForCausalLM",
    "qwen36moe":  "Qwen36MoeForCausalLM",
    "deepseek3":  "DeepseekV3ForCausalLM",  # llama.cpp short name for V3
    "deepseek2":  "DeepseekV3ForCausalLM",  # V3 inherits V2 arch in some GGUFs
    "deepseekv31":"DeepseekV31ForCausalLM",
    "deepseekv32":"DeepseekV32ForCausalLM",
    "deepseekv4": "DeepseekV4ForCausalLM",
    "llama4":     "Llama4ForCausalLM",
    "kimik2":     "KimiK2ForCausalLM",
}


# ── canonical family mapping ──────────────────────────────────────────────
# Maps full vLLM arch name → lowercase family root. This replaces the
# `arch[:5]` prefix-slice that collided Gemma3 / Gemma4 (audit bug 2).
# Two archs share a family iff they share a tokenizer family AND can
# potentially share spec-decode drafters.

_ARCH_TO_FAMILY: dict[str, str] = {
    # Gemma 3 and 4 are DELIBERATELY different families — different
    # tokenizer (Gemma 4 has 512-dim global heads, sliding-window pattern
    # change), different chat templates, incompatible vocabularies for
    # spec-decoding.
    "Gemma3ForCausalLM":              "gemma3",
    "Gemma3ForConditionalGeneration": "gemma3",
    "Gemma4ForCausalLM":              "gemma4",
    "Gemma4ForConditionalGeneration": "gemma4",
    # Qwen3 base, 3.5, 3.6, MoE, A3B, 3.6 MoE all share tokenizer family.
    "Qwen3ForCausalLM":               "qwen3",
    "Qwen35ForCausalLM":              "qwen3",
    "Qwen36ForCausalLM":              "qwen3",
    "Qwen3MoeForCausalLM":            "qwen3",
    "Qwen36MoeForCausalLM":           "qwen3",
    "Qwen3A3BForCausalLM":            "qwen3",
    # Qwen Coder family is a separate tokenizer (extra code tokens).
    "Qwen3CoderForCausalLM":          "qwen3_coder",
    "Qwen3XmlForCausalLM":            "qwen3_xml",
    # Llama 3 / 3.1 / 3.2 / 3.3 share tokenizer; Llama 4 is separate.
    "LlamaForCausalLM":               "llama3",
    "Llama4ForCausalLM":              "llama4",
    "Llama4MoeForCausalLM":           "llama4",
    # DeepSeek V2 / V3 / V3.1 / V3.2 share tokenizer (some inherit V2 arch).
    "DeepseekV2ForCausalLM":          "deepseek_v3",
    "DeepseekV3ForCausalLM":          "deepseek_v3",
    "DeepseekV31ForCausalLM":         "deepseek_v3",
    "DeepseekV32ForCausalLM":         "deepseek_v3",
    # DeepSeek V4 has a different tokenizer.
    "DeepseekV4ForCausalLM":          "deepseek_v4",
    # Kimi K2 instruct + thinking share tokenizer.
    "KimiK2ForCausalLM":              "kimi_k2",
    "KimiK2ThinkingForCausalLM":      "kimi_k2",
    # GLM 4.5 / 4.7 share family.
    "Glm4MoeForCausalLM":             "glm4",
    "Glm47MoeForCausalLM":            "glm4",
    # Granite 3 / 4 are different tokenizers (Granite-4 added BPE tokens).
    "GraniteForCausalLM":             "granite",
    "Granite4ForCausalLM":            "granite4",
    "CohereForCausalLM":              "cohere",
    "Ernie4ForCausalLM":              "ernie",
    "JambaForCausalLM":               "jamba",
    "XlamForCausalLM":                "xlam",
    "Lfm2ForCausalLM":                "lfm",
    "Lfm25ForCausalLM":               "lfm",
    "MistralForCausalLM":             "mistral",
    "MistralThinkingForCausalLM":     "mistral",
    "GptOssForCausalLM":              "gpt_oss",
    "InternLMForCausalLM":            "internlm",
    "InternLM2ForCausalLM":           "internlm",
    "LongcatFlashForCausalLM":        "longcat",
}


def family_of(arch: str | None) -> str | None:
    """Return the canonical family root (lowercase) for an arch name, or
    None if the arch is unknown.

    Use this instead of slicing or pattern-matching arch strings — the
    table is authoritative and collisions like Gemma3 vs Gemma4 are
    resolved by listing them as distinct families. Both args and return
    are intentionally None-tolerant so callers don't have to guard.
    """
    if not arch:
        return None
    return _ARCH_TO_FAMILY.get(arch)


# Architectures whose default (no-flag) behavior is to generate a thinking
# trace. Used by the picker UX to suggest enabling/disabling reasoning
# parsing without the user having to know per-model.
_THINKING_DEFAULT_ARCHS: frozenset[str] = frozenset({
    "Qwen35ForCausalLM",
    "Qwen36ForCausalLM",
    "Qwen36MoeForCausalLM",  # 0.6.3: was missing from reasoning-parser table — bug fix 1
    "DeepseekV3ForCausalLM",
    "DeepseekV31ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV4ForCausalLM",
    "KimiK2ThinkingForCausalLM",
    "MistralThinkingForCausalLM",
    "GptOssForCausalLM",
})


def is_thinking_default(arch: str | None) -> bool:
    """True iff this arch defaults to producing a thinking/reasoning trace.

    The picker uses this to decide whether to suggest `--thinking` on by
    default, and the tuner uses it to size context windows appropriately
    (thinking traces eat tokens).
    """
    return bool(arch) and arch in _THINKING_DEFAULT_ARCHS
