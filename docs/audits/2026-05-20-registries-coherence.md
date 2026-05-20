# Registries coherence audit — 2026-05-20

Read-only sweep of every arch-keyed / lookup table under `src/vserve/`.

**Beyond the prompt's list**: `_KV_DTYPE_BYTES_PER_ELEMENT`, `_LLAMACPP_KV_DTYPE_QUALITY` (llamacpp.py:29,49); `_STRATEGY_FREE_FRACTION` (ot_strategies.py:57); `_NAME_FALLBACKS`, `KNOWN_TOOL_PARSERS`, `KNOWN_REASONING_PARSERS` (tools.py:44,64,47,67).

## Union-of-archs matrix

T=tool, R=reasoning, B=force-backend, S=sampling, M=spec-method, X=spec-blocklist, G=GGUF-map. `.` = absent.

| Arch | T | R | B | S | M | X | G |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Gemma3ForCausalLM | y | y | . | y | . | . | y |
| Gemma4ForCausalLM | y | y | y | y | . | . | y |
| Gemma4ForConditionalGeneration | y | y | y | y | . | . | . |
| Qwen3MoeForCausalLM | y | y | . | y | . | y | y |
| Qwen3A3BForCausalLM | y | y | . | y | . | y | y |
| Qwen36MoeForCausalLM | y | **.** | . | y | y | . | y |
| Llama4ForCausalLM | y | . | . | y | . | . | y |
| DeepseekV2ForCausalLM | . | . | y | . | . | y | . |
| KimiK2ThinkingForCausalLM | y | y | y | y | . | . | . |
| GptOssForCausalLM | y | y | y | **.** | . | . | . |
| GptOssMoeForCausalLM | . | . | . | . | . | y | . |
| LongcatFlashForCausalLM | . | . | y | . | . | . | . |

## Coverage gaps

1. **Qwen36MoeForCausalLM absent from `_ARCH_TO_REASONING_PARSER`** (vllm.py:99-117). Sibling Qwen36ForCausalLM is on line 110. Qwen 3.6 MoE is thinking-default → `<think>` leaks into `message.content`.
2. **GptOss, Mistral, MistralThinking absent from `SAMPLING_DEFAULTS`** (sampling.py:35) despite having tool + reasoning rows.
3. **DeepseekV2, LongcatFlash** in `_ARCH_FORCES_BACKEND` / `SPEC_BLOCKLIST` only — no sampling, no tool parser.
4. **`_GGUF_ARCH_TO_HF_ARCH` missing**: `glm4moe`, `glm47moe`, `granite`, `granite4`, `cohere`, `ernie4`, `jamba`, `xlam`, `lfm2`, `lfm25`, `internlm`, `internlm2`, `mistral`, `gptoss`, `kimik2thinking`, `qwen3xml`, `llama4moe`.
5. **`seed_oss` / `hunyuan`** parsers in `_REASONING_MARKER_TABLE` (tools.py:53,57) have no arch row anywhere — orphan fallbacks.

## Naming-convention violations

- **spec_decode.py:45** uses `GptOssMoeForCausalLM`; vllm.py:87,116,169,210 use `GptOssForCausalLM`. Blocklist entry is dead.
- **sampling.py:91-92**: `"deepseek2"` and `"deepseek3"` both map to `DeepseekV3ForCausalLM`, but `SPEC_BLOCKLIST` separately has `DeepseekV2ForCausalLM`. A `deepseek2` GGUF gets V3 sampling and V2 spec-blocking.
- **spec_decode.py:73**: ad-hoc `arch[:5]` family check would collide `Gemma3`/`Gemma4`, `KimiK2`/other `Kimi*`. Needs a canonical `family_of(arch)`.

## Hardware-gating scatter

- **`SPEC_BLOCKLIST` (spec_decode.py:42) is unconditional**. Comment cites llamacpp#19493 on sm86 but applies to all hardware. A3B-style flips net-positive at sm≥9.0 — convert to `{arch: {"block_below_sm": 90}}` or pass `gpu_compute_cap` into `pick_spec_config`.
- **`QUANT_ENV_VARS` (models.py:91) emits `VLLM_USE_FLASHINFER_MOE_FP4` unconditionally** for nvfp4/modelopt. FlashInfer MoE FP4 is sm≥100 only → SM-gate it.
- `_ARCH_FORCES_BACKEND` SM-routing (vllm.py:186-213) is correctly centralized.

## Hoist candidates → new `src/vserve/arch_registry.py`

1. `_ARCH_TO_TOOL_PARSER`, `_ARCH_TO_REASONING_PARSER`, `_ARCH_FORCES_BACKEND` out of `backends/vllm.py` — per-arch policy, not engine constants.
2. `_GGUF_ARCH_TO_HF_ARCH` out of `recipes/sampling.py` — used for any GGUF-arch normalization.
3. Canonical `family_of(arch)` replacing `spec_decode.vocab_compatible`'s `arch[:5]`.

## Unsloth UD-2.0 tiers

`UNSLOTH_QUANT_TIERS` covers all canonical UD-2.0 tiers (`-UD-` stripped by `_UD_TIER_PATTERN`). **Missing**: dense `MXFP4` (regex line 165 has only `MXFP4_MOE`). **Tie-break**: `quality_rank` duplicates (5,5; 9,9; 10,10; 12,12) — pickers must tie-break on `bits_per_weight`.

---

**Top-3**

1. `Qwen36MoeForCausalLM` missing from `_ARCH_TO_REASONING_PARSER` (vllm.py:99-117).
2. `GptOssMoeForCausalLM` in `SPEC_BLOCKLIST` (spec_decode.py:45) does not match canonical `GptOssForCausalLM`.
3. `SPEC_BLOCKLIST` (spec_decode.py:42) is not hardware-gated despite sm-dependent research.
