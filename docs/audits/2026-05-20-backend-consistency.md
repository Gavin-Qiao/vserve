# Backend consistency audit: vllm.py vs llamacpp.py (2026-05-20)

Drift between `VllmBackend` (872 LOC) and `LlamaCppBackend` (1,484 LOC).

## 1. Per-method comparison

| Method | vLLM | llama.cpp | Drift |
|---|---|---|---|
| `__init__` | 385 | — | vllm caches parser registry. |
| `runtime_info` | 420 (kwargs) | 120 | vllm takes `prefer_cache=`/`with_pip_check=`; llamacpp ignores. |
| `compatibility` | 428 | 148 | vllm returns `CompatibilityResult`; llamacpp returns dict. Protocol violation. |
| `tune` | 435 | 178 | vllm: `recommended_profile`; llamacpp: `recommended_kv_dtype`+`kv_cache_dtypes`. |
| `available_tool_parsers`/`_reasoning_parsers` | 462/467 | — | vllm-only; llamacpp drops `tool_parser`/`reasoning_parser` silently. |
| `start`/`stop`/`is_running` | 833/837/841 (delegates) | 1215/1362/1374 (140 LOC inline) | Biggest spaghetti signal. |
| `_assert_unit_safe`, `_write_failed_manifest`, `_restore_active_links` | — | 95/1470/1455 | llamacpp safety net; vllm equivalents in `serve.py:160-220`. |
| `detect_tools` | 853 | 1398 | vllm: `{tool_call_parser, reasoning_parser}`; llamacpp: `{supports_tools, supports_reasoning}`. Disjoint. |
| `quant_flag` | 829 | 1211 | vllm delegates; llamacpp returns `""`. |

## 2. `build_config(choices)` key drift

- slot count: vllm `slots` (l.622) vs llamacpp `parallel` (l.1009).
- KV dtype: vllm `kv_dtype` single (l.623) vs llamacpp `kv_cache_k`+`kv_cache_v` pair (l.1025).
- vllm-only consumed (dropped by llamacpp): `tool_parser`, `reasoning_parser`, `gpu_mem_util`, `attention_backend`, `gpu_compute_cap`, `performance_mode`, `optimization_level`, `block_size`, `kv_cache_memory_bytes`, `trust_remote_code`, `override_generation_config`.
- llamacpp-only consumed (dropped by vllm): `n_gpu_layers`, `n_cpu_moe`, `mmproj`, `flash_attn`, `cache_reuse`, `slot_save_path`, `cram_mb`, `swa_full`, `override_tensors`, `reasoning_format`, `reasoning_budget`, `embedding`, `pooling`.
- shared: `context`, `port`, `tools`, `thinking`, `chat_template_kwargs`, `recipe_sampling`, `spec`.
- No `_validate_choices` — unknown keys vanish.

## 3. Hoist candidates

- `_ARCH_TO_TOOL_PARSER`/`_ARCH_TO_REASONING_PARSER`/`_ARCH_FORCES_BACKEND` (vllm.py:39/99/157) — llamacpp re-encodes via `gguf_arch.lower().startswith(...)` at llamacpp.py:1112/1119/1125. Lift to `backends/_arch.py`.
- `_KV_DTYPE_BYTES_PER_ELEMENT` (llamacpp.py:29) belongs next to vLLM KV math in `probe.py`.
- `_read_model_config` (vllm.py:120) + `_read_gguf_metadata` (llamacpp.py:639) — hoist `read_arch(ModelInfo) -> ArchFacts`.
- systemctl `is_running` ladder duplicated at `serve.py:188` and `llamacpp.py:1374` — move to `vserve/systemd.py`.
- `BACKEND_INCOMPATIBLE_KV_DTYPES` (vllm.py:177) and `_FA_ALL_QUANTS_GATED_DTYPES` (llamacpp.py:47) share `dict[variant, frozenset[dtype]]`.

## 4. Asymmetric features

- Parser-registry probing + validation (vllm.py:472, 570) — none in llamacpp.
- Lifecycle safety net (llamacpp.py:95/1455/1470) — vllm equivalents in `serve.py:160-220` under different names.
- KV-dtype filtering: vllm post-tune (`_filter_incompatible_kv_dtypes` vllm.py:338) vs llamacpp pre-tune (`_candidate_kv_dtypes` l.159).
- Forced-KV-dtype-on-quant: vllm fp8 on NVFP4 (vllm.py:679); llamacpp f16 V on Gemma-4 (llamacpp.py:1127). Same pattern, separate code.

## 5. Recommended refactors

1. **`vserve/systemd.py:is_active(unit)`** — removes duplicate state machine (`serve.py:188`, `llamacpp.py:1374`).
2. **`backends/_arch.py`** with `read_arch()` + the three `_ARCH_TO_*`/`_ARCH_FORCES_*` tables + family helpers — replaces the prefix-match chain at `llamacpp.py:1111-1125`.
3. **llamacpp returns `CompatibilityResult`/`RuntimeIdentity`** (`llamacpp.py:120, 148`); drop `| Any` at `protocol.py:87, 91`.
4. **Symmetrize `build_config`**: rename llamacpp `parallel`→`slots`; accept `kv_dtype` and split to `(kv_cache_k, kv_cache_v)`. Add `_validate_choices` (mirror vllm.py:570) rejecting unknown keys.
5. **`LifecycleMixin`** (`_assert_unit_safe`, `_snapshot_active`, `_restore_active`, `_write_failed_manifest`). 1-line `vllm.start` vs 147-line `llamacpp.start` is the strongest spaghetti signal — both delegate or both inline.

## 6. Patch sediment

Surgical and well-commented: `cudagraph_mode: NONE` (vllm.py:761-771), Gemma-4 V→f16 (llamacpp.py:1125-1136), SM120/GPT-OSS conditional (vllm.py:186-213). Needs cleanup: `--fit off` (llamacpp.py:1302) hardcoded in `start` not `build_config`; `--n-cpu-moe` rewrite (llamacpp.py:1064-1069) silently mutates `override_tensors`.

Bottom line: both agree on Protocol's 15 methods but diverge on lifecycle placement, `build_config` keys, and `tune`/`detect_tools` shapes. All fixable without behavior change.
