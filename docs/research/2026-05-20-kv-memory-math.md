# KV-Cache Memory Math, Block-Allocator Overhead, OOM Heuristics

**2026-05-20.** Single-GPU budget prediction (RTX PRO 5000 48GB Blackwell) for vLLM + llama.cpp.

## 1. KV bytes-per-token: textbook formula no longer suffices

The canonical `2 * n_layers * n_kv_heads * head_dim * bytes_per_elem * ctx * slots` is correct only for *plain GQA, all-global*. Three 2025-2026 shifts break it:

- **MLA (DeepSeek-V3/R1):** per-token cache collapses to `(d_c + d_rope) * n_layers * dtype_bytes` (512+64 in V3) -- ~98.6% reduction vs naive MHA. ([MLA writeup](https://mccormickml.com/2025/04/26/inner-workings-of-mla/))
- **Hybrid SWA (Gemma 3/4, Llama 4):** 5:1 local:global. Local layers allocate only `min(ctx, sliding_window)` tokens. vLLM's Hybrid KV Cache Manager (v0.10+) uses `page_size = block_size * kv_hidden_size * num_layers_per_group`, grouping by `min(n_sliding, n_global)` (e.g. 10 in Gemma-3-27B). Reported overhead drops from 60% to <15% vs global-only. ([HMA design](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/), [Gemma 3 report](https://arxiv.org/pdf/2503.19786))
- **FP8 KV on Blackwell:** halves bytes/token; vLLM's April 2026 two-level FP32 accumulator restored 128k NIAH from 13% -> 89% accuracy, but `head_dim=256` (Gemma global, Qwen3) still regresses on prefill -- FP8 auto-pin should skip SWA layers. ([vLLM FP8 KV blog 2026-04-22](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html))

## 2. vLLM PA block-allocator overhead

PagedAttention zeros external fragmentation but bounds internal frag by `block_size`:
- vLLM converged on **block_size=16**; size 32 has up to **1.9x kernel-time variance**; PA adds ~20-26% per-kernel latency but yields 2-4x end-to-end throughput; total waste <4%. ([Kwon et al. 2309.06180](https://arxiv.org/pdf/2309.06180); [Red Hat 2025-07](https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems))
- Concrete (vllm#39133): Gemma-4-31B-INT4 BF16-KV on 2x24GB at `gpu_memory_utilization=0.96` -> **25,200 tokens / 23.08 GiB = ~1,093 tokens/GiB** -- vLLM under-uses SWA without HMA. ([vllm#39133](https://github.com/vllm-project/vllm/issues/39133))
- Published accounting: `available = total_vram * gpu_memory_utilization - weights - non_torch - activation_peak`. Default `gpu_memory_utilization=0.92` in V1. ([vllm#13803](https://github.com/vllm-project/vllm/discussions/13803))

## 3. llama.cpp compute-buffer: no closed form

ggerganov (#10068) explicitly states **no closed-form formula exists**; community guidance is to fit linear `mem = k*n_ctx + offset` across 2k/4k/8k probes with `-ngl 0`, **per model, per backend, per FA flag**. Buffer sizes change with any backend update. Knobs: `-b` (logical, default 2048), `-ub` (physical, default 512); `-b 4096 -ub 4096` is the high-throughput pin. Flash-attention eliminates the quadratic attention-score buffer. ([#10068](https://github.com/ggml-org/llama.cpp/discussions/10068))

## 4. Practical max-(ctx x slots) heuristics

- vLLM V1 budgets `max_num_seqs * max_model_len`; defaults `max_num_seqs=1024`, `max_num_batched_tokens=8192` online / 16384 offline; full-cudagraph throughput recommends **>=32k batched**. ([vLLM optim docs](https://docs.vllm.ai/en/stable/configuration/optimization/))
- `vllm/benchmarks/auto_tune` sweeps `(max_num_seqs, max_num_batched_tokens)` with latency-SLA + prefix-hit constraints; v0.8.5 -> v0.11.0 tuned configs delivered ~2x throughput on identical HW. ([auto_tune README](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md))
- AI21 measured **0.16 GB/request** empirical KV on Qwen2.5-7B-Int8 at 3072 ctx -- ~29x lower than textbook 4.59 GB, because PA + prompt-length variance leaves blocks partially full. Driving lesson: **measure on real prompt distribution**. ([AI21 scaling vLLM](https://www.ai21.com/blog/scaling-vllm-without-oom/))

## 5. Prefix-cache footprint

vLLM APC is near-free metadata-wise (1M:1 data:metadata, 8-byte hash per 128-token block) but **consumes KV-budget linearly** on retained prefixes; global LRU eviction. T-LRU (RFC #37823, 2026) cuts P95 TTFT 27.4% on conversation workloads. llama.cpp's `--cache-reuse` + `/slot/save` stores Q4_K_M at ~8 B/token in host RAM (e.g. 10x10k = 800MB); per-slot save adds ~232 B/token overhead vs raw KV. ([llm-d blog](https://llm-d.ai/blog/kvcache-wins-you-can-see); [llama.cpp#20574](https://github.com/ggml-org/llama.cpp/discussions/20574); [vllm#37823](https://github.com/vllm-project/vllm/issues/37823))

## 6. Empirical OOM curves

Closed-form formulas are unreliable; production guides converge on **measure-then-fit**. AI21 drives recipes from prompt-length probes; llama.cpp fits linear per-model curves; vLLM's auto_tune is empirical grid search. Lyceum's interactive calculator is the most-cited bytes/token reference. ([Lyceum calc](https://lyceum.technology/magazine/kv-cache-memory-calculation-llm/))

---

## vserve action items

1. **Per-arch KV-bytes-per-token predictor** (`kv_bpt.py`): table-driven; MLA, SWA-hybrid, GQA branches; consumes `config.json` head_dim/n_kv_heads/sliding_window/layer_types. Today vserve has per-arch head_dim only for Gemma-4 (item E) -- extend to MLA + Llama-4 + Qwen3.5. *Justification:* vllm#39133 shows naive math is off ~5x on SWA models.
2. **Hybrid-KV-aware vLLM budget math**: when HMA is enabled, compute budget per *kv-cache-group* (full vs sliding) and sum, mirroring HMA design doc. *Justification:* avoids over-pinning `max_model_len` on Gemma-3/4 -- [HMA docs](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/).
3. **`vserve tune` wrapping `vllm/benchmarks/auto_tune`**: sweep `(max_num_seqs, max_num_batched_tokens)` under latency-SLA + prefix-hit floor; emit `tuned.toml`. *Justification:* tuned v0.11 yields ~2x throughput vs defaults -- [auto_tune README](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md).
4. **Prefix-cache accounting in `vserve doctor`**: subtract `--reserved-prefix-tokens` from KV before computing slots; surface LRU vs T-LRU. *Justification:* APC is linear in retained prefixes -- [llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see).
5. **llama.cpp compute-buffer 3-point linear fit**: replace analytic math (#34) with 2k/4k/8k probe + `mem = k*n_ctx + offset` per (model, FA, kv-quant). *Justification:* ggerganov says no closed form is correct -- [#10068](https://github.com/ggml-org/llama.cpp/discussions/10068).
