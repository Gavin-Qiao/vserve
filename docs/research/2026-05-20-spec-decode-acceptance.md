# Spec-Decode Acceptance Rates and Net-Speedup Curves

**Date:** 2026-05-20. **Scope:** empirical (target, draft) acceptance, domain/temp/KV-quant sensitivity, per-arch `n_speculative_tokens` defaults for single-GPU RTX PRO 5000 48GB Blackwell sm120.

## 1. Method-level acceptance (chat, T=0, batch=1)

| Target / draft | Method | tau | alpha | Speedup |
|---|---|---|---|---|
| Vicuna-13B / EAGLE-2 head | EAGLE-2 | 4.83 MT / 5.41 HE | ~0.62 | 4.26x / 4.96x |
| Vicuna-13B / EAGLE-3 head | EAGLE-3 | 6.65 MT / 7.54 HE | ~0.80 | 5.58x / 6.47x |
| Llama-3.1-8B / EAGLE-3 head | EAGLE-3 | 6.13 MT / 6.23 GSM | 0.75-0.85 | 4.40x / 4.48x; 1.8x@QPS=1 |
| Llama-3.1-70B / 8B classic draft | draft | ~5 | **0.765** (1K) -> **0.509** (2K) | 2.31x@c=1, parity@c>=16 |
| Llama-3.1-70B / Qwama-0.5B | draft | ~3 | **0.535** | <1.0x@c>=2 |
| Llama4-Maverick / EAGLE | EAGLE | -- | -- | 1.4-2.0x prod; 4ms/tok@B=1, 8xH100 |
| DeepSeek-V3 / MTP d=1 | mtp | -- | **alpha_1=0.85-0.90** | 1.8x decode |
| DeepSeek-V3 / MTP d=2,3 (SGLang) | mtp | 2.18 (k=3) / 2.44 (k=4) | drops after pos 1 | 1.25-2.11x |
| Qwen3.6-35B-A3B FP8 / DFlash n=15 (B200) | DFlash | **5.3-7.7**/block; pos-0 0.85, pos-14 0.12-0.20 | -- | 2.29x@c=1, 2.49x@c=3 |
| Qwen3.5 / DFlash | DFlash | -- | -- | 5.2x HE / 4.7x Math500 / 3.0x MT@c=1 |
| Gemma-4 / EAGLE-3 | EAGLE-3 | k=3: 2.07-3.15; k=5: 2.20-3.93 | -- | -- |
| Qwen3.6-35B-A3B / Qwen3.5-0.8B on RTX 3090 | draft | 1.0 | 1.0 (100% accept) | **-10.7% to -14.6%** |

Sources: [EAGLE-3 Tab.1](https://arxiv.org/html/2503.01840v1); [RH 2025-07](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding); [SqueezeBits](https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301); Meta [2508.08192](https://arxiv.org/abs/2508.08192); [DS-V3 TR](https://arxiv.org/html/2412.19437v1); [LMSYS 2025-07](https://www.lmsys.org/blog/2025-07-17-mtp/); [A.Kuo](https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b); [Kaitchup](https://kaitchup.substack.com/p/dflash-for-qwen35-eagle-for-gemma); [thc1006](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090).

## 2. Domain sensitivity (generic untuned draft, 99,768 nodes)

| Domain | alpha | E[L]/step | Verdict |
|---|---|---|---|
| Chat | 0.565 | **1.065** | positive |
| Code | 0.538 | 0.975 | marginal |
| Reasoning | 0.532 | 0.956 | marginal |
| Math | 0.518 | 0.914 | **net-negative** |

Source: [Goyal 2604.14682](https://arxiv.org/html/2604.14682). EAGLE-3's *trained* head inverts this (code tau=7.54 > chat tau=6.65 -- fixed templates). **Draft quality dominates domain only when draft is tuned on similar data.**

## 3. Temperature dependence

Llama-2-13B + Llama-68M / Alpaca ([Yang 2410.10141](https://arxiv.org/html/2410.10141v1)):

| Temp | 0.0 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| Speedup | 2.23x | 1.72x (-23%) | 1.60x (-28%) | 1.35x (-39%) |

Distill draft at serving temp. T=0.6 chat: ~10-15% degradation vs T=0 unless tuned.

## 4. KV-quant interaction

- **DFlash + any KV-quant = broken** ([vllm#41559](https://github.com/vllm-project/vllm/issues/41559)). DFlash needs non-causal cross-attn; FLASH_ATTN/FLEX reject fp8 on non-causal, FLASHINFER/TRITON/TURBOQUANT reject non-causal entirely. Forces BF16 KV (halves capacity, 28K->14K on 24GB). [PR #39995](https://github.com/vllm-project/vllm/pull/39995) in-flight.
- **MTP on Qwen3.6-27B-FP8 long-ctx:** illegal memory access ([vllm#40756](https://github.com/vllm-project/vllm/issues/40756)).
- **QuantSpec** (Q4 KV in drafter only): alpha>0.90 vs FP16 -- drafter-side quant safe ([2502.10424](https://arxiv.org/pdf/2502.10424)).
- **N-gram + FP8 KV** composes fine, 84% lower $/serve prefill-heavy ([Kwak](https://medium.com/@injae.kwak/part-2-optimizing-llm-inference-speculative-decoding-and-quantization-on-vllm-with-google-cloud-54b91f018496)).

## 5. Net-speedup inversion

- **MoE-A3B on Ampere:** every config net-negative (-1.5% to -14.6%) despite 100% accept; experts need batch ~94 ([thc1006](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)). DFlash on B200/sm120 inverts to 2.29x@c=1 -- blocklist is **hardware-conditional**.
- **Llama-3.1-70B + 8B draft:** parity at conc=16 (1K) / conc=8 (2K); above, vanilla wins ([SqueezeBits](https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301)).
- **EAGLE-3 alpha falls with k** (gpt-oss-120b, [RH 2026-04](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss)): k=2 0.454/1.91, k=3 0.356/2.07, k=4 0.283/2.13. **k=3 wins**, persists to 200 conc.
- **MTP n=1 is worst non-zero** on Qwen3.6-27B FP8 (-19% decode); min useful depth **n=2** ([A.Kuo](https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b)).

## 6. 2026 defaults

vLLM has **no global default** for `num_speculative_tokens` ([docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)).

| Method | `n_spec` | Source |
|---|---|---|
| MTP | start 1, bump 2-3 | vLLM MTP docs |
| EAGLE-3 | **3** (prod), 5 if alpha>0.75 | RH 2026-04 |
| Draft | 3-5, never >5 | BentoML, SqueezeBits |
| N-gram | 5 (range 1-5) | vLLM ngram default |
| DFlash | 15 | A.Kuo, Kaitchup |

---

## vserve action items

1. **Per-method `n_max`** (today fixed at 5 for all). Set MTP n_max=3 n_min=2; EAGLE-3 n_max=3; draft n_max=3; ngram n_max=5; DFlash n_max=15. Cite [RH 2026-04](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss) k=3 alpha curve.
2. **Add Llama3/4 + EAGLE-3** to `SPEC_METHOD_BY_ARCH` as `"eagle3"` when a `-EAGLE3` sibling head is detected. tau=6.13/4.4x on 8B is the strongest reproduced ([2503.01840 Tab.1](https://arxiv.org/html/2503.01840v1)).
3. **Add DeepSeek V3/V4** as `"mtp"` with `n_max=3, n_min=2` (n=1 degenerate). [DS-V3 TR](https://arxiv.org/html/2412.19437v1) alpha_1=0.85-0.90; [LMSYS](https://www.lmsys.org/blog/2025-07-17-mtp/) k=3 tau=2.18.
4. **Hardware-condition the A3B/MoE blocklist** by compute capability. Block sm<=8.9 ([thc1006](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) -10% to -14%); allow sm>=9.0 with DFlash ([A.Kuo](https://allenkuo.medium.com/when-speculative-decoding-helps-local-llms-and-when-it-doesnt-5c41dd804e4b) B200 2.29x@c=1). PRO 5000 = sm120 -> permit.
5. **Move kv_cache_dtype check into `pick_spec_config()`.** Refuse dtype not in {auto, bfloat16, fp16} AND method in {dflash, eagle3} until [vllm#39995](https://github.com/vllm-project/vllm/pull/39995) lands. Today the check is only in `start()`; the chooser silently returns an unservable config.
