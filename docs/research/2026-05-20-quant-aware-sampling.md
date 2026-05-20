# Should `SAMPLING_DEFAULTS` Become Quant-Aware?

**2026-05-20.** Audit of `recipes/sampling.py` (vserve 0.6.1b1) against published guidance through May 2026. Today's defaults are arch-keyed only; UNSLOTH_QUANT_TIERS does not influence sampler emission. The question: is that wrong?

## 1. Logit-distribution sharpness vs bit depth

KL-divergence to FP16 is the canonical sharpness proxy; llama.cpp's `perplexity` tool emits per-token KL ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)). "An Empirical Study of Qwen3 Quantization" ([2505.02214](https://arxiv.org/abs/2505.02214)) reports notable degradation only below 4 bits; Q4-Q8 PPL stays within ~6% of FP16, and the paper does **not** examine logit entropy or recommend sampler changes. DeepSeek's "Quantitative Analysis" ([2505.02390](https://arxiv.org/abs/2505.02390)) finds Q4 maintains "little performance degradation versus FP8" across MATH/AIME/MBPP, silent on sampling. No paper publishes a sharpness-vs-quant curve recommending a temperature delta.

## 2. KV-cache quantization

vLLM's April 2026 blog ([fp8-kvcache](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)) reports FP8 KV + FP8 attention degrades reasoning by 1-2 points after the FP32-accumulator fix; evaluations explicitly "adopt the default non-greedy sampling parameters suggested by model creators" -- i.e., no sampling adjustment was needed to recover. [Localbench's KL-divergence benchmark](https://localbench.substack.com/p/kv-cache-quantization-benchmark) shows Gemma is far more sensitive to KV-quant than Qwen (Gemma q8_0 KV ~= Q5 weight quant), but again offers no sampling guidance.

## 3. Unsloth per-model recipes are NOT quant-tiered

The single most important finding: every Unsloth model card and the Dynamic-2.0 docs ([unsloth-dynamic-2.0-ggufs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)) publish one sampling recipe per `(model, mode)` pair. UD-Q4_K_XL, Q5_K_M, Q6_K, Q8_0 all share identical recommended `temp/top_p/top_k`. Verified across [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5), [Qwen3.6](https://unsloth.ai/docs/models/qwen3.6), [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF), [DeepSeek-R1 RedHat deploy](https://developers.redhat.com/articles/2025/03/03/deployment-ready-reasoning-quantized-deepseek-r1-models). NVIDIA's NVFP4 blog ([developer.nvidia.com](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)) reports DeepSeek-R1 MMLU drop of 0.1pt FP8 -> NVFP4 and recommends no sampler change.

## 4. Min_p literature

Nguyen et al.'s original min-p paper ([2407.01082](https://arxiv.org/html/2407.01082v8)) recommends 0.05-0.1 baseline tuned to **temperature, not quant**. The 2026 critique "Min-p, Max Exaggeration" ([2506.13681](https://arxiv.org/pdf/2506.13681)) further attacks min-p's claimed benefits without mentioning quant. llama.cpp's default `min_p=0.1` is widely considered too aggressive; Unsloth's [GLM-4.7-Flash thread](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/13) explicitly recommends `--min-p 0.01` (which matches our current value).

## 5. Repetition penalty

A strong empirical signal: Unsloth's pinned [GLM-4.7 thread](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/13) (40+ users) concludes `repeat-penalty` *causes* looping at all GGUF quant tiers and should default to 1.0/off. Qwen3-VL Thinking [discussion #1](https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/discussions/1) converged on `presence_penalty=1.5, repeat_penalty=1.0` across UD-Q4_K_XL through Q8 -- again, **not** quant-tiered. Our `Qwen3CoderForCausalLM` row currently emits `repeat_penalty=1.05`; this contradicts current community guidance.

## 6. Thinking-mode sensitivity

Reasoning models DO degrade more under aggressive quant: NVIDIA's [NVFP4 QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf) notes that pure NVFP4 PTQ "breaks the model's capabilities" on reasoning, requiring distillation rather than sampling tweaks to recover. There is no evidence that raising temperature compensates for quant-induced reasoning loss.

## 7. vLLM `override-generation-config` x prefix caching / spec decode

`--override-generation-config` merges into `generation_config.json` parsed at engine init -- it is a load-time argument, not per-request, so it does NOT participate in prefix-cache hashing (hashes are over token IDs only -- [prefix caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/)). EAGLE3 spec-decode is "algorithmically validated to be lossless" at any temperature ([speculators v0.3 blog](https://blog.vllm.ai/2025/12/13/speculators-v030.html)); acceptance rate falls but sampling distribution is preserved. No surprising interactions documented.

## vserve action items

Net call: **do not introduce a `(arch, quant_tier)` table.** Published empirical evidence is uniform: keep sampling arch-keyed. Concrete edits to `recipes/sampling.py`:

1. **Drop the Qwen3-Coder `repeat_penalty=1.05`** on line 43 -- contradicts the Unsloth pinned recommendation that repeat-penalty causes looping on Qwen3-family GGUFs ([GLM-4.7 thread](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/13), [Qwen3-Coder-Next card](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)).
2. **Set Qwen3-Coder `top_k=40, top_p=0.95, temp=1.0`** -- the Unsloth Coder-Next card now publishes a non-thinking recipe distinct from generic Qwen3 ([Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)).
3. **Align `min_p=0.01` on Qwen3.5 row** (currently 0.0) -- Unsloth's published Qwen3.5 guide states `min_p=0.0` matches the model's `generation_config.json`, but llama.cpp's default of 0.1 truncates too aggressively; vserve's existing `0.01` floor is well-justified by [min-p paper](https://arxiv.org/html/2407.01082v8) tail-truncation guidance and protects against quant-noise outliers at no cost.
4. **Add a startup-banner note** when the loaded GGUF is below Q4_K (Q2, Q3, IQ2, IQ3) flagging "sub-Q4 quant; sampling defaults assume Q4+. Reasoning quality may regress." Source: [Qwen3 empirical study](https://arxiv.org/abs/2505.02214) showing notable PPL break below 4-bit.
5. **No FP8 KV-quant temperature delta.** vLLM's own benchmarks ([fp8-kvcache blog](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)) recovered accuracy via the FP32-accumulator kernel fix without sampler change; pinning FP8 KV is a memory decision, not a sampling one.

`SAMPLING_DEFAULTS` stays a `dict[arch -> SamplingDefaults]`. The `UNSLOTH_QUANT_TIERS` table should remain orthogonal -- it informs VRAM math, not sampler emission.
