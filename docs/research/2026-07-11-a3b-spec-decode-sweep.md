# Qwen3.6-35B-A3B-NVFP4 spec-decode sweep — every method is net-negative

**Date:** 2026-07-11. **Hardware:** 1× RTX PRO 5000 Blackwell (48 GB, sm120).
**Runtime:** vLLM 0.24.0. **Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE, ~35B
total / ~3B active, ModelOpt NVFP4, in-checkpoint MTP `mtp_num_hidden_layers=1`).
**Base profile (`qwen64k`):** 64k context, 8 slots, fp8 KV, gpu-util 0.90,
text-only (`--language-model-only`), 8192 batched tokens. **Bench:** `vserve
bench`, default prompt, max-tokens 512, c1 = 60 s / c8 = 75 s.

## Result

| variant | c1 tok/s | c1 Δ | c1 accept | c8 tok/s | c8 Δ | c8 accept |
|---|---|---|---|---|---|---|
| **off (no spec)** | **220.0** | base | — | **1101.1** | base | — |
| ngram (k=5) | 134.8 | −39% | 9% | 352.2 | **−68%** | 14% |
| MTP k=1 | 135.6 | −38% | **82%** | 728.5 | −34% | 82% |
| MTP k=2 | 107.0 | −51% | 70% | 528.5 | −52% | 70% |
| MTP k=3 | 105.0 | −52% | 60% | 517.0 | −53% | — |

(MTP k=3 measured 2026-07-10, identical base; the rest 2026-07-11.)

## Reading

- **Off wins decisively at every concurrency.** No speculative-decoding method
  — ngram or MTP at any depth — is net-positive on this model on this card.
- **The k=1 lead is refuted.** The SGLang research surfaced a report of vLLM
  MTP ~+27.5% at k=1 on "this model class"; here k=1 is the *least bad* MTP
  depth but still −38% at c1. It does not transfer to this checkpoint.
- **High acceptance does not save it.** MTP k=1 accepts 82% of drafted tokens
  and is still net-negative — direct confirmation that the cost is the
  extra-forward + expert-traffic tax on the verify step, not poor drafting.
  This is the mechanism the 2026-05-20 acceptance research and the SGLang c6
  verifier both flagged as *architectural* for extreme-sparsity MoE on a
  bandwidth-bound card, not an implementation artifact.
- **Depth hurts monotonically.** Acceptance falls 82% → 70% → 60% from k=1→3
  and throughput tracks it down; deeper trees multiply wasted expert reads.
- **ngram is worst under load.** 9–14% acceptance (generative text has little
  to look up) and a −68% collapse at c8 — the verify step steals batch slots
  from real decode.

## Consequence for vserve

- `Qwen3_5MoeForConditionalGeneration` (and its GGUF twin) stay in
  `recipes.spec_decode.SPEC_BLOCKLIST`: `--spec auto` never recommends spec
  decoding here. Explicit `--mtp` / `--spec mtp` remain available for
  experiments. The `qwen64k` serving profile keeps spec **off**.
- This sweep is now reproducible as a first-class command:
  `vserve tune <model> --sweep spec` (0.6.8b3) automates the boot→bench→
  scrape→rank→restore loop this document was produced by hand.

## Open threads

- **DFlash** on sm120 is still unmeasured (2026-05-20 research reported
  2.29×@c1 on B200; unknown whether the sm120 kernel path exists in vLLM 0.24).
- **Dense Qwen3.6-27B-FP8** MTP is untested on this fleet — the expert-traffic
  mechanism is MoE-specific, so a dense model may behave differently; worth a
  `vserve tune qwen3.6 27b --sweep spec` when that model is resident.
