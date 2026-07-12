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

## Follow-up: "fastest 8×64k" — regime, MoE backend, and MTP-at-saturation

Same box/model/runtime, 2026-07-11. Distinct-prompt concurrent decode
(`scratchpad/longctx_bench.py`, 8 streams, no prefix sharing).

**Throughput has two very different numbers depending on actual KV occupancy:**

| workload | aggregate | per-stream TPOT | bottleneck |
|---|---|---|---|
| 8× short prompt (64k *window*, short convo) | ~1100 tok/s | 7 ms | MoE experts |
| 8× **~54k actual context** (distinct) | **58 tok/s** | 36 ms | attention over long KV |

At real 54k occupancy, per-token cost is dominated by reading 8×54k of KV every
decode step — fp8 KV (which we already run) is the single biggest lever there,
halving that bandwidth vs bf16 (and bf16 doesn't even fit 8×64k). TTFT for
distinct 54k prompts is 20–33 s; prefix caching (on) makes that ~free on the
reuse that real agent sessions have, so the effective number for shared-prefix
multi-turn sits well above 58.

**MoE backend is MARLIN, and that is the ceiling.** vLLM's NvFp4 oracle auto-picks
`MARLIN` (W4A16 dequant→bf16) on sm120. Forcing the native FP4 kernels fails hard:
`VLLM_CUTLASS` → "kernel does not support quantization scheme QuantKey(u8,
scale(f8e4m3fn…))"; `flashinfer_cutlass`/`flashinfer_trtllm`/`flashinfer_cutedsl`
→ crash-loop (unsupported on capability 120). So no faster MoE path exists on this
card — consistent with the SGLang research's sm120-NVFP4-broken finding.

**MTP at saturating context is *worse*, not better (hypothesis refuted).** The
short-context verdict (MTP net-negative, expert-traffic tax) raised the question:
at long context, where attention dominates, does MTP's one-forward-yields-many
amortize the expensive KV read? Measured at 8×54k:

| config | aggregate | per-stream TPOT | acceptance |
|---|---|---|---|
| off (MARLIN) | **58 tok/s** | 36 ms | — |
| MTP k=1 | 29.7 tok/s (**−49%**) | 55 ms | 86% |

The amortization doesn't happen: the Qwen MTP draft layer is a full decoder layer
that reads KV to *generate* the draft, then the verify forward reads KV again —
~2 KV-reading passes per step. At long context each read costs more, so 86%
acceptance (1.86× tokens) can't cover ~2× KV bandwidth + the expert tax. MTP is
net-negative at short (−38%) *and* saturating (−49%) context; the loss grows with
context length. k=1 was the best case (highest acceptance), so k=2/k=3 were not
run.

**Bottom line — fastest 8×64k A3B:** the `qwen64k` profile as-is (NVFP4 + MARLIN
MoE + fp8 KV + gpu-util 0.90 + text-only + **no spec decode**). Every knob is at
its optimum: fp8 KV is mandatory (bf16 fits only 4 slots at 64k) and doubly
valuable at long context; ns=8 matches the concurrency; MARLIN is the only working
MoE kernel; native FP4 and all spec methods are ruled out empirically.

## Fastest 8-concurrent **32k** (production ask, 2026-07-11)

Relaxing 64k→32k halves the KV, so true-context throughput ~doubles. Guide
research (4-agent workflow) confirmed the config sits on the **official
recipes.vllm.ai recipe for this exact checkpoint** (fp8 KV, flashinfer attn,
marlin MoE, bt 8192, prefix-caching). Every additional knob was tested:

| config (8 concurrent) | short-prompt agg | true 8×29k agg | 8×29k TPOT | verdict |
|---|---|---|---|---|
| **qwen32k** (fp8, ns8, util0.90, bt8192) | **1091 tok/s** | **129.5 tok/s** | 21.7 ms | **winner (= recipe)** |
| bf16/auto KV | 1086 | 136.7* | 23.3 ms | worse decode + 2× KV; *agg is TTFT-noise |
| max-num-batched-tokens 16384 | 1108 | 129.6 | 20.4 ms | flat (prefill FLOPs fixed); more VRAM, no gain |
| max-num-partial-prefills 2 | — | — | — | **NotImplementedError** on 0.24/hybrid-attn |

- **fp8 e4m3 KV stays** — lower decode TPOT (21.7 vs 23.3 ms, the KV-bandwidth
  halving) and half the memory. #39137 (auto-KV-on-NVFP4) is fixed in 0.24, so
  bf16 is *correct* but not faster. nvfp4 KV still crashes on sm120 (#43562).
- **max-num-seqs stays 8** — inert above the concurrency ceiling (the "ns is the
  lever" result only held when >ns requests were offered).
- **gpu-util stays 0.90** — at ns=8 KV is ~3% used, so extra VRAM is unused;
  raising util *shrinks* the activation headroom that caused the 0.97 crash.
- **GPU boost verified** — 2257–2587 MHz / P1 / ~300 W under load (a headless
  sm120 throttle-to-180 MHz bug the research flagged does NOT apply here).
- **The one real future lever: `flashinfer_b12x`** — a native NVFP4 fused-MoE
  that replaces the MARLIN dequant, +6% at 8-way concurrency on the sibling
  Qwen3-30B-A3B-NVFP4 — but it landed in **vLLM 0.25** (PR #40082), outside the
  pinned `<0.25` range. A pin bump + FlashInfer-from-source build is the only
  path to beat MARLIN on decode; modest and not "soon"-safe.

**Production answer — fastest 8×32k A3B:** the `qwen32k` profile as-is. Decode is
maxed on 0.24 (~130 tok/s true-32k, ~1090 short); no config change beats it. The
movable production levers are workload-shaped: prefix caching (structure shared
preambles → skip most of the 32k prefill) and staggered arrivals (the ~9s TTFT
is worst-case: 8 distinct 32k prompts hitting a cold server at once).

## Open threads

- **DFlash** on sm120 is still unmeasured (2026-05-20 research reported
  2.29×@c1 on B200; unknown whether the sm120 kernel path exists in vLLM 0.24).
- **Dense Qwen3.6-27B-FP8** MTP is untested on this fleet — the expert-traffic
  mechanism is MoE-specific, so a dense model may behave differently; worth a
  `vserve tune qwen3.6 27b --sweep spec` when that model is resident.

## 2026-07-12 update — vLLM 0.25.0 adopted; flashinfer_b12x measured on sm120

The "one real future lever" above (native `flashinfer_b12x` NVFP4 MoE, gated on
vLLM 0.25) was tested end-to-end after 0.25.0 shipped (2026-07-11). Same box,
same qwen32k profile — only the MoE backend varied.

**Adoption.** vserve pinned-stable → 0.25.0 (range `>=0.20,<0.26`). qwen32k on
0.25 with **MARLIN** benched **c1 235.1 / c8 1155.9 tok/s** (short prompt,
max-tokens 512, 60/75 s) — no regression vs 0.24 (~220 / ~1091), a ~6% uplift.
MARLIN is still what 0.25's NVFP4 MoE oracle auto-selects on sm120
(`Using 'MARLIN' NvFp4 MoE backend`).

**flashinfer_b12x A/B (first sm120 RTX PRO 5000 data point).**
`--moe-backend flashinfer_b12x` — oracle-*excluded*, opt-in; the checkpoint has
no `swiglu_limit` so it is not blocked; CuteDSL-JIT at boot, ~7 GB host (far
lighter than the MARLIN nvcc storm). Confirmed active in-log
(`Using 'FLASHINFER_B12X' NvFp4 MoE backend`), output correct (' Paris, …'):

| backend | c1 tok/s | c8 tok/s | c1 TPOT | c8 TPOT |
|---|---|---|---|---|
| MARLIN | 235.1 | 1155.9 | 4.1 ms | 6.6 ms |
| flashinfer_b12x | 235.9 | 1120.4 | 4.1 ms | 6.9 ms |

Parity at c1, ~3% slower at c8. The "+6%" from PR #40082 is b12x vs
flashinfer-cutlass (which doesn't run on sm120), on SM121/DGX-Spark — never vs
Marlin. **Verdict: MARLIN stays the sm120 decode ceiling; b12x is not a win.**
(b12x's first request pays a one-time ~8.7 s autotune stall → cold c1 aggregate
202.8, warm 235.9; steady-state TPOT is identical to MARLIN.)

**Host-RAM footnote.** The first 0.25 boot re-JITs FlashInfer's sm120 kernels
(flashinfer 0.6.12→0.6.13 cache miss); with the `MAX_JOBS`/`NVCC_THREADS` caps
missing from `.env`, uncapped `cicc` (~4 GB each) hit the 50 GB cgroup guard and
was OOM-killed. Fix: the caps are now emitted by `vserve run` (0.6.8b4). Capped
at `MAX_JOBS=4`, the compile peaks ~48 GB transiently (reclaimable) and completes
in ~6.5 min; the cache makes subsequent boots ~1 min.
