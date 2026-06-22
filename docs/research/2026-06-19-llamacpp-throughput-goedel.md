# llama.cpp Batched-Serving Throughput: Goedel-Prover-V2-32B

**Date:** 2026-06-19. **Scope:** empirical, single RTX PRO 5000 48GB Blackwell sm120, llama.cpp b583 (`67ace02`). Target = `mradermacher/Goedel-Prover-V2-32B-i1-GGUF-Q4_K_M` (Qwen2.5-32B dense: 64 layers, 8 KV heads, head_dim 128). Triggered by "why is GPU utilization only ~50% serving Goedel?" Extends [2026-05-20-spec-decode-acceptance.md](2026-05-20-spec-decode-acceptance.md).

## 1. Why utilization reads ~50%

Single-stream and small-batch autoregressive decode is gated by a **fixed per-decode-step overhead** (sampling, scheduling, kernel launch, sync), not by raw compute or bandwidth. `nvidia-smi`'s "GPU-Util" counts *time ≥1 kernel ran*, so it sits ~50–65% even when little is reclaimable. The bubble does **not** shrink with batch size, so larger batches amortize it over more tokens.

Measured np→throughput scaling (shallow context ~300 tok, CTX=16384, C=np, warmup+2 trials, `ignore_eos`):

| np | agg decode tok/s | per-slot tok/s | GPU util avg/max | power | SM clock |
|---|---|---|---|---|---|
| 5 (prod) | ~92 | 18.5–19.4 | 58% / 78% | 254 W (capped) | 2327 MHz |
| 10 | ~158 | 15.8–16.7 | 44% / 97% | 200 W | 2558 MHz |
| 16 | ~210 | 13.4–13.6 | 41% / 100% | 188 W | 2561 MHz |

As np rises, **throughput climbs while avg power falls and clocks rise** — proof the np=5 limiter is the per-step bubble + the 300 W SW power cap (`SW Power Cap : Active`, clocks pulled 3090→~2320 MHz), not the SMs. The card's **Max Power Limit = Default = 300 W** (min 250) — immovable.

At realistic proof depth the win shrinks (attention grows, becomes power-bound). Decode at ~6.2K depth: np=5 → ~43 agg (8.6/slot, util 68%, 262 W max 306); np=10 → ~54 agg (5.4/slot) — only **1.25×** vs 1.7× shallow.

## 2. Doc "speed-ups" that were tested and rejected (llama.cpp, batched)

| Lever | Flag | Result vs baseline | Why |
|---|---|---|---|
| Backend (GPU) sampling | `-bs` | **−5%** (C=1 35.8 vs 37.5; C=5 94.2 vs 99.4) | CPU sampling already overlaps on 24 cores; moving it onto the already-bottlenecked GPU costs time. |
| n-gram spec-decode | `--spec-type ngram-mod` | **−9%** @ C=5 (90 vs 99); ~−4% @ C=1 | Default `n-match=24` barely fires (19 drafts / ~1000 tok), ~26% accept; draft+verify competes with the batch & power cap. |
| Draft-model spec-decode | `--spec-type draft-simple -md …` | not run | Acceptance/economics dominated by §1 of the acceptance note; net-negative at batch and fights the power cap. |

**Verdict:** none of the doc speed-tricks help llama.cpp batched serving on this fleet. They are best-case at batch=1 with high acceptance — the opposite of a best-of-N prover at np≥5. This is why [`recipes/spec_decode.py`](../../src/vserve/recipes/spec_decode.py) keeps spec-decode **off** for llama.cpp (ngram *exists* as of b583 — the old "llama.cpp has no ngram" comment was stale — but stays disabled by measurement, mirroring the A3B-MoE blocklist).

## 3. The real lever, and its hard limits

Throughput scales with concurrency (np), but two hard caps bound it:

- **Model trained context = 40960 tokens.** Requesting more per slot logs `n_ctx_seq (…) > n_ctx_train (40960) … capping` and degrades quality. **49152/slot is not achievable** without unvalidated RoPE/YaRN scaling (not advisable for a prover).
- **VRAM.** In llama.cpp `-c` is *total* context split across `-np` slots, so KV = f(total `-c`), independent of np. KV ≈ **133 KB/token** (q8_0; confirmed: np=6 needed a 33.69 GB KV buffer = 137109 B/tok). Weights (Q4_K_M) ≈ 18.8 GiB. Budget on 48 GB:

| Config | total `-c` | per-slot | VRAM | Fits? |
|---|---|---|---|---|
| np=5 @ 32768 (old prod) | 163840 | 32768 | ~40.5 GB | yes (under-provisioned ctx) |
| **np=5 @ 40960** | **204800** | **40960** | **46.7 GB** | **yes (2.3 GB headroom)** ← optimal |
| np=4 @ 40960 | 163840 | 40960 | ~45.6 GB | yes (fewer slots) |
| np=6 @ 40960 | 245760 | 40960 | ~50 GB | **OOM** (33.7 GB KV buffer) |
| np=5 @ 49152 | 245760 | (caps to 40960) | ~53 GB | **OOM** |

"fp8 KV" = **q8_0** — llama.cpp has no true fp8 KV type (allowed: f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1).

## 4. Optimal config (applied 2026-06-19)

```
-c 204800 -np 5 -ctk q8_0 -ctv q8_0 -fa on   # + temp 0.7 top_p 0.8 top_k 20 min_p 0.01
```

`-np 5 @ 40960/slot` is the unique best point: max slots that fit at the **model's full trained context** (np=6 OOMs, 40960 is the model cap). It **dominates** the old prod config (same 5-way throughput — verified 101.7 agg tok/s, no regression — but +25% per-proof context: 40960 vs 32768, free). To raise throughput further the workload must accept smaller per-slot context (more slots) — a context↔throughput trade the prover's 49152 requirement forecloses.
