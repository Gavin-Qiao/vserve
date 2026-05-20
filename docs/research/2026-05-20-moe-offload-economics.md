# MoE Expert-Offload Economics on RTX PRO 5000 48 GB (Blackwell)

Date: 2026-05-20. Target: vserve 0.6.1b1+, single sm120 GPU, PCIe gen5 x16.

## 1. `-ncmoe` vs `-ot` regex

`--cpu-moe` and `--n-cpu-moe N` (PR #11397; `common/arg.cpp` ~2284) are sugar
over the buffer-override path as `-ot`. `-ncmoe` does *not* deprecate the
regex form: only *repeated* `-ot` invocations are deprecated, comma lists
preferred. `--n-cpu-moe` counts from the *highest* layer down — opposite of
intuition for DeepSeek-style models with dense FFNs in the first layers (#18049).

Single-GPU numbers:
- Qwen3-Coder-30B-A3B, dual RTX 5060 Ti: **31 tok/s with `--cpu-moe`** (forced
  to fit) (llmkube).
- Qwen3.6-35B-A3B same hw: **21.7 tok/s with `--cpu-moe`** vs **107.8 tok/s
  without** — 5x penalty when the model fits.
- DeepSeek R1 q6 + `-ot exps=CPU`: 6.95 tok/s @ `-ngl 40` vs 4.65 @ `-ngl 0`
  (PR #11397).
- Qwen3-235B-A22B, surgical layer-selective `-ot`: **~16.7 tok/s** at 32k ctx
  (Sanftenberg, 2025).

`-ncmoe` and the equivalent `-ot` regex are throughput-equivalent. The win
condition is *not* offloading when the model fits.

## 2. Which experts to offload first

Community consensus (Unsloth, Doctor-Shotgun, LM-Kit) on FFN-projection
ordering for MoE blocks:
- `down_exps` is most throughput-critical — keep on GPU longest;
- `up_exps` is the safest to evict first;
- `gate_exps` is small but per-step — keep on GPU.

Optimal ladder: **up → up+gate → up+down → all**. No public per-expert
activation-frequency dataset exists for Mixtral / DeepSeek / Qwen3 / Llama 4.
PreScope (arXiv 2509.23638) reports **Top-4 hit rate >= 94 %**, input/output
layers most skewed.

## 3. PCIe-bandwidth crossover

PCIe gen5 x16 ≈ 64 GB/s. Doctor-Shotgun: 300 GB CPU-resident weights over
PCIe 4.0 x16 ≈ 10 s per sweep if batch < a few hundred tokens. PreScope:
**79 % PCIe utilisation** with pinned host memory + async copy. On gen5 x16,
<8 GB routed-expert traffic/token gives ~100 ms/token overhead — acceptable
above ~5 tok/s.

## 4. Cached-expert pinning

llama.cpp #20757: two-tier VRAM-slot + pinned-RAM SLRU cache; on RTX PRO 2000
8 GB + GPT-OSS-120B (~57 GB) it reports **12-14 tok/s steady @ 98-100 % hit**
vs **0.5-1 tok/s** pure CPU offload. vLLM RFC #38256: LFRU variant, 30 tok/s
on RTX PRO 2000 + GPT-OSS-20B, 160x over HF baseline. Neither has landed in
main. Off-tree: MoE-Infinity, Fiddler, PreScope, HOBBIT.

## 5. vLLM expert-parallelism vs CPU-offload

vLLM EP (`--enable-expert-parallel`) targets multi-GPU; docs recommend
H200/H100 8-GPU for DeepSeek-V3. For single-GPU 48 GB, only `--cpu-offload-gb`
is available; RFC #33869 GPU-cached / CPU-pinned expert pool is in-flight.
**llama.cpp remains the better back-end for MoE > 48 GB on single Blackwell**;
vLLM wins only when the active set fits.

## 6. `--no-mmap` in 2026

Issue #14999 (open since 2025-07-31): from b6051 onward `--no-mmap` on MoE
models triggers HSA memory-critical aborts even with ample host RAM.
**`--no-mmap` is no longer a uniformly safe auto-default.**

## vserve action items

For the RTX PRO 5000 48 GB target in `recipes/ot_strategies.py`:

1. **Insert an `up+gate` rung** between `partial-up` and `moderate`. Empirical
   FFN-projection hierarchy: `down_exps` belongs on GPU last (Unsloth canon;
   discussion #18049). Re-order: `none → partial-up → up+gate → up+down →
   max → layered`.
2. **Default `--n-cpu-moe N` over the regex form** for layer-counted offloads
   — same throughput, simpler diff against llama.cpp's built-in fitter
   (PR #11397, #18049). Keep `-ot` only for the surgical `layered` strategy
   (DeepSeek-V3 671B; Llama 4 Maverick 128-expert).
3. **Skip offload when est. weight + KV + 2 GB margin <= 48 GB** — the 5x
   penalty on Qwen3.6-A3B (107.8 -> 21.7 tok/s) is the canonical signal that
   offload is pure overhead when the model fits.
4. **Make `--no-mmap` opt-in for llama.cpp builds >= b6051** to avoid #14999;
   when set explicitly, pair with `--mlock` and document the host-RAM cost.
5. **Calibrate `_STRATEGY_FREE_FRACTION`** against the new rung:
   `partial-up ≈ 12 %`, `up+gate ≈ 18 %` (new), `up+down ≈ 26 %`,
   `max ≈ 35 %`, `layered ≈ 30 %`. Ship a `scripts/bench-ncmoe-sweep.sh`
   recipe so users can re-fit constants from `llama-bench`.

## Sources

llama.cpp PRs/issues: #11397, #14999, #18049, #18949, #20703, #20757,
#22183. vLLM RFC #33869, RFC #38256. Papers: PreScope (arXiv 2509.23638),
Pre-gated MoE (ISCA 2024), HOBBIT (arXiv 2411.01433). Guides: Doctor-Shotgun
on HF, DocShotgun gist a02a4c0c, llmkube.com hybrid-moe, Sanftenberg
Qwen3-235B Medium guide.
