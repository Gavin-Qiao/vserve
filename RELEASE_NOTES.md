# vserve 0.6.8b2 Release Notes (beta)

**Beta.** Adds a user-facing **MTP speculative-decoding toggle** to `vserve run`,
riding the in-checkpoint Multi-Token-Prediction draft layers that Qwen3.5/3.6-class
checkpoints ship (`mtp_num_hidden_layers`) via vLLM ≥ 0.24's unified
`speculative-config: {method: mtp}`. Install with `pip install --pre vserve` or
`uv tool install vserve --prerelease allow`.

## MTP on/off toggle for `vserve run`

- **`--mtp` / `--no-mtp`** — explicit toggle, off by default (speculative decoding
  shifts the perf profile toward low-concurrency latency and costs a little VRAM,
  so it stays a deliberate choice). **`--mtp-tokens N`** sets the speculation depth
  and implies `--mtp`.
- **In-checkpoint detection** mirrors vLLM 0.24's `SpeculativeConfig` rewriting:
  `mtp_num_hidden_layers` (Qwen3.5/3.6), `num_nextn_predict_layers`
  (DeepSeek/GLM/Qwen3-Next/Nemotron-H-style), `num_mtp_modules` (MiniMax) — read
  from `config.json` top level and `text_config`. The emitted block carries **no
  `model` key**, so vLLM loads the draft layers from the target checkpoint itself.
  Checkpoints without native layers fall back to a sibling `-MTP` variant directory;
  on llama.cpp that sibling GGUF is the only supported flavor.
- **Depth default is 3, not 1** — depth 1 is the worst non-zero setting on this
  class of hardware (−19% decode on Qwen3.6-27B-FP8) and k=3 wins the
  acceptance-vs-depth curve (docs/research/2026-05-20-spec-decode-acceptance.md).
  Explicit depths beyond the checkpoint's layer count are validated for vLLM's
  module-reuse divisibility rule at config time instead of crashing at engine boot.
- **Guard rails:** requires a *known* vLLM runtime ≥ 0.24
  (`runtime.VLLM_UNIFIED_MTP_VERSION`) — older or undeterminable runtimes are
  refused with the repair command rather than risking a crash-looping service.
  Models with no MTP weights get a precise refusal, never a silent substitute
  method. The existing Gemma-4 + tools + MTP refusal (vllm#41967) and the
  spec-decode × quantized-KV CUDA-graph gate (vllm#41559/#42692) apply unchanged.
- **Interactive wizard** offers MTP (default No) when the checkpoint carries draft
  layers and the runtime supports them; `vserve tune` now stamps
  `supports_mtp`/`mtp_num_layers` so the model picker shows an `mtp` tag and
  `vserve list <model>` points at the flag.
- **Recipe fix:** `pick_spec_config`'s MTP step was gated on a sibling `-MTP`
  directory existing, so vLLM safetensors checkpoints whose draft layers live
  *inside* the checkpoint never got their arch-table MTP recommendation. Native
  layers are now checked first (vLLM only); auto-picked MTP depth drops 5 → 3 per
  the research defaults.

## `--spec` — the spec-decode recipe, wired into `vserve run`

The recommendation engine (`pick_spec_config`) existed since 0.6.x but nothing in
the CLI ever called it. `vserve run --spec METHOD` now exposes it:

- **`auto`** — recommend per model/backend (blocklist-aware, best-effort): native
  MTP → sibling MTP variant → same-family ≤1.5B local draft → ngram (vLLM only).
  May resolve to nothing; that's a note, not an error. On a pre-0.24/unknown
  runtime an mtp recommendation degrades to ngram instead of blocking the launch.
- **`off` / `ngram` / `mtp` / `draft`** — explicit methods; precise refusal when
  impossible (ngram is refused on llama.cpp — benchmarked net-negative for batched
  serving). `--mtp`/`--no-mtp` remain as shorthand for `--spec mtp`/`--spec off`;
  combining `--spec` with the shorthand is an error.
- **Draft discovery** — `--spec draft`/`auto` scan local models for same-family
  ≤1.5B candidates with an identical tokenizer (BOS/EOS from config.json /
  generation_config.json, text_config included; size parsed from the `<N>B` name
  token, largest wins so `35B-A3B` reads as 35B).

## Spec-decode observability + host-RAM guard rails

- **`vserve bench` reports acceptance** — it snapshots vLLM's cumulative
  spec-decode counters around the run and prints the window's acceptance rate
  and accepted-tokens-per-step (also in `--json` as a `spec_decode` block).
  Silent when speculative decoding is off or the backend isn't vLLM. This is
  the measurement loop the 64k validation used by hand.
- **`vserve status` shows what the server is actually running** — a
  `Speculative: mtp (k=3)` line (with the draft checkpoint name when one is
  used) and `Multimodal: text-only (encoder skipped)` for
  `language-model-only` configs; both surface in `--json` too.
- **`vserve doctor` verifies the host-RAM OOM guards** — the managed unit must
  carry a `MemoryMax` cap and `configs/.env` must exist with the JIT compile
  caps (`MAX_JOBS`/`NVCC_THREADS`); both checks are vacuous on machines
  without a vserve-managed unit. `vserve run` additionally warns (launch still
  proceeds) when booting a vLLM unit with no `MemoryMax` — an uncapped boot
  can freeze the host via nvcc/FlashInfer JIT storms.

## Wizard parity: text-only serving + no more silently-dropped flags

- The interactive wizard now offers **text-only serving** for multimodal
  checkpoints (the `--language-model-only` recipe: vision/audio encoder never
  loads, VRAM goes to KV cache; image/audio requests are rejected) and shows the
  choice in the summary.
- `--thinking` and `--moe-backend` used to be **silently dropped** unless `--yes`
  (the wizard never consumed them). Any flag the wizard doesn't consume —
  including the new `--spec`/`--mtp` — now forces the scripted path, so what you
  pass is what you get.

**GPU-validated (2026-07-10, RTX PRO 5000 sm120, vLLM 0.24.0):** the toggle
boots, serves, and soaks cleanly end-to-end — `Qwen3.6-35B-A3B-NVFP4` at 64k ctx
with `--mtp --mtp-tokens 3` loaded the in-checkpoint `Qwen3_5MoeMTP` draft
(embeddings/lm_head shared) and survived a 54.8k-token prompt with no
vllm#40756-class crash. **But the benchmark says keep it off for the MoE-A3B**:
59.9% acceptance (τ≈2.8 tokens/step) still decoded at −52% (c1: 105 vs 220 tok/s)
/ −53% (c8: 517 vs 1103 tok/s) — the (k+1)-token verify step multiplies MoE
expert weight traffic and the bf16 quant-excluded draft layer adds sequential
forwards. `Qwen3_5MoeForConditionalGeneration` (and its GGUF twin) is therefore
in the auto blocklist; explicit `--mtp` remains available for experiments, and
the dense Qwen3.5/3.6 recommendation stands (untested at k≥2 on this fleet).

---

# vserve 0.6.8b1 Release Notes (beta)

**Beta.** Makes the pre-launch tuner **model-aware** so the auto
`gpu-memory-utilization` leaves headroom for transient runtime memory — fixing
serve-time CUDA OOMs on MoE and multimodal models that the old fixed-overhead math
couldn't see. Not yet GPU-validated across the model zoo, hence beta; install with
`pip install --pre vserve` or `uv tool install vserve --prerelease allow`.

## Model-aware GPU-memory headroom

The tuner derived `gpu-memory-utilization` as `(vram_total − fixed_overhead) /
vram_total` and then handed *all* the remaining VRAM to the KV cache. That fixed
reserve (base 3.5 GB) can't cover the transient memory the static KV math never
models — per-step activations, the CUDA-graph capture pool, and the FlashInfer
fp8/NVFP4 kernel autotuner workspace. These spike under concurrent load (MoE expert
dispatch) and at multimodal vision/audio encoder profiling, so an over-filled GPU
boots fine single-stream but OOMs under real load.

- **`runtime_headroom_gb(model)`** reserves extra VRAM by model class on top of the
  base overhead: **+1.5 GB for MoE**, **+2.5 GB for multimodal** (cumulative). On a
  48 GB card this lands auto util at ~0.895 for a MoE and ~0.843 for a MoE+vision
  model — matching configs verified stable under load (0.90 / 0.85). The old math
  produced ~0.93–0.96, which OOM'd under concurrency.
- **`resolve_gpu_memory_utilization(..., model=...)`** applies the headroom on the
  **auto path only** — an explicit `--gpu-util` or a configured
  `gpu.memory_utilization` is honored unchanged (backward compatible).
- **Multimodal detection** — new `ModelInfo.is_multimodal`, from the standard HF
  config keys (`vision_config` / `audio_config` / `image_token_id` / …).
- `vserve run` and `vserve tune` now resolve util **per-model**.

**Not yet validated (why beta):** the headroom constants are calibrated on RTX PRO
5000 (48 GB, sm120) and not re-benchmarked across other GPUs/models; they err toward
more headroom. Serving a multimodal model text-only (`--language-model-only`) still
receives the multimodal reserve — harmless, just leaves extra headroom.

---

# vserve 0.6.7 Release Notes

**Patch.** Moves the pinned-stable vLLM runtime to **0.24.0**, which serves
block-diffusion (dLLM) models like DiffusionGemma natively — so dLLMs no longer
need a separate newer runtime. Install with `pip install vserve` or `uv tool
install vserve`.

## vLLM 0.24.0 support

- Support range widens to `>=0.20,<0.25` and the pinned-stable runtime moves
  from `0.23.0` to **`0.24.0`** (`vserve runtime upgrade vllm --stable`).
- **Native block-diffusion (dLLM) on the stable runtime.** vLLM 0.24 lands
  DiffusionGemma in-tree on the V2 model-runner ModelState hooks (vllm#45163).
  vserve's dLLM serve recipe is unchanged and now runs on the *stable* service:
  `VLLM_USE_V2_MODEL_RUNNER=1`, `--trust-remote-code`, the entropy-bound
  diffusion sampler (via `--hf-overrides`), `--attention-backend TRITON_ATTN`, a
  host-RAM-bounded `runai_streamer` load, and the tuned `max-num-seqs 16` /
  `gpu-memory-utilization 0.70` diffusion-state caps.
- **Removed the `dllm_service_name` second-service compromise (added 0.6.6).**
  With native dLLM support in the stable runtime, the separate newer-vLLM
  service is gone — dLLMs serve on the single `service_name` like every other
  model. The `dllm_service_name` config key is no longer read; a leftover key in
  an existing `config.yaml` is silently ignored. Operational cleanup once on
  0.24.0: drop `dllm_service_name` from your config and decommission the
  `vllm-nightly` service and its venv (the temporary runtime is no longer
  needed).

## vLLM 0.24.0 compatibility notes

- **`CUDA_VISIBLE_DEVICES` (vllm#45026):** vLLM 0.24 no longer *sets* it
  internally (it adds an optional `--device-ids` for topology) but still
  *respects* an externally-set value. vserve pins the GPU via
  `CUDA_VISIBLE_DEVICES` in the systemd EnvironmentFile, so single-GPU serving
  is unaffected.
- **Transformers v5 required:** 0.24 drops the Transformers v4 compatibility
  shim (the box already runs 5.12.1). Point the runtime at a Transformers 5.x
  environment.
- **Removed models:** 0.24 removes ERNIE, Xverse, Dots1, Bamba, Mono-InternVL,
  and first-generation Qwen/QwenVL. vserve's arch-registry entries for these are
  inert unless such a model is present, so no action is needed.

## Runtime note

- Bumping the pinned-stable runtime assumes a GPU soak-test of vLLM 0.24.0 on
  the target box (the same gate used for the 0.23.0 pin). vserve emits the
  correct config regardless of the installed build; `vserve runtime check vllm`
  accepts the full `>=0.20,<0.25` range.

## Dependencies

- Bumped dependency floors to the newest compatible versions and refreshed the
  lockfile — verified against the full test suite, `ruff`, and `mypy`. Runtime:
  `huggingface-hub>=1.21` (was `>=0.30`), `rich>=15` (was `>=14`), `typer>=0.25`
  (was `>=0.15`; capped below 0.26 by huggingface-hub 1.21's own `typer<0.26`
  pin), `packaging>=26` (was `>=24`), `nvidia-ml-py>=13.610.43`. Optional:
  `gguf>=0.19` (was `>=0.6`). Dev: `pytest>=9` (was `>=8`), `mypy>=2.1` (was
  `>=1.15`; pulls the new `ast-serialize` transitive). `rich` 15 and `mypy` 2
  are major bumps — `ruff` and `mypy` both pass clean on the bumped versions.
- vLLM is not a declared dependency (it's detected at runtime); its pin already
  tracks the newest release, 0.24.0 (no 0.24.x patch or 0.25 exists yet).

# vserve 0.6.6 Release Notes

**Patch.** Adds per-model vLLM runtime selection so block-diffusion (dLLM)
models can serve on a separate, newer vLLM service while the pinned-stable
runtime stays untouched. Install with `pip install vserve` or `uv tool install
vserve`.

## Per-model vLLM runtime (dLLM on a separate service)

- vserve can now drive a **second vLLM systemd service** for block-diffusion
  models, configured via a new optional `dllm_service_name`:
  ```yaml
  backends:
    vllm:
      service_name: vllm               # stable runtime — used for everything by default
      dllm_service_name: vllm-nightly  # used only for block-diffusion (dLLM) models
  ```
- When `vserve run` launches a dLLM (e.g. DiffusionGemma), it resolves to the
  `dllm_service_name` service; every other model uses the stable `service_name`.
  `vserve stop`/`status` operate on whichever runtime is live (both services are
  checked), and a stop tears down both so the GPU is never left occupied.
- This lets you serve a dLLM on a vLLM build newer than the pinned-stable one
  (native block-diffusion support needs vLLM ≥ 0.23.1) **without changing your
  stable runtime** — the default `vllm.service` is never touched. The second
  service is operator-provided: a copy of `vllm.service` whose `ExecStart` points
  at the newer venv. Leaving `dllm_service_name` unset preserves the previous
  single-runtime behavior exactly.

# vserve 0.6.5 Release Notes

**Patch.** Adds serving support for block-diffusion language models
(DiffusionGemma), hardens `vserve run` against startup crash-loops, and keeps
ngram speculative decoding off on llama.cpp where it costs throughput. Install
with `pip install vserve` or `uv tool install vserve`.

## Block-diffusion (dLLM) serving

- **vserve now serves block-diffusion LLMs** such as
  `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`. These are not autoregressive
  causal LMs — they ride vLLM's V2 model-runner ModelState hooks — so `vserve
  run` now detects them (from `model_type` / architecture) and emits the right
  serve recipe instead of the causal-LM config that fails at load with
  `ValueError: Argument input_ids not found`:
  - `VLLM_USE_V2_MODEL_RUNNER=1` (written to the systemd EnvironmentFile),
    `--trust-remote-code`, the entropy-bound diffusion sampler (via
    `--hf-overrides`), `--attention-backend TRITON_ATTN`, and a host-RAM-bounded
    `runai_streamer` load.
  - Autoregressive-only knobs are suppressed for these models: the NVFP4
    fp8-KV-cache forcing (KV stays `auto`) and ngram speculative decoding.
  - **Tuned defaults** for the diffusion-state VRAM profile: `max-num-seqs 16`
    and `gpu-memory-utilization 0.70`. On a 48 GB sm120 GPU at 16k context this
    is the throughput knee — ~540 tok/s vs ~373 at the model card's `ns=4`
    (+45%). The KV cache is not the concurrency limiter (it supports ~24×); the
    large diffusion-state tensors are, so utilization is lowered to leave room.
- **Runtime note:** native block-diffusion support requires a vLLM build newer
  than the pinned stable 0.23.0 (a 0.23.1+ nightly, or 0.24 when it ships).
  vserve emits the correct config regardless — point the runtime at a capable
  build to actually serve.

## `vserve run` no longer spins on a failed start

- A model that fails fatally at startup (e.g. an unsupported architecture) could
  put the systemd unit into an unbounded `Restart=on-failure` loop: because a
  mid-restart service reads as "active", `vserve run` treated it as "still
  warming" and returned, leaving it to restart roughly once a minute forever
  (the `StartLimitBurst` guard is tuned for fast storms and a slow ~60 s-cycle
  failure slips under it).
- `vserve run` now tracks the unit's restart counter and, when it climbs during
  the health wait, **stops the unit and prints the engine diagnosis** instead of
  spinning. Every non-ready exit path now best-effort stops the unit, so a
  failed launch can no longer keep auto-restarting after the command returns.

## llama.cpp speculative decoding

- **ngram speculative decoding stays off on llama.cpp.** The pinned runtime
  gained `--spec-type ngram-*`, but benchmarking (Goedel-Prover-V2 / Qwen2.5-32B
  at batch 5) showed it costs ~9% decode throughput for batched serving, so
  vserve keeps it disabled there. It remains enabled on vLLM, where it is a net
  win. See `docs/research/2026-06-19-llamacpp-throughput-goedel.md`.

# vserve 0.6.4 Release Notes

**Stable.** Promotes 0.6.4b1 after an on-GPU sweep, and adds support for the
vLLM 0.23 line. Install with `pip install vserve` or `uv tool install vserve`.

## vLLM 0.23 support

- **0.23 is now supported and is the pinned stable runtime.** The accepted
  range widens to `>=0.20,<0.24`, and fresh installs pin **vLLM 0.23.0**. The
  upgrade is drop-in: 0.23 changes none of the serve flags, tool/reasoning
  parser names, or KV-cache dtypes vserve emits, and its dependencies are
  unchanged from 0.22 (`torch==2.11.0`, Python `>=3.10,<3.15`). The deprecated
  FlashInfer MoE env vars remain warn-and-redirect shims in 0.23 and stay
  suppressed. The bundled vLLM architecture fixture is refreshed to 0.23
  (365 archs).

## Verified on-GPU

This clears the beta's on-GPU gate. The 0.23 runtime was soak-tested on an
RTX PRO 5000 (Blackwell / sm120) across both backends and four model families:

- **vLLM** — Qwen3.6-35B-A3B FP8 (text + vision, 127 tok/s), gpt-oss-20b mxfp4
  (177 tok/s), Qwen3.5 smoke.
- **llama.cpp** — Qwen3.6-35B-A3B-MTP GGUF (122 tok/s), which exercises the
  GGUF `RuntimeIdentity.fingerprint` fix at serve time, not just tune.

## Carried from 0.6.4b1

- **GGUF tune/run crash fix** — no more `AttributeError: 'RuntimeIdentity'
  object has no attribute 'fingerprint'` on `vserve tune` / uncached
  `vserve run` for llama.cpp models.
- **`vserve run --language-model-only`** — serve a natively-multimodal model
  (Gemma-4, Qwen3.5/3.6) text-only, skipping the vision/audio encoder.
- **Qwen3.5 / 3.6 sampler disambiguation** by model name, with canonical-arch
  coverage for the spec-decode recipe and a sniffer↔arch-table drift guard.

See the 0.6.4b1 notes below for full detail on the carried items.

## Gemma-4 multimodal

`Gemma-4-31B-IT-NVFP4` serves full omni-modal (text + image, both verified) on
a single Blackwell card on the pinned **0.23.0** with no special flags — no
vserve change required. The previously-planned `--limit-mm-per-prompt`
auto-emit proved unnecessary on 0.23 (the 0.22-era multimodal-profiling OOM
does not reproduce) and has been dropped. (vLLM *nightlies* after 0.23.0 carry
a temporary `tie_weights` regression —
[vllm#45543](https://github.com/vllm-project/vllm/issues/45543), fixed by
[#45544](https://github.com/vllm-project/vllm/pull/45544) — that breaks
quantized Gemma-4 load; the pinned stable 0.23.0 is unaffected.)

## Known follow-ups

- Minor: precise vision/audio tower-size subtraction in the
  `--language-model-only` capacity estimate.

# vserve 0.6.4b1 Release Notes

**Beta — not yet verified on-GPU.** Bundles the llama.cpp runtime fix plus the
text-only-serving and Qwen3.5/3.6 work accumulated since 0.6.3. Install with
`pip install --pre vserve` or `uv tool install vserve --prerelease allow`;
promote to a stable 0.6.4 only after an on-GPU sweep.

## Bug fixes

- **GGUF tune/run crash.** `vserve tune` — and any uncached `vserve run` — on a
  llama.cpp/GGUF model raised `AttributeError: 'RuntimeIdentity' object has no
  attribute 'fingerprint'`. The 0.6.3 change that made the llama.cpp runtime
  descriptor a `RuntimeIdentity` (from a dict) wasn't matched in
  `build_tuning_fingerprint`, which calls `.fingerprint()`; `RuntimeIdentity`
  now provides it. Regression-tested.

## Features

- **`vserve run --language-model-only`** — serve a natively-multimodal model
  (Gemma-4, Qwen3.5/3.6, …) in text-only mode: emits vLLM's
  `--language-model-only` to skip the vision/audio encoder (freeing VRAM for KV
  cache) and drops the multimodal `max-num-batched-tokens` floor.

## Qwen 3.5 / 3.6

- **Sampler disambiguation by model name.** Qwen3.5 and Qwen3.6 share the
  canonical arch `Qwen3_5MoeForConditionalGeneration` but want different
  samplers (3.5: temp 0.6 / pp 1.0; 3.6: temp 1.0 / pp 1.5).
  `get_sampling_defaults()` now resolves the right one from the version token in
  the model name.
- **Canonical-arch coverage** for the spec-decode recipe — the same
  canonical-vs-synthetic-arch gap the 0.6.3 tool-parser fix closed, now closed
  for `SPEC_METHOD_BY_ARCH` too. Adds a cross-registry coverage guard and a
  sniffer↔arch-table consistency test so the two parser-selection paths can't
  silently drift back to `hermes` / `deepseek_r1`.

## Notes

- This beta has **not** been through an on-GPU sweep. Known follow-ups deferred
  to a stable 0.6.4: precise tower-size subtraction in the capacity math under
  `--language-model-only`, and auto-emitting `limit-mm-per-prompt` for Gemma-4
  multimodal serving.

# vserve 0.6.3 Release Notes

**First stable release of the 0.6 line** — promotes the 0.6.3b3 beta after a
full on-GPU verification sweep on vLLM 0.22.0 (RTX PRO 5000, sm120), and adds
vLLM 0.22 support.

## vLLM 0.22 support

- Supported runtime range widened to `>=0.20,<0.23`. An `installed_vllm_version`
  probe drives three version gates, each conservative on an unknown runtime
  (behaves pre-0.22):
  - **`default-chat-template-kwargs`** emitted on >=0.22 (vLLM renamed the
    flag); thinking toggles keep working across 0.20–0.22.
  - **Deprecated FlashInfer MoE env vars** (`VLLM_USE_FLASHINFER_MOE_*`)
    suppressed on >=0.22 in favor of vLLM's hardware-aware `--moe-backend`
    default; new `vserve run --moe-backend` escape hatch.
  - **spec-decode + fp8 KV keeps CUDA graphs** on >=0.22 (DFlash fp8 fix,
    vllm#42692); turboquant and other quantized KV stay conservatively
    un-graphed.
- vLLM arch fixture refreshed to 0.22.0; runtime-emitted issue citations
  corrected (#42808/#43357 workspace-lock, #39137 NVFP4-KV).
- **Pinned stable runtime stays `0.21.0`.** 0.22 is supported, but Gemma-4
  NVFP4 startup OOMs on 0.22 (see caveat), so `runtime upgrade --stable`
  installs the runtime where every bundled model serves unflagged.

## Bug fixes (found during the on-GPU sweep)

- **Qwen 3.5 / 3.6 tool calling** routed to the wrong parser (`hermes`). These
  models emit the XML `<function=…><parameter=…>` tool format and need
  `qwen3_coder`; tool calls were silently leaking into `message.content` with
  `tool_calls=[]`. Confirmed and fixed on-GPU (`arch_registry.py`).
- **`vserve bench`** under-counted thinking-model output — vLLM 0.22 streams
  reasoning tokens as `delta.reasoning` (renamed from `reasoning_content`), so
  bench reported 0/N against a perfectly healthy server. Now counts all
  channels.
- **Test isolation**: the suite exercised the real `/run/lock/vserve`, so a
  `pytest` run on a workstation with a live backend could delete the operator's
  session marker. Now redirected to a per-test directory.

## On-GPU sweep (vLLM 0.22.0, sm120)

Verified: runtime gate; the Qwen3.5-4B @128k×11 `turboquant_3bit_nc`
crash-repro (`cudagraph_mode=NONE` applied, round-trips); streaming bench;
gpt-oss-20b (TRITON_ATTN forced, `speculative_config=None`); spec-decode+fp8
CUDA-graph capture; the renamed thinking kwarg; and Qwen3.6-35B-A3B MoE
(text + image/VLM + tool calling, under a retuned host-RAM guard).

## Known caveat: Gemma-4 NVFP4 on vLLM 0.22

vLLM 0.22's batched vision encoder (#43169) makes V1 startup profiling run a
real encoder forward over 3 max-size dummy **videos**, OOMing host RAM on a
62 GiB box. Serve Gemma-4 on 0.21, or pass
`--limit-mm-per-prompt '{"image":1,"video":0}'` plus
`--mm-processor-kwargs '{"max_soft_tokens":560}'`. Automatic multimodal caps
for video-capable models land in 0.6.4.

---

# vserve 0.6.3b3 Release Notes (beta)

> **Beta gate updated.** `0.6.3b3` is the **on-GPU-verified beta**. The
> original failing config — Qwen3.5-4B at 128k × 11 with
> `turboquant_3bit_nc`, the crash that motivated the AA fix in 0.6.1 —
> was re-run on idle RTX PRO 5000 (sm120) and loads cleanly. The
> `cudagraph_mode=NONE` AA fix works as designed (`Maximum concurrency
> for 131,072 tokens per request: 39.44x`). Inference round-trips
> correctly on all four exercised models (Qwen3.5-0.8B, Qwen3.5-4B,
> Gemma-4-31B-IT-NVFP4, gpt-oss-20b). Streaming bench confirmed
> end-to-end (763.6 tok/s on Qwen3.5-0.8B; TPOT 11.8ms on Qwen3.5-4B
> at 128k × 11 turboquant_3bit_nc). The "beta" label remains only
> until one more soak / production-traffic cycle.

## 0.6.3b3 — on-GPU sweep findings (since b2)

The comprehensive GPU sweep surfaced two real bugs and prompted two
new test classes. **Both bugs are fixed in this release.**

### Bug 1 (CRITICAL): `vserve bench` failed on live YAML configs

`_find_running_backend_and_cfg` read the running backend's config with
`json.loads`. vserve writes vLLM configs as **YAML**. The bench command
silently swallowed the parse error and returned "Could not resolve
served-model name". The old test mocked `Path.read_text` to return
JSON, masking the bug.

- `cli.py:_find_running_backend_and_cfg` now dispatches on suffix
  (`.json` → `json.loads`, otherwise `try_read_profile_yaml`).
- `tests/test_bench_command.py::TestFindRunningBackendAndCfgYamlLoader`
  adds two regression tests using real YAML and JSON temp files (no
  `read_text` mock).

### Bug 2 (LATENT): arch_registry references arch names that don't exist in vLLM

Cross-checking every registry key against
`vllm.model_executor.models.ModelRegistry.get_supported_archs()`
revealed **16 of 38 keys reference vLLM arch names that don't exist**.
The user's Qwen 3.5 / 3.6 family models (4 of 9 installed) currently
route correctly only because the limits cache stores
`reasoning_parser: qwen3` from prior tunes — new tunes or
cache-clear flows would route those models to `None`.

- The 0.6.3b1 "Bug fix #1" (adding `Qwen36MoeForCausalLM` to the
  reasoning-parser table) is structurally invalid — that arch name
  doesn't exist in vLLM. The real canonical name is
  `Qwen3_5MoeForConditionalGeneration`.
- 0.6.3b3 adds the real canonical names alongside the legacy ones in
  every arch table and in `_THINKING_DEFAULT_ARCHS`. Existing tunes
  keep working; fresh tunes now route correctly via the new entries.
- New test `tests/test_arch_registry_vllm_canonical.py` cross-checks
  every registry key against the captured vLLM 0.21.0 fixture
  (`tests/fixtures/vllm_archs.json`, 361 supported archs) with an
  explicit allowlist for the 14 remaining forward-compat entries.
  `test_suggested_parsers_fire_for_qwen3_5_canonical` proves the new
  entries actually fire end-to-end through `_suggested_*_parser`.

### On-GPU verifications (5 models, sm120)

| What | Evidence |
| --- | --- |
| Original crash repro: Qwen3.5-4B @ 128k × 11 + turboquant_3bit_nc | Loads cleanly; AA fix applies (`cudagraph_mode=NONE`) |
| Streaming bench end-to-end | 763.6 tok/s on Qwen3.5-0.8B, 11.8ms TPOT on Qwen3.5-4B |
| family_of vs arch[:5] (Gemma3/Gemma4 collision) | Gemma-4-NVFP4 routes to gemma4 family, not Gemma3 prefix |
| FP4 env gating on sm120 | NVFP4 path active (`FlashInferCutlassNvFp4LinearKernel`) |
| Forced TRITON_ATTN backends | Both Gemma-4 and gpt-oss-20b force TRITON_ATTN |
| GptOssForCausalLM SPEC_BLOCKLIST | `speculative_config=None` in engine config |

### Deferred to 0.6.4

- Retire 14 remaining fictional arch names (needs limits-cache schema bump
  to avoid breaking existing tunes).
- Cross-check llama.cpp GGUF arch names against installed builds.
- Unit test for `_resolve_quant_envs` sm < 100 branch.

## 0.6.3b2 — CI fix + hook hardening (since b1)

The `0.6.3b1` ship surfaced a real CI gap: 9 `--help` tests passed
locally and failed in GitHub Actions. Root cause was Rich/Typer
emitting per-character style spans
(`\x1b[1;36m-\x1b[0m\x1b[1;36m-flag\x1b[0m`) when `CI=true` +
`GITHUB_ACTIONS=true` are set, which breaks naive
`"--flag" in stdout` substring assertions.

**Fixes:**
- `tests/_helpers.py` adds `strip_ansi()`. The 9 affected
  `--help` assertions in `tests/test_run_llamacpp_flags.py` and
  `tests/test_bench_command.py` now run `strip_ansi(result.stdout)`
  before substring checks. Regression verified: under
  `CI=true GITHUB_ACTIONS=true COLUMNS=80`, all 951 tests pass.

**Hook hardening (so this class of regression can't ship again):**
- `.githooks/pre-commit` rewritten — mirrors the CI workflow exactly:
  `ruff` + `mypy --check-untyped-defs` + full `pytest` (no `-x`) under
  `CI=true GITHUB_ACTIONS=true COLUMNS=80`. What's green locally is
  green in CI.
- `.githooks/pre-push` (new) — same gates plus a `pyproject.toml` ↔
  `__init__.py` version-sync check and an `uv build` dry-run, so a
  packaging error can't make it onto the wire.
- `scripts/install-hooks.sh` (new) — wires
  `git config --local core.hooksPath .githooks`. README's Development
  section documents the one-line install.

`0.6.3b1` is the cumulative beta. It rolls in **0.6.1** (research bundle —
21 items across vLLM 0.21+ / llama.cpp b9222+ / Unsloth Dynamic 2.0 /
Gemma 4 family), **0.6.2** (CLI plumbing for the 0.6.1 recipe modules),
and the **0.6.3 de-spaghetti refactor** that consolidates patch sediment
accumulated since 0.5.x.

**Major changes since 0.6.0** (organised by phase):

### 0.6.1 — research bundle
21 structural items across five areas — see the corrections, vLLM-side,
llama.cpp-side, tuning+bench, and measurement-layer sections below.

### 0.6.2 — CLI plumbing for the recipe modules
- `vserve bench` subcommand — drives the streaming benchmark (TTFT / TPOT /
  ITL / E2E percentiles) against the live backend; writes to perf cache.
- New CLI flags reach the 0.6.1 backend plumbing:
  `--cache-reuse`, `--cram-mb`, `--slot-save-path`, `--swa-full`,
  `--n-cpu-moe`, `--reasoning-budget`, `--thinking/--no-thinking`.
- Five forward-looking research reports while the GPU was unavailable
  (KV memory math, spec-decode acceptance, MoE offload economics, LoRA,
  quant-aware sampling) — see `docs/research/2026-05-20-*.md`.

### 0.6.3 — de-spaghetti refactor
A 5-agent audit (`docs/audits/2026-05-20-*.md`) surfaced 4 real runtime
bugs and structural cruft predating 0.6.1. The refactor plan is in
`docs/plans/2026-05-20-v063-refactor-plan.md`. Highlights:

**Bug fixes:**
- `Qwen36MoeForCausalLM` was missing from the reasoning-parser registry —
  `<think>` was leaking into `message.content` on Qwen 3.6 MoE.
- `arch[:5]` family check collided Gemma3 / Gemma4 in spec-decode vocab
  compatibility — replaced by a canonical `family_of()` helper in the
  new `arch_registry.py`.
- FlashInfer MoE FP4 env vars were emitted unconditionally; now gated on
  `gpu.compute_cap >= 100`.
- `GptOssMoeForCausalLM` was a typo in `SPEC_BLOCKLIST` (every other
  registry uses `GptOssForCausalLM`); corrected.

**Structural:**
- New `src/vserve/arch_registry.py` — single canonical home for all
  arch-keyed tables (tool-parser, reasoning-parser, forced-backend,
  GGUF→HF arch mapping) plus `family_of(arch)` and `is_thinking_default(arch)`.
- `cli.py` shrunk **6,136 → 5,089 lines (−17%)**. Domain logic moved to:
  - `vserve.diagnostics` — engine-log failure pattern matching.
  - `vserve.downloader` — pure HF-download helpers (~10 functions).
  - `vserve.picker` — limits-table parsing + scripted-config default
    chooser (vLLM + llama.cpp).
  - `vserve.cli_doctor` — the entire `vserve doctor` body (514 lines
    incl. 364-line nested helper closure).
- Backends symmetrised:
  - `vserve.systemd_helpers` — shared systemctl-call primitive and
    unit-safety asserter; both backends use them.
  - `llamacpp.runtime_info` / `compatibility` migrated from ad-hoc
    dicts to the canonical `RuntimeIdentity` / `CompatibilityResult`
    dataclasses.
  - Protocol return types tightened (dropped `| Any` papering).
  - New `Choices` TypedDict in `protocol.py` documents the per-backend
    key-ownership split flagged by audit.
- Helpers + Pattern cleanup:
  - 5× repeated `read_limits(limits_path(...))` collapsed into a
    `read_limits_for(provider, model_name)` helper in `config.py`.
- Dead code removed: `compare.py`, `_is_multimodal_model()`,
  `read_timing` / `write_timing` / `timing_path`, dead
  `recipes/__init__.py` re-exports.

The probe-based tune redesign (`docs/plans/2026-05-19-tune-redesign-probe.md`)
stays separate as the next minor release — items below remove ~90% of the
cases the probe would otherwise have to catch, making that PR meaningfully
smaller.

## Bundled corrections to 0.6.0

### AA. `cudagraph_mode: NONE`, not `--enforce-eager`, for TurboQuant workspace lock

The 0.6.0 diagnoser hinted `--enforce-eager` for the
`AssertionError: Workspace is locked but allocation from
'turboquant_attn.py:879:_decode_attention' requires N MB` failure. The
canonical maintainer-recommended fix is
`compilation-config: {cudagraph_mode: NONE}` (per vllm#40807 and #41403
maintainer comments) — that workaround keeps `torch.compile` fusions
while skipping CUDA-graph capture, where `--enforce-eager` overshoots and
disables both (lose ~10-30% decode tokens/sec for no extra benefit).

Two changes:

1. `_diagnose_engine_failure` now emits a citation-bearing recommendation
   that names `cudagraph_mode: NONE` first and references the issue
   numbers inline.
2. `VllmBackend.build_config` pre-emits `compilation-config:
   {cudagraph_mode: NONE}` automatically when the chosen KV dtype is any
   `turboquant_*` variant — and again when speculative decoding is
   requested with quantized KV (vllm#41559, DFlash spec-decode breaks
   with any KV quantization). User-set explicit values win via setdefault.

### BB. `--fit off` for llama.cpp launches

Every llama-server start logged `common_fit_params: failed to fit params
to free device memory: n_gpu_layers already set by user to N, abort`
because vserve always pins `-ngl` and the auto-fitter aborts. Cosmetic
but noisy in `journalctl -u llama-cpp.service`. `start()` now appends
`--fit off` to silence the auto-fitter explicitly (llamacpp#21801).

## vLLM-side hosting

### A. Architecture-derived sampling-defaults registry

New module `vserve/recipes/sampling.py` maps architectures to the
sampler parameters Unsloth and the model vendors document — e.g. Gemma 4
`temp=1.0 top_p=0.95 top_k=64 min_p=0.01`, Qwen3-Thinking `temp=0.6
top_p=0.95 top_k=20 presence_penalty=1.0`, DeepSeek V3.1
`temp=0.6 min_p=0.01`, Kimi K2 Thinking `temp=1.0`. Vendor defaults are
documented to be load-bearing for thinking models (Unsloth: "NEVER use
greedy on thinking variants — loop trap"); shipping the stock backend
default silently degraded output. vLLM `build_config` now emits an
`override-generation-config:` block; llama.cpp `build_config` writes the
flags directly into the launch script. Per-request sampler overrides
still win at inference time. Opt out per launch with `recipe_sampling:
False` in the choices dict.

### B. Reasoning-parser auto-discovery

0.6.0 added architecture-aware tool-parser auto-discovery; 0.6.1 mirrors
it for `reasoning-parser`. `_ARCH_TO_REASONING_PARSER` maps the
architectures vserve might see (Gemma 3/4 → `gemma4`, DeepSeek V3-V4 →
`deepseek_r1`, Qwen3 family → `qwen3`, GPT-OSS → `openai_gptoss`, etc.).
Both `tune()` and `build_config` now use the lookup, so users no longer
need to pass `--reasoning-parser` explicitly for the architectures
vserve recognizes. Runtime registry filtering (the same path used by
tool-parser detection) only emits parsers the installed vLLM build
actually has.

### C. `chat-template-kwargs` plumbing

Gemma 4, Qwen 3.x, and DeepSeek V3.1+ all toggle their thinking channel
via a chat-template kwarg (`enable_thinking` for the first two,
`thinking` for DeepSeek), not a CLI flag. vserve now accepts
`choices["thinking"]: bool | "auto"` and routes it to the right kwarg
based on architecture. The escape hatch `choices["chat_template_kwargs"]:
dict` passes arbitrary kwargs through. On the llama.cpp side, the kwargs
serialize as a single JSON string passed to `--chat-template-kwargs`;
Kimi-K2 models also auto-add `--special` so `<think>` tokens reach the
reasoning parser.

### D. Expanded `_ARCH_TO_TOOL_PARSER` to vLLM 0.21's full table

0.6.0's table mapped four architectures (Gemma 3/4 only). 0.6.1 expands
to vLLM 0.21's complete parser set — Llama 3/4, Qwen 3 family (incl.
Coder + XML), DeepSeek V3/V3.1/V3.2/V4, Kimi K2 (instruct + thinking),
GLM-4.5/4.7, IBM Granite 3/4, Cohere Command R+, Baidu ERNIE 4.5, AI21
Jamba, Salesforce xLAM, Liquid LFM 2/2.5, Mistral, GPT-OSS, InternLM.
The chat-template-marker fallback in `tools.py:_MARKER_TABLE` was also
extended with the new vendor-specific tags (`<|tool_calls_section_begin|>`
for Kimi K2, `<|START_TOOL|>` for Cohere, `<|TOOL_CALL|>` for LFM, etc.).

### E. Hybrid head_dim KV math (Gemma 4)

Gemma 4 has `head_dim=256` on sliding-window layers and `head_dim=512`
on global-attention layers (5:1 pattern). vserve's fallback
`head_dim = hidden_size // num_attention_heads` rounded to 288, missing
both. `extract_model_info` now uses the weighted average over
`sliding_window_pattern` — 5×256 + 1×512 / 6 = 298 for Gemma 4's 5+1
pattern. `ModelInfo` gains a `global_head_dim` field so downstream
upper-bound budgeting (e.g. cudaMalloc estimates) can use
`max(head_dim, global_head_dim)`.

### F. Gemma-4 chat template auto-resolution

The vLLM-bundled `gemma4` tool parser scans for a custom encoding
(`<|"|>` string delimiter, `<|tool_call>` outer tag, bare unquoted JSON
keys) that the stock HF chat template does not emit. Without the
vendored `tool_chat_template_gemma4.jinja`, every tool-call request
silently produces malformed output that the parser rejects. vserve now
auto-resolves the template: first checks the configured vLLM install
under `examples/`, falls back to a packaged copy shipped in
`vserve/templates/tool_chat_template_gemma4.jinja` (the upstream-canonical
file pinned at the 2026-05 vLLM main snapshot). Explicit
`choices["chat_template"]` wins over auto-resolution.

### Q. NVFP4 / FlashInfer / MXFP4 quant flags + envs

`QUANT_FLAGS` now covers NVFP4 (Blackwell-required), ModelOpt-NVFP4
checkpoints (nvidia/* repos), MXFP4 (MoE), bitsandbytes (single-card
4-bit iteration), and gguf (vLLM 0.21 can serve GGUFs directly).
`QUANT_ENV_VARS` plumbs `VLLM_USE_FLASHINFER_MOE_FP4=1` and
`VLLM_FLASHINFER_MOE_BACKEND=throughput` into the systemd
EnvironmentFile when an NVFP4/ModelOpt model is launched — without
those envs, MoE inference on FlashInfer falls back to the slow path.
NVFP4 models with `kv_dtype=auto` are auto-pinned to `fp8` because vLLM
otherwise treats fp8-KV as fp8-checkpoint (vllm#39133). A new
`recommend_quant_for_arch(sm, is_moe, available)` helper exposes the
canonical (compute_cap, MoE) → quant routing table.

### R. MLA attention-backend awareness

MLA architectures (DeepSeek V2-V4, Kimi K2, LongCat-Flash) ship their
own attention layout; vLLM's auto-pick can pick a slower fallback.
`_ARCH_FORCES_BACKEND` now pins FLASHMLA on Hopper / TOKENSPEED_MLA on
Blackwell for MLA models, keeps TRITON_ATTN for heterogeneous-head_dim
architectures (Gemma 4), and forces TRITON_ATTN for GPT-OSS on SM120
specifically (vllm#40153 — FlashInfer doesn't support attention sinks
on that compute capability). Other compute caps use the default
backend. A new `_forced_attention_backend(model_path, compute_cap)`
helper does the routing and consults the per-backend
`BACKEND_INCOMPATIBLE_KV_DTYPES` table to filter cells the backend
would otherwise refuse. `GpuInfo` now carries the parsed compute cap.

### M. Speculative decoding (ngram + draft + MTP, with MoE blocklist)

New module `vserve/recipes/spec_decode.py` covers three independent
shapes: vLLM-only `ngram` (zero extra-model cost), `draft` (pair a
small same-family GGUF), and `mtp` (Unsloth Qwen3.6 MTP variants).
`pick_spec_config()` walks the (architecture, backend, available drafts)
state and returns a `SpecConfig`. The blocklist refuses spec-decode by
default on A3B-style MoE (Qwen3-A3B, Qwen3-MoE, gpt-oss-MoE, Mixtral,
DeepSeek V2) because spec-decode is net-negative on consumer hardware
for that family per llamacpp#19493 benchmarks; `force=True` bypasses.
vLLM `build_config` refuses MTP + Gemma-4 + tools (vllm#41967 — first
call's args corrupted); llama.cpp `start()` emits `-md` plus
`--spec-draft-n-max/n-min/p-min`. `vocab_compatible()` enforces
identical BOS/EOS + family before pairing a draft.

## llama.cpp-side hosting

### H. `--n-cpu-moe N` over the regex `-ot`

`-ncmoe N` (alias `--n-cpu-moe`) is the modern equivalent of
`-ot ".ffn_.*_exps.=CPU"` — counts from the highest-numbered layer down,
no regex required. `build_config` now converts the all-experts `-ot`
pattern to `n_cpu_moe=99` (clamps to actual expert layer count) and
drops the `-ot` to avoid emitting both. Surgical layer-range patterns
(e.g. `\.([1-5][0-9])\.ffn_(up|gate)_exps.=CPU`) stay on the `-ot` path.
The `--no-mmap` auto-add (perf gate from 0.6.0 L7) now fires for both
`-ot` and `-ncmoe`.

### I. K==V invariant + widened KV-quant matrix

Fused Flash-Attention requires symmetric K and V cache dtypes
(llamacpp#22411 — asymmetric pairs silently fall back to a much slower
dequant-on-the-fly path). `build_config` now refuses asymmetric pairs
when `flash_attn=True`; callers wanting asymmetric pairs must pass
`flash_attn=False` explicitly. The candidate KV-dtype matrix widens
from `(f16, q8_0, q4_1, q4_0)` to `(f16, bf16, q8_0, q5_1, q4_1, q4_0)`,
and `iq4_nl` is added conditionally when the binary was built with
`GGML_CUDA_FA_ALL_QUANTS=ON` (probed via `--version` output). For
Gemma-4-31B (60-layer dense), V is force-pinned to `f16` regardless of
picker output to avoid illegal-memory-access after the second SWA
checkpoint (llamacpp#22527).

### J. `mmproj-*.gguf` auto-detection + audio MM-token floor

llama.cpp silently serves multimodal models as text-only when `--mmproj`
is missing. `find_mmproj(model_dir)` now discovers the projector GGUF
(prefers `mmproj-BF16.gguf`; quantized variants are documented to
produce garbage); `ModelInfo.mmproj_path` carries the result; the
llama.cpp `start()` emits `--mmproj <path>` automatically. Callers can
override with `choices["mmproj"]` (path string) or opt out with
`mmproj=""`. On the vLLM side, `_mm_batched_tokens_floor` now returns
8192 for audio-capable models (Gemma 4 E-class audio: 750 tokens/segment
× multi-segment requests exceed the 4096 vision-only floor).

### K. `--reasoning-format` auto-emission

llama.cpp's `--reasoning-format` controls how `<think>` / `<|channel|>`
blocks split from final content in the response JSON. `build_config`
now reads the GGUF chat template and picks `harmony` for `<|channel|>`
markers, `deepseek` for `<think>` markers, none otherwise. The
companion `--reasoning-budget N` (hard cap on thinking tokens) is
passed through from `choices["reasoning_budget"]`.

### L. llama.cpp build-version gate

New module `vserve/llamacpp_probe.py` runs `llama-server --version`
and parses build number, commit, CUDA runtime version, and the
`GGML_CUDA_FA_ALL_QUANTS` flag. `check_build_compat()` issues
warnings for known-bad combos: CUDA 13.2 + any GGUF produces gibberish
(Unsloth + llamacpp#21371); UD-IQ4_XS / UD-IQ4_NL pre-b8808 produces
gibberish; b8661 has a Windows tokenizer regression. Warnings print
before `backend.start()` in the run banner; the probe failing silently
means "build info unavailable" — never blocks launch.

### S. Prompt-caching primitives

`build_config` now accepts `cache_reuse` (minimum chunk for KV-shift
reuse within a slot; 256 is the canonical default), `slot_save_path`
(persistent slot snapshots for warm-restart of long-context agents),
`cram_mb` (host-memory prompt cache for shared prefixes; 93% TTFT
reduction per llamacpp#20574), and `swa_full` (required for Gemma 4 +
cache-reuse per llamacpp#21468). `start()` routes them to
`--cache-reuse`, `--slot-save-path`, `--cram`, `--swa-full`. The
Gemma-4-cache-reuse-without-swa_full combination raises a config-time
ValueError citing the issue.

### N. Graduated `-ot` strategies

New module `vserve/recipes/ot_strategies.py` exposes a tiered offload
hierarchy (`none → partial-up → moderate → max → layered`) instead of
0.6.0's binary "all experts to CPU or nothing." `pick_ot_strategy()`
walks the ladder and returns the least-aggressive strategy that fits
the VRAM budget with a configurable safety margin. The layered fallback
covers very-large-MoE cases (DeepSeek-V3 671B et al.) with a
layer-range regex.

## Tuning + benchmarking

### O. Empirical `llama-bench` tuner backend

New module `vserve/recipes/llama_bench.py` wraps `llama-bench` for
sweep-and-pick tuning. `run_sweep()` drives the binary with a
caller-supplied axis matrix (e.g. `-p 512,4096,8192 -n 128,256 -fa 1
-ctk f16,q8_0 -ctv f16,q8_0`); `parse_llama_bench_jsonl()` parses the
JSONL output; `pick_best_cell()` selects by weighted prefill/decode
throughput under the `throughput | latency | balanced` profile.
`cache_key()` content-addresses results by `(model, gpu, build)` so
reruns reuse cached sweeps.

### P. `bench.py` TTFT / TPOT / ITL streaming rewrite

`run_streaming_benchmark()` issues concurrent streaming
`/v1/chat/completions` requests, parses SSE chunks, timestamps each
token, and returns a `BenchResult` with TTFT (first-token latency)
p50+p99, TPOT (time-per-output-token) p50+p99, ITL (inter-token-latency)
p99, throughput (tokens/sec + requests/sec), and E2E p99. Accepts a
`max_latency_ms` ceiling for early termination. The legacy sequential
`run_openai_completion_benchmark` and `run_openai_embedding_benchmark`
helpers stay for backward compat.

### T. Persistent decode-tok/s in the picker matrix + `vserve status`

Three pieces:

1. **Perf cache** (`vserve/perf_cache.py`) — per-user JSON under
   `~/.cache/vserve-perf/`, keyed on `(model, GPU UUID, backend,
   build-id, config-hash)`. Schema captures decode tok/s p50, TTFT p50,
   sample count, served name, timestamp. Atomic writes (tmp + rename)
   survive concurrent launches. Cache entries from a different build
   are filtered out — we never show stale numbers.
2. **Measurement-at-launch** — after `vserve run` reaches health-OK,
   a 5-second streaming probe records the actual decode tok/s for the
   exact config that just launched and persists it. The launch banner
   now prints "Decode: 78 tok/s · TTFT 180 ms (measured at launch)".
3. **Picker matrix annotation** — the limits matrix now has a
   companion "Measured decode tok/s" sub-table that shows the cached
   number for every cell with a prior measurement; cells without data
   show "—" (we never show a math-derived estimate because the
   estimate is wrong on exactly the cases the user most needs to know
   about — expert spill, build regressions).
4. **`vserve status` live probe** — when the service is running,
   `vserve status` now fires a 3-second streaming probe and prints
   "Decode: X tok/s   TTFT Y ms (live, 3s probe)" plus the
   launch-time baseline from the cache for comparison. Closes the
   "I don't know if my inference is healthy" loop without leaving the
   shell.

The math-formula option ("Expected tok/s: 80") was explicitly
rejected: a closed-form estimate is misleading on the worst-case
configs (5-8× wrong on expert-spill cliff cases). Measured-or-nothing
is more honest.

### G. Unsloth Dynamic 2.0 quant-tier classification

`models.py` parses the `-UD-<tier>` segment from GGUF filenames and
records the tier on `ModelInfo.quant_tier` (Q4_K_XL, IQ4_XS, MXFP4_MOE,
TQ1_0, Q5_K_XL, Q8_K_XL, etc.). The `UNSLOTH_QUANT_TIERS` table records
approximate bits-per-weight, min VRAM per billion params, and a
relative quality ranking — useful for per-card recommendation and for
the build-version gate (item L) to fire per-tier (e.g. UD-IQ4_XS
specifically requires b8808+). MXFP4_MOE-tagged files set
`is_moe=True` and `quant_method="mxfp4_moe"` regardless of upstream
metadata.

## Discipline notes

This release was assembled in dependency order behind a single
test/lint/mypy gate. The release-notes correction in item AA cites
the maintainer-canonical fix inline (vllm#40807, #41403) per the
`feedback_release_discipline` memory update — diagnoser hints, recipe
recommendations, and "Try this:" suggestions are claims about the world
and need an upstream citation, not just internal reasoning. Every new
recipe-table entry above is paired with a Unsloth doc / vLLM doc /
issue link in the source comments.

---

# vserve 0.6.0 Release Notes

`v0.6.0` ships seven tuner-correctness fixes discovered through a live
session debugging Gemma-4 (both the 26B-A4B MoE on llama.cpp and the 31B
NVFP4 multimodal on vLLM). Every defect shared the same root cause: the
tuner was a memory-budget calculator pretending to be a config validator.
It emitted cells the engine actually refused — for backend-incompatible
KV dtypes, undersized multimodal batches, default served-name path
issues, missing tool-parser wiring, MoE auto-offload that hurt
throughput, slot ceilings read from the wrong KV-dtype column, and
compute-buffer optimism. Each defect is now closed by code; the
underlying memory-math-vs-probe redesign is tracked separately for a
later release.

## 1. llama.cpp slot-ceiling clamped to the *effective* KV-dtype column

Previously the interactive picker showed `max()` across all KV-dtype
columns of `limits[ctx]`. For Gemma-4-26B-A4B on q8_0, `limits[128k] =
{f16: 11, q8_0: 22, q4_1: 38, q4_0: 42}` displayed `max = 42` — but the
runtime defaulted to f16 and OOM'd at slot 12. The picker now uses the
runtime-effective dtype column (default `recommended_kv_dtype`, which is
q8_0 here) and the slot prompt reads `entry[effective_kv]`.

## 2. llama.cpp `-ot` auto-applied only when needed

Defect: vserve 0.5.8 auto-added `-ot ".ffn_.*_exps.=CPU"` for *every*
MoE model. For a 12.5 GB model on a 48 GB GPU, that pushed hot expert
weights to system RAM — every expert lookup traversed PCIe at ~32 GB/s
instead of HBM at ~1 TB/s, collapsing tokens/s by ~8× (Unsloth's own
benchmark: 119 → 30 tok/s). Auto-apply is now gated on
`_llamacpp_needs_moe_offload(limits, ctx, slots, kv)`, which returns
True only when `full_offload=False` OR the chosen slots exceed the
no-`-ot` capacity. Models that fit get the fast path.

## 3. vLLM backend × dtype × architecture compatibility

Defect: `vserve run` on Gemma-4-31B-IT-NVFP4 picked
`kv-cache-dtype: turboquant_3bit_nc` (the tuner's high-slot
recommendation). Engine init crashed with:

```
ValueError: Selected backend TRITON_ATTN is not valid for this
configuration. Reason: ['kv_cache_dtype not supported']
```

Gemma-4 has heterogeneous head dimensions (`head_dim=256,
global_head_dim=512`) → vLLM forces the TRITON_ATTN backend → TRITON_ATTN
does not accept any `turboquant_*` dtype. The tuner now detects this
architecture trigger and filters every `turboquant_*` cell out of
`limits` and `kv_cache_dtypes` in the tune output. Compatible dtypes
(`auto`, `fp8`) survive untouched.

## 4. Multimodal `max-num-batched-tokens` floor

Defect: Gemma-4-31B-IT-NVFP4 (vision + audio) crashed at startup with:

```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048).
```

vLLM auto-disables MM-input chunking for bidirectional-attention vision
encoders — a single image must fit in a single batch. vserve emitted no
`max-num-batched-tokens`, so vLLM used its 2048 default. The tuner now
sets a conservative 4096 floor whenever the model has a `vision_config`
or `audio_config` block (covers Gemma-4 vision's 2496 with margin).

## 5. `served-model-name` aliases auto-emitted

Defect: vLLM defaulted the served model id to the full filesystem path
`/opt/vllm/models/nvidia/Gemma-4-31B-IT-NVFP4`. Every OpenAI-compatible
client sending a short name (`"model": "gemma-4-31b"` or
`"model": "nvidia/Gemma-4-31B-IT-NVFP4"`) got HTTP 400 "model not
found." `build_config` now emits both the canonical
`provider/model` id and a lowercased slug with chat suffixes stripped:

```yaml
served-model-name:
  - nvidia/Gemma-4-31B-IT-NVFP4
  - gemma-4-31b
```

## 6. Tool-call parser auto-discovery (vLLM 0.21+)

Defect: vLLM 0.21 ships a per-architecture `gemma4_tool_parser` (and
many others) but vserve had no mapping from architecture → parser name.
A client sending `{"tool_choice": "auto", "tools": [...]}` got HTTP
400: `'auto' tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`. `tune()` now maps `Gemma4*ForCausalLM`
and `Gemma4*ForConditionalGeneration` to `gemma4` (using vLLM's lazy
registry as ground truth for what's actually installed), and
`build_config()` consumes the suggested parser when `--tools` is
requested.

## 7. llama.cpp compute-buffer reserve

Defect: tuner held back only 10 % of VRAM for the layer-fit decision,
then handed every remaining byte to KV-cache math. Reality on
high-parallel runs is that llama.cpp also needs ~5–8 GB for prompt
prefill compute buffers (scale with `-np × ubatch`), CUDA workspace,
pinned host-side transfer buffers, and the embedding/output projection.
Without that reserve, the tuner approved `22 slots × 128k × q8_0` on a
48 GB card; at runtime llama.cpp silently moved tensors back to host
RAM (40.9 GB RAM peak observed, decode collapsed). The tune budget now
subtracts a fixed 10 % compute reserve from the KV-cache pool. The
gemma 256 k slot ceiling drops from 6 → 5, qwen3.5-style models from
their previous ceiling by 1; everything else by 0–1. Headroom in
exchange for not silently OOM-spilling.

## Tests

`587 → 603 passed` (+16). New coverage:

- `tests/test_cli.py` — slot-ceiling regression, `-ot` gating helper
  across (within-cap, exceeds-cap, partial-offload, non-MoE) cases.
- `tests/test_backends.py` — TRITON_ATTN compat filter (heterogeneous
  head_dim detection, TurboQuant removal, no-op for Llama, recommended
  profile invalidation), MM token floor (text-only / vision / audio),
  `served-model-name` alias generation, tool-parser arch-to-name
  mapping with runtime-registry cross-check.
- `tests/test_llamacpp.py` — Gemma-4 SWA test recalibrated to 5-slot
  ceiling (was 6) to reflect the new compute reserve.

`ruff`, `mypy` clean.

## Migration

Profiles saved by 0.5.x may carry settings the new tuner would reject
(TurboQuant on a TRITON_ATTN architecture, MoE `-ot` on a model that
fits). To regenerate cleanly:

```bash
rm /opt/{vllm,llama-cpp}/configs/models/<provider>--<model>.<profile>.{yaml,json,sh}
vserve run <provider>/<model> --save-profile <name>
```

## Tracking — what's deferred to 0.7.0

The defects above are still all closed by code, but the structural
redesign that prevents the entire defect class — **probe-based tune**:
launch the engine for ~5 s per candidate cell, drop cells that
crash — is tracked in
`docs/plans/2026-05-19-tune-redesign-probe.md`. Closing that ticket
turns the per-defect band-aids into a single test that runs at tune
time.

---

# vserve 0.5.9 Release Notes

`v0.5.9` is a hotfix for a context-sizing bug discovered after 0.5.8 shipped.

## Fixed: llama.cpp per-slot context was divided by `--parallel`

`llama-server`'s `-c / --ctx-size` is the *total* KV-cache size across all
slots — per-slot window = `ctx_size / parallel`. vserve 0.5.8 wrote the
user-facing per-slot context into `ctx_size` directly, so a request like
`--context 32768 --slots 4` ended up serving with `n_ctx_seq = 8192`
(32768 ÷ 4) per request. With 8 slots, a 24 k context shrank to 3 k.

`build_config()` now multiplies the per-slot context by `parallel` when
writing the JSON launch config:

```python
ctx_size      = per_slot_context * parallel     # llama-server -c value
ctx_per_slot  = per_slot_context                # informational, what user asked for
```

`vserve status` and the `Starting with …` line now prefer `ctx_per_slot`
so the displayed value matches the tune output and the user's intent.

**Verified live on `gemma-4-26B-A4B-it-GGUF`** (4 slots, requested
context 32 k):

```
exec llama-server … -c 131072 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ot '.ffn_.*_exps.=CPU' --no-mmap
journal:  slot load_model: id 0 | task -1 | new slot, n_ctx = 32768
journal:  slot load_model: id 1 | task -1 | new slot, n_ctx = 32768
journal:  slot load_model: id 2 | task -1 | new slot, n_ctx = 32768
journal:  slot load_model: id 3 | task -1 | new slot, n_ctx = 32768
```

Each slot now has the full 32 k window the user asked for.

## Tests

- 3 new tests: `test_build_config_ctx_size_is_per_slot_times_parallel`,
  `test_build_config_single_slot_ctx_size_equals_context`, and updated
  `test_start_emits_ctk_ctv_b_ub_and_ot` to assert `-c 32768` (8192 × 4) +
  `-np 4` end up in the launch script. The basic-config test also
  updated to reflect the new dual `ctx_size` / `ctx_per_slot` shape.
- 579 → **581 passed**; ruff + mypy clean.

## Migration

Profiles saved by 0.5.8 carry the wrong `ctx_size` (per-slot, not total).
Re-run `vserve run … --save-profile <name>` to regenerate; or hand-edit
the JSON to set `ctx_size = ctx_per_slot * parallel`.

---

# vserve 0.5.8 Release Notes

`v0.5.8` is the first release on top of `v0.5.7`. It widens the supported
vLLM runtime range to include vLLM 0.21, ships three independent UX defect
fixes for `vserve run` / `vserve stop`, and lifts the llama.cpp tuning
surface to roughly parity with the vLLM side (KV-cache dtype matrix, MoE
expert-CPU offload auto-detect, batch-size knobs). All work was driven by
the 2026-05-19 audit session.

## Upgrade

```bash
uv tool upgrade vserve
```

or:

```bash
pip install --upgrade vserve
```

After upgrading vserve, bring the runtime to the new pinned stable (the local
cache is invalidated automatically on successful install):

```bash
vserve stop
vserve runtime upgrade vllm --stable
vserve runtime check vllm
```

For optional GGUF metadata support from the upstream `gguf` package:

```bash
pip install --upgrade 'vserve[llamacpp]'
```

## vLLM 0.21 runtime support

- `SUPPORTED_VLLM_RANGE` widened from `>=0.20,<0.21` to `>=0.20,<0.22`.
- `PINNED_STABLE_VLLM` advanced from `0.20.0` to `0.21.0`.
- `vserve runtime check vllm` accepts any `0.20.x` or `0.21.x` release; the
  existing pre-release / dev-build guard still rejects `rc` and `dev` builds.
- `vserve runtime upgrade vllm --stable` installs `vllm==0.21.0` and clears
  the cached `RuntimeInfo` so the next `vserve run` re-probes once.
- README and Prerequisites table updated to describe the wider support window.

### What 0.21 brings to existing serving profiles

The generated vLLM YAML profiles produced by `v0.5.6+` continue to work
without edits — none of the flags vserve emits (`max-model-len`, `max-num-seqs`,
`kv-cache-dtype`, `enable-prefix-caching`, `max-num-batched-tokens`,
`performance-mode`, `optimization-level`, `block-size`, `kv-cache-memory-bytes`,
`gpu-memory-utilization`, `quantization`, `enable-auto-tool-choice`,
`tool-call-parser`, `reasoning-parser`, `trust-remote-code`, `dtype`) changed
semantics or were renamed in 0.21.

Notable upstream additions that are now available behind the existing config
plumbing (no vserve changes required to opt in via `extra_args` or a
hand-edited profile):

- TOKENSPEED_MLA attention backend for DeepSeek-R1 / Kimi-K2.5 prefill+decode
  on Blackwell GPUs (`tokenspeed-mla` and `tokenspeed-triton` wheels are
  pulled in by `pip install vllm==0.21.0`).
- KV Offload + Hybrid Memory Allocator (HMA) with scheduler-side
  sliding-window group support and multi-connector store completion.
- Speculative decoding now respects reasoning / thinking budgets — correct
  spec decode against reasoning models such as Qwen3-Thinking and DeepSeek-R1.
- Multi-stream pre-attention GEMM with a configurable
  `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` knob and BF16 / MXFP8 all-to-all
  support for FlashInfer one-sided communication.
- New tool parsers (Cohere reasoning + tool, LFM2 / LFM2.5 tool) and an
  upgraded XGrammar 0.2.0 with structural tags for strict tool calling.
  Detection through `vserve run --tools` picks these up automatically because
  parser selection has always queried the installed vLLM runtime's registry
  rather than a hard-coded vserve list.

### Behaviour change to flag to API clients

vLLM 0.21 renamed the field that exposes a thinking-model's reasoning channel
in `/v1/chat/completions` responses from `message.reasoning_content` to
`message.reasoning`. vserve itself does not parse responses, so the change is
transparent to all vserve commands, but external clients that previously read
`message.reasoning_content` need to update.

### Tuning, Probe, and Backend Protocol

No semantic changes — the PagedAttention block-rounded KV math, the FP8 /
TurboQuant KV-dtype tables, the chunked-prefill scheduler profiles, and the
runtime parser registry probe in `VllmBackend` all carry over unchanged.

The tuning fingerprint already includes `vllm_version`, so cached limits for
models tuned under 0.20 are correctly invalidated and re-tuned on first 0.21
run. No user action is required.

## UX fixes

### `vserve stop` printed the unknown-owner warning twice

When the GPU session marker was missing — typical after a service was started
outside vserve (`sudo systemctl start vllm.service`, an older vserve version
predating session markers, or a stray `_clear_all_session_markers` run) —
`vserve stop` printed `"vLLM is running, but the owning user is unknown."`
twice before the `Stopped: vLLM.` line. Root cause: `cli.py:stop()` invokes
`_session_or_exit(allow_unknown_owner=True)` twice (initial pre-flight +
TOCTOU re-check under flock); both calls fell through to the
warn-and-return branch.

- `_session_or_exit` gained a `quiet=False` parameter. The two re-check call
  sites in `stop()` (probe-failed fallback at line 3472, main TOCTOU re-check
  at line 3507) pass `quiet=True`; the initial check stays loud.
- `stop()` now claims the orphan session via `write_session("orphan-claim")`
  immediately after the first warning, so the underlying condition is
  resolved rather than just suppressed. The next `vserve stop` / `vserve run`
  from the same user resolves cleanly without any warning at all.
- The warning copy is now actionable: `"<backend> is running, but no vserve
  session marker exists." → "Proceeding as <user>. Claim future sessions by
  starting via 'vserve run'."`

### `vserve run --profile <path>` lagged 10–60 seconds before starting

Every `vserve run` called `_check_backend_runtime_or_exit`, which spawned
three subprocesses against `/opt/vllm/venv` on every invocation:
`vllm --version` (15 s timeout), a metadata-import script that imports
vllm/torch/transformers just to read their versions (20 s timeout), and
`pip check` (60 s timeout). None of this changes between two consecutive
`vserve run` calls unless the venv itself is rewritten.

- New `~/.cache/vserve/runtime/vllm.json` cache file keyed by the mtimes of
  `vllm_bin`, `vllm_python`, and the venv's `site-packages` directory. Any
  pip install/uninstall/upgrade changes one of those mtimes, invalidating
  the cache cleanly.
- `collect_vllm_runtime_info` gained `prefer_cache=False` and
  `with_pip_check=True` keyword arguments. The hot path (`vserve run`) calls
  with `prefer_cache=True, with_pip_check=False` and gets a cached
  `RuntimeInfo` plus zero subprocesses on cache hits. Diagnostic commands
  (`vserve doctor`, `vserve runtime check vllm`) keep `prefer_cache=False`
  so they always re-probe in full, including `pip check`.
- `vserve runtime upgrade vllm --stable` invalidates the cache after a
  successful install (belt-and-suspenders alongside the mtime key).
- **Measured on this box: 4.9 s cold → 57 ms warm** (~85× speed-up on the
  hot path).

### `vserve run` re-printed the same 5 journal lines every 3 seconds

The health-wait loop in `_launch_backend` tailed `journalctl -u <unit> -n 5`
on each 3-second poll and unconditionally re-printed all 5 lines under a
fresh `Latest service logs:` header. A model that took 60 s to load
produced ~20 identical blocks of output.

- New private `_wait_for_health` helper extracted from `_launch_backend`.
  It hashes the journal tail (SHA-256 of the joined lines) and only prints
  the block when the hash changes. Between unchanged polls a single `.`
  (no newline) confirms liveness; every 30 s a `still starting... (Ns)`
  line flushes on its own. Final-state green/red banner stays byte-identical
  so external scrapers that match `"vLLM is running"` continue to work.
- The helper returns `"ready"` / `"stopped"` / `"timeout"`; `_launch_backend`'s
  outer logic decides what to render for each terminal state, keeping the
  state machine clear and testable. `urlopen_fn` and `sleep_fn` are
  injectable for tests.

## llama.cpp tuning uplift

Background: the audit found vserve's llama.cpp side emitting only ~12 % of
the upstream `llama-server` flag surface. The biggest gaps were KV-cache
quantization (`--cache-type-k/v`) and the Unsloth-recommended MoE
expert-CPU offload (`-ot ".ffn_.*_exps.=CPU"`). Both are now first-class.
See `docs/plans/2026-05-19-llamacpp-tuning-uplift.md` for the full audit.

### Context × KV-dtype matrix in `vserve tune`

Tune output for GGUF models is now a 2-D table mirroring the vLLM side.
Columns: F16 KV (legacy default), Q8 KV (≈2 × more slots, near-zero
quality loss), Q4_1 KV, Q4_0 KV (≈4 × more slots, validate before
adopting). Recommendation line: `-ctk q8_0 -ctv q8_0` whenever Q8 strictly
beats F16 at the largest fitting context. Symmetric K/V is enforced
because llama.cpp's fused Flash-Attention falls back to the slow non-fused
implementation on asymmetric pairs.

New CLI flags on `vserve run`:

- `--kv-cache-k {f16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl, bf16, f32}`
- `--kv-cache-v {…}` (defaults to `--kv-cache-k` for the fused-FA path)

The launcher emits `-ctk <K> -ctv <V>` to `llama-server`. Asymmetric pairs
print a yellow performance-degradation warning.

**Measured on this box:**

| Model | Context | F16 slots | Q8 slots | Q4_0 slots |
|---|---|---|---|---|
| `Qwen3.6-27B-GGUF-Q4_K_XL` | 262 k | 1 | **3** | 6 |
| `Qwen3.6-27B-GGUF-Q4_K_XL` | 131 k | 3 | **6** | 12 |
| `gemma-4-26B-A4B-it-GGUF` | 131 k | 11 | **22** | 42 |
| `gemma-4-31B-it-GGUF` | 131 k | 2 | **4** | 9 |

### MoE expert CPU offload (`-ot`) auto-detect

When a GGUF reports `expert_count > 1`, `vserve tune` now computes a
second slot table assuming `-ot ".ffn_.*_exps.=CPU"` is applied. The
estimate is element-count-based — per-layer attention + shared FFN stays
on GPU, expert FFN moves to CPU. `vserve run` auto-applies the same
pattern unless `--no-moe-offload` is passed or `--override-tensor`
patterns are supplied explicitly.

**Measured on `gemma-4-26B-A4B-it-GGUF` (128 experts, 8 active):**

- Analytic prediction: `-ot` frees ~11.1 GB → GPU resident ~1.4 GB.
- Actual live `vserve run gemma-4-26B-A4B-it-GGUF --yes --replace --context 32768 --slots 4`:
  - Generated `active.sh` contains
    `-c 32768 -ngl 30 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ot '.ffn_.*_exps.=CPU'`.
  - `nvidia-smi` reports **4284 MiB used / 44120 MiB free** while
    serving — model + KV cache combined.
  - Round-trip `/v1/chat/completions` returns valid output
    (`content: "READY."`, `reasoning_content` carries the Gemma
    thinking channel).

Before this change the same model used ~24 GB at the previous default
(no `-ot`, F16 KV). Net: ~**5.5 × VRAM reduction at the same context**,
matching the analytic estimate to within 100 MB.

New CLI flags on `vserve run`:

- `--override-tensor PATTERN` / `-ot PATTERN` (repeatable)
- `--no-moe-offload` (opt-out)

`vserve run` also auto-appends `--no-mmap` whenever the generated config
contains any `override_tensors` entry. Modern llama.cpp (b583+) emits a
perf warning when `-ot` and mmap are combined; the auto-flag keeps the
MoE-offload path on the fast loader without user intervention. Profiles
that explicitly set `"mmap": true` in JSON suppress the auto-flag.

### `--batch-size` / `--ubatch-size` surfaced

`vserve run` now accepts `--batch-size N` (`-b`) and `--ubatch-size N`
(`-ub`). Defaults remain llama.cpp's own (`2048` / `512`) when the flags
are omitted — vserve only emits them when set explicitly, so existing
profiles continue to behave identically.

### Unsloth UD-2.0 quant detection

`ModelInfo.is_unsloth_ud` now returns `True` for models whose provider is
`unsloth` and whose directory contains a `*-UD-*.gguf` file (e.g.
`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`,
`Qwen3.6-27B-UD-Q4_K_XL.gguf`). Unsloth UD- variants are imatrix-
calibrated against Unsloth's `calibration_v5` and are the recommended
download for serving quality. The property is exposed for downstream
badging.

### Tuning fingerprint

`LIMITS_SCHEMA_VERSION` bumped from 4 → 5 to force a one-shot re-tune of
every GGUF model the first time it's used under 0.5.8 — the limits file
now stores a 2-D `{ctx: {dtype: slots}}` matrix rather than the legacy
`{ctx: slots}` flat form. The renderer keeps a backward-compatible path
that reads the old flat form too, so old cached files are still readable
during the transition.

## Tests

- 26 new tests across `tests/test_cli.py`, `tests/test_runtime.py`, and
  `tests/test_llamacpp.py`:
  - **vLLM 0.21 range:** `test_check_vllm_compatibility_accepts_stable_021`,
    `test_check_vllm_compatibility_rejects_022`
  - **Unknown-owner warning:** `test_stop_warns_once_when_owner_unknown`,
    `test_stop_claims_orphan_session_before_release`,
    `test_stop_does_not_claim_when_session_already_exists`,
    `test_stop_warns_once_under_probe_failed_fallback`
  - **Runtime cache:**
    `test_collect_vllm_runtime_info_returns_cached_when_prefer_cache_and_key_matches`,
    `test_collect_vllm_runtime_info_repopulates_cache_on_key_drift`,
    `test_collect_vllm_runtime_info_does_not_run_pip_check_on_cheap_path`,
    `test_invalidate_vllm_runtime_cache_removes_file`
  - **Health-wait dedup:** `test_wait_for_health_returns_ready_on_first_200`,
    `test_wait_for_health_deduplicates_log_block`,
    `test_wait_for_health_prints_new_block_when_tail_changes`,
    `test_wait_for_health_returns_stopped_when_service_dies`,
    `test_wait_for_health_returns_timeout_after_budget`
  - **llama.cpp KV-dtype matrix (L1):**
    `test_llamacpp_kv_cache_bytes_q8_halves_attention_bytes`,
    `test_llamacpp_kv_cache_bytes_q4_smaller_than_q8`,
    `test_tune_emits_kv_dtype_matrix`,
    `test_tune_recommends_q8_when_strictly_more_slots`,
    `test_build_config_emits_kv_cache_dtypes`
  - **llama.cpp MoE -ot + batch/ubatch + start (L2, L3, L7):**
    `test_build_config_emits_batch_and_ubatch`,
    `test_build_config_records_override_tensors`,
    `test_start_emits_ctk_ctv_b_ub_and_ot`,
    `test_start_omits_no_mmap_when_no_override_tensors`,
    `test_start_respects_explicit_mmap_true`
  - **Unsloth UD-2.0 (L4):**
    `test_is_unsloth_ud_detects_UD_prefix`,
    `test_is_unsloth_ud_false_for_plain_quant`,
    `test_is_unsloth_ud_false_for_non_unsloth_provider`
- Existing `test_upgrade_vllm_stable_force_reinstalls_pinned_stable` updated
  to assert both the new `vllm==0.21.0` pin and the cache invalidation hook.
- Existing
  `test_tune_qwen35_counts_only_full_attention_layers_for_context_capacity`
  and `test_tune_gemma4_caps_swa_cache_by_sliding_window` updated to assert
  on the F16 column of the new 2-D matrix (the F16 numbers match the legacy
  single-int values byte-for-byte — confirms the refactored math is
  equivalent at F16).

## Verification

Fresh local release checks before tagging:

- `uv run ruff check src/ tests/` — clean
- `uv run mypy src/vserve/ --ignore-missing-imports --check-untyped-defs` — clean
- `uv run pytest tests/ -q --tb=short` — **579 passed** (547 baseline + 4
  churn since 0.5.6 + 15 UX patch + 11 llama.cpp uplift + 2 L7)
- Live smoke against `/opt/vllm/venv` after `pip install vllm==0.21.0`:
  - `vserve runtime check vllm` reports `0.21.0 is within supported range >=0.20,<0.22`
  - `vserve doctor` — 27 OK, 2 pre-existing warnings (stale RPC sockets,
    optional `gguf` package), 0 fail
  - `vserve run --profile <custom>.yaml --yes --replace` restarts the
    service and `/v1/models` + `/v1/chat/completions` return responses with
    `system_fingerprint: vllm-0.21.0-…`
- Analytic tuning verified across **every downloaded model** — 9 of 9 with
  weights produce a context × KV-dtype slot table and recommended profiles:
  - vLLM: `Qwen3.5-0.8B`, `Qwen3.5-4B`, `Qwen3-Embedding-8B`, `Qwen3.6-27B-FP8`,
    `Qwen3.6-35B-A3B-FP8` (MoE), `openai/gpt-oss-20b`
  - llama.cpp: `unsloth/gemma-4-26B-A4B-it-GGUF` (MoE),
    `unsloth/gemma-4-31B-it-GGUF`, `unsloth/Qwen3.6-27B-GGUF-Q4_K_XL`
  - (A 10th directory `unsloth/Qwen3.6-27B-GGUF-Q5_K_XL` is an aborted
    download with no `.gguf` weights and is correctly excluded by vserve's
    model scanner — not a tuning failure.)
- Live `vserve run` against `gemma-4-26B-A4B-it-GGUF` with MoE auto-default:
  - Generated `active.sh` contains
    `-ctk q8_0 -ctv q8_0 -fa on -ot '.ffn_.*_exps.=CPU'`
  - `nvidia-smi` after startup: **4284 MiB used** (vs ~24 GB at the
    previous default — a 5.5× reduction)
  - Round-trip `/v1/chat/completions` returns valid output.
