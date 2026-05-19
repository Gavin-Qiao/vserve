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
