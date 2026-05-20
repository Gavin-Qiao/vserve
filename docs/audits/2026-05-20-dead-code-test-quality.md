# Dead Code & Test Quality Audit — 2026-05-20

Scope: `src/vserve/` (dead code) and `tests/` (test quality). Read-only.

## Dead-code findings

**Top hits** — zero consumers in `src/`:

- `src/vserve/compare.py` (43 lines) — `filter_models_for_workload`, `rank_models`. Only `tests/test_compare.py` references them. Wire into a CLI command or delete.
- `src/vserve/recipes/ot_strategies.py` (90 lines) — `pick_ot_strategy`, `OT_STRATEGIES`. Only its own test imports it.
- `src/vserve/recipes/llama_bench.py` (148 lines) — `parse_llama_bench_jsonl`, `run_llama_bench`. Only its own test imports it.
- `src/vserve/recipes/__init__.py:3-9` — re-exports `SAMPLING_DEFAULTS`, `SamplingDefaults`, `get_sampling_defaults`. No external consumer.

**Lower-impact**:

- `src/vserve/backends/vllm.py:216` — `_is_multimodal_model()` dead; `_mm_batched_tokens_floor` (L225) duplicates the same vision/audio-config lookup.
- `src/vserve/config.py:588,593` — `read_timing`/`write_timing`. Only `tests/test_imports.py:26` references them.
- `src/vserve/cli.py:4610` — sole ERA001 lint hit (orphan `# action == "curve"` comment).

Unused imports (`ruff --select F401`): **none**.

## Test-quality issues

- **Substring-on-human-output assertions are pervasive.** Highest density: `tests/test_cli.py` (77 occurrences of `" in ` in `stdout`), `tests/test_llamacpp.py` (62), `tests/test_variants.py` (26), `tests/test_welcome.py` (17). Each is one Rich-format tweak away from a false failure. Convert the load-bearing ones to JSON-mode assertions (`--json`) and demote help-text checks to a single smoke test.
- **Parametrize candidates** — `tests/test_run_llamacpp_flags.py:62-87` has 6 nearly identical `test_help_lists_*` tests, and `:90-126` has 5 `*_threads_through` flag tests with the same body shape. `tests/test_bench_command.py:110-156` has 4 identical "respects flag X" tests. Whole suite uses `@pytest.mark.parametrize` only **once** (in `test_models.py`).
- **No-assertion tests** in `tests/test_lock.py` (`test_release_idempotent` at L50, `test_lock_released_on_process_death` at L160, `test_different_model_locks_independent` at L198, and four `test_check_session_*` tests at L364/376/404/415). They only verify "does not raise" — add explicit positive assertions on lock-state or session-state.
- **Mock-target ambiguity** (no live break, but a footgun): `tests/test_picker_perf_block.py:194` patches `vserve.bench.run_streaming_benchmark`, which is correct only because the SUT (`_print_status_inference_probe`) does a local `from vserve.bench import ...`. Three other call sites in `cli.py` (lines 32, 2694, 4718) use the top-level binding and require `vserve.cli.run_streaming_benchmark`. One refactor away from silent test bypass. Pick one convention.

## Test-file split proposals (>1000 lines)

- `tests/test_cli.py` (2913 lines, 134 tests) — split into `test_cli_run.py`, `test_cli_profile.py`, `test_cli_download.py`, `test_cli_misc.py`.
- `tests/test_llamacpp.py` (2111 lines, 116 tests) — already class-organised; promote each class to its own file (`test_llamacpp_tune.py`, `test_llamacpp_embedding.py`, `test_llamacpp_metadata.py`, ...).
- `tests/test_backends.py` (1531 lines, 98 tests) — split vLLM-specific classes (`TestVllm*`, lines 568-1535) into `test_vllm.py`; keep registry + protocol tests here.

## Coverage gaps (no test sibling)

- `src/vserve/model_files.py` — used by 6 source files (`cli.py`, `models.py`, `runtime.py`, `backends/llamacpp.py`, `backends/vllm.py`); no `tests/test_model_files.py`. Path-classification regressions here would surface as confusing failures elsewhere.
- `src/vserve/backends/protocol.py` — Protocol-only; coverage is indirect via implementer tests.
- `src/vserve/backends/vllm.py` — tested inside `tests/test_backends.py` rather than `test_vllm.py` (see split proposal above).
- `src/vserve/llamacpp_probe.py` — covered indirectly by `tests/test_llamacpp.py:1747-1799`; consider promoting to `test_llamacpp_probe.py`.

## Fixture overlap (conftest.py)

`fake_model_dir`, `fake_moe_model_dir`, `fake_gguf_model_dir`, `fake_embedding_model_dir` are correctly distinct (HF safetensors / HF-MoE / GGUF generative / GGUF embedding). No duplication; keep as-is.
