# cli.py sprawl audit — 2026-05-20

`src/vserve/cli.py`: **6,136 lines**, 103 top-level `def`s (81 private helpers,
13 Typer commands, 2 sub-apps, 1 dashboard callback). Top-level imports: 11;
**inline imports inside function bodies: 167** — nearly every command
re-imports `vserve.*` lazily.

## Top-10 longest functions

| # | Function | Lines | `cli.py:LINE` | Suggested home |
|---|---|---|---|---|
| 1 | `doctor` | 513 | cli.py:5423 | `commands/doctor.py` — already-nested `_ok/_warn/_fail/_fail_or_warn` (364-line nested helper at L5476) is a self-contained framework |
| 2 | `status` | 331 | cli.py:4778 | `commands/status.py`; nested `_read_backend_config` (161 L) + `_running_next_action` (149 L) should be module-level |
| 3 | `_launch_backend` | 283 | cli.py:2700 | `serve_runner.py` — orchestrates lock, stop-existing, start, health-wait, diagnostics |
| 4 | `fan` | 280 | cli.py:4355 | `commands/fan.py` (logic already lives in `fan.py`; CLI wrapper is huge) |
| 5 | `tune` | 250 | cli.py:1591 | `commands/tune.py` — single body fuses validation, picker, loop, pre-cache UX |
| 6 | `init` | 243 | cli.py:5169 | `commands/init.py` — pure setup wizard |
| 7 | `_scripted_config` | 219 | cli.py:3005 | `recipes/scripted.py` — **22 kwargs**, half are `llamacpp_*`-prefixed |
| 8 | `run` | 216 | cli.py:4044 | `commands/run.py` — **33 Typer flags**, ~23 if/elif branches |
| 9 | `_download_model` | 204 | cli.py:1276 | `model_files.py` — domain logic, not CLI |
| 10 | `cache_clean` | 200 | cli.py:5936 | `commands/cache.py` — nested `_unsafe_cache_root_reason` (177 L) is its own concern |

Five functions exceed 200 lines; nested helpers `_fail_or_warn` (doctor, 364 L),
`_fail` (init, 225 L), `_read_backend_config` (status, 161 L), and
`_running_next_action` (status, 149 L) are also each oversized.

## Patch-sediment markers

`# TODO|FIXME|XXX|HACK|workaround|kludge` returns **0 matches**. Patch sediment
shows up as **version-referencing prose** inside `_diagnose_engine_failure`
(cli.py:2403, 2408, 2414, 2432, 2441 — "turboquant_* dtypes starting in
0.6.1", "vserve 0.6.0+ does this automatically", "vserve 0.6.0+ reserves more
compute headroom"…), "Pre-0.5.8 llama.cpp cache shape" (cli.py:2156), "Prefer
ctx_per_slot (vserve 0.5.9+)" (cli.py:5021). These belong in a diagnostics
module with versioned rule tables, not inline strings.

## Extraction proposals

1. **`commands/doctor.py`** (cli.py:5412-5934, ~520 L). `_emit/_ok/_warn/_fail/_fail_or_warn` mini-framework + check stanzas form a self-contained pipeline. Move `_doctor_summary_label` (cli.py:5412) with it.
2. **`commands/status.py`** + **`status/backend_view.py`** (cli.py:4635-5167, ~530 L). Lift `_backend_manifest` and `_read_backend_config` to module scope — three callers (cli.py:4635, 4783, 5117) duplicate manifest-read+config-load.
3. **`serve/runner.py`** (cli.py:2700-2982). `_launch_backend` (40 branches) + `_wait_for_health` (2522), `_diagnose_engine_failure` (2365), `_fetch_engine_log_for_diagnosis` (2456), `_print_engine_diagnosis` (2506), `_measure_and_cache` (2686), `_write_bench_to_perf_cache` (2637).
4. **`recipes/scripted.py` / `recipes/picker.py`** (cli.py:3005-4042, ~1,040 L). `_scripted_config`, `_custom_config*`, `_choose_*_scripted_defaults`, `_print_limits_table`, `_print_measured_cells_block`, `_print_llamacpp_moe_block`, `_vllm_*` helpers — model/recipe domain, not CLI.
5. **`commands/tune.py`** (cli.py:1591-1997, ~410 L). `tune` + `_run_tuning_benchmarks`, `_benchmark_candidate_names`, `_llamacpp_benchmark_candidates`, `_exception_is_startup_timeout`, `_measurement_succeeded`, `_wait_backend_stopped`, `_print_benchmark_summary`. Pre-cache block (cli.py:1774-1837) extracts cleanly.

Repeated patterns: `read_limits(limits_path(m.provider, m.model_name))` appears 15× (cli.py:365, 688, 740, 1508, 1555, 1706, 3519, 3527, 3703…); `perf_cache.gpu_uuid_or_index` + `lookup_*` repeats at 2249, 2637, 5109; `_pick()` items-list construction is hand-rolled in 7 sites.

## Flag-naming inconsistencies

- `--kv-cache-dtype` (vLLM) vs `--kv-cache-k`/`--kv-cache-v` (llama.cpp): `-dtype` is suffix-key, `-k`/`-v` is positional. Unify.
- MoE knobs mix polarities: `--n-cpu-moe N` (llama.cpp) vs `--no-moe-offload` (disable auto). Consider `--moe-offload/--no-moe-offload` + `--moe-cpu-experts N`.
- `--batch-size` (llama.cpp `-b`), `--ubatch-size`, and `--batched-tokens` (vLLM) collide conceptually and none name their backend. Prefix or rename.
- `--max-tokens` (bench, per-request) vs `--batched-tokens` (vLLM scheduler) read as plural forms of the same knob.
- Bench unit-suffix mix: `--duration-s` and `--max-latency-ms` vs `--bench-seconds`/`--bench-startup-seconds` (no unit). Pick one.
- `--pre`/`--prerelease` dual-alias on `update`; `--upgrade` overloads with `runtime upgrade` sub-app.
