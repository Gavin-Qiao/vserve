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
