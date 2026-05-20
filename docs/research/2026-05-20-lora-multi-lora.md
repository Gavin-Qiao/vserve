# LoRA / Multi-LoRA serving — research for future vserve minor release

Date: 2026-05-20. Hardware: RTX PRO 5000 48GB Blackwell sm120. vserve baseline: 0.6.1b1 (base-model only).

## 1. vLLM multi-LoRA plumbing (2026)

Stable flags ([1]): `--enable-lora`, `--lora-modules`, `--max-loras` (concurrent GPU-pool), `--max-lora-rank`, `--max-cpu-loras` (CPU LRU), `--lora-target-modules` (suffix filter). New JSON `--lora-modules` form carries `base_model_name` and populates `parent`/`root` on `/v1/models`. Legacy `name=path` still works. `--max-lora-rank` must equal the largest actual rank; oversizing wastes VRAM and degrades throughput. Common defaults: 16/32, kernels go up to 256.

## 2. Hot-swap — runtime load/unload

vLLM: `POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter`, body `{"lora_name","lora_path","load_inplace"}`. Needs env `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; server logs an explicit "local development only" warning. No replica consistency, no post-restart persistence ([2]). Bug #42125 ([8]): same-name reload can reuse stale prefix-cache blocks. `LoRAResolver` plugins (`lora_filesystem_resolver`, `lora_hf_hub_resolver`) resolve unknown model names on first request.

llama.cpp: `llama-server` accepts repeated `--lora`, `--lora-init-without-apply`, plus `POST /lora-adapters` for runtime scale/switch ([3]). PEFT adapters converted via `convert_lora_to_gguf.py`.

## 3. LoRA + quantization compatibility (vLLM, May 2026)

| Quant | LoRA | Notes |
|---|---|---|
| FP8 W8A8 | Supported (FP8 LoRA dense kernel) | sm89/sm90/sm120 |
| AWQ INT4 | Dense supported; MoE patchy | — |
| GPTQ-Marlin | Recommended INT4+LoRA path ([5]) | Tested in current release |
| NVFP4 | **Not supported with LoRA** ([5]) | No FP4 LoRA kernel yet |
| MXFP4 / MXFP4_MOE | Experimental; LoRA undocumented ([6]) | GPT-OSS path |
| FP8 KV cache | Orthogonal, composes | vserve already auto-pins |

MoE LoRA shipped for GPT-OSS, Qwen3-MoE, DeepSeek, Llama-MoE in 2026 ([7]).

## 4. LoRA + speculative decoding

Historically broken (LoRA applied to draft → kernel/vocab mismatches). PR #11966 ([9]) disables LoRA on the draft, pads vocab on target+draft. Pipeline-parallel + spec decode still disallowed (≤0.15). EAGLE3 (target-feature reuse) is the recommended draft path with LoRA.

## 5. RLHF/DPO fine-tunes — sampling and template

PEFT adapters inherit the base tokenizer; DPO/SFT typically does not change the template ([10]). Tülu-3 DPO ships its own chat template baked into `tokenizer_config.json` (temp 0.7, top_p 0.95). Zephyr-DPO uses Mistral-Instruct. **vserve must load the adapter-shipped tokenizer/template, not the base's** — otherwise generations break silently.

## 6. Multi-LoRA throughput overhead (published)

- S-LoRA (Sheng 2023, [11]): 2,000 adapters at ~constant 7 req/s; vLLM-naive OOMs above a handful.
- Punica (Chen 2023, [12]): SGMV adds +2 ms/token, ≈12× over naive; mixed-batch ≈ same-batch.
- vLLM 2026 production: ~50% max-throughput drop with one LoRA on A100 in poorly-tuned configs ([13]); tuned 10–25%.
- Rank curve: doubling rank ≈ 8–12% extra latency in SGMV regime.

## 7. Adapter naming — PEFT is the de-facto standard

PEFT `adapter_config.json` + `adapter_model.safetensors` is the only widely-adopted layout ([14]). Both vLLM and llama.cpp (after conversion) consume it. vserve should not invent its own.

## 8. llama.cpp adapters

Self-contained GGUF, multi-adapter via repeated `--lora` / `--lora-scaled file:scale`, hot-swap via `POST /lora-adapters`. No SGMV equivalent — weighted-merge, not concurrent batched serving across distinct adapters.

## vserve action items

1. **Flag surface**: adopt vLLM-native `--enable-lora`, `--max-loras` (default 4), `--max-lora-rank` (default 32), `--max-cpu-loras` (default 8). Picker syntax: `--lora a=/path/a,b=/path/b` (legacy form, simplest TUI parse). [1]
2. **Picker UX**: each adapter exposed as its own `served-model-name`; vserve `model` selector lists base + adapters; routing is automatic via OpenAI `model` field. [1]
3. **Hot-swap surface**: expose `vserve lora add NAME PATH` / `vserve lora rm NAME` wrapping `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`. Print upstream's "dev-only" warning and refuse unless `VSERVE_LORA_HOTSWAP=1`. [2][8]
4. **Quant-compat preflight**: block NVFP4 + LoRA with a clear error; warn (don't block) on MXFP4 + LoRA; allow FP8/AWQ/GPTQ-Marlin + LoRA. Recommend GPTQ-Marlin as canonical INT4+LoRA path. [5]
5. **Spec-decode interlock**: if `--enable-lora` and spec-decode are both on, refuse pipeline-parallel and require EAGLE3 draft head. [9]
6. **Tokenizer/template autoload**: when an adapter ships `tokenizer_config.json` / `chat_template.jinja`, load *those* and override the base — fixes DPO/Tülu silent breakage. [10]
7. **Llama.cpp parity**: in the llama.cpp backend, mirror `--lora` flag list and surface `POST /lora-adapters` via the same `vserve lora` subcommand. Document that llama.cpp does NOT do SGMV — concurrent multi-tenant LoRA is vLLM-only. [3]
8. **PEFT-only adapters**: validate `adapter_config.json` exists at the path; refuse non-PEFT layouts. Do not invent a vserve adapter format. [14]

[1]: https://docs.vllm.ai/en/stable/features/lora/
[2]: https://github.com/vllm-project/vllm/issues/6275
[3]: https://github.com/ggml-org/llama.cpp/discussions/10123
[5]: https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/
[6]: https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more
[7]: https://vllm.ai/blog/2026-02-26-multi-lora
[8]: https://github.com/vllm-project/vllm/issues/42125
[9]: https://github.com/vllm-project/vllm/pull/11966
[10]: https://huggingface.co/docs/transformers/main/en/chat_templating
[11]: https://arxiv.org/abs/2311.03285
[12]: https://arxiv.org/abs/2310.18547
[13]: https://github.com/vllm-project/vllm/issues/10062
[14]: https://huggingface.co/docs/peft/developer_guides/checkpoint
