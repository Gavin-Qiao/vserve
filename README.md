<div align="center">

# vserve

**A CLI for managing LLM inference on GPU workstations.**

Download models. Auto-tune limits. Serve with one command. Multiple backends.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.20–0.24](https://img.shields.io/badge/vLLM-0.20%E2%80%930.24-ff6f00?style=flat-square)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-purple?style=flat-square)
![Tests](https://img.shields.io/badge/tests-1005%20passed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

</div>

---

## Release: 0.6.3

`vserve` is now **stable** — first non-beta release of the 0.6 line.

Highlights in `0.6.3`:

- vLLM 0.22 support: range `>=0.20,<0.23` (pinned stable runtime stays `0.21.0` — see caveats)
- migrates the deprecated `VLLM_USE_FLASHINFER_MOE_*` env vars to vLLM 0.22's hardware-aware backend auto-selection, with a `--moe-backend` escape hatch
- emits `default-chat-template-kwargs` on 0.22 (the flag was renamed upstream); thinking toggles keep working across 0.20–0.22
- spec-decode with fp8 KV cache keeps CUDA graphs on 0.22 (DFlash fp8 fix verified upstream); TurboQuant stays conservatively un-graphed
- fixes Qwen 3.5 / 3.6 tool calling — these emit the XML tool format and now route to the `qwen3_coder` parser instead of `hermes` (which silently dropped tool calls into message content)
- the 0.6.1–0.6.3b3 line: research bundle, `vserve bench`, de-spaghetti refactor, arch-registry canonicalization — GPU-verified on 0.21 and re-verified on 0.22

Known caveats:

- **Gemma-4 NVFP4 on vLLM 0.22**: startup OOMs from multimodal video-encoder profiling (vLLM #43169 batched encoder; the V1 profiler allocates max-size dummy videos). Serve it on 0.21, or pass `--limit-mm-per-prompt '{"image":1,"video":0}'` plus `--mm-processor-kwargs '{"max_soft_tokens":560}'`. Automatic multimodal caps land in 0.6.4. This is why the pinned stable runtime stays `0.21.0`.
- vLLM 0.20.0–0.22.0 compute RMSNorm weights in FP32 upstream (vllm#42325, fixed after the 0.22.0 cut) — accuracy-sensitive users should adopt the next vLLM patch
- non-interactive startup remains intentionally strict: if the backend never reaches a healthy API state within the timeout window, `run` exits nonzero even if the service is still warming
- multi-user coordination is best-effort operational safety, not a security boundary

---

## Install

Install from PyPI:

```bash
uv tool install vserve
pip install vserve
```

For llama.cpp GGUF tuning support:

```bash
pip install 'vserve[llamacpp]'
```

---

## Quick Start

```bash
vserve init                        # scan GPU, backends, CUDA, systemd — write config
vserve runtime check vllm          # verify the external vLLM runtime
vserve add                         # search HuggingFace, pick variant, download
vserve run <model>                 # auto-tune + interactive config + serve
vserve run <model> --tools         # enable tool calling (auto-detected)
vserve run <model> --backend llamacpp  # force a specific backend
```

Scriptable serving:

```bash
vserve run qwen fp8 --yes --context 32768 --slots 4 --kv-cache-dtype fp8 --port 8888
vserve run qwen fp8 --yes --replace              # safe non-interactive restart
vserve run qwen fp8 --save-profile fast --yes
vserve run qwen fp8 --profile fast
vserve run --profile /opt/vllm/configs/models/provider--Model.fast.yaml --yes
```

Runtime repair and GGUF-only setup:

```bash
vserve runtime check vllm
vserve runtime upgrade vllm --stable
vserve add TheBloke/some-model-GGUF
vserve run some model q4 --backend llamacpp --yes --gpu-layers 999
```

Automation:

```bash
vserve run qwen fp8 --profile fast --yes
vserve status
vserve stop
```

---

## Backends

vserve auto-detects the right backend from the model format:

| Format | Backend | Engine |
|:-------|:--------|:-------|
| safetensors, GPTQ, AWQ, FP8 | **vLLM** | PagedAttention, continuous batching |
| GGUF | **llama.cpp** | CPU/GPU offload, quantized inference |

No configuration needed — download a model and `vserve run` picks the right engine.

### vLLM

The default for transformer models in safetensors format. Optimized for high-throughput serving with PagedAttention, KV cache management, and automatic batching.

- Auto-tunes `--max-model-len`, `--max-num-seqs`, `--kv-cache-dtype` based on your GPU
- Calculates PagedAttention block-rounded capacity for native, FP8, and TurboQuant KV-cache dtypes
- Recommends scheduler profiles with chunked-prefill-oriented token budgets and vLLM 0.20 optimization knobs
- Tool calling with parser auto-detection (Qwen, Llama, Mistral, DeepSeek, Gemma, GPT-OSS)
- Systemd service management via `vllm.service`

### llama.cpp

For GGUF quantized models. Serves via `llama-server` with an OpenAI-compatible API.

- Auto-calculates `--n-gpu-layers`, `--ctx-size`, `--parallel` based on VRAM
- Reads GGUF metadata without the optional `gguf` package and accounts for layerwise KV heads, sliding-window attention, and recurrent state
- Partial GPU offload — serve models that don't fully fit in VRAM
- Tool calling via `--jinja` (no parser configuration needed)
- Systemd service management via `llama-cpp.service`

---

## What It Does

**vserve** manages the full lifecycle of serving LLMs on a GPU workstation:

- **Download** — search HuggingFace, see available weight variants (FP8, NVFP4, BF16, GGUF) with sizes, download only one backend format at a time, and materialize each runnable variant into its own model root
- **Auto-tune** — calculate exactly what context lengths and concurrency your GPU can handle, based on model architecture and available VRAM
- **Benchmark** — opt into bounded backend microbenchmarks with `vserve tune --bench`
- **Tool calling** — auto-detects the correct parser from the model's chat template (vLLM) or uses `--jinja` (llama.cpp)
- **Run/Stop** — interactive config wizard, systemd service management, health check with timeout
- **Fan control** — temperature-based curve daemon with quiet hours, or hold a fixed speed
- **Multi-user** — best-effort session coordination warns other `vserve` users before they disrupt your running model
- **Doctor** — diagnose GPU, CUDA, backend, systemd issues with actionable fix suggestions

---

## Commands

| Command | Description |
|:--------|:------------|
| `vserve` | Dashboard — GPU, models, status |
| `vserve init` | Auto-discover backends and write config |
| `vserve list [name]` | List models with backend, tools, and limits |
| `vserve add [model]` | Search and download from HuggingFace with variant picker |
| `vserve rm <name>` | Remove a downloaded model |
| `vserve tune [model]` | Calculate context/concurrency limits |
| `vserve tune [model] --bench` | Run bounded benchmarks for tuned vLLM or llama.cpp profiles |
| `vserve run [model]` | Configure and start serving (auto-tunes if needed) |
| `vserve run MODEL... --yes --context N --slots N` | Non-interactive serving from flags |
| `vserve run MODEL... --yes --replace` | Non-interactive restart; without `--replace`, running backends are refused |
| `vserve run MODEL... --profile NAME_OR_PATH` | Serve a saved profile by name or explicit path |
| `vserve run MODEL... --tools --tool-parser hermes --reasoning-parser qwen3` | Start with explicit parsers |
| `vserve run MODEL... --trust-remote-code` | Opt in to vLLM remote model code execution |
| `vserve run MODEL... --backend llamacpp --gpu-layers 999` | Force llama.cpp for GGUF |
| `vserve profile list\|show\|rm` | Manage saved serving profiles |
| `vserve stop` | Stop the running server |
| `vserve status [--json]` | Show current serving config and probe uncertainty |
| `vserve fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vserve doctor [--json] [--strict]` | Check system readiness; strict exits nonzero on failures |
| `vserve cache clean [--dry-run] [--all] [--yes]` | Preview or clean stale sockets and JIT caches |
| `vserve runtime check vllm` | Check vLLM version/dependency compatibility |
| `vserve runtime upgrade vllm --stable` | Reinstall vserve's pinned stable vLLM runtime |
| `vserve version` | Show current version and check for updates |
| `vserve update [--nightly]` | Update vserve, optionally allowing pre-releases |

Model-taking commands support **fuzzy matching** — `vserve run qwen fp8` finds the right model.

Profile rules: names saved with `--save-profile` must match `[A-Za-z0-9._-]+` and cannot be `.`, `..`, or include path separators. Profile names resolve inside configured vserve profile roots. Explicit external `--profile` paths are accepted only by `run` and infer backend from YAML/JSON when possible. `profile show` and `profile rm` never read or delete arbitrary external paths, even with `--force`.

Automation note: `run --yes` is fully non-interactive. If it needs to stop or start systemd services it uses non-prompting service operations; configure passwordless service control for the vserve operator or run without `--yes`.

---

## Tool Calling

### vLLM

Auto-detects the correct vLLM parser by reading the model's chat template:

| Model Family | Tool Parser | Reasoning Parser |
|:-------------|:------------|:-----------------|
| Qwen 2.5 | `hermes` | — |
| Qwen 3 | `hermes` | `qwen3` |
| Qwen 3.5 | `qwen3_coder` | `qwen3` |
| Llama 3.1 / 3.2 / 3.3 | `llama3_json` | — |
| Llama 4 | `llama4_pythonic` | — |
| Mistral / Mixtral | `mistral` | `mistral` |
| DeepSeek V3 / R1 | `deepseek_v3` | `deepseek_r1` |
| Gemma 4 | `gemma4` | `gemma4` |
| GPT-OSS | `openai` | `openai_gptoss` |

Detection is template-based (not model-name regex), so it works for fine-tunes and community uploads.

Remote model code is disabled by default. Use `--trust-remote-code` only for repositories you trust; generated profiles include `trust-remote-code` only when that flag is explicitly set.

### llama.cpp

Uses `--jinja` to read the model's chat template directly. No parser selection needed — one flag covers all model families.

---

## Prerequisites

| Requirement | Check | Install |
|:------------|:------|:--------|
| NVIDIA GPU + drivers | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/drivers) |
| CUDA toolkit | `nvcc --version` | `sudo apt install nvidia-cuda-toolkit` |
| systemd | (most Linux servers) | See [troubleshooting](docs/troubleshooting.md) |
| sudo access | for systemctl, fan control | |

**For vLLM backend:**

| Requirement | Check | Install |
|:------------|:------|:--------|
| stable vLLM 0.20.x–0.24.x | `vserve runtime check vllm` | `vserve runtime upgrade vllm --stable` or [docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) |

**For llama.cpp backend:**

| Requirement | Check | Install |
|:------------|:------|:--------|
| llama-server | `llama-server --version` | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |

---

## Configuration

Auto-discovered on first run. Override at `~/.config/vserve/config.yaml`:

```yaml
schema_version: 2
cuda_home: /usr/local/cuda
gpu:
  index: 0
  memory_utilization: 0.91
backends:
  vllm:
    root: /opt/vllm
    service_name: vllm
    service_user: vllm
    port: 8888
  llamacpp:
    root: /opt/llama-cpp
    service_name: llama-cpp
    service_user: llama-cpp
```

Legacy top-level `vllm_root`, `service_name`, `llamacpp_root`, and GPU memory keys still load, but newly saved config uses the backend-indexed schema above.

`gpu.index` is part of runtime truth, not only a tuning hint. vserve records it in active manifests and tuning fingerprints. llama.cpp launch scripts export `CUDA_VISIBLE_DEVICES=<index>`. vLLM writes `configs/.env` with the same value and `doctor` expects the systemd unit to load that environment file.

---

## Directory Layout

```
/opt/vllm/                     # vLLM backend
├── venv/bin/vllm              # Python venv
├── .venv/bin/vllm             # alternate Python venv location
├── models/                    # safetensors models
├── configs/
│   ├── .env                   # service environment
│   ├── active.yaml            # active profile symlink
│   └── models/                # limits + YAML profiles
├── tmp/                       # RPC sockets / runtime temp files
├── .cache/
│   ├── flashinfer/            # FlashInfer JIT cache
│   ├── torch_extensions/      # torch extension cache
│   └── vllm/                  # vLLM/torch.compile cache
├── run/
│   └── active-manifest.json   # active backend state
└── logs/

/opt/llama-cpp/                # llama.cpp backend
├── bin/llama-server           # compiled binary
├── models/                    # GGUF models
├── configs/
│   ├── active.sh              # active launch script symlink
│   ├── active.json            # active config symlink
│   └── models/                # JSON profiles
├── run/
│   └── active-manifest.json   # active backend state
└── logs/
```

GGUF downloads create one runnable model root per selected quant/subdirectory, so `Q4_K_M` and `Q8_0` variants do not share a directory. Source roots left only for materialization are ignored by model scanning.

---

## Fan Control

```bash
vserve fan              # show status, interactive menu
vserve fan auto         # temp-based curve with quiet hours
vserve fan 80           # hold at 80% (persistent daemon)
vserve fan off          # stop daemon, restore NVIDIA auto
```

The auto curve ramps with temperature and caps fan speed during quiet hours (configurable). Emergency override at 88C ignores quiet hours.

---

## Architecture

vserve uses a **Backend Protocol** pattern. Each inference engine implements the same interface:

```
Backend Protocol
├── VllmBackend        — safetensors, AWQ, FP8, GPTQ
├── LlamaCppBackend    — GGUF
└── (future: SGLang, etc.)
```

The registry auto-detects the right backend from the model format. Runtime checks, tuning fingerprints, profile/config generation, service lifecycle, active manifests, and status summaries live behind the backend protocol so the command layer can stay focused on user workflows.

---

## Development

```bash
git clone https://github.com/Gavin-Qiao/vserve.git
cd vserve
uv sync --dev
./scripts/install-hooks.sh        # wire up .githooks/ pre-commit + pre-push
uv run pytest tests/              # full suite
uv run ruff check src/ tests/     # lint
uv run mypy src/vserve/ --ignore-missing-imports --check-untyped-defs
```

The `install-hooks.sh` step points `core.hooksPath` at the tracked
`.githooks/` directory. Once set, every `git commit` runs the same
gates CI runs (ruff + mypy + pytest under `CI=true GITHUB_ACTIONS=true`
and `COLUMNS=80`, so that anything green locally is green in CI), and
every `git push` re-runs those plus a version-sync check and an
`uv build` dry-run. Bypass with `--no-verify` only when you've already
verified CI separately.

---

## License

[MIT](LICENSE)
