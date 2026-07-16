# Qwen3.6-35B-A3B-NVFP4 — optimal 8×32k serving (vLLM 0.25.0, sm120)

Exact launch config for the production profile `qwen32k` (RTX PRO 5000 Blackwell,
sm120). Every engine-level knob below is set **at server start** — none of it is
reachable over the OpenAI HTTP API, so this file is the source of truth.

- **Runtime:** vLLM **0.25.0** (flashinfer 0.6.13, torch 2.11.0). MoE kernel =
  **MARLIN** (auto-selected NVFP4 MoE on sm120; `flashinfer_b12x` measured at
  parity — do not switch). KV = **fp8 e4m3** (nvfp4 KV still crashes on sm120).
- **Measured:** c1 235 / c8 1156 tok/s (short prompt); true 8×29k ≈ 130 tok/s.
- **Host-RAM safety:** `MAX_JOBS=4` + `NVCC_THREADS=1` cap the first-boot
  FlashInfer/nvcc JIT (uncapped it OOM-kills at ~50 GB). Keep a mem limit of
  ~50 GB and a persistent cache so the JIT compiles **once**.

---

## 1. Bare-metal `vllm serve` (what the systemd unit runs, expanded)

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_HOME=/usr/local/cuda \
TMPDIR=/opt/vllm/tmp VLLM_RPC_BASE_PATH=/opt/vllm/tmp \
MAX_JOBS=4 NVCC_THREADS=1 \
/opt/vllm/venv/bin/vllm serve /opt/vllm/models/nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 qwen3.6-35b-a3b-nvfp4 \
  --host 0.0.0.0 --port 8888 \
  --dtype bfloat16 \
  --quantization modelopt \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5}'
```

The managed unit is equivalent — it runs `vllm serve --config
/opt/vllm/configs/active.yaml` with `EnvironmentFile=/opt/vllm/configs/.env`,
under `MemoryHigh=48G / MemoryMax=50G / MemorySwapMax=0 / OOMPolicy=stop`.

### Flag notes (the HTTP-invisible knobs)
| flag | value | note |
|---|---|---|
| `--enable-prefix-caching` | on | reuses shared preambles; huge TTFT win on multi-turn |
| `--kv-cache-dtype` | `fp8` | halves KV bandwidth; nvfp4 KV crashes on sm120 |
| `--max-num-seqs` | `8` | the concurrency ceiling (8×32k) |
| `--max-num-batched-tokens` | `8192` | chunked-prefill token budget |
| chunked prefill | **on (default)** | implied by V1 engine + the token budget; no flag needed (`--enable-chunked-prefill` to be explicit) |
| `--reasoning-parser` | `qwen3` | separates `<think>` reasoning from the answer |
| MoE backend | **auto → MARLIN** | omit for auto; pin with `--moe-backend marlin` if you want it explicit. **Never** `flashinfer_b12x` (parity/slower on sm120) |
| MTP / spec decode | **OFF** | net-negative on this MoE. To enable anyway: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| `--language-model-only` | on | skips the vision encoder → frees VRAM for KV |

---

## 2. `docker run` (official image)

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -p 8888:8888 \
  -v /opt/vllm/models:/models:ro \
  -v vllm-cache:/root/.cache \
  -e MAX_JOBS=4 -e NVCC_THREADS=1 \
  --memory=50g --memory-swap=50g \
  vllm/vllm-openai:v0.25.0 \
  /models/nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 qwen3.6-35b-a3b-nvfp4 \
  --host 0.0.0.0 --port 8888 \
  --dtype bfloat16 --quantization modelopt \
  --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 \
  --enable-prefix-caching --language-model-only \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5}'
```

- `--ipc=host` (or `--shm-size=16g`): vLLM needs large shared memory.
- `-v vllm-cache:/root/.cache`: persist the FlashInfer/torch.compile JIT cache so
  the compile happens **once** — otherwise every fresh container re-storms.
- `--memory=50g --memory-swap=50g`: mirrors the systemd cgroup guard (swap off).
- The image entrypoint is `vllm serve`, so the model path is the **first
  positional arg**. (If your image pins the older `api_server` entrypoint, pass
  `--model /models/...` instead.)
- To let vLLM download from HF instead of mounting: drop the `/models` mount, use
  the tag `nvidia/Qwen3.6-35B-A3B-NVFP4`, and mount a HF cache
  (`-v hf-cache:/root/.cache/huggingface`, `-e HF_TOKEN=...`).

---

## 3. Docker Compose

```yaml
services:
  qwen32k:
    image: vllm/vllm-openai:v0.25.0
    command:
      - /models/nvidia/Qwen3.6-35B-A3B-NVFP4
      - --served-model-name
      - nvidia/Qwen3.6-35B-A3B-NVFP4
      - qwen3.6-35b-a3b-nvfp4
      - --host=0.0.0.0
      - --port=8888
      - --dtype=bfloat16
      - --quantization=modelopt
      - --max-model-len=32768
      - --max-num-seqs=8
      - --max-num-batched-tokens=8192
      - --gpu-memory-utilization=0.9
      - --kv-cache-dtype=fp8
      - --enable-prefix-caching
      - --language-model-only
      - --reasoning-parser=qwen3
      - '--override-generation-config={"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5}'
    ports:
      - "8888:8888"
    volumes:
      - /opt/vllm/models:/models:ro
      - vllm-cache:/root/.cache          # persist JIT cache (compile once)
    environment:
      MAX_JOBS: "4"                       # host-RAM JIT-storm cap
      NVCC_THREADS: "1"
    ipc: host                             # large shared memory for vLLM
    mem_limit: 50g                        # mirror the cgroup guard
    memswap_limit: 50g                    # => swap off (MemorySwapMax=0)
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

volumes:
  vllm-cache:
```

`docker compose up -d` → OpenAI API on `:8888`. First boot JIT-compiles (~6 min,
capped ~48 GB); the `vllm-cache` volume makes subsequent boots ~1 min.

---

## Smoke test

```bash
curl -s localhost:8888/health && echo OK
curl -s localhost:8888/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.6-35B-A3B-NVFP4","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
# -> " Paris, a city renowned for its rich"
```
