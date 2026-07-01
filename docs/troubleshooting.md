# Troubleshooting

Hard-won lessons from running vLLM on NVIDIA workstation GPUs.

## Supported vLLM Runtime

vserve currently supports stable vLLM `>=0.20,<0.25`. Release candidates, dev builds, and older minor versions should be replaced unless you intentionally bypass the guard for local testing.

```bash
vserve runtime check vllm
vserve runtime upgrade vllm --stable
```

`vserve runtime check vllm` reports the external vLLM version plus torch, torch CUDA, Transformers, Hugging Face Hub, and `pip check` results. Tuning caches include these runtime facts, so changing vLLM or torch causes vserve to recalculate limits instead of reusing stale capacity numbers.

`vserve runtime upgrade vllm --stable` requires a configured vLLM virtualenv, a stopped backend service, and installs vserve's pinned runtime (`vllm==0.24.0`). The full range `>=0.20,<0.25` is accepted by `runtime check`. 0.24 serves block-diffusion (dLLM) models like DiffusionGemma natively on the stable runtime — no separate service needed. On 0.22+, serving Gemma-4 multimodal can OOM during vision/video profiling — use `--language-model-only` for text-only (first-class in 0.23) or pass manual `--limit-mm-per-prompt`/`--mm-processor-kwargs` caps until vserve auto-emits them (0.6.4). It refuses to mutate the environment if backend state is active or uncertain.

For beta/pre-release vserve builds:

```bash
uv tool install --prerelease allow vserve
pip install --pre vserve
vserve update --nightly
```

## GPU Crashes (Xid Errors)

### Xid 8 — GPU Stopped Processing

**Symptom:** vLLM dies after hours of stable inference. Kernel log shows:
```
NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!
NVRM: Xid (PCI:0000:02:00): 8, pid=..., name=VLLM::EngineCor
```

**Root cause:** The GPU's recovery counter watchdog detected a hang. On Blackwell (SM120), this correlates with CUDA graphs under sustained FP8 load (see [vllm-project/vllm#35659](https://github.com/vllm-project/vllm/issues/35659)).

**Recovery:**
1. The GPU usually recovers automatically after the faulting process exits — no reset needed.
2. `nvidia-smi --gpu-reset` is deprecated as of driver 570+.
3. If the GPU is unresponsive, reload kernel modules: `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia`
4. Last resort: reboot.

**Prevention:**
- Use the production driver branch (595.x), not the new-feature branch (590.x).
- `vserve fan auto` keeps temps below throttle threshold.
- systemd `Restart=always` with exponential backoff recovers automatically.
- If crashes persist, try `--enforce-eager` (disables CUDA graphs, ~2.3x throughput cost).

### vLLM exits cleanly but systemd doesn't restart

**Cause:** vLLM's APIServer shuts down cleanly (exit 0) after an EngineCore crash. `Restart=on-failure` ignores exit 0.

**Fix:** Use `Restart=always` in the systemd unit.

## Fan Control

### Why override NVIDIA's auto curve?

The default auto curve on workstation GPUs (RTX PRO series) caps fan speed conservatively for noise. Under sustained 300W inference, this can allow temps to reach 85-90°C, causing:
- Thermal throttling (reduced inference throughput)
- Accelerated component aging (electromigration, capacitor degradation)
- Increased risk of Xid errors

### Coolbits setup

Fan control via `nvidia-settings` requires Coolbits enabled in X11 config:

```bash
# /etc/X11/xorg.conf
Section "Device"
    Identifier     "GPU0"
    Driver         "nvidia"
    BusID          "PCI:2:0:0"    # check with: nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader
    Option         "Coolbits" "4"  # bit 2 = manual fan control
EndSection
```

On headless systems, `vserve fan` uses a temporary Xvfb virtual display — no persistent X server needed.

### Hardware thermal failsafe

Even if the fan daemon crashes and the fan is stuck at 30%, the GPU has hardware protection:
- GPU Boost reduces clocks starting at ~83°C
- Aggressive power limiting at ~90°C
- Hardware shutdown at ~100-105°C (cannot be overridden by software)

The `vserve fan` emergency override (100% fan at 88°C regardless of quiet hours) is software-level defense. The hardware failsafe is always present underneath.

## JIT Compilation

### FlashInfer JIT cache

vLLM 0.18+ uses FlashInfer with just-in-time compiled CUDA kernels. First run for each model/KV-dtype combination triggers JIT compilation that takes 2-5 minutes.

**Where the cache lives:** `$VLLM_ROOT/.cache/flashinfer/`

**Problem:** If vLLM runs as a different user (e.g., systemd `User=vllm`), the JIT cache must be writable by that user. First-time startup will be slow.

**The pre-cache option:** after an interactive `vserve tune`, vserve may offer to start the configured `vllm.service` briefly, wait for the health endpoint, and stop it again. This builds service-user JIT artifacts through the same systemd path used for real serving. Non-interactive `vserve tune MODEL` does not pre-cache; the first `vserve run` may still spend several minutes compiling.

Preview or clear caches safely:

```bash
vserve cache clean --dry-run --all
vserve stop
vserve cache clean --all --yes
```

`cache clean` refuses to mutate caches while any backend is running or when backend state cannot be confirmed.

`vserve cache clean --all` deletes stale sockets under `$VLLM_ROOT/tmp` plus `$VLLM_ROOT/.cache/flashinfer`, `$VLLM_ROOT/.cache/torch_extensions`, and `$VLLM_ROOT/.cache/vllm`. It does not delete downloaded models, profile YAML/JSON, limits caches, logs, `.env`, or active manifests.

For automation, `vserve run --yes` never prompts and refuses to replace a running backend unless you pass `--replace`.

When `--yes` needs systemd lifecycle changes, vserve uses non-prompting service calls. If sudo would ask for a password, automation fails fast; either configure a narrow passwordless rule for the backend service operations or run interactively.

### torch.compile cache

vLLM also uses `torch.compile` which has its own cache at `$VLLM_ROOT/.cache/vllm/torch_compile_cache/`. Same ownership considerations apply.

### CUDA_HOME and PATH

The JIT compiler needs:
- `nvcc` accessible (CUDA toolkit bin directory in PATH)
- `CUDA_HOME` set to the CUDA toolkit root
- Matching CUDA version between the toolkit and the PyTorch build

If JIT compilation fails with `nvcc not found` or architecture mismatches, check:
```bash
which nvcc
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### ProtectSystem=strict breaks JIT

Do NOT use `ProtectSystem=strict` in the vLLM systemd unit — it makes `/usr` read-only, preventing nvcc from writing temporary files during JIT compilation. The `gpu-fan.service` can use it (fan control doesn't need JIT), but the main `vllm.service` cannot.

## Configured GPU Index

Set `gpu.index` in `~/.config/vserve/config.yaml` when the serving GPU is not device 0. vserve uses that index for GPU probing, tuning fingerprints, active manifests, fan helpers, and launch-time CUDA visibility.

For llama.cpp, generated `active.sh` exports `CUDA_VISIBLE_DEVICES=<index>`. For vLLM, vserve writes `$VLLM_ROOT/configs/.env` with the same value. The vLLM systemd unit must load that file with `EnvironmentFile=.../configs/.env`; otherwise `vserve doctor` reports a failure and startup refuses to pretend the configured GPU is enforced.

## Driver Management

### Which driver branch to use

| Branch | Status | Use |
|--------|--------|-----|
| 570.x | Old production | Pre-Blackwell only |
| 575.x | Early Blackwell | Known issues, avoid |
| 590.x | New-feature branch | Experimental, not for production |
| 595.x | **Current production** | Recommended for Blackwell |

Check your branch: `nvidia-smi` shows the driver version in the header.

### Open vs. proprietary kernel modules

Blackwell GPUs (RTX 50-series, RTX PRO 5000/6000) **require open kernel modules**. The proprietary modules do not support Blackwell at all. Always use packages with `-open` suffix (e.g., `nvidia-driver-595-server-open` or `nvidia-open` from the CUDA repo).

### nvidia-settings version mismatch

If `nvidia-settings --version` shows a different version than `nvidia-smi`, you have a partial driver upgrade. This can cause subtle issues. Fix by upgrading the full driver stack to match.

## Headless Operation

### Disabling the display manager

For 24/7 inference, disable the display manager to free GPU memory and simplify driver reloads:

```bash
sudo systemctl disable gdm    # or lightdm/sddm
sudo systemctl set-default multi-user.target
```

GUI is still available on demand: `sudo systemctl start gdm`

### GPU memory with no display

With no display manager: ~2 MiB GPU memory used (driver overhead only).
With Xorg/GDM running: ~15 MiB (Xorg frame buffer).

This is negligible for 48 GB, but matters for driver reload — fewer processes holding the GPU open means cleaner `rmmod`.
