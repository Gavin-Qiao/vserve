"""Engine-failure log scanning.

Reads recent engine log output (vLLM stdout/stderr or llama.cpp's systemd
journal) and matches known failure signatures to actionable suggestions.

Each pattern reflects a failure mode actually observed in the 2026-05-19
debugging session that drove 0.5.x / 0.6.0 tuner-correctness fixes — the
goal is to surface "do X" instead of leaving the user to grep journalctl.

Extracted from `cli.py` in 0.6.3 per audit
`docs/audits/2026-05-20-cli-sprawl.md` (cli.py at 6,136 lines had domain
logic mixed with CLI plumbing).
"""

from __future__ import annotations

import subprocess


def diagnose_engine_failure(log_text: str, backend_name: str) -> list[tuple[str, str]]:
    """Scan engine logs for known failure signatures.

    Returns a list of ``(cause, suggestion)`` pairs the CLI can print.
    Empty list means "no recognized pattern" — caller should fall back to
    the generic "Check: sudo journalctl ..." hint.
    """
    if not log_text:
        return []
    findings: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(cause: str, suggestion: str) -> None:
        if cause in seen:
            return
        seen.add(cause)
        findings.append((cause, suggestion))

    # vLLM-side patterns
    if backend_name == "vllm":
        if "Selected backend" in log_text and "is not valid for this configuration" in log_text \
                and "kv_cache_dtype not supported" in log_text:
            _add(
                "vLLM forced an attention backend that does not accept the requested kv-cache-dtype.",
                "Re-run with --kv-cache-dtype fp8 (or auto). Architectures like Gemma-4 force "
                "TRITON_ATTN which rejects every turboquant_* dtype.",
            )
        if "Workspace is locked but allocation" in log_text and "turboquant" in log_text:
            _add(
                "TurboQuant decode kernel needed more workspace than CUDA-graph capture sized.",
                "Add `compilation-config: {cudagraph_mode: NONE}` to the YAML "
                "(keeps torch.compile fusions, only skips graph capture — the "
                "community-verified fix per vllm#42808/43357). Alternative: drop "
                "--kv-cache-dtype to fp8. --enforce-eager also works but unnecessarily "
                "disables torch.compile (vserve auto-emits cudagraph_mode: NONE for "
                "turboquant_* dtypes starting in 0.6.1).",
            )
        if "Chunked MM input disabled" in log_text and "max_tokens_per_mm_item" in log_text:
            _add(
                "Multimodal model: one image/audio item is larger than max-num-batched-tokens.",
                "Set --batched-tokens 4096 (or higher). vserve 0.6.0+ does this automatically "
                "for any model with vision_config / audio_config.",
            )
        if "auto" in log_text and "tool choice requires --enable-auto-tool-choice" in log_text:
            _add(
                "Tool calling was requested but vLLM is not configured with a tool-call parser.",
                "Pass --tools and re-run; vserve 0.6.0+ auto-maps known architectures (gemma4, "
                "qwen3_coder, etc.) to a parser. Older configs need both "
                "enable-auto-tool-choice and tool-call-parser in the YAML.",
            )
        if "CUDA out of memory" in log_text or "torch.cuda.OutOfMemoryError" in log_text:
            _add(
                "Engine ran out of GPU memory while allocating buffers.",
                "Reduce --slots, --context, or use a smaller-bytes KV dtype (fp8). "
                "Re-run `vserve tune <model> --recalc` to refresh the limits cache.",
            )

    # llama.cpp patterns (failure surface is in journal, not vllm.log)
    if backend_name == "llamacpp":
        if "cudaMalloc failed: out of memory" in log_text \
                or "failed to allocate buffer for kv cache" in log_text:
            _add(
                "llama.cpp ran out of GPU memory allocating the KV cache.",
                "Reduce --slots or --context. The product (slots × context) drives KV size. "
                "vserve 0.6.0+ reserves more compute headroom in tune to prevent this.",
            )
        if "common_fit_params: failed to fit params to free device memory" in log_text \
                and "n_gpu_layers already set by user" in log_text:
            _add(
                "llama.cpp's auto-fitter wanted to spill layers to CPU but --n-gpu-layers was "
                "explicit; engine aborts.",
                "Either reduce --n-gpu-layers below the model's total, OR enable MoE expert "
                "CPU offload via --override-tensor '.ffn_.*_exps.=CPU' (vserve auto-applies "
                "this in 0.6.0+ only when needed).",
            )
        if "core-dump" in log_text or "SIGSEGV" in log_text or "status=11/SEGV" in log_text:
            _add(
                "llama-server segfaulted (typically a model/runtime mismatch or KV alloc failure).",
                "Check `journalctl -u llama-cpp.service -n 200` for the preceding error. "
                "Common cause is requesting more KV than fits; try smaller --slots or --context.",
            )

    return findings


def fetch_engine_log(backend, *, max_lines: int = 400) -> str:
    """Read recent engine log output for diagnosis.

    For vLLM the engine writes stdout/stderr to ``<logs_dir>/vllm.log`` via the
    systemd unit's ``StandardOutput=append:`` redirect (journal only contains
    systemd state-change lines). For llama.cpp logs land in the journal.

    Silent fallthrough on any error — returns empty string so callers can
    proceed to the generic "journalctl ..." hint.
    """
    from vserve.config import cfg

    if backend.name == "vllm":
        log_path = cfg().logs_dir / "vllm.log"
        try:
            # `sudo tail` reads the vllm-owned file; readable for the
            # vserve-running user since the directory permissions allow it
            # on the supported install. Fall back to journalctl if not.
            result = subprocess.run(
                ["sudo", "-n", "tail", "-n", str(max_lines), str(log_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass

    try:
        result = subprocess.run(
            [
                "journalctl", "-u", backend.service_name,
                "--no-pager", "-n", str(max_lines), "-o", "cat",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout or ""
    except Exception:
        pass
    return ""


def print_engine_diagnosis(backend, console, *, header: str | None = None) -> bool:
    """Run diagnosis on the recent engine log and print findings.

    Returns True if at least one finding was surfaced (caller can suppress
    the generic journalctl hint when we already gave a precise answer).
    """
    log_text = fetch_engine_log(backend)
    findings = diagnose_engine_failure(log_text, backend.name)
    if not findings:
        return False
    if header:
        console.print(header)
    for cause, suggestion in findings:
        console.print(f"  [bold]Cause:[/bold] {cause}")
        console.print(f"  [bold]Try:[/bold]   {suggestion}\n")
    return True
