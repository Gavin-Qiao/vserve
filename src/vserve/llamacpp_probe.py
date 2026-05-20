"""Build-info probes for the installed llama-server / llama-bench binary.

Surfaces build number, commit hash, CUDA runtime version, and whether the
binary was compiled with ``GGML_CUDA_FA_ALL_QUANTS=ON``. Used by:

- item I: gate the ``iq4_nl`` KV-cache dtype candidate behind the build flag
- item L: warn at config time on known-bad combos (UD-IQ4_XS < b8808; CUDA 13.2)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Minimum llama.cpp build number that produces correct output for an
# Unsloth UD-IQ4_XS / UD-IQ4_NL GGUF. Earlier builds output gibberish
# (per Unsloth maintainer + llamacpp#21423).
MIN_BUILD_PER_TIER: dict[str, int] = {
    "IQ4_XS": 8808,
    "IQ4_NL": 8808,
}


@dataclass(frozen=True)
class LlamaCppBuildInfo:
    """Parsed output of ``llama-server --version``.

    ``build_number`` is the integer the binary advertises (e.g. ``9222``).
    The numbering varies by upstream tag — cross-check with commit SHA
    before firing per-tier gates so we don't false-warn.
    """

    build_number: int | None
    commit: str | None
    cuda_version: str | None
    fa_all_quants: bool
    raw_version_text: str = ""


_BUILD_NUMBER_RE = re.compile(r"version[^0-9]*(?P<n>\d+)\s*\(([0-9a-f]{7,})\)")
_BUILD_NUMBER_FALLBACK_RE = re.compile(r"build[\s:=]*(?P<n>\d+)")
_COMMIT_RE = re.compile(r"\(([0-9a-f]{7,40})\)")
_CUDA_RE = re.compile(r"CUDA[^\d]*(\d+\.\d+)")


def probe_llama_cpp_build(entrypoint: Path | str) -> LlamaCppBuildInfo | None:
    """Run ``<entrypoint> --version`` and parse build metadata.

    Returns None on timeout / subprocess failure. Callers should treat None
    as "build info unavailable — silent fallthrough", not as a hard error.
    """
    try:
        result = subprocess.run(
            [str(entrypoint), "--version"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    text = (result.stdout or "") + (result.stderr or "")
    if not text.strip():
        return None
    return _parse_version_text(text)


def _parse_version_text(text: str) -> LlamaCppBuildInfo:
    """Extract build info from llama-server --version output.

    Public for testability — pass arbitrary version strings to verify
    parsing without spawning a subprocess.
    """
    build_number: int | None = None
    commit: str | None = None
    m = _BUILD_NUMBER_RE.search(text)
    if m:
        build_number = int(m.group("n"))
        commit = m.group(2)
    else:
        # Some builds emit only `build: 583` without a parenthesized commit.
        m2 = _BUILD_NUMBER_FALLBACK_RE.search(text)
        if m2:
            build_number = int(m2.group("n"))
        cm = _COMMIT_RE.search(text)
        if cm:
            commit = cm.group(1)
    cuda_match = _CUDA_RE.search(text)
    cuda_version = cuda_match.group(1) if cuda_match else None
    fa_all_quants = "GGML_CUDA_FA_ALL_QUANTS" in text or "FA_ALL_QUANTS" in text
    return LlamaCppBuildInfo(
        build_number=build_number,
        commit=commit,
        cuda_version=cuda_version,
        fa_all_quants=fa_all_quants,
        raw_version_text=text,
    )


def check_build_compat(info: LlamaCppBuildInfo, quant_tier: str | None) -> list[str]:
    """Produce config-time warnings about known-bad llama.cpp build combos.

    Sources: llamacpp#21371 (CUDA 13.2 gibberish), #21423 (b8661 tokenizer
    break), #21655 (b8680 Apple regression), Unsloth Dynamic 2.0 docs.

    Returns a list of human-readable warning strings (possibly empty).
    Callers print them in the vserve run startup banner.
    """
    warnings: list[str] = []
    if info.cuda_version == "13.2":
        warnings.append(
            "CUDA 13.2 + GGUF produces gibberish output per Unsloth + "
            "llamacpp#21371. Switch to CUDA 12.8 or 13.0."
        )
    if quant_tier and quant_tier in MIN_BUILD_PER_TIER and info.build_number:
        min_build = MIN_BUILD_PER_TIER[quant_tier]
        if info.build_number < min_build:
            warnings.append(
                f"llama.cpp build b{info.build_number} produces gibberish with "
                f"UD-{quant_tier} (fixed in b{min_build}, llamacpp#21423). "
                f"Either upgrade llama.cpp or switch to UD-Q4_K_XL."
            )
    if info.build_number == 8661:
        # llamacpp#21423 Windows tokenizer regression — affects every quant.
        warnings.append(
            "llama.cpp b8661 has a Windows tokenizer regression "
            "(llamacpp#21423). Avoid this exact build."
        )
    return warnings
