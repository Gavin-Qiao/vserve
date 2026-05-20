"""Empirical llama-bench tuner wrapper (item O).

Drives ``llama-bench`` with a sweep matrix and selects the cell that maximizes
a weighted (prefill, decode) throughput, subject to user-selectable profiles.

Closes the structural gap that motivated 0.8.0's probe-based redesign — the
memory-only math tuner can be wrong about real-world throughput at high -np
because llama.cpp silently sheds tensors to CPU. Measuring removes that gap
for the matrix dimensions we sweep.

Source: R2 §15 (llama-bench docs, JSONL output schema).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class BenchCell:
    """One row of llama-bench output."""

    pp_avg_ts: float
    tg_avg_ts: float
    config: dict[str, Any] = field(default_factory=dict)

    def score(self, prefill_weight: float = 0.3, decode_weight: float = 0.7) -> float:
        return self.pp_avg_ts * prefill_weight + self.tg_avg_ts * decode_weight


@dataclass
class BenchSweep:
    """Container for a multi-row sweep result."""

    cells: list[BenchCell]
    model_path: Path
    build_commit: str | None = None
    gpu_uuid: str | None = None
    duration_s: float = 0.0


# Profile → (prefill weight, decode weight). User-selectable to bias the
# winner toward prefill-heavy (RAG / batch) or decode-heavy (interactive chat).
PROFILE_WEIGHTS: dict[str, tuple[float, float]] = {
    "throughput": (0.5, 0.5),
    "latency":    (0.1, 0.9),
    "balanced":   (0.3, 0.7),
}


def parse_llama_bench_jsonl(text: str) -> list[BenchCell]:
    """Parse the JSONL output of ``llama-bench -o jsonl``.

    Each line is a JSON object with at minimum ``pp_avg_ts`` and ``tg_avg_ts``;
    other fields (n_threads, batch_size, etc.) go into ``cell.config``.
    Lines that can't be parsed are skipped (llama-bench mixes progress lines
    into the stream in some versions).
    """
    cells: list[BenchCell] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        pp = record.get("pp_avg_ts")
        tg = record.get("tg_avg_ts")
        if pp is None or tg is None:
            continue
        try:
            pp_f = float(pp)
            tg_f = float(tg)
        except (TypeError, ValueError):
            continue
        cells.append(BenchCell(
            pp_avg_ts=pp_f,
            tg_avg_ts=tg_f,
            config={k: v for k, v in record.items() if k not in {"pp_avg_ts", "tg_avg_ts"}},
        ))
    return cells


def pick_best_cell(
    cells: Sequence[BenchCell],
    *,
    profile: str = "balanced",
) -> BenchCell | None:
    """Return the cell with the highest weighted (pp, tg) score, or None
    when ``cells`` is empty."""
    if not cells:
        return None
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"unknown profile {profile!r}; valid: {sorted(PROFILE_WEIGHTS)}")
    pp_w, tg_w = PROFILE_WEIGHTS[profile]
    return max(cells, key=lambda c: c.score(prefill_weight=pp_w, decode_weight=tg_w))


def cache_key(*, model_path: Path, gpu_uuid: str | None, build_commit: str | None) -> str:
    """Build a content-addressable cache key for a sweep result.

    Hashes (model SHA prefix, GPU UUID, llama.cpp build commit) so reruns
    against the same hardware/build/quant reuse the cached sweep.
    """
    parts = [
        f"model={model_path.resolve()}",
        f"gpu={gpu_uuid or 'unknown'}",
        f"build={build_commit or 'unknown'}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def run_sweep(
    *,
    entrypoint: Path | str,
    model_path: Path,
    sweep_axes: dict[str, list[int | float | str]],
    timeout_s: int = 3600,
) -> BenchSweep:
    """Run llama-bench with the given sweep matrix and return parsed cells.

    ``sweep_axes`` keys are the bench command-line flags (``-p``, ``-n``,
    ``-b``, ``-ub``, ``-ngl``, ``-fa``, ``-ctk``, ``-ctv``, ``--n-cpu-moe``);
    values are lists of values to sweep. ``llama-bench`` cross-products them.
    """
    cmd: list[str] = [str(entrypoint), "-m", str(model_path), "-o", "jsonl"]
    for axis, values in sweep_axes.items():
        if not values:
            continue
        cmd.extend([axis, ",".join(str(v) for v in values)])
    start = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s,
    )
    duration = time.monotonic() - start
    cells = parse_llama_bench_jsonl((result.stdout or "") + (result.stderr or ""))
    return BenchSweep(
        cells=cells,
        model_path=model_path,
        duration_s=duration,
    )
