"""Graduated MoE expert-CPU offload strategies for llama.cpp (item N).

Today vserve auto-applies the most aggressive pattern (all FFN experts to
CPU). When VRAM headroom allows it, keeping more experts on GPU yields
materially better throughput. This module exposes a tiered set of -ot
patterns and a picker that climbs the ladder until the estimated VRAM fits.

Source: R3 (Unsloth canonical hierarchy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OtStrategy:
    name: str
    patterns: tuple[str, ...]
    description: str


# Order matters — picker walks from least-CPU-offload to most. ``layered``
# is the surgical layer-range escape (highest layers only on CPU).
OT_STRATEGIES: tuple[OtStrategy, ...] = (
    OtStrategy(
        name="none",
        patterns=(),
        description="No expert offload — full model on GPU.",
    ),
    OtStrategy(
        name="partial-up",
        patterns=(".ffn_(up)_exps.=CPU",),
        description="Up-projection experts on CPU; gate + down on GPU.",
    ),
    OtStrategy(
        name="moderate",
        patterns=(".ffn_(up|down)_exps.=CPU",),
        description="Up + down experts on CPU; gate on GPU.",
    ),
    OtStrategy(
        name="max",
        patterns=(".ffn_.*_exps.=CPU",),
        description="All expert FFNs on CPU (most aggressive).",
    ),
    OtStrategy(
        name="layered",
        patterns=(r"\.([0-9]|[1-9][0-9])\.ffn_(gate|up|down)_exps.=CPU",),
        description="Surgical: experts on layers 6+ to CPU (DeepSeek-V3 671B etc.).",
    ),
)

# Rough VRAM fraction freed by each strategy. Calibrated against the gemma-4
# A4B at 22 slots × 128k: max ≈ 35-40% of model weight, moderate ≈ 24%,
# partial-up ≈ 12%. Treat as approximations; pair with a safety margin.
_STRATEGY_FREE_FRACTION: dict[str, float] = {
    "none":        0.0,
    "partial-up":  0.12,
    "moderate":    0.24,
    "max":         0.35,
    "layered":     0.30,
}


def pick_ot_strategy(
    *,
    model_vram_gb: float,
    budget_vram_gb: float,
    safety_margin_gb: float = 1.0,
    candidates: Sequence[OtStrategy] | None = None,
) -> OtStrategy:
    """Return the least-aggressive OT strategy that fits the VRAM budget.

    ``model_vram_gb``: total model weight in GB (pre-offload).
    ``budget_vram_gb``: VRAM available for weights (after KV / activations).
    ``safety_margin_gb``: subtract from the budget so we don't bind tight.

    Walks the ladder (none → partial-up → moderate → max → layered) and
    returns the first strategy whose estimated post-offload weight fits.
    """
    effective_budget = max(0.0, budget_vram_gb - safety_margin_gb)
    pool = candidates if candidates is not None else OT_STRATEGIES
    for strategy in pool:
        freed = _STRATEGY_FREE_FRACTION.get(strategy.name, 0.0)
        post_offload = model_vram_gb * (1.0 - freed)
        if post_offload <= effective_budget:
            return strategy
    # Nothing fit — fall back to layered (last item).
    return pool[-1]
