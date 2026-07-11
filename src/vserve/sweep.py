"""Spec-decode sweep: the measurement loop as a first-class operation.

``vserve tune <model> --sweep spec`` boots a base profile once per
speculative-decoding variant (off / ngram / MTP at several depths), benches
each at the concurrencies the fleet actually runs, records draft acceptance,
and reports which variant wins — then restores the prior state. It automates
exactly the manual boot→bench→scrape→compare that found in-checkpoint MTP to
be 2.1x net-negative on the A3B MoE (docs/plans/2026-07-10-qwen36-64k-mtp-speed.md).

This module is the pure core (variant enumeration, ranking, table rendering);
the GPU orchestration lives in the CLI so this stays unit-testable without a
running service.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SweepVariant:
    """One speculative-decoding config to measure. ``spec`` is None for the
    off baseline; otherwise a ``recipes.spec_decode.SpecConfig``."""

    key: str
    label: str
    spec: object | None  # SpecConfig | None (kept loose to avoid a hard import)


@dataclass
class SweepPoint:
    """One (variant, concurrency) measurement."""

    variant: str
    concurrency: int
    decode_tps: float
    tpot_ms_p50: float | None = None
    ttft_ms_p50: float | None = None
    acceptance_rate: float | None = None
    accepted_per_step: float | None = None
    error: str | None = None


# Default MTP depths to sweep. k=1 is included deliberately: the depth-vs-sign
# relationship is model-specific (the 2026-05-20 research measured k=1 as the
# worst depth on the dense 27B, but community reports put k=1 net-positive on
# some MoE-NVFP4 checkpoints), so the sweep settles it empirically per model.
DEFAULT_MTP_DEPTHS: tuple[int, ...] = (1, 2, 3)
DEFAULT_CONCURRENCIES: tuple[int, ...] = (1, 8)


def enumerate_spec_variants(
    *,
    native_mtp_layers: int | None,
    backend: str = "vllm",
    mtp_depths: tuple[int, ...] = DEFAULT_MTP_DEPTHS,
    include_ngram: bool = True,
) -> list[SweepVariant]:
    """Build the variant list for a model.

    Always starts with the ``off`` baseline. Adds ngram (vLLM only), then one
    MTP variant per depth when the checkpoint carries in-checkpoint MTP layers,
    filtering depths that violate vLLM's module-reuse divisibility rule.
    """
    from vserve.recipes.spec_decode import SpecConfig

    variants: list[SweepVariant] = [SweepVariant(key="off", label="off (no spec)", spec=None)]

    if include_ngram and backend == "vllm":
        from vserve.recipes.spec_decode import NGRAM_NUM_SPECULATIVE_TOKENS

        variants.append(
            SweepVariant(
                key="ngram",
                label="ngram (prompt-lookup)",
                spec=SpecConfig(method="ngram", n_max=NGRAM_NUM_SPECULATIVE_TOKENS, n_min=1),
            )
        )

    if native_mtp_layers is not None and native_mtp_layers > 0:
        for k in mtp_depths:
            # vLLM reuses the MTP module for depths beyond n_predict and needs
            # divisibility — skip depths that would be rejected at engine init.
            if k > native_mtp_layers and k % native_mtp_layers != 0:
                continue
            variants.append(
                SweepVariant(
                    key=f"mtp-k{k}",
                    label=f"MTP k={k}",
                    spec=SpecConfig(method="mtp", draft_model_path=None, n_max=k),
                )
            )

    return variants


@dataclass
class RankedVariant:
    key: str
    decode_tps: float
    delta_pct: float  # vs the off baseline; 0.0 for baseline, None-safe
    acceptance_rate: float | None
    is_winner: bool = False


def rank_by_concurrency(
    points: list[SweepPoint],
    *,
    concurrency: int,
    baseline_key: str = "off",
) -> list[RankedVariant]:
    """Rank variants by decode throughput at one concurrency, as a delta vs the
    off baseline. Errored points are dropped. Highest tps first."""
    at_c = [p for p in points if p.concurrency == concurrency and p.error is None]
    baseline = next((p.decode_tps for p in at_c if p.variant == baseline_key), None)
    ranked: list[RankedVariant] = []
    for p in at_c:
        if baseline and baseline > 0:
            delta = (p.decode_tps - baseline) / baseline * 100.0
        else:
            delta = 0.0
        ranked.append(
            RankedVariant(
                key=p.variant,
                decode_tps=p.decode_tps,
                delta_pct=delta,
                acceptance_rate=p.acceptance_rate,
            )
        )
    ranked.sort(key=lambda r: r.decode_tps, reverse=True)
    if ranked:
        ranked[0].is_winner = True
    return ranked


def recommend_variant(
    points: list[SweepPoint],
    *,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
    baseline_key: str = "off",
    min_gain_pct: float = 5.0,
) -> tuple[str, str]:
    """Pick the recommended variant and a one-line rationale.

    A spec variant is only recommended if it beats the off baseline by at
    least ``min_gain_pct`` at the LOW concurrency (spec decoding's sweet spot)
    AND does not regress the HIGH concurrency below baseline — otherwise the
    conservative ``off`` wins. This encodes the fleet's hard-won rule that
    acceptance rate alone does not justify spec decoding on a bandwidth-bound
    MoE.
    """
    if not points:
        return baseline_key, "no measurements"
    lo = min(concurrencies)
    hi = max(concurrencies)
    lo_ranked = rank_by_concurrency(points, concurrency=lo, baseline_key=baseline_key)
    hi_ranked = rank_by_concurrency(points, concurrency=hi, baseline_key=baseline_key)
    hi_delta = {r.key: r.delta_pct for r in hi_ranked}

    best = baseline_key
    best_gain = 0.0
    for r in lo_ranked:
        if r.key == baseline_key:
            continue
        regresses_hi = hi_delta.get(r.key, 0.0) < 0.0
        if r.delta_pct >= min_gain_pct and not regresses_hi and r.delta_pct > best_gain:
            best = r.key
            best_gain = r.delta_pct

    if best == baseline_key:
        # Explain why nothing beat off.
        contender = next((r for r in lo_ranked if r.key != baseline_key), None)
        if contender is None:
            return baseline_key, "only the baseline was measured"
        why = (
            f"best spec variant '{contender.key}' gained {contender.delta_pct:+.0f}% at c{lo}"
            f" ({hi_delta.get(contender.key, 0.0):+.0f}% at c{hi})"
            f" — below the +{min_gain_pct:.0f}% bar or regresses high concurrency"
        )
        return baseline_key, why
    return best, f"'{best}' gained +{best_gain:.0f}% at c{lo} without regressing c{hi}"
