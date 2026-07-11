"""Tests for the spec-decode sweep pure core (vserve.sweep)."""

from __future__ import annotations

from vserve.sweep import (
    SweepPoint,
    enumerate_spec_variants,
    rank_by_concurrency,
    recommend_variant,
)


class TestEnumerateVariants:
    def test_off_always_first(self):
        variants = enumerate_spec_variants(native_mtp_layers=None)
        assert variants[0].key == "off"
        assert variants[0].spec is None

    def test_ngram_vllm_only(self):
        with_ngram = enumerate_spec_variants(native_mtp_layers=None, backend="vllm")
        assert any(v.key == "ngram" for v in with_ngram)
        no_ngram = enumerate_spec_variants(native_mtp_layers=None, backend="llamacpp")
        assert not any(v.key == "ngram" for v in no_ngram)

    def test_mtp_depths_when_native_layers_present(self):
        variants = enumerate_spec_variants(native_mtp_layers=1, mtp_depths=(1, 2, 3))
        keys = [v.key for v in variants]
        assert keys == ["off", "ngram", "mtp-k1", "mtp-k2", "mtp-k3"]
        mtp = next(v for v in variants if v.key == "mtp-k2")
        assert mtp.spec.method == "mtp"
        assert mtp.spec.n_max == 2
        assert mtp.spec.draft_model_path is None

    def test_no_mtp_when_checkpoint_lacks_layers(self):
        variants = enumerate_spec_variants(native_mtp_layers=None)
        assert not any(v.key.startswith("mtp") for v in variants)

    def test_divisibility_filter_on_depths(self):
        # native n_predict=2: depth 3 (3 % 2 != 0) is skipped; 2 and 4 kept.
        variants = enumerate_spec_variants(native_mtp_layers=2, mtp_depths=(2, 3, 4))
        keys = [v.key for v in variants]
        assert "mtp-k2" in keys and "mtp-k4" in keys
        assert "mtp-k3" not in keys


def _pts(*rows):
    return [SweepPoint(*r) for r in rows]


class TestRankByConcurrency:
    def test_delta_vs_baseline_and_winner(self):
        points = _pts(
            ("off", 1, 220.0),
            ("mtp-k1", 1, 135.6),
            ("ngram", 1, 240.0),
        )
        ranked = rank_by_concurrency(points, concurrency=1)
        assert ranked[0].key == "ngram" and ranked[0].is_winner
        ngram = ranked[0]
        assert abs(ngram.delta_pct - (240 - 220) / 220 * 100) < 0.01
        mtp = next(r for r in ranked if r.key == "mtp-k1")
        assert mtp.delta_pct < 0 and not mtp.is_winner

    def test_errors_dropped(self):
        points = _pts(("off", 1, 220.0))
        points.append(SweepPoint("mtp-k3", 1, 0.0, error="OOM"))
        ranked = rank_by_concurrency(points, concurrency=1)
        assert [r.key for r in ranked] == ["off"]

    def test_filters_by_concurrency(self):
        points = _pts(("off", 1, 220.0), ("off", 8, 1100.0))
        ranked = rank_by_concurrency(points, concurrency=8)
        assert len(ranked) == 1 and ranked[0].decode_tps == 1100.0


class TestRecommendVariant:
    def test_recommends_off_when_spec_net_negative(self):
        """The measured A3B reality: MTP high acceptance but net-negative."""
        points = _pts(
            ("off", 1, 220.0), ("off", 8, 1100.0),
            ("mtp-k1", 1, 135.6), ("mtp-k1", 8, 520.0),
            ("mtp-k3", 1, 105.0), ("mtp-k3", 8, 517.0),
        )
        pick, why = recommend_variant(points)
        assert pick == "off"
        assert "below the" in why or "regress" in why

    def test_recommends_spec_when_it_wins_both(self):
        points = _pts(
            ("off", 1, 200.0), ("off", 8, 1000.0),
            ("ngram", 1, 260.0), ("ngram", 8, 1010.0),  # +30% c1, +1% c8
        )
        pick, why = recommend_variant(points)
        assert pick == "ngram"
        assert "+30%" in why

    def test_rejects_spec_that_regresses_high_concurrency(self):
        # Big c1 win but c8 regresses → conservative off.
        points = _pts(
            ("off", 1, 200.0), ("off", 8, 1000.0),
            ("mtp-k1", 1, 280.0), ("mtp-k1", 8, 900.0),  # +40% c1 but -10% c8
        )
        pick, _ = recommend_variant(points)
        assert pick == "off"

    def test_below_bar_stays_off(self):
        points = _pts(
            ("off", 1, 200.0), ("off", 8, 1000.0),
            ("ngram", 1, 205.0), ("ngram", 8, 1005.0),  # only +2.5% c1
        )
        pick, _ = recommend_variant(points, min_gain_pct=5.0)
        assert pick == "off"

    def test_empty_points(self):
        pick, why = recommend_variant([])
        assert pick == "off"
