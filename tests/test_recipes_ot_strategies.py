"""Tests for the graduated MoE expert-CPU-offload picker (item N)."""

from __future__ import annotations


class TestPickOtStrategy:
    """Climb the ladder from no-offload to most-aggressive until budget fits."""

    def test_picks_none_when_model_fits(self):
        from vserve.recipes.ot_strategies import pick_ot_strategy
        # 10 GB model on a 24 GB budget → no offload needed.
        s = pick_ot_strategy(model_vram_gb=10, budget_vram_gb=24)
        assert s.name == "none"
        assert s.patterns == ()

    def test_picks_partial_up_when_slight_overshoot(self):
        from vserve.recipes.ot_strategies import pick_ot_strategy
        # 25 GB model on a 24 GB budget → ~4% over. partial-up frees ~12%.
        s = pick_ot_strategy(model_vram_gb=25, budget_vram_gb=24)
        assert s.name == "partial-up"

    def test_picks_moderate_when_partial_not_enough(self):
        from vserve.recipes.ot_strategies import pick_ot_strategy
        # 32 GB model. partial-up frees 12% → 28.16 GB. moderate frees 24%
        # → 24.32 GB. With default 1-GB safety margin, budget=26 means
        # effective_budget=25; moderate (24.32 GB) fits, partial (28.16) doesn't.
        s = pick_ot_strategy(model_vram_gb=32, budget_vram_gb=26)
        assert s.name == "moderate"

    def test_picks_max_when_moderate_not_enough(self):
        from vserve.recipes.ot_strategies import pick_ot_strategy
        # 40 GB model. moderate (24% off) = 30.4 GB. max (35% off) = 26 GB.
        # budget=28 with default 1-GB margin → effective 27. Only max fits.
        s = pick_ot_strategy(model_vram_gb=40, budget_vram_gb=28)
        assert s.name == "max"

    def test_falls_back_to_layered_when_nothing_fits(self):
        from vserve.recipes.ot_strategies import pick_ot_strategy
        # 100 GB model on 20 GB budget — no strategy fits. Default to layered.
        s = pick_ot_strategy(model_vram_gb=100, budget_vram_gb=20)
        assert s.name == "layered"

    def test_safety_margin_pushes_to_more_aggressive(self):
        from vserve.recipes.ot_strategies import pick_ot_strategy
        # With no margin, model fits at "none"; with 1 GB safety, we should
        # tip into partial-up.
        s_nomargin = pick_ot_strategy(model_vram_gb=23.5, budget_vram_gb=24, safety_margin_gb=0)
        s_margin = pick_ot_strategy(model_vram_gb=23.5, budget_vram_gb=24, safety_margin_gb=1.0)
        assert s_nomargin.name == "none"
        assert s_margin.name == "partial-up"

    def test_patterns_match_unsloth_canonical_forms(self):
        from vserve.recipes.ot_strategies import OT_STRATEGIES
        by_name = {s.name: s for s in OT_STRATEGIES}
        assert by_name["max"].patterns == (".ffn_.*_exps.=CPU",)
        assert by_name["moderate"].patterns == (".ffn_(up|down)_exps.=CPU",)
        assert by_name["partial-up"].patterns == (".ffn_(up)_exps.=CPU",)
