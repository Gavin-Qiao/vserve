"""Tests for the per-architecture sampling-defaults recipe registry."""

from __future__ import annotations


class TestSamplingDefaults:
    """0.7.0 item A: emit recipe-correct sampler params per architecture so
    Gemma-4 doesn't run with llama.cpp's stock temp 0.8 (degraded output)
    and Qwen3-Thinking doesn't enter loop traps under low-temp / greedy."""

    def test_gemma4_uses_unsloth_recipe(self):
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("Gemma4ForCausalLM")
        assert d is not None
        assert d.temp == 1.0
        assert d.top_p == 0.95
        assert d.top_k == 64
        assert d.min_p == 0.01

    def test_gemma3_inherits_gemma4_sampler(self):
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("Gemma3ForCausalLM")
        assert d is not None
        assert d.temp == 1.0
        assert d.top_k == 64

    def test_qwen3_non_thinking(self):
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("Qwen3ForCausalLM")
        assert d is not None
        assert d.temp == 0.7
        assert d.top_p == 0.8
        assert d.top_k == 20

    def test_qwen35_thinking_has_presence_penalty(self):
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("Qwen35ForCausalLM")
        assert d is not None
        assert d.temp == 0.6
        assert d.presence_penalty == 1.0

    def test_canonical_qwen35_arch_defaults_to_qwen35_sampler(self):
        """Real Qwen3.5/3.6 report Qwen3_5MoeForConditionalGeneration. Without a
        name to disambiguate, default to the Qwen3.5 sampler (the arch is 3_5)."""
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("Qwen3_5MoeForConditionalGeneration")
        assert d is not None
        assert d.temp == 0.6
        assert d.presence_penalty == 1.0

    def test_canonical_qwen35_arch_disambiguates_qwen36_by_name(self):
        """Qwen3.5 and Qwen3.6 share the arch; the model name picks the sampler.
        Guards against the digits in '397B'/'A17B'/'35B'/'A3B' being mistaken
        for the '3.6' version token."""
        from vserve.recipes.sampling import get_sampling_defaults
        arch = "Qwen3_5MoeForConditionalGeneration"
        d36 = get_sampling_defaults(arch, "Qwen3.6-35B-A3B")
        assert d36 is not None
        assert d36.temp == 1.0
        assert d36.presence_penalty == 1.5
        d35 = get_sampling_defaults(arch, "Qwen3.5-397B-A17B")
        assert d35 is not None
        assert d35.temp == 0.6
        assert d35.presence_penalty == 1.0

    def test_qwen3_coder_repeat_penalty(self):
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("Qwen3CoderForCausalLM")
        assert d is not None
        assert d.repeat_penalty == 1.05

    def test_deepseek_v31(self):
        from vserve.recipes.sampling import get_sampling_defaults
        d = get_sampling_defaults("DeepseekV31ForCausalLM")
        assert d is not None
        assert d.temp == 0.6
        assert d.min_p == 0.01

    def test_kimi_k2_instruct_vs_thinking(self):
        from vserve.recipes.sampling import get_sampling_defaults
        instruct = get_sampling_defaults("KimiK2ForCausalLM")
        thinking = get_sampling_defaults("KimiK2ThinkingForCausalLM")
        assert instruct is not None and thinking is not None
        # Unsloth: instruct=0.6 (deterministic), thinking=1.0 (high-creativity)
        assert instruct.temp == 0.6
        assert thinking.temp == 1.0

    def test_unknown_architecture_returns_none(self):
        from vserve.recipes.sampling import get_sampling_defaults
        assert get_sampling_defaults("SomeFutureArchForCausalLM") is None

    def test_empty_or_none_architecture_returns_none(self):
        from vserve.recipes.sampling import get_sampling_defaults
        assert get_sampling_defaults(None) is None
        assert get_sampling_defaults("") is None


class TestGgufArchMapping:
    """llama.cpp emits lowercase GGUF arch names — verify the map routes
    them to the same SamplingDefaults rows the HF mapping uses."""

    def test_gguf_gemma4_maps(self):
        from vserve.recipes.sampling import get_sampling_defaults_from_gguf_arch
        d = get_sampling_defaults_from_gguf_arch("gemma4")
        assert d is not None
        assert d.temp == 1.0

    def test_gguf_qwen3_maps(self):
        from vserve.recipes.sampling import get_sampling_defaults_from_gguf_arch
        d = get_sampling_defaults_from_gguf_arch("qwen3")
        assert d is not None
        assert d.temp == 0.7

    def test_gguf_uppercase_input_ok(self):
        from vserve.recipes.sampling import get_sampling_defaults_from_gguf_arch
        d = get_sampling_defaults_from_gguf_arch("GEMMA4")
        assert d is not None

    def test_gguf_unknown_returns_none(self):
        from vserve.recipes.sampling import get_sampling_defaults_from_gguf_arch
        assert get_sampling_defaults_from_gguf_arch("brandnew_arch") is None


class TestRenderRecipeSummary:
    """Pretty-print sampler params for the run banner."""

    def test_includes_temp_and_optional_fields(self):
        from vserve.recipes.sampling import SamplingDefaults, render_recipe_summary
        s = render_recipe_summary(SamplingDefaults(1.0, 0.95, 64, 0.01))
        assert "temp=1.0" in s
        assert "top_p=0.95" in s
        assert "top_k=64" in s
        assert "min_p=0.01" in s

    def test_skips_unset_fields(self):
        from vserve.recipes.sampling import SamplingDefaults, render_recipe_summary
        s = render_recipe_summary(SamplingDefaults(0.6, min_p=0.01))
        assert "temp=0.6" in s
        assert "min_p=0.01" in s
        assert "top_p" not in s
