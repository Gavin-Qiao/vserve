"""Sanity checks for the architecture-keyed registries.

These tests don't assert per-row business logic (that's covered by the
domain-specific tests). They just verify the tables stay internally
consistent: every entry has the required fields, names are valid
identifiers, etc. Catches typos and partial-row entries that the
domain tests would miss because they only exercise a few rows.
"""

from __future__ import annotations


class TestSamplingDefaultsTable:
    def test_all_entries_have_temperature(self):
        from vserve.recipes.sampling import SAMPLING_DEFAULTS
        for arch, defaults in SAMPLING_DEFAULTS.items():
            assert isinstance(defaults.temp, float), f"{arch}: temp must be float"
            assert 0.0 <= defaults.temp <= 2.0, f"{arch}: temp {defaults.temp} out of range"

    def test_all_entries_are_finite(self):
        from vserve.recipes.sampling import SAMPLING_DEFAULTS
        for arch, defaults in SAMPLING_DEFAULTS.items():
            for field_name in ("top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"):
                value = getattr(defaults, field_name)
                if value is not None:
                    assert value == value, f"{arch}: {field_name} is NaN"

    def test_arch_keys_follow_naming_convention(self):
        from vserve.recipes.sampling import SAMPLING_DEFAULTS
        for arch in SAMPLING_DEFAULTS:
            # vLLM-style architecture names always end with "ForCausalLM" or
            # "ForConditionalGeneration". Catches typos like "Gemma4Causal".
            assert (
                arch.endswith("ForCausalLM")
                or arch.endswith("ForConditionalGeneration")
                or arch.endswith("ForMaskedLM")
            ), f"Unexpected arch name: {arch}"

    def test_gguf_arch_map_resolves_to_valid_defaults(self):
        from vserve.recipes.sampling import (
            SAMPLING_DEFAULTS, _GGUF_ARCH_TO_HF_ARCH,
        )
        for gguf_name, hf_arch in _GGUF_ARCH_TO_HF_ARCH.items():
            assert hf_arch in SAMPLING_DEFAULTS, (
                f"GGUF arch {gguf_name!r} points at unknown HF arch {hf_arch!r}"
            )

    def test_qwen3_thinking_temp_is_load_bearing(self):
        """Unsloth: 'NEVER use greedy on thinking variants'. Verify Qwen3.5
        (thinking) is not at 0.0."""
        from vserve.recipes.sampling import SAMPLING_DEFAULTS
        assert SAMPLING_DEFAULTS["Qwen35ForCausalLM"].temp > 0.0

    def test_kimi_k2_thinking_vs_instruct_temps_diverge(self):
        """Kimi K2 instruct=0.6, thinking=1.0 per Unsloth docs."""
        from vserve.recipes.sampling import SAMPLING_DEFAULTS
        instruct = SAMPLING_DEFAULTS["KimiK2ForCausalLM"]
        thinking = SAMPLING_DEFAULTS["KimiK2ThinkingForCausalLM"]
        assert instruct.temp != thinking.temp


class TestToolParserTable:
    def test_no_empty_parser_names(self):
        from vserve.backends.vllm import _ARCH_TO_TOOL_PARSER
        for arch, parser in _ARCH_TO_TOOL_PARSER.items():
            assert isinstance(parser, str) and parser, f"{arch}: empty parser name"

    def test_gemma_family_uses_gemma4_parser(self):
        from vserve.backends.vllm import _ARCH_TO_TOOL_PARSER
        for arch in (
            "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration",
            "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
        ):
            assert _ARCH_TO_TOOL_PARSER[arch] == "gemma4"

    def test_deepseek_per_version_parsers(self):
        from vserve.backends.vllm import _ARCH_TO_TOOL_PARSER
        # Each DeepSeek version uses a version-specific parser.
        assert _ARCH_TO_TOOL_PARSER["DeepseekV3ForCausalLM"] == "deepseek_v3"
        assert _ARCH_TO_TOOL_PARSER["DeepseekV31ForCausalLM"] == "deepseek_v31"
        assert _ARCH_TO_TOOL_PARSER["DeepseekV32ForCausalLM"] == "deepseek_v32"
        assert _ARCH_TO_TOOL_PARSER["DeepseekV4ForCausalLM"] == "deepseek_v4"

    def test_table_size_grew_to_at_least_25(self):
        """0.6.0 had 4 entries; 0.6.1 expanded to 25+. Regression-guard
        against accidental shrinkage."""
        from vserve.backends.vllm import _ARCH_TO_TOOL_PARSER
        assert len(_ARCH_TO_TOOL_PARSER) >= 25


class TestReasoningParserTable:
    def test_no_empty_parser_names(self):
        from vserve.backends.vllm import _ARCH_TO_REASONING_PARSER
        for arch, parser in _ARCH_TO_REASONING_PARSER.items():
            assert isinstance(parser, str) and parser, f"{arch}: empty parser name"

    def test_qwen3_family_uses_qwen3_parser(self):
        from vserve.backends.vllm import _ARCH_TO_REASONING_PARSER
        for arch in (
            "Qwen3ForCausalLM", "Qwen35ForCausalLM", "Qwen36ForCausalLM",
            "Qwen3MoeForCausalLM", "Qwen36MoeForCausalLM", "Qwen3A3BForCausalLM",
        ):
            assert _ARCH_TO_REASONING_PARSER[arch] == "qwen3"

    def test_qwen36moe_routes_to_qwen3_reasoning_parser(self):
        """0.6.3 bug fix 1: Qwen36MoeForCausalLM was missing from the table;
        sibling registries (sampling, spec_decode, tool-parser) all include it,
        so its absence here caused <think> to leak into message.content for
        Qwen 3.6 MoE."""
        from vserve.backends.vllm import _ARCH_TO_REASONING_PARSER
        assert _ARCH_TO_REASONING_PARSER["Qwen36MoeForCausalLM"] == "qwen3"

    def test_deepseek_family_uses_deepseek_r1_parser(self):
        from vserve.backends.vllm import _ARCH_TO_REASONING_PARSER
        for arch in (
            "DeepseekV3ForCausalLM", "DeepseekV31ForCausalLM",
            "DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM",
        ):
            assert _ARCH_TO_REASONING_PARSER[arch] == "deepseek_r1"


class TestArchForcesBackend:
    def test_mla_archs_default_to_flashmla(self):
        from vserve.backends.vllm import _ARCH_FORCES_BACKEND
        for arch in (
            "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
            "DeepseekV4ForCausalLM", "KimiK2ForCausalLM",
            "LongcatFlashForCausalLM",
        ):
            assert _ARCH_FORCES_BACKEND[arch] == "FLASHMLA"

    def test_gemma_family_forces_triton_attn(self):
        from vserve.backends.vllm import _ARCH_FORCES_BACKEND
        assert _ARCH_FORCES_BACKEND["Gemma4ForCausalLM"] == "TRITON_ATTN"
        assert _ARCH_FORCES_BACKEND["Gemma4ForConditionalGeneration"] == "TRITON_ATTN"

    def test_blackwell_upgrades_mla_to_tokenspeed(self, tmp_path):
        from vserve.backends.vllm import _forced_attention_backend
        import json

        def _w(parent, arch):
            d = parent / arch
            d.mkdir(parents=True)
            (d / "config.json").write_text(json.dumps({"architectures": [arch]}))
            return d

        # Both sm100 (DC Blackwell) and sm120 (RTX Blackwell) get TOKENSPEED_MLA.
        for sm in (100, 120):
            d = _w(tmp_path / f"sm{sm}", "DeepseekV3ForCausalLM")
            assert _forced_attention_backend(d, sm) == "TOKENSPEED_MLA"
        # Hopper stays on FLASHMLA.
        d_h = _w(tmp_path / "sm90", "DeepseekV3ForCausalLM")
        assert _forced_attention_backend(d_h, 90) == "FLASHMLA"


class TestBackendIncompatKvDtypesTable:
    def test_triton_attn_rejects_turboquant_family(self):
        from vserve.backends.vllm import BACKEND_INCOMPATIBLE_KV_DTYPES
        triton = BACKEND_INCOMPATIBLE_KV_DTYPES["TRITON_ATTN"]
        assert "turboquant_k8v4" in triton
        assert "turboquant_3bit_nc" in triton

    def test_tokenspeed_mla_is_permissive(self):
        from vserve.backends.vllm import BACKEND_INCOMPATIBLE_KV_DTYPES
        # Blackwell variant supports more dtypes; expect empty or near-empty.
        assert len(BACKEND_INCOMPATIBLE_KV_DTYPES["TOKENSPEED_MLA"]) == 0


class TestQuantFlagsTable:
    def test_includes_0_6_1_additions(self):
        from vserve.models import QUANT_FLAGS
        for key in ("nvfp4", "modelopt", "mxfp4", "mxfp4_moe", "bitsandbytes", "gguf"):
            assert key in QUANT_FLAGS

    def test_legacy_entries_still_present(self):
        from vserve.models import QUANT_FLAGS
        for key in ("gptq", "awq", "fp8", "compressed-tensors"):
            assert key in QUANT_FLAGS


class TestQuantEnvVarsTable:
    def test_nvfp4_routes_flashinfer_envs(self):
        from vserve.models import QUANT_ENV_VARS
        envs = QUANT_ENV_VARS["nvfp4"]
        assert envs["VLLM_USE_FLASHINFER_MOE_FP4"] == "1"
        assert envs["VLLM_FLASHINFER_MOE_BACKEND"] == "throughput"

    def test_modelopt_mirrors_nvfp4(self):
        from vserve.models import QUANT_ENV_VARS
        assert QUANT_ENV_VARS["modelopt"] == QUANT_ENV_VARS["nvfp4"]

    def test_mxfp4_explicit_empty_not_missing(self):
        from vserve.models import QUANT_ENV_VARS
        # 0 envs is meaningful — Humming backend default; reserve the entry.
        assert "mxfp4" in QUANT_ENV_VARS


class TestUnslothQuantTiersTable:
    def test_all_tiers_have_bits_per_weight(self):
        from vserve.models import UNSLOTH_QUANT_TIERS
        for tier, props in UNSLOTH_QUANT_TIERS.items():
            assert "bits_per_weight" in props, f"{tier}: missing bits_per_weight"
            assert 1.0 <= props["bits_per_weight"] <= 17.0

    def test_tiers_in_regex(self):
        """Every tier in the docs table is parseable by the regex."""
        from vserve.models import UNSLOTH_QUANT_TIERS, parse_unsloth_quant_tier
        for tier in UNSLOTH_QUANT_TIERS:
            filename = f"model-UD-{tier}.gguf"
            assert parse_unsloth_quant_tier(filename) == tier, (
                f"Regex doesn't match docs-table tier {tier}"
            )

    def test_mxfp4_moe_flagged_moe_only(self):
        from vserve.models import UNSLOTH_QUANT_TIERS
        assert UNSLOTH_QUANT_TIERS["MXFP4_MOE"].get("moe_only") is True


class TestCanonicalQwen35ArchCoverage:
    """Real Qwen3.5 AND Qwen3.6 checkpoints both report arch
    `Qwen3_5MoeForConditionalGeneration` (model_type qwen3_5_moe). It must be
    present in every behavior-driving arch-keyed registry — the synthetic
    Qwen35/Qwen36* short-names only match the GGUF path. A missing canonical
    entry is the exact bug class that bit the tool parser (b94a823) and the 3.6
    reasoning parser (0.6.3 bug fix 1).

    NOTE: SAMPLING_DEFAULTS has no direct key for this arch because Qwen3.5 and
    Qwen3.6 share it but want different samplers (temp 0.6/pp 1.0 vs 1.0/pp 1.5);
    get_sampling_defaults() disambiguates by model name instead (covered in
    test_recipes_sampling).
    """

    ARCH = "Qwen3_5MoeForConditionalGeneration"

    def test_present_in_tool_parser_table(self):
        from vserve.backends.vllm import _ARCH_TO_TOOL_PARSER
        assert _ARCH_TO_TOOL_PARSER[self.ARCH] == "qwen3_coder"

    def test_present_in_reasoning_parser_table(self):
        from vserve.backends.vllm import _ARCH_TO_REASONING_PARSER
        assert _ARCH_TO_REASONING_PARSER[self.ARCH] == "qwen3"

    def test_present_in_family_table(self):
        from vserve.arch_registry import family_of
        assert family_of(self.ARCH) == "qwen3"

    def test_is_thinking_default(self):
        from vserve.arch_registry import is_thinking_default
        assert is_thinking_default(self.ARCH) is True

    def test_present_in_spec_decode_table(self):
        from vserve.recipes.spec_decode import SPEC_METHOD_BY_ARCH
        assert SPEC_METHOD_BY_ARCH[self.ARCH] == "mtp"
