"""Tests for ``vserve.arch_registry``.

0.6.3 consolidated arch-keyed tables that had drifted across multiple
modules (see audit ``docs/audits/2026-05-20-registries-coherence.md``).
This test suite enforces:

- ``family_of()`` doesn't suffer the 5-char prefix-slice collision that
  the 0.6.3 audit caught (bug 2 — Gemma3 vs Gemma4).
- Every architecture in the tool-parser registry has a family entry.
- Thinking-default archs all have a reasoning parser.
- Re-exports from ``backends/vllm.py`` and ``recipes/sampling.py`` still
  resolve to the canonical objects in ``arch_registry``.
"""

from __future__ import annotations


class TestFamilyOf:
    def test_gemma3_and_gemma4_have_distinct_families(self):
        """0.6.3 bug fix 2: ``arch[:5]`` collided Gemma3 and Gemma4 because
        they share a 5-letter prefix. ``family_of()`` correctly
        distinguishes them."""
        from vserve.arch_registry import family_of
        assert family_of("Gemma3ForCausalLM") == "gemma3"
        assert family_of("Gemma4ForCausalLM") == "gemma4"
        assert family_of("Gemma3ForCausalLM") != family_of("Gemma4ForCausalLM")

    def test_qwen3_variants_share_family(self):
        from vserve.arch_registry import family_of
        for arch in (
            "Qwen3ForCausalLM", "Qwen35ForCausalLM", "Qwen36ForCausalLM",
            "Qwen3MoeForCausalLM", "Qwen36MoeForCausalLM", "Qwen3A3BForCausalLM",
        ):
            assert family_of(arch) == "qwen3", f"{arch}: expected qwen3"

    def test_qwen3_coder_distinct_from_base_qwen3(self):
        from vserve.arch_registry import family_of
        # Coder has extra code tokens — different tokenizer family.
        assert family_of("Qwen3CoderForCausalLM") != family_of("Qwen3ForCausalLM")

    def test_deepseek_v2_v3_share_family(self):
        from vserve.arch_registry import family_of
        for arch in (
            "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
            "DeepseekV31ForCausalLM", "DeepseekV32ForCausalLM",
        ):
            assert family_of(arch) == "deepseek_v3"

    def test_deepseek_v4_distinct_from_v3(self):
        from vserve.arch_registry import family_of
        assert family_of("DeepseekV4ForCausalLM") != family_of("DeepseekV3ForCausalLM")

    def test_unknown_arch_returns_none(self):
        from vserve.arch_registry import family_of
        assert family_of("NeverHeardOfThisModel") is None
        assert family_of("") is None
        assert family_of(None) is None

    def test_llama3_and_llama4_distinct_families(self):
        from vserve.arch_registry import family_of
        assert family_of("LlamaForCausalLM") == "llama3"
        assert family_of("Llama4ForCausalLM") == "llama4"
        assert family_of("Llama4MoeForCausalLM") == "llama4"
        assert family_of("LlamaForCausalLM") != family_of("Llama4ForCausalLM")


class TestIsThinkingDefault:
    def test_qwen35_is_thinking_default(self):
        from vserve.arch_registry import is_thinking_default
        assert is_thinking_default("Qwen35ForCausalLM") is True

    def test_qwen36moe_is_thinking_default(self):
        """0.6.3 bug fix 1: this arch was missing from the reasoning-parser
        table; thinking-default registry must also include it."""
        from vserve.arch_registry import is_thinking_default
        assert is_thinking_default("Qwen36MoeForCausalLM") is True

    def test_base_qwen3_is_not_thinking_default(self):
        from vserve.arch_registry import is_thinking_default
        assert is_thinking_default("Qwen3ForCausalLM") is False

    def test_gemma4_is_not_thinking_default(self):
        """Gemma 4 has thinking via chat-template-kwargs, not always-on."""
        from vserve.arch_registry import is_thinking_default
        assert is_thinking_default("Gemma4ForCausalLM") is False

    def test_unknown_arch_is_not_thinking_default(self):
        from vserve.arch_registry import is_thinking_default
        assert is_thinking_default("UnknownForCausalLM") is False
        assert is_thinking_default(None) is False
        assert is_thinking_default("") is False


class TestRegistryCoverageMatrix:
    """Every arch in the tool-parser table should also have a family entry
    so that spec-decode and other family-routed logic works for it."""

    def test_every_tool_parser_arch_has_a_family(self):
        from vserve.arch_registry import _ARCH_TO_TOOL_PARSER, family_of
        unmapped: list[str] = []
        for arch in _ARCH_TO_TOOL_PARSER:
            if family_of(arch) is None:
                unmapped.append(arch)
        assert not unmapped, f"Archs missing family entries: {unmapped}"

    def test_every_reasoning_parser_arch_has_a_family(self):
        from vserve.arch_registry import _ARCH_TO_REASONING_PARSER, family_of
        unmapped: list[str] = []
        for arch in _ARCH_TO_REASONING_PARSER:
            if family_of(arch) is None:
                unmapped.append(arch)
        assert not unmapped, f"Archs missing family entries: {unmapped}"

    def test_every_forces_backend_arch_has_a_family(self):
        from vserve.arch_registry import _ARCH_FORCES_BACKEND, family_of
        unmapped: list[str] = []
        for arch in _ARCH_FORCES_BACKEND:
            if family_of(arch) is None:
                unmapped.append(arch)
        assert not unmapped, f"Archs missing family entries: {unmapped}"

    def test_every_thinking_default_arch_has_reasoning_parser(self):
        """Any arch marked thinking-default MUST have a reasoning-parser
        entry or its <think> trace will leak into message.content (the
        symptom of 0.6.3 bug fix 1)."""
        from vserve.arch_registry import (
            _ARCH_TO_REASONING_PARSER, _THINKING_DEFAULT_ARCHS,
        )
        missing = [
            arch for arch in _THINKING_DEFAULT_ARCHS
            if arch not in _ARCH_TO_REASONING_PARSER
        ]
        assert not missing, (
            f"Thinking-default archs missing reasoning parser (would leak "
            f"<think> into message.content): {missing}"
        )

    def test_every_gguf_short_name_maps_to_known_family(self):
        """If a GGUF short name maps to an HF arch, that HF arch should
        have a family entry."""
        from vserve.arch_registry import _GGUF_ARCH_TO_HF_ARCH, family_of
        broken: list[tuple[str, str]] = []
        for gguf, hf_arch in _GGUF_ARCH_TO_HF_ARCH.items():
            if family_of(hf_arch) is None:
                broken.append((gguf, hf_arch))
        assert not broken, (
            f"GGUF short names mapping to HF archs without families: {broken}"
        )


class TestReExportsStillResolve:
    """``backends/vllm.py`` and ``recipes/sampling.py`` re-export the
    arch_registry tables. Ensure the re-export point still resolves to the
    canonical dict object (same identity, not a copy)."""

    def test_vllm_re_exports_tool_parser_table(self):
        from vserve.arch_registry import _ARCH_TO_TOOL_PARSER as canonical
        from vserve.backends.vllm import _ARCH_TO_TOOL_PARSER as reexport
        assert reexport is canonical

    def test_vllm_re_exports_reasoning_parser_table(self):
        from vserve.arch_registry import _ARCH_TO_REASONING_PARSER as canonical
        from vserve.backends.vllm import _ARCH_TO_REASONING_PARSER as reexport
        assert reexport is canonical

    def test_vllm_re_exports_forces_backend_table(self):
        from vserve.arch_registry import _ARCH_FORCES_BACKEND as canonical
        from vserve.backends.vllm import _ARCH_FORCES_BACKEND as reexport
        assert reexport is canonical

    def test_sampling_re_exports_gguf_arch_map(self):
        from vserve.arch_registry import _GGUF_ARCH_TO_HF_ARCH as canonical
        from vserve.recipes.sampling import _GGUF_ARCH_TO_HF_ARCH as reexport
        assert reexport is canonical


class TestSpecDecodeUsesFamilyOf:
    """The bug 2 fix: ``vocab_compatible()`` must correctly distinguish
    Gemma3 and Gemma4 targets/drafts so it doesn't pick a wrong-vocab
    drafter (which would crash with KV mismatch at the first speculation
    step)."""

    def test_gemma3_target_rejects_gemma4_drafter(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="Gemma4ForCausalLM", size_b=4.0,
            bos_token_id=2, eos_token_id=1,
        )
        # Same BOS/EOS but different family → reject.
        assert vocab_compatible("Gemma3ForCausalLM", 2, 1, draft) is False

    def test_gemma4_target_rejects_gemma3_drafter(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="Gemma3ForCausalLM", size_b=4.0,
            bos_token_id=2, eos_token_id=1,
        )
        assert vocab_compatible("Gemma4ForCausalLM", 2, 1, draft) is False

    def test_qwen3_target_accepts_qwen35_drafter(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="Qwen35ForCausalLM", size_b=4.0,
            bos_token_id=2, eos_token_id=1,
        )
        assert vocab_compatible("Qwen3ForCausalLM", 2, 1, draft) is True

    def test_unknown_target_arch_refuses_spec_decode(self, tmp_path):
        """Conservative: unknown target arch → no family → refuse."""
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="Qwen3ForCausalLM", size_b=4.0,
            bos_token_id=2, eos_token_id=1,
        )
        assert vocab_compatible("NeverHeardOfThis", 2, 1, draft) is False
