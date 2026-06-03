"""Cross-check arch_registry keys against the vLLM 0.22 ModelRegistry.

This is the test that would have caught the 0.6.3 defect: tests passed
because both code-under-test and tests used the same fictional arch
names (``Qwen36MoeForCausalLM`` etc.), but **no model on disk produces
those names** — vLLM 0.21 registers ``Qwen3_5MoeForConditionalGeneration``
instead. The on-GPU sweep at 0.6.3b2 ship-time surfaced the gap.

The fixture (``tests/fixtures/vllm_archs.json``) is captured from
``vllm.model_executor.models.ModelRegistry.get_supported_archs()`` on the
workstation install. Refresh it when bumping the vLLM requirement.

We don't require *every* registry key to be in vLLM — some are
forward-compat aspirational entries for archs that vLLM doesn't ship yet
(KimiK2, Llama4Moe, GLM 4.7, etc.). The allowlist below tracks those
intentional misses. Adding a new key without adding it to either the
fixture or the allowlist will fail this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "vllm_archs.json"

# Archs that are intentionally in arch_registry but NOT in the vLLM 0.22
# ModelRegistry. These are either forward-compat (upstream support
# pending) or aspirational. Audit any addition here. (Re-verified at the
# 0.22.0 fixture refresh: all 16 entries are still absent upstream.)
_ALLOWLIST_NOT_IN_VLLM: frozenset[str] = frozenset({
    # Forward-compat — vLLM doesn't ship support for these yet
    "DeepseekV31ForCausalLM",
    "Ernie4ForCausalLM",
    "Glm47MoeForCausalLM",
    "Granite4ForCausalLM",
    "KimiK2ForCausalLM",
    "KimiK2ThinkingForCausalLM",
    "Lfm25ForCausalLM",
    "Llama4MoeForCausalLM",
    "MistralThinkingForCausalLM",
    "XlamForCausalLM",
    # Qwen3 fictional variants — predate the audit; kept for backward
    # compat with old limits caches. Real canonical names
    # (Qwen3_5*ForConditionalGeneration) are in the registry too.
    # TODO(0.6.4): retire once cache schema is bumped.
    "Qwen35ForCausalLM",
    "Qwen36ForCausalLM",
    "Qwen36MoeForCausalLM",
    "Qwen3A3BForCausalLM",
    "Qwen3CoderForCausalLM",
    "Qwen3XmlForCausalLM",
})


def _load_vllm_archs() -> set[str]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture missing: {FIXTURE_PATH}")
    with FIXTURE_PATH.open() as f:
        data = json.load(f)
    return set(data["archs"])


def _all_registry_keys() -> set[str]:
    from vserve.arch_registry import (
        _ARCH_FORCES_BACKEND,
        _ARCH_TO_FAMILY,
        _ARCH_TO_REASONING_PARSER,
        _ARCH_TO_TOOL_PARSER,
        _GGUF_ARCH_TO_HF_ARCH,
        _THINKING_DEFAULT_ARCHS,
    )
    return (
        set(_ARCH_TO_TOOL_PARSER)
        | set(_ARCH_TO_REASONING_PARSER)
        | set(_ARCH_FORCES_BACKEND)
        | set(_ARCH_TO_FAMILY)
        | set(_GGUF_ARCH_TO_HF_ARCH.values())
        | set(_THINKING_DEFAULT_ARCHS)
    )


class TestRegistryKeysAgainstVllm:
    def test_no_unexpected_keys_missing_from_vllm(self):
        """Every arch_registry key must either be in the vLLM fixture or
        on the audited allowlist. New keys absent from both fail."""
        vllm_archs = _load_vllm_archs()
        registry_keys = _all_registry_keys()
        missing = registry_keys - vllm_archs - _ALLOWLIST_NOT_IN_VLLM
        assert not missing, (
            f"Registry keys not in vLLM 0.22 and not on the allowlist: {sorted(missing)}. "
            "Either fix the key to match vLLM's canonical arch name, or add it to "
            "_ALLOWLIST_NOT_IN_VLLM with a justification."
        )

    def test_qwen3_5_canonical_names_registered(self):
        """0.6.3b3 added the real vLLM canonical names for Qwen 3.5 / 3.6
        family alongside the legacy fictional ones. Verify they resolve."""
        from vserve.arch_registry import (
            _ARCH_TO_REASONING_PARSER,
            _ARCH_TO_TOOL_PARSER,
            family_of,
            is_thinking_default,
        )
        for arch in (
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ):
            assert _ARCH_TO_TOOL_PARSER.get(arch) == "hermes", arch
            assert _ARCH_TO_REASONING_PARSER.get(arch) == "qwen3", arch
            assert family_of(arch) == "qwen3", arch
            assert is_thinking_default(arch), arch

    def test_allowlist_is_actually_missing(self):
        """Sanity check: every entry on the allowlist really is absent
        from vLLM. Catches stale allowlist entries."""
        vllm_archs = _load_vllm_archs()
        still_in_vllm = _ALLOWLIST_NOT_IN_VLLM & vllm_archs
        assert not still_in_vllm, (
            f"Allowlist entries that ARE in vLLM now: {sorted(still_in_vllm)}. "
            "Remove them from _ALLOWLIST_NOT_IN_VLLM — vLLM ships support."
        )

    def test_suggested_parsers_fire_for_qwen3_5_canonical(self, tmp_path):
        """End-to-end proof that the new canonical entries actually route.

        Writes a synthetic config.json with the canonical Qwen3.5 arch
        and calls the real ``_suggested_*_parser`` functions — closes the
        gap left by ``test_qwen3_5_canonical_names_registered`` (which
        only checks dict membership, not the lookup pipeline)."""
        import json as _json
        from vserve.backends.vllm import (
            _suggested_reasoning_parser,
            _suggested_tool_parser,
        )

        for arch, expect_tp, expect_rp in (
            ("Qwen3_5ForConditionalGeneration",    "hermes", "qwen3"),
            ("Qwen3_5MoeForConditionalGeneration", "hermes", "qwen3"),
        ):
            model_dir = tmp_path / arch
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                _json.dumps({"architectures": [arch]})
            )
            # available_parsers=None bypasses the runtime-registry filter
            # so we exercise just the arch-table lookup.
            assert _suggested_tool_parser(model_dir, available_parsers=None) == expect_tp, arch
            assert _suggested_reasoning_parser(model_dir, available_parsers=None) == expect_rp, arch

    def test_fixture_freshness_visible(self):
        """Surface the fixture's recorded vLLM version so failures can
        be diagnosed without opening the JSON."""
        if not FIXTURE_PATH.exists():
            pytest.skip("fixture missing")
        with FIXTURE_PATH.open() as f:
            data = json.load(f)
        assert "vllm_version" in data
        assert data["vllm_version"], "vllm_version is empty"
