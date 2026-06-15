"""Docs must state the live vLLM support policy.

troubleshooting.md shipped stale twice (it still said ``<0.21`` and
``vllm==0.20.0`` at 0.6.3b3, one full minor behind the code). These
assertions make ``runtime.py`` constants and the docs move together.
"""

from pathlib import Path

from vserve.runtime import PINNED_STABLE_VLLM, SUPPORTED_VLLM_RANGE

ROOT = Path(__file__).parent.parent


def test_troubleshooting_states_live_range():
    text = (ROOT / "docs" / "troubleshooting.md").read_text()
    assert SUPPORTED_VLLM_RANGE in text, (
        f"docs/troubleshooting.md must mention the supported range "
        f"{SUPPORTED_VLLM_RANGE!r}"
    )
    assert f"vllm=={PINNED_STABLE_VLLM}" in text, (
        f"docs/troubleshooting.md must mention the pinned runtime "
        f"vllm=={PINNED_STABLE_VLLM}"
    )


def test_readme_badge_matches_range():
    text = (ROOT / "README.md").read_text()
    # SUPPORTED_VLLM_RANGE ">=0.20,<0.24" → badge "0.20–0.23"
    assert "vLLM-0.20%E2%80%930.23" in text, "README vLLM badge is stale"
    assert "0.20.x or 0.21.x" not in text, "README prerequisites row is stale"
