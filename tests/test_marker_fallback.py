"""Tests for the chat-template marker-based parser detection (tools.py).

When the architecture name isn't in ``_ARCH_TO_TOOL_PARSER``, vserve
falls back to scanning the model's chat template for known tag markers
(``<|tool_call|>``, ``<|tool_calls_section_begin|>``, etc.). This is
the escape hatch that makes vserve degrade gracefully for unknown
architectures — they get the right parser even without a table entry.

These tests verify end-to-end via ``detect_tool_parser`` that each
0.6.1-added marker routes to the correct parser name. If a marker
gets accidentally removed from ``_MARKER_TABLE``, these tests fail.
"""

from __future__ import annotations


def _make_model_dir(tmp_path, name, *, chat_template):
    """Synthesize a model directory with a tokenizer_config.json containing
    the given chat_template string."""
    import json
    d = tmp_path / name
    d.mkdir()
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": chat_template}))
    return d


class TestKimiK2MarkerFallback:
    def test_tool_calls_section_begin_routes_to_kimi_k2(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Kimi-K2",
            chat_template="{%- if tools %}<|tool_calls_section_begin|>...{%- endif %}",
        )
        assert detect_tool_parser(d) == "kimi_k2"

    def test_tool_call_begin_also_routes_to_kimi_k2(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Kimi-K2-alt",
            chat_template="<|tool_call_begin|>{{ name }}<|tool_call_argument_begin|>",
        )
        assert detect_tool_parser(d) == "kimi_k2"


class TestCohereMarkerFallback:
    def test_start_tool_marker_routes_to_cohere_command4(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Cohere-CmdR",
            chat_template="<|START_TOOL|>{{ tools }}<|END_TOOL|>",
        )
        assert detect_tool_parser(d) == "cohere_command4"


class TestLfmMarkerFallback:
    def test_tool_call_marker_routes_to_lfm2(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "LFM2",
            chat_template="<|TOOL_CALL|>{{ name }}<|/TOOL_CALL|>",
        )
        assert detect_tool_parser(d) == "lfm2"


class TestGraniteMarkerFallback:
    def test_tool_call_routes_to_granite(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Granite4",
            chat_template="<|tool_call|>{...}<|/tool_call|>",
        )
        assert detect_tool_parser(d) == "granite"


class TestGlmMarkerFallback:
    def test_arg_name_inner_routes_to_glm45(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "GLM-4.5",
            chat_template="<tool_call><arg_name>city</arg_name></tool_call>",
        )
        assert detect_tool_parser(d) == "glm45"


class TestErnieMarkerFallback:
    def test_ernie_inner_routes_to_ernie45(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "ERNIE-4.5",
            chat_template="<tool_call>ERNIE function call</tool_call>",
        )
        assert detect_tool_parser(d) == "ernie45"


class TestExistingMarkersStillResolve:
    """Regression — the 0.6.1 additions must not shadow the 0.6.0 entries."""

    def test_tool_calls_routes_to_mistral(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Mistral-Large",
            chat_template="{% if tools %}[TOOL_CALLS]{{ tools }}{% endif %}",
        )
        assert detect_tool_parser(d) == "mistral"

    def test_python_start_routes_to_llama4(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Llama-4",
            chat_template="<|python_start|>{{ function }}<|python_end|>",
        )
        assert detect_tool_parser(d) == "llama4_pythonic"

    def test_gemma4_marker_still_resolves(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Gemma-4-26B",
            chat_template="<|tool_call>name<tool_call|>",
        )
        assert detect_tool_parser(d) == "gemma4"

    def test_hermes_still_default_for_plain_tool_call(self, tmp_path):
        from vserve.tools import detect_tool_parser
        d = _make_model_dir(
            tmp_path, "Hermes",
            chat_template="<tool_call>{ \"name\": \"x\" }</tool_call>",
        )
        # Generic <tool_call> + JSON inner → hermes (default fallback).
        assert detect_tool_parser(d) == "hermes"


class TestReasoningMarkerAdditions:
    def test_im_thinking_routes_to_cohere(self, tmp_path):
        from vserve.tools import detect_reasoning_parser
        d = _make_model_dir(
            tmp_path, "Cohere-thinking",
            chat_template="<|im_thinking|>{{ thought }}<|/im_thinking|>",
        )
        assert detect_reasoning_parser(d) == "cohere"

    def test_hun_thinking_routes_to_hunyuan(self, tmp_path):
        from vserve.tools import detect_reasoning_parser
        d = _make_model_dir(
            tmp_path, "Hunyuan",
            chat_template="<|hun_thinking|>thought<|/hun_thinking|>",
        )
        assert detect_reasoning_parser(d) == "hunyuan"

    def test_channel_routes_to_gemma4_reasoning(self, tmp_path):
        from vserve.tools import detect_reasoning_parser
        d = _make_model_dir(
            tmp_path, "Gemma-4-thinking",
            chat_template="<|channel>thought<channel|>",
        )
        assert detect_reasoning_parser(d) == "gemma4"
