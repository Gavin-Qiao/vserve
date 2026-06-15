import json

from vserve.tools import (
    _read_chat_template,
    detect_reasoning_parser,
    detect_tool_parser,
    supports_tools,
)


def test_read_chat_template_ignores_non_mapping_tokenizer_config(tmp_path):
    model_dir = tmp_path / "neutral-model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(json.dumps(["bad"]))

    assert _read_chat_template(model_dir) is None
    assert detect_tool_parser(model_dir) is None
    assert detect_reasoning_parser(model_dir) is None
    assert supports_tools(model_dir) is False


def test_read_chat_template_ignores_non_string_template_entries(tmp_path):
    model_dir = tmp_path / "plain-model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": [
            {"name": "tool_use", "template": 123},
            {"name": "default", "template": {"jinja": True}},
        ]
    }))

    assert _read_chat_template(model_dir) is None
    assert detect_tool_parser(model_dir) is None
    assert detect_reasoning_parser(model_dir) is None
    assert supports_tools(model_dir) is False


def test_read_chat_template_ignores_non_utf8_file(tmp_path):
    model_dir = tmp_path / "broken-template-model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_bytes(b"\x80\x81\x82")

    assert _read_chat_template(model_dir) is None
    assert detect_tool_parser(model_dir) is None
    assert detect_reasoning_parser(model_dir) is None
    assert supports_tools(model_dir) is False


def test_sniffer_agrees_with_arch_table_for_qwen35_canonical(tmp_path):
    """Regression guard for the two-path parser selection. In the --tools serve
    flow the parser comes from the chat-template *sniffer* (detect_tool_parser),
    NOT _ARCH_TO_TOOL_PARSER; b94a823 only patched the arch table. For the
    canonical arch real Qwen3.5/3.6 checkpoints report
    (Qwen3_5MoeForConditionalGeneration) the two paths must agree, so they can't
    silently drift back to hermes / deepseek_r1."""
    from vserve.arch_registry import (
        _ARCH_TO_REASONING_PARSER,
        _ARCH_TO_TOOL_PARSER,
    )

    arch = "Qwen3_5MoeForConditionalGeneration"
    model_dir = tmp_path / "Qwen3.5-MoE"
    model_dir.mkdir()
    # Same markers the real Qwen3.5/3.6 templates carry: XML tool calls
    # (<tool_call> + <function=) and an enable_thinking-gated <think> block.
    template = (
        "{%- if enable_thinking is defined and enable_thinking is false %}"
        "{{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}"
        "{%- for tc in message.tool_calls %}"
        "{{- '<tool_call>\\n<function=' + tc.name + '>\\n<parameter=x>\\n' }}{%- endfor %}"
    )
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))

    assert detect_tool_parser(model_dir) == _ARCH_TO_TOOL_PARSER[arch] == "qwen3_coder"
    assert detect_reasoning_parser(model_dir) == _ARCH_TO_REASONING_PARSER[arch] == "qwen3"
