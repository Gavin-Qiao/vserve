"""Auto-detect vLLM tool-call and reasoning parsers from model chat template.

Reads the model's tokenizer_config.json and checks for known markers
in the chat template to determine which --tool-call-parser and
--reasoning-parser to use.  Falls back to model-name heuristics for
models that use token-level routing (e.g. GPT-OSS).
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path


# Ordered by specificity — first match wins.
# Each entry: (marker, extra_marker, parser_name)
# When extra_marker is set, both must be present for a match.
_MARKER_TABLE: list[tuple[str, str | None, str]] = [
    # Unique markers — unambiguous
    ("[TOOL_CALLS]",          None,          "mistral"),
    ("<|python_start|>",      None,          "llama4_pythonic"),
    ("<|python_tag|>",        None,          "llama3_json"),
    ("tool\u2581calls\u2581begin",  None,    "deepseek_v3"),
    ("<|tool_call>",          None,          "gemma4"),       # asymmetric tag
    ("<|action_start|>",      None,          "internlm"),
    ("<|tool_calls_section_begin|>", None,   "kimi_k2"),
    ("<|tool_call_begin|>",   None,          "kimi_k2"),
    ("<|START_TOOL|>",        None,          "cohere_command4"),
    ("<|TOOL_CALL|>",         None,          "lfm2"),
    # IBM Granite — generic <|tool_call|> tag (Granite 3 + 4 share it; the
    # parser layer disambiguates by checksum of the surrounding schema).
    ("<|tool_call|>",         None,          "granite"),
    # Shared <tool_call> tag — disambiguate by inner format
    ("<tool_call>",           "<function=",  "qwen3_coder"),  # XML inner
    ("<tool_call>",           "<arg_name>",  "glm45"),         # GLM-4.5 attribute-list inner
    ("<tool_call>",           "ERNIE",       "ernie45"),       # ERNIE-4.5 inner schema
    ("<tool_call>",           None,          "hermes"),        # JSON inner — default
]

# Name-based fallback for models that use token-level routing
# (no text markers in the chat template).
_NAME_FALLBACKS: list[tuple[str, str]] = [
    (r"gpt-oss",    "openai"),
]
KNOWN_TOOL_PARSERS = frozenset(parser for _marker, _extra, parser in _MARKER_TABLE) | frozenset(
    parser for _pattern, parser in _NAME_FALLBACKS
)

# Reasoning parser markers — ordered by specificity.
_REASONING_MARKER_TABLE: list[tuple[str, str | None, str]] = [
    # Unique markers
    ("<seed:think>",      None,              "seed_oss"),
    ("[THINK]",           None,              "mistral"),
    ("<|channel>",        None,              "gemma4"),
    ("<|im_thinking|>",   None,              "cohere"),
    ("<|hun_thinking|>",  None,              "hunyuan"),
    # Shared <think> — disambiguate
    ("<think>",           "enable_thinking",  "qwen3"),        # Qwen3/3.5
    ("<think>",           None,              "deepseek_r1"),   # DeepSeek, QwQ, KimiK2Thinking, etc.
]

_REASONING_NAME_FALLBACKS: list[tuple[str, str]] = [
    (r"gpt-oss",    "openai_gptoss"),
]
KNOWN_REASONING_PARSERS = frozenset(parser for _marker, _extra, parser in _REASONING_MARKER_TABLE) | frozenset(
    parser for _pattern, parser in _REASONING_NAME_FALLBACKS
)

# Models whose chat templates contain tool markers but are not
# generative — embedding, reranker, reward, etc.
_NON_GENERATIVE_RE = re.compile(
    r"\b(embed|rerank|reward|classifier|retriev)", re.IGNORECASE,
)


def detect_tool_parser(model_path: Path) -> str | None:
    """Detect the vLLM tool-call parser for a model.

    Returns the parser name (e.g. 'hermes', 'qwen3_coder') or None
    if the model does not support tool calling.
    """
    if _NON_GENERATIVE_RE.search(model_path.name):
        return None

    # Tier 1: chat template marker matching
    template = _read_chat_template(model_path)
    if template is not None:
        parser = _match_template(template)
        if parser is not None:
            return parser

    # Tier 2: model-name heuristic (for token-level parsers like 'openai')
    name = model_path.name.lower()
    for pattern, parser in _NAME_FALLBACKS:
        if re.search(pattern, name):
            return parser

    return None


def detect_reasoning_parser(model_path: Path) -> str | None:
    """Detect the vLLM reasoning parser for a model.

    Returns the parser name (e.g. 'qwen3', 'deepseek_r1') or None
    if the model does not support reasoning output.
    """
    if _NON_GENERATIVE_RE.search(model_path.name):
        return None

    template = _read_chat_template(model_path)
    if template is not None:
        parser = _match_reasoning_template(template)
        if parser is not None:
            return parser

    name = model_path.name.lower()
    for pattern, parser in _REASONING_NAME_FALLBACKS:
        if re.search(pattern, name):
            return parser

    return None


def supports_tools(model_path: Path) -> bool:
    """Check if a model's chat template references tools at all."""
    if _NON_GENERATIVE_RE.search(model_path.name):
        return False
    template = _read_chat_template(model_path)
    if template is None:
        # No template markers, but check name fallback
        name = model_path.name.lower()
        return any(re.search(p, name) for p, _ in _NAME_FALLBACKS)
    # Check for Jinja variable reference, not just substring
    return bool(re.search(r"\btools\b", template))


@functools.lru_cache(maxsize=32)
def _read_chat_template(model_path: Path) -> str | None:
    """Extract the chat template string from tokenizer_config.json."""
    tok_config = model_path / "tokenizer_config.json"
    if not tok_config.exists():
        return None
    try:
        data = json.loads(tok_config.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    ct = data.get("chat_template")
    if ct is None:
        return None
    if isinstance(ct, str):
        return ct
    if isinstance(ct, list):
        def _template_from_entry(entry: object) -> str | None:
            if not isinstance(entry, dict):
                return None
            template = entry.get("template")
            if isinstance(template, str) and template:
                return template
            return None

        # Multi-template format — prefer 'tool_use' template if present.
        for entry in ct:
            if isinstance(entry, dict) and entry.get("name") == "tool_use":
                template = _template_from_entry(entry)
                if template is not None:
                    return template
        # Fall back to 'default' template only (avoid merging unrelated variants).
        for entry in ct:
            if isinstance(entry, dict) and entry.get("name") in ("default", None):
                template = _template_from_entry(entry)
                if template is not None:
                    return template
        # Last resort: first template
        if ct:
            return _template_from_entry(ct[0])
    return None


def _match_template(template: str) -> str | None:
    """Match a chat template against known tool parser markers."""
    for marker, extra_marker, parser in _MARKER_TABLE:
        if marker in template:
            if extra_marker is not None:
                if extra_marker in template:
                    return parser
                continue
            return parser
    return None


def _match_reasoning_template(template: str) -> str | None:
    """Match a chat template against known reasoning parser markers."""
    for marker, extra_marker, parser in _REASONING_MARKER_TABLE:
        if marker in template:
            if extra_marker is not None:
                if extra_marker in template:
                    return parser
                continue
            return parser
    return None
