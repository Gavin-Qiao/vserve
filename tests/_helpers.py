"""Shared test helpers (importable as ``from _helpers import ...``)."""

from __future__ import annotations

import re

_ANSI_ESCAPE = re.compile(r"\x1b\[[\d;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from CLI output.

    Typer/Rich renders option flags with per-character style spans
    (e.g. ``\\x1b[1;36m-\\x1b[0m\\x1b[1;36m-flag\\x1b[0m``), which breaks
    naive ``"--flag" in stdout`` assertions when Rich detects CI mode
    (``CI=true``, ``GITHUB_ACTIONS=true``). Use this when asserting on
    ``CliRunner().invoke(...).stdout``.
    """
    return _ANSI_ESCAPE.sub("", text)
