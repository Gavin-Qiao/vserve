"""Shared systemd state-machine primitives.

vLLM and llama.cpp both ship as systemd units that vserve start/stop via
``sudo systemctl``. Before 0.6.3, each backend (serve.py for vLLM,
llamacpp.py for llama.cpp) carried its own near-identical implementation
of the safety asserter + the systemctl call wrapper.

This module centralises the primitives. Both backends now call into the
same code path; their high-level `start()` / `stop()` methods still own
their own pre/post hooks (env-file write, manifest emission, etc.).

Extracted in 0.6.3 per audit
`docs/audits/2026-05-20-backend-consistency.md` finding #1 (lifecycle
placement asymmetry).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from vserve.config import (
    find_systemd_unit_path,
    unit_content_matches_backend,
    validate_systemd_service_name,
)


# Privileged actions that need the safety asserter to fire before invoking
# `sudo systemctl ...`. Other actions (is-active, status) are read-only.
PRIVILEGED_SYSTEMCTL_ACTIONS = frozenset({"start", "stop", "restart", "reload"})


def assert_unit_safe(
    *,
    service_name: str,
    backend_name: str,
    root: Path,
    expected_paths: list[Path],
) -> None:
    """Verify the systemd unit file looks like a vserve-managed unit before
    invoking a privileged systemctl action.

    The check prevents vserve from accidentally restarting an unrelated
    service that happens to share the configured ``service_name`` (e.g.
    if someone manually wrote a unit with that name).

    Raises ``RuntimeError`` when the unit exists but doesn't match — caller
    should abort instead of executing the action.
    """
    validate_systemd_service_name(service_name)
    unit = find_systemd_unit_path(service_name)
    if unit is None:
        # No unit file → nothing to verify (caller's systemctl will fail
        # later if that's wrong).
        return
    try:
        content = unit.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Cannot verify {service_name}.service before privileged systemctl action: {exc}"
        ) from None
    if not unit_content_matches_backend(
        content,
        backend_name=backend_name,
        root=root,
        expected_paths=expected_paths,
    ):
        raise RuntimeError(
            f"{service_name}.service does not look like a vserve {backend_name} unit"
        )


def systemctl_call(
    service_name: str,
    action: str,
    *,
    timeout: int = 30,
    non_interactive: bool = False,
    asserter: Callable[[], None] | None = None,
) -> tuple[bool, str, str]:
    """Invoke ``systemctl <action> <service_name>``, with ``sudo`` for
    privileged actions and an optional pre-action safety check.

    ``asserter``: callable that raises on unsafe state. Only invoked for
    actions in :data:`PRIVILEGED_SYSTEMCTL_ACTIONS`. If it raises, the
    systemctl call is NOT executed — returned ``(ok=False, stdout="",
    stderr=str(exc))``.

    Returns ``(ok, stdout, stderr)``. ``ok`` is ``True`` iff systemctl
    exited 0 within the timeout window.
    """
    command = ["systemctl", action, service_name]
    if action in PRIVILEGED_SYSTEMCTL_ACTIONS:
        if asserter is not None:
            try:
                asserter()
            except Exception as exc:
                return False, "", str(exc)
        if non_interactive:
            command.insert(0, "-n")
        command.insert(0, "sudo")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"systemctl {action} timed out after {timeout}s"
    except Exception as exc:
        return False, "", str(exc)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def unit_memory_max(service_name: str) -> str | None:
    """The unit's effective ``MemoryMax`` (e.g. ``"50G"``), or None.

    None means "no cap": the property is unset/``infinity``, the unit is
    unknown, or systemctl failed. Read-only (`systemctl show`), no sudo.
    This is the host-RAM OOM guard vserve expects on managed inference
    units — an uncapped vLLM boot can freeze the host via JIT/compile
    storms (see docs/plans/2026-07-10-qwen36-64k-mtp-speed.md, RAM notes).
    """
    try:
        validate_systemd_service_name(service_name)
    except Exception:
        return None
    try:
        result = subprocess.run(
            ["systemctl", "show", service_name, "-p", "MemoryMax", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    if not value or value == "infinity":
        return None
    return value


def parse_is_active_output(stdout: str, stderr: str, ok: bool, *, service_name: str) -> bool | None:
    """Interpret ``systemctl is-active`` output as a tri-state.

    Returns:
      * ``True``  — unit is active.
      * ``False`` — unit is inactive, failed, or not found.
      * ``None``  — unit is in a transitional state; caller should retry.

    Caller is expected to ``raise RuntimeError(...)`` on transitional /
    error states it can't handle. We return ``None`` rather than raising
    so the caller chooses the wording.
    """
    status = stdout.strip().lower()
    if ok and status == "active":
        return True
    if status in {"inactive", "failed"}:
        return False
    if status in {"activating", "deactivating", "reloading"}:
        return None
    if "could not be found" in stderr.lower():
        return False
    return False
