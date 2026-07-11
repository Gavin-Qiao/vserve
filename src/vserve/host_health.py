"""Host-level readiness checks for `vserve doctor`.

These guard against two failure modes that took the fleet down but live
*outside* the inference stack, so neither backend's ``doctor_checks`` is the
right home:

- **Driver package drift.** An unattended ``apt upgrade`` that bumps the
  NVIDIA driver mid-session causes an NVML "Driver/library version mismatch"
  that kills the GPU until reboot (observed 2026-06-15). Holding the driver
  packages prevents it.
- **Unbounded logs.** vLLM / llama-server append to a single log the process
  holds open; with no rotation it grows without limit (observed 232 MB).

Each check returns a plain dict ``{"ok", "message", "fix"}`` so the doctor
command can route it to its ok/warn renderer, and so the logic is unit-testable
without standing up the whole doctor.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _installed_nvidia_packages() -> set[str]:
    """Names of installed apt packages containing 'nvidia' (empty on non-apt hosts)."""
    if shutil.which("dpkg-query") is None:
        return set()
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package}\n"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "nvidia" in line.lower()
    }


def _held_packages() -> set[str] | None:
    """Names of apt-held packages, or None when apt-mark is unavailable/failed."""
    if shutil.which("apt-mark") is None:
        return None
    try:
        result = subprocess.run(
            ["apt-mark", "showhold"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def check_nvidia_driver_held() -> dict:
    """Are the installed NVIDIA driver packages apt-held against upgrades?

    Skips cleanly (ok) on hosts without apt / without nvidia packages — the
    check is advisory hardening, not a hard requirement.
    """
    held = _held_packages()
    if held is None:
        return {"ok": True, "message": "Driver hold: skipped (no apt-mark)", "fix": ""}
    nvidia = _installed_nvidia_packages()
    if not nvidia:
        return {"ok": True, "message": "Driver hold: no apt nvidia packages", "fix": ""}
    held_nvidia = nvidia & held
    if held_nvidia:
        return {
            "ok": True,
            "message": f"NVIDIA driver packages held ({len(held_nvidia)}/{len(nvidia)}) — safe from unattended upgrades",
            "fix": "",
        }
    return {
        "ok": False,
        "message": "NVIDIA driver packages NOT apt-held — an unattended upgrade can break NVML mid-session (needs reboot)",
        "fix": r"sudo apt-mark hold $(dpkg-query -W -f='${Package}\n' | grep -i nvidia)",
    }


def _logrotate_covers(log_dir: Path, rotate_d: Path = Path("/etc/logrotate.d")) -> bool:
    """True when some /etc/logrotate.d rule names this log directory."""
    if not rotate_d.is_dir():
        return False
    needle = str(log_dir)
    try:
        entries = list(rotate_d.iterdir())
    except OSError:
        return False
    for rule in entries:
        try:
            if rule.is_file() and needle in rule.read_text():
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def check_log_rotation(log_dir: Path, *, rotate_d: Path = Path("/etc/logrotate.d")) -> dict:
    """Is ``log_dir`` covered by logrotate, or accumulating unbounded logs?

    ok when the dir is absent, a logrotate rule names it, or it holds no
    ``*.log`` yet; warn (with the largest current size) otherwise.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return {"ok": True, "message": "Log rotation: no log dir yet", "fix": ""}
    if _logrotate_covers(log_dir, rotate_d):
        return {"ok": True, "message": f"Log rotation configured for {log_dir}", "fix": ""}
    biggest = 0
    try:
        for f in log_dir.glob("*.log"):
            try:
                biggest = max(biggest, f.stat().st_size)
            except OSError:
                continue
    except OSError:
        pass
    if biggest == 0:
        return {"ok": True, "message": f"Log rotation: none, but {log_dir} has no active logs", "fix": ""}
    biggest_mb = biggest / (1024 * 1024)
    return {
        "ok": False,
        "message": f"No logrotate rule for {log_dir} — largest log {biggest_mb:.0f} MB and growing",
        "fix": f"Add an /etc/logrotate.d rule (copytruncate) for {log_dir}/*.log",
    }
