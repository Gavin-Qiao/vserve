"""File-based locking for multi-user concurrency.

Uses fcntl.flock() on files in /run/lock/vserve/ — advisory locks that auto-release
on process exit or crash. Lock metadata (pid, user, timestamp) is written for
diagnostics when a lock is held by another process.
"""

from __future__ import annotations

import fcntl
import json
import os
import errno
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LOCK_DIR = Path("/run/lock/vserve")
SESSION_PATH = LOCK_DIR / "vserve-session.json"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass
class LockInfo:
    pid: int
    user: str
    since: str
    command: str


class LockHeld(Exception):
    """Raised when a lock is already held by another process."""

    def __init__(self, name: str, info: LockInfo | None) -> None:
        self.name = name
        self.info = info
        super().__init__(self.message())

    def message(self) -> str:
        if self.info:
            elapsed = _format_elapsed(self.info.since)
            return (
                f"Locked by {self.info.user} (pid {self.info.pid})"
                f" since {self.info.since} ({elapsed} ago)"
                f"\n  {self.info.command}"
            )
        return f"Lock '{self.name}' is held by another process"


class VserveLock:
    """Advisory file lock via fcntl.flock().

    Usage::

        with VserveLock("fan", "fan daemon"):
            ...  # exclusive access

    The lock is held as long as the file descriptor is open.
    On process crash the OS releases it automatically.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._path = LOCK_DIR / f"vserve-{name}.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        _ensure_lock_dir()
        try:
            self._fd = _open_regular_file(self._path, os.O_RDWR | os.O_CREAT, 0o666)
        except PermissionError as exc:
            if "Refusing" in str(exc):
                raise
            info = _read_lock_info(self._path)
            if info and _pid_alive(info.pid):
                raise LockHeld(self.name, info)
            raise PermissionError(
                f"Cannot create lock file: {self._path}\n"
                f"Run: sudo chmod 1777 {LOCK_DIR}"
            ) from None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            info = _read_lock_info(self._path)
            os.close(self._fd)
            self._fd = None
            raise LockHeld(self.name, info)

        # Write metadata for other processes to read
        info = LockInfo(
            pid=os.getpid(),
            user=os.environ.get("USER", "?"),
            since=time.strftime("%Y-%m-%d %H:%M:%S"),
            command=self.description,
        )
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, json.dumps(asdict(info)).encode() + b"\n")

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.ftruncate(self._fd, 0)
                os.lseek(self._fd, 0, os.SEEK_SET)
            except OSError:
                pass
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> VserveLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _ensure_lock_dir() -> None:
    try:
        st = os.lstat(str(LOCK_DIR))
    except FileNotFoundError:
        LOCK_DIR.mkdir(mode=0o1777, parents=True, exist_ok=True)
    else:
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError(f"Refusing to use symlink as vserve lock directory: {LOCK_DIR}")
        if not stat.S_ISDIR(st.st_mode):
            raise PermissionError(f"Refusing to use non-directory vserve lock directory: {LOCK_DIR}")
    try:
        os.chmod(str(LOCK_DIR), 0o1777)
    except PermissionError:
        pass


def _raise_if_unsafe_file(path: Path) -> None:
    try:
        st = os.lstat(str(path))
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f"Refusing to use symlink as vserve state file: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise PermissionError(f"Refusing to use non-regular vserve state file: {path}")


def _open_regular_file(path: Path, flags: int, mode: int = 0o666) -> int:
    _raise_if_unsafe_file(path)
    try:
        fd = os.open(str(path), flags | _O_NOFOLLOW, mode)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PermissionError(f"Refusing to use symlink as vserve state file: {path}") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"Refusing to use non-regular vserve state file: {path}")
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            try:
                os.fchmod(fd, mode)
            except PermissionError:
                pass
    except Exception:
        os.close(fd)
        raise
    return fd


def _unlink_regular_state_file(path: Path) -> bool:
    try:
        _raise_if_unsafe_file(path)
    except PermissionError:
        raise
    try:
        path.unlink(missing_ok=True)
        return True
    except PermissionError:
        return False
    except OSError:
        return True


def _read_lock_info(path: Path) -> LockInfo | None:
    fd: int | None = None
    try:
        fd = _open_regular_file(path, os.O_RDONLY)
        with os.fdopen(fd) as f:
            fd = None
            data = json.loads(f.read().strip())
        return LockInfo(**data)
    except Exception:
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def wait_for_release(name: str, timeout: float = 5.0) -> bool:
    """Block until the named lock is free, or timeout. Returns True if free."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lock = VserveLock(name, "wait-probe")
            lock.acquire()
            lock.release()
            return True
        except (LockHeld, PermissionError):
            time.sleep(0.2)
    return False


def notify_user(username: str, message: str) -> bool:
    """Send a message to a user's terminal(s) via /dev/pts. Requires tty group.

    Returns True if at least one terminal was written to.
    """
    import subprocess
    sent = False
    try:
        result = subprocess.run(
            ["who"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == username:
                tty = f"/dev/{parts[1]}"
                try:
                    with open(tty, "w") as f:
                        f.write(f"\r\n\033[1;36m[vserve]\033[0m {message}\r\n")
                    sent = True
                except OSError:
                    continue
    except Exception:
        pass
    return sent


@dataclass
class SessionInfo:
    user: str
    since: str
    model: str


class SessionHeld(Exception):
    """Raised when another user owns the active session."""

    def __init__(self, info: SessionInfo) -> None:
        self.info = info
        super().__init__(self.message())

    def message(self) -> str:
        elapsed = _format_elapsed(self.info.since)
        return (
            f"GPU session owned by {self.info.user}"
            f" since {self.info.since} ({elapsed} ago)"
            f"\n  Model: {self.info.model}"
        )


class SessionUnknown(Exception):
    """Raised when a backend is active but no reliable owner marker exists."""

    def __init__(self, reason: str = "A backend is running, but no vserve session marker exists.") -> None:
        self.reason = reason
        super().__init__(reason)

    def message(self) -> str:
        return self.reason


def _safe_session_user(user: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in user)


def _session_path(user: str | None = None) -> Path:
    user = user or os.environ.get("USER", "?")
    safe_user = _safe_session_user(user)
    return SESSION_PATH.with_name(f"{SESSION_PATH.stem}-{safe_user}{SESSION_PATH.suffix}")


def _session_candidates() -> list[Path]:
    paths: list[Path] = []
    if SESSION_PATH.exists():
        paths.append(SESSION_PATH)
    if LOCK_DIR.exists():
        paths.extend(LOCK_DIR.glob(f"{SESSION_PATH.stem}-*{SESSION_PATH.suffix}"))
    uniq: dict[str, Path] = {}
    for path in paths:
        uniq[str(path)] = path
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(uniq.values(), key=_mtime, reverse=True)


def _read_session_file(path: Path) -> SessionInfo | None:
    fd: int | None = None
    try:
        fd = _open_regular_file(path, os.O_RDONLY)
        with os.fdopen(fd) as f:
            fd = None
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                text = f.read().strip()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if not text:
            return None
        data = json.loads(text)
        return SessionInfo(**data)
    except FileNotFoundError:
        return None
    except Exception:
        import sys
        print(f"[vserve] warning: corrupt session file {path}", file=sys.stderr)
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def write_session(model: str) -> None:
    """Record current user as the advisory GPU session owner."""
    _ensure_lock_dir()
    user = os.environ.get("USER", "?")
    info = SessionInfo(
        user=user,
        since=time.strftime("%Y-%m-%d %H:%M:%S"),
        model=model,
    )
    payload = json.dumps(asdict(info)).encode() + b"\n"
    session_path = _session_path(user)
    fd = _open_regular_file(session_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def read_session() -> SessionInfo | None:
    for path in _session_candidates():
        info = _read_session_file(path)
        if info is not None:
            return info
    return None


def clear_session() -> None:
    session_path = _session_path()
    try:
        if _unlink_regular_state_file(session_path):
            return
    except PermissionError:
        raise
    except OSError:
        return

    try:
        fd = _open_regular_file(session_path, os.O_WRONLY)
    except OSError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.ftruncate(fd, 0)
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _clear_all_session_markers() -> None:
    for path in _session_candidates():
        try:
            if _unlink_regular_state_file(path):
                continue
        except PermissionError:
            continue
        except OSError:
            continue

        try:
            fd = _open_regular_file(path, os.O_WRONLY)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.ftruncate(fd, 0)
            os.fsync(fd)
        except OSError:
            pass
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def check_session(*, fail_on_probe_uncertainty: bool = False) -> None:
    """Raise only when a backend is confirmed running for another user."""
    from vserve.backends import probe_running_backend

    current_user = os.environ.get("USER", "?")
    running_backend, probe_failed = probe_running_backend()
    info = read_session()
    if running_backend is None:
        if not probe_failed:
            _clear_all_session_markers()
            return
        if info is not None and fail_on_probe_uncertainty:
            if info.user == current_user:
                return
            raise SessionHeld(info)
        if info is not None and info.user == current_user:
            return
        return
    if info is None:
        backend_name = getattr(running_backend, "display_name", "A backend")
        raise SessionUnknown(f"{backend_name} is running, but no vserve session marker exists.")
    if info.user == current_user:
        return
    raise SessionHeld(info)


def _format_elapsed(since: str) -> str:
    try:
        t = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M:%S"))
        delta = int(time.time() - t)
        if delta < 60:
            return f"{delta}s"
        if delta < 3600:
            return f"{delta // 60}m"
        return f"{delta // 3600}h {(delta % 3600) // 60}m"
    except Exception:
        return "?"
