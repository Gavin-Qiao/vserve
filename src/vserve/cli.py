"""vserve — LLM inference manager CLI."""


import pathlib
import re
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vserve.config import (
    CONFIG_FILE,
    MODELS_DIR,
    limits_path,
    profile_path,
    read_limits,
)
from vserve.models import scan_models, fuzzy_match, ModelInfo
from vserve.lock import (
    VserveLock,
    LockHeld,
    SessionHeld,
    SessionUnknown,
    notify_user,
    check_session,
    read_session,
    write_session,
    clear_session,
)

from vserve import __version__

app = typer.Typer(help="LLM inference manager")
cache_app = typer.Typer(help="Cache management")
runtime_app = typer.Typer(help="Runtime environment management")
profile_app = typer.Typer(help="Manage saved serving profiles")
app.add_typer(cache_app, name="cache")
app.add_typer(runtime_app, name="runtime")
app.add_typer(profile_app, name="profile")
console = Console()
_TITLE = f"[bold cyan]vserve[/bold cyan] [dim]{__version__}[/dim]"
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")



def _session_or_exit(
    *,
    fail_on_probe_uncertainty: bool = True,
    allow_unknown_owner: bool = False,
    quiet: bool = False,
) -> None:
    """Block if another user owns the active GPU session, DM the holder.

    When `allow_unknown_owner` is True and the owner is unknown, return silently
    if `quiet=True`; otherwise emit a single yellow warning. The `quiet` flag is
    intended for follow-up TOCTOU re-checks within a single command so we don't
    repeat the warning every time we re-acquire the lock.
    """
    import os
    try:
        check_session(fail_on_probe_uncertainty=fail_on_probe_uncertainty)
    except SessionHeld as exc:
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]"
            f"\n\nAsk [cyan]{exc.info.user}[/cyan] to run [cyan]vserve stop[/cyan] first.",
            title=f"[red]vserve {__version__}: session locked[/red]",
            border_style="red",
        ))
        me = os.environ.get("USER", "?")
        if exc.info.user != me:
            notify_user(
                exc.info.user,
                f"{me} wants the GPU (you: {exc.info.model})",
            )
        raise typer.Exit(1) from None
    except SessionUnknown as exc:
        if allow_unknown_owner:
            if quiet:
                return
            me = os.environ.get("USER", "?")
            console.print(f"[yellow]{exc.message()}[/yellow]")
            console.print(
                f"  [dim]Proceeding as {me}. "
                "Claim future sessions by starting via `vserve run`.[/dim]"
            )
            return
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]"
            "\n\nUse [cyan]vserve status[/cyan] or inspect the backend service before starting another model.",
            title=f"[red]vserve {__version__}: session owner unknown[/red]",
            border_style="red",
        ))
        raise typer.Exit(1) from None


def _safe_path_exists(path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_resolve_path(path):
    try:
        path.stat()
    except FileNotFoundError:
        try:
            return path.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
    except (OSError, RuntimeError):
        return None
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _record_backend_manifest(backend, cfg_path: pathlib.Path, *, label: str, status: str, port: int | None = None, error: str | None = None) -> None:
    """Best-effort write of backend runtime state for status/doctor."""
    from datetime import datetime, timezone
    from vserve.config import write_active_manifest

    path_fn = getattr(backend, "active_manifest_path", None)
    if not callable(path_fn):
        return
    manifest = {
        "backend": backend.name,
        "service_name": backend.service_name,
        "config_path": str(cfg_path.resolve()),
        "label": label,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from vserve.config import cfg as _cfg

        gpu_index = int(getattr(_cfg(), "gpu_index", 0) or 0)
        manifest["gpu_index"] = gpu_index
        manifest["cuda_visible_devices"] = str(gpu_index)
    except Exception:
        pass
    if port is not None:
        manifest["port"] = port
    if error:
        manifest["error"] = error
    try:
        write_active_manifest(manifest, path_fn())
    except Exception:
        pass


def _check_backend_runtime_or_exit(
    backend,
    *,
    allow_unsupported_runtime: bool = False,
    prefer_cache: bool = False,
):
    """Validate the backend's runtime version. On `vserve run`, pass
    `prefer_cache=True` to serve a cached RuntimeInfo and skip the slow
    `pip check` step — diagnostic commands (doctor, runtime check) leave it
    False to always re-probe in full."""
    if getattr(backend, "name", None) != "vllm":
        return None
    runtime_info = None
    check = None
    runtime_error: Exception | None = None
    try:
        runtime_info_fn = getattr(backend, "runtime_info", None)
        if not callable(runtime_info_fn):
            raise RuntimeError("runtime_info is unavailable")
        try:
            runtime_info = runtime_info_fn(
                prefer_cache=prefer_cache,
                with_pip_check=not prefer_cache,
            )
        except TypeError:
            # Older backend stubs (test doubles) may not accept the kwargs.
            runtime_info = runtime_info_fn()
        from vserve.runtime import RuntimeInfo, check_vllm_compatibility
        if not isinstance(runtime_info, RuntimeInfo):
            raise RuntimeError("runtime_info did not return RuntimeInfo")
        check = check_vllm_compatibility(runtime_info)
    except Exception as exc:
        runtime_info = None
        runtime_error = exc
        compatibility_fn = getattr(backend, "compatibility", None)
        if callable(compatibility_fn):
            try:
                check = compatibility_fn()
            except Exception as compat_exc:
                runtime_error = compat_exc
        if check is None:
            assert runtime_error is not None
            exc = runtime_error
            console.print(f"[red]Could not check {backend.display_name} runtime compatibility:[/red] {exc}")
            console.print("  Use [cyan]vserve runtime check vllm[/cyan] for details.")
            if not allow_unsupported_runtime:
                raise typer.Exit(1) from None
            return runtime_info
    if check is None:
        return runtime_info
    if check.supported:
        return runtime_info
    console.print(f"[red]Unsupported {backend.display_name} runtime.[/red]")
    console.print(f"  Supported: {check.range}")
    for error in check.errors:
        console.print(f"  [red]FAIL[/red]  {error}")
    console.print("  Repair: [cyan]vserve runtime upgrade vllm --stable[/cyan]")
    if not allow_unsupported_runtime:
        raise typer.Exit(1)
    return runtime_info


def _lock_or_exit(name: str, description: str) -> VserveLock:
    """Acquire a lock or DM the holder and exit."""
    import os
    lock = VserveLock(name, description)
    try:
        lock.acquire()
    except LockHeld as exc:
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]",
            title=f"[red]vserve {__version__}: {name} locked[/red]",
            border_style="red",
        ))
        # DM the lock holder
        me = os.environ.get("USER", "?")
        if exc.info and exc.info.user != me:
            notify_user(
                exc.info.user,
                f"{me} is waiting for {name} (you: {exc.info.command})",
            )
        raise typer.Exit(1) from None
    except PermissionError as exc:
        console.print(Panel(
            f"[bold]{exc}[/bold]",
            title=f"[red]vserve {__version__}: permission denied[/red]",
            border_style="red",
        ))
        raise typer.Exit(1) from None
    return lock


def _all_models() -> list[ModelInfo]:
    """Scan model directories for all registered backends."""
    from vserve.backends import _BACKENDS
    all_m: list[ModelInfo] = []
    seen: set = set()
    # Always include the legacy MODELS_DIR (vLLM)
    for m in scan_models(MODELS_DIR):
        if m.path not in seen:
            all_m.append(m)
            seen.add(m.path)
    # Scan additional backend model dirs
    for b in _BACKENDS:
        models_dir = b.root_dir / "models"
        if models_dir == MODELS_DIR or not models_dir.exists():
            continue
        for m in scan_models(models_dir):
            if m.path not in seen:
                all_m.append(m)
                seen.add(m.path)
    return all_m


def _resolve_model(query: str) -> ModelInfo:
    models = _all_models()
    if not models:
        console.print("[red]No models found.[/red] Run: vserve add")
        raise typer.Exit(1)
    matches = fuzzy_match(query, models)
    if len(matches) == 0:
        console.print(f"[red]No model matching '{query}'[/red]")
        for m in models:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[yellow]'{query}' matches multiple models:[/yellow]")
        for m in matches:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    return matches[0]


def _join_model_terms(terms: str | list[str] | tuple[str, ...] | None) -> str | None:
    if not terms:
        return None
    if isinstance(terms, str):
        query = terms.strip()
        return query or None
    query = " ".join(str(term) for term in terms).strip()
    return query or None


def _backend_format_guidance(backend_name: str) -> str:
    if backend_name == "llamacpp":
        return "llama.cpp serves GGUF models. Use vserve add to select a GGUF variant or omit --backend."
    if backend_name == "vllm":
        return "vLLM serves Hugging Face safetensors/bin model roots. Use a non-GGUF variant or omit --backend."
    return "Use vserve list to inspect model formats, or omit --backend for automatic selection."


def _validate_profile_name(profile: str) -> str:
    if profile in {".", ".."} or "/" in profile or "\\" in profile or not _PROFILE_NAME_RE.match(profile):
        console.print("[red]Profile name must contain only letters, numbers, '.', '_', and '-'.[/red]")
        console.print("  It cannot be '.' or '..' and cannot include path separators.")
        raise typer.Exit(1)
    return profile


def _show_update_notice() -> None:
    """Print Wrangler-style update box if a newer version is available."""
    from vserve.version import update_available
    info = update_available()
    if info:
        console.print(Panel(
            f"  vserve [bold]{info.current}[/bold] → [bold green]{info.latest}[/bold green]\n"
            f"  Run [cyan]vserve update[/cyan] to upgrade",
            title=f"[yellow]vserve {__version__}: update available[/yellow]",
            border_style="yellow",
        ))


@app.callback(invoke_without_command=True)
def dashboard(ctx: typer.Context):
    """Show status dashboard when called with no subcommand."""
    if ctx.invoked_subcommand is not None:
        import sys
        if "--json" in sys.argv or ctx.invoked_subcommand in {"status", "doctor"}:
            return
        from vserve.version import background_refresh
        background_refresh()
        from rich.rule import Rule
        console.print(Rule(title=f"[bold cyan]vserve[/bold cyan] [dim]{__version__}[/dim]", style="cyan"))
        if ctx.invoked_subcommand not in ("version", "update"):
            def _close_border() -> None:
                _show_update_notice()
                console.print(Rule(style="cyan"))
            ctx.call_on_close(_close_border)
        else:
            ctx.call_on_close(lambda: console.print(Rule(style="cyan")))
        return

    if not CONFIG_FILE.exists():
        console.print(Panel(
            "[bold]First time? Run [cyan]vserve init[/cyan] to set up your system.[/bold]",
            title=_TITLE,
            border_style="yellow",
        ))

    try:
        from vserve.gpu import get_gpu_info
        gpu = get_gpu_info()
        gpu_line = f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, CUDA {gpu.cuda})"
    except Exception:
        gpu_line = "[dim]unavailable[/dim]"

    from vserve.backends import running_backend as _running_backend
    from vserve.config import active_yaml_path, try_read_profile_yaml

    models = _all_models()
    probed = sum(
        1 for m in models if read_limits(limits_path(m.provider, m.model_name))
    )

    serving_line = "[dim]not running[/dim]"
    rb = _running_backend()
    if rb is not None:
        active = active_yaml_path()
        if active.is_symlink():
            cfg = try_read_profile_yaml(active)
            model_path = cfg.get("model", "?") if cfg else "?"
            model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path
            port = cfg.get("port", 8888) if cfg else 8888
            if cfg:
                serving_line = f"[green]{model_name}[/green] at :{port} ({rb.display_name})"
            else:
                serving_line = f"[yellow]config unreadable[/yellow] ({rb.display_name})"
        else:
            serving_line = f"[green]active[/green] ({rb.display_name})"

    lines = [
        f"  [bold]GPU[/bold]       {gpu_line}",
        f"  [bold]Models[/bold]    {len(models)} downloaded, {probed} probed",
        f"  [bold]Serving[/bold]   {serving_line}",
        "",
    ]

    from rich.table import Table as _Tbl
    from rich.text import Text as _Txt
    cmd_tbl = _Tbl(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    cmd_tbl.add_column(min_width=26)
    cmd_tbl.add_column(style="dim")
    for cmd, desc in [
        ("list", "List models with limits & capabilities"),
        ("add [model]", "Search & download from HuggingFace"),
        ("rm [model]", "Remove a downloaded model"),
        ("tune [model]", "Calculate context & concurrency limits"),
        ("run [model]", "Start serving (interactive config)"),
        ("stop", "Stop the inference server"),
        ("status", "Show what's currently serving"),
        ("fan [auto|off|30-100]", "GPU fan control with temp curve"),
        ("doctor", "Check system readiness"),
        ("init", "Auto-discover backends and write config"),
        ("version", "Show current version"),
        ("update", "Update to the latest version"),
    ]:
        cmd_tbl.add_row(_Txt(cmd, style="bold cyan"), desc)

    from rich.console import Group
    body = Group(
        "\n".join(lines) + "\n  [bold]Commands[/bold]",
        cmd_tbl,
    )
    console.print(Panel(body, title=_TITLE, border_style="cyan"))
    from vserve.version import background_refresh
    background_refresh()
    _show_update_notice()


@app.command()
def version():
    """Show current version and check for updates."""
    from vserve.version import update_available, background_refresh

    background_refresh()
    info = update_available()

    if info:
        console.print(f"  vserve [bold]{info.current}[/bold]")
        console.print(f"  [yellow]Update available: {info.latest}[/yellow]")
        console.print("  Run [cyan]vserve update[/cyan] to upgrade")
    else:
        console.print(f"  vserve [bold]{__version__}[/bold] — up to date")


@app.command()
def update(
    nightly: bool = typer.Option(False, "--nightly", help="Install latest pre-release version"),
):
    """Update vserve to the latest version."""
    import shutil
    import subprocess
    import sys

    from vserve.version import check_pypi, write_cache, _compare_versions
    command_timeout = 300

    def _run_checked(cmd: list[str], label: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=command_timeout)
        except subprocess.TimeoutExpired:
            console.print(f"[red]{label} timed out after {command_timeout}s.[/red]")
            raise typer.Exit(1)
        if result.returncode != 0:
            console.print(f"[red]{label} failed.[/red]")
            details = result.stderr.strip() or result.stdout.strip()
            if details:
                console.print(f"  {details}")
            raise typer.Exit(1)
        return result

    def _current_importable_version() -> str | None:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import vserve; print(vserve.__version__)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        version = result.stdout.strip()
        return version or None

    def _finish_update(installer: str, expected: str | None) -> None:
        _refresh_banner()
        installed = _current_importable_version()
        if installed is None:
            console.print(f"[yellow]{installer} completed, but vserve could not verify the installed module version.[/yellow]")
            return
        if expected is not None and _compare_versions(installed, __version__) <= 0:
            console.print(f"[red]{installer} completed, but the running environment still imports vserve {installed}.[/red]")
            console.print("  The installer may have upgraded a different environment; check `which vserve` and your Python path.")
            raise typer.Exit(1)
        if expected is not None and _compare_versions(installed, expected) < 0:
            console.print(f"[red]{installer} completed, but vserve {installed} is older than expected {expected}.[/red]")
            raise typer.Exit(1)
        if expected is None and installed == __version__:
            console.print(f"[yellow]{installer} completed, but the currently importable vserve remains {installed}.[/yellow]")
            console.print("  This can mean the pre-release channel had no newer build, or the installer targeted another environment.")
            return
        console.print(f"[green]vserve {installed} is installed.[/green]")

    console.print(f"[dim]Current version: {__version__}[/dim]")

    if not nightly:
        latest = check_pypi()
        if not latest:
            console.print("[red]Could not determine the latest published version.[/red]")
            console.print("  Retry later, or use [cyan]vserve update --nightly[/cyan] if you intentionally want a pre-release.")
            raise typer.Exit(1)
        write_cache(latest)
        if _compare_versions(latest, __version__) <= 0:
            console.print("[green]Already up to date.[/green]")
            return
        console.print(f"[yellow]New version available: {latest}[/yellow]\n")

    uv = shutil.which("uv")
    if uv:
        try:
            result = subprocess.run([uv, "tool", "list"], capture_output=True, text=True, timeout=command_timeout)
        except subprocess.TimeoutExpired:
            console.print(f"[yellow]uv inspection timed out after {command_timeout}s; trying pip fallback if available.[/yellow]")
            result = None
        if result is not None and result.returncode != 0:
            console.print("[yellow]uv inspection failed; trying pip fallback if available.[/yellow]")
            details = result.stderr.strip() or result.stdout.strip()
            if details:
                console.print(f"  {details}")
            result = None
        if result is not None and "vserve" in result.stdout:
            if nightly:
                console.print("[dim]Upgrading to latest pre-release via uv...[/dim]")
                _run_checked([uv, "tool", "upgrade", "vserve", "--prerelease", "allow"], "uv upgrade")
            else:
                console.print("[dim]Upgrading via uv...[/dim]")
                _run_checked([uv, "tool", "upgrade", "vserve"], "uv upgrade")
            _finish_update("uv upgrade", None if nightly else latest)
            return

    pip = shutil.which("pip") or shutil.which("pip3")
    if pip:
        pip_cmd = [pip]
        if nightly:
            console.print("[dim]Upgrading to latest pre-release via pip...[/dim]")
            _run_checked([*pip_cmd, "install", "--upgrade", "--pre", "vserve"], "pip install")
        else:
            console.print("[dim]Upgrading via pip...[/dim]")
            _run_checked([*pip_cmd, "install", "--upgrade", "vserve"], "pip install")
        _finish_update("pip install", None if nightly else latest)
        return

    console.print("[red]Could not find uv or pip to perform upgrade.[/red]")
    console.print("Run manually: [cyan]uv tool upgrade vserve[/cyan] or [cyan]pip install -U vserve[/cyan]")
    raise typer.Exit(1)


@runtime_app.command("check")
def runtime_check(
    backend: str = typer.Argument("vllm", help="Runtime backend to check"),
    allow_unsupported: bool = typer.Option(False, "--allow-unsupported-runtime", help="Exit 0 even if runtime is outside vserve's support range"),
):
    """Check external backend runtime compatibility."""
    if backend != "vllm":
        console.print(f"[red]Unsupported runtime check backend: {backend}[/red]")
        console.print("  Available: vllm")
        raise typer.Exit(1)

    from vserve.runtime import check_vllm_compatibility, collect_vllm_runtime_info

    info = collect_vllm_runtime_info()
    check = check_vllm_compatibility(info)

    if check.supported:
        console.print("[green]vLLM runtime supported[/green]")
    else:
        console.print("[red]Unsupported vLLM runtime[/red]")
    console.print(f"  vLLM:            {info.vllm_version or '?'}")
    console.print(f"  torch:           {info.torch_version or '?'}")
    console.print(f"  torch CUDA:      {info.torch_cuda or '?'}")
    console.print(f"  transformers:    {info.transformers_version or '?'}")
    console.print(f"  huggingface_hub: {info.huggingface_hub_version or '?'}")
    console.print(f"  supported range: {check.range}")

    for message in check.messages:
        console.print(f"  [green]OK[/green]    {message}")
    for warning in check.warnings:
        console.print(f"  [yellow]WARN[/yellow]  {warning}")
    for error in check.errors:
        console.print(f"  [red]FAIL[/red]  {error}")

    if not check.supported and not allow_unsupported:
        raise typer.Exit(1)


@runtime_app.command("upgrade")
def runtime_upgrade(
    backend: str = typer.Argument(..., help="Runtime backend to upgrade"),
    stable: bool = typer.Option(False, "--stable", help="Install vserve's pinned stable runtime"),
):
    """Upgrade an external backend runtime managed by vserve."""
    if backend != "vllm":
        console.print(f"[red]Unsupported runtime upgrade backend: {backend}[/red]")
        console.print("  Available: vllm")
        raise typer.Exit(1)
    if not stable:
        console.print("[red]Choose the upgrade policy explicitly.[/red]")
        console.print("  Run: [cyan]vserve runtime upgrade vllm --stable[/cyan]")
        raise typer.Exit(1)

    lock = _lock_or_exit("runtime", "upgrading vLLM runtime")
    try:
        from vserve.backends import get_backend_by_name

        try:
            backend_obj = get_backend_by_name("vllm")
            if backend_obj.is_running():
                console.print("[red]Stop vLLM before upgrading the runtime.[/red]")
                console.print("  Run: [cyan]vserve stop[/cyan]")
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as exc:
            console.print("[red]Could not confirm vLLM is stopped.[/red]")
            console.print(f"  {exc}")
            raise typer.Exit(1) from None

        from vserve.runtime import (
            PINNED_STABLE_VLLM,
            check_vllm_compatibility,
            collect_vllm_runtime_info,
            upgrade_vllm_stable,
        )

        try:
            upgrade_vllm_stable()
        except RuntimeError as exc:
            console.print("[red]vLLM runtime upgrade failed.[/red]")
            console.print(f"  {exc}")
            raise typer.Exit(1) from None

        info = collect_vllm_runtime_info()
        check = check_vllm_compatibility(info)
        if check.supported:
            console.print(f"[green]vLLM runtime upgraded[/green] to {info.vllm_version or PINNED_STABLE_VLLM}")
            return
        console.print(f"[yellow]vLLM install completed, but runtime is not supported ({info.vllm_version or '?'}).[/yellow]")
        for error in check.errors:
            console.print(f"  [red]FAIL[/red]  {error}")
        raise typer.Exit(1)
    finally:
        lock.release()


def _refresh_banner() -> None:
    """Silently update the login banner if installed."""
    from pathlib import Path as _P
    banner_src = _P(__file__).parent / "welcome.sh"
    banner_dest = CONFIG_FILE.parent / "welcome.sh"
    if banner_dest.exists() and banner_src.exists():
        import shutil
        shutil.copy2(banner_src, banner_dest)


@app.command(name="list")
def list_models(model_terms: list[str] = typer.Argument(None, help="Model name terms for detail view")):
    """List downloaded models."""
    from vserve.backends import _BACKENDS

    model = _join_model_terms(model_terms)
    all_models = _all_models()
    if not all_models:
        console.print("[dim]No models found.[/dim] Run: vserve add")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    _backend_colors = {"vLLM": "cyan", "llama.cpp": "magenta"}

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Backend")
    table.add_column("Disk", justify="right")
    table.add_column("Limits")
    table.add_column("Max Context", justify="right")
    table.add_column("Tools", style="green")
    table.add_column("Reasoning", style="green")

    for m in all_models:
        lim = read_limits(limits_path(m.provider, m.model_name))

        # Detect backend
        backend_name = "\u2014"
        backend_obj = None
        for b in _BACKENDS:
            if b.can_serve(m):
                backend_name = b.display_name
                backend_obj = b
                break

        max_ctx = "\u2014"
        if lim:
            for ctx_str in sorted(lim.get("limits", {}).keys(), key=lambda x: int(x), reverse=True):
                entry = lim["limits"][ctx_str]
                if isinstance(entry, dict):
                    if any(v is not None for v in entry.values()):
                        max_ctx = f"{int(ctx_str) // 1024}k"
                        break
                elif entry is not None:
                    max_ctx = f"{int(ctx_str) // 1024}k"
                    break

        # Use cached limits for tool/reasoning info (fast) — only live-detect for untuned models
        tp = "\u2014"
        rp = "\u2014"
        if lim:
            # vLLM: parser names in limits
            tp = lim.get("tool_call_parser") or ("jinja" if lim.get("supports_tools") else "\u2014")
            rp = lim.get("reasoning_parser") or ("jinja" if lim.get("supports_reasoning") else "\u2014")
        elif backend_obj:
            # Untuned — live detect (slow for GGUF, but only for untuned models)
            tool_info = backend_obj.detect_tools(m.path)
            tp = tool_info.get("tool_call_parser") or ("jinja" if tool_info.get("supports_tools") else "\u2014")
            rp = tool_info.get("reasoning_parser") or ("jinja" if tool_info.get("supports_reasoning") else "\u2014")

        bc = _backend_colors.get(backend_name, "white")
        table.add_row(
            m.full_name, f"[{bc}]{backend_name}[/{bc}]", f"{m.model_size_gb} GB",
            "\u2713" if lim else "\u2717", max_ctx, tp, rp,
        )

    console.print(table)


@app.command(name="ls", hidden=True)
def ls(model_terms: list[str] = typer.Argument(None, help="Model name terms for detail view")):
    """List downloaded models (alias for list)."""
    list_models(model_terms)


def _show_model_detail(m: ModelInfo):
    lim = read_limits(limits_path(m.provider, m.model_name))

    console.print(f"\n[bold]{m.full_name}[/bold]")
    console.print(f"  Arch: {m.architecture}  Quant: {m.quant_method or 'none'}  MoE: {m.is_moe}")
    console.print(f"  Weight files: {m.model_size_gb} GB  Max positions: {m.max_position_embeddings}\n")

    if not lim:
        console.print(f"  [dim]Not probed yet.[/dim] Run: vserve tune {m.model_name.lower()}\n")
        return

    limits = lim.get("limits", {})
    is_flat = any(isinstance(v, (int, type(None))) for v in limits.values())

    if is_flat:
        table = Table(title="Context / Concurrency Limits")
        table.add_column("Context", justify="right")
        table.add_column("Parallel slots", justify="right")
        for ctx_str in sorted(limits.keys(), key=int):
            entry = limits[ctx_str]
            slot_str = str(entry) if entry is not None else "OOM"
            table.add_row(f"{int(ctx_str) // 1024}k", slot_str)
    else:
        table = Table(title="Context / Concurrency Limits")
        table.add_column("Context", justify="right")
        dtype_order = _vllm_limit_dtype_order(lim, limits)
        for dtype in dtype_order:
            table.add_column(_vllm_kv_label(dtype), justify="right")
        for ctx_str in sorted(limits.keys(), key=int):
            entry = limits[ctx_str]
            choices = _vllm_limits_entry(entry)
            row = [f"{int(ctx_str) // 1024}k"]
            for dtype in dtype_order:
                value = choices.get(dtype)
                row.append(str(value) if value is not None else "OOM")
            table.add_row(*row)

    console.print(table)
    console.print()


def _is_interactive() -> bool:
    """True when stdin is a real terminal (not CI, not CliRunner)."""
    import sys
    return sys.stdin.isatty()


def _has_gum() -> bool:
    import shutil
    return shutil.which("gum") is not None


def _restore_terminal() -> None:
    """Reset terminal to sane state after menu tools may have altered it."""
    import sys
    try:
        import termios
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


# Save terminal state at import time so we can restore after menus
try:
    import sys as _sys
    import termios as _termios
    _SAVED_TERM_ATTRS = _termios.tcgetattr(_sys.stdin.fileno())
except Exception:
    _SAVED_TERM_ATTRS = None  # type: ignore[assignment]

# Enable readline for input() — arrow keys, history, backspace
try:
    import readline as _readline  # noqa: F401
except ImportError:
    pass


def _pick(items: list[str], title: str = "") -> int | None:
    """Single-select menu. Returns index or None on cancel.

    Uses gum → simple-term-menu → numbered prompt (best available).
    """
    if not _is_interactive():
        # CI / CliRunner fallback
        for i, item in enumerate(items, 1):
            console.print(f"  {i}) {item}")
        while True:
            choice = typer.prompt(title or "Choice")
            try:
                n = int(choice)
                if 1 <= n <= len(items):
                    return n - 1
            except ValueError:
                pass
            console.print(f"[red]Enter a number 1-{len(items)}.[/red]")

    if _has_gum():
        import subprocess
        cmd = ["gum", "choose", "--cursor.foreground", "6", "--item.faint"]
        if title:
            cmd.extend(["--header", title])
        cmd.extend(items)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        _restore_terminal()
        if r.returncode != 0 or not r.stdout.strip():
            return None
        selected = r.stdout.strip()
        for i, item in enumerate(items):
            if item == selected:
                return i
        return None

    try:
        from simple_term_menu import TerminalMenu  # type: ignore[import-untyped]
        menu = TerminalMenu(
            items, title=title,
            menu_cursor="❯ ",
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("standout",),
            cycle_cursor=True,
            status_bar="  ↑↓ navigate · enter select · q cancel",
            status_bar_style=("fg_gray",),
        )
        idx = menu.show()
        _restore_terminal()
        return idx  # type: ignore[return-value]
    except Exception:
        _restore_terminal()

    # Final fallback: numbered prompt
    for i, item in enumerate(items, 1):
        console.print(f"  {i}) {item}")
    while True:
        choice = typer.prompt(title or "Choice")
        try:
            n = int(choice)
            if 1 <= n <= len(items):
                return n - 1
        except ValueError:
            pass
        console.print(f"[red]Enter a number 1-{len(items)}.[/red]")


def _parse_multi_select(answer: str, item_count: int) -> list[int] | None:
    """Parse comma/space-separated menu indices.

    Returns None when any token is invalid so callers can re-prompt instead of
    silently dropping part of the user's selection.
    """
    text = answer.strip()
    if not text:
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for part in text.replace(",", " ").split():
        try:
            n = int(part)
        except ValueError:
            return None
        if not 1 <= n <= item_count:
            return None
        idx = n - 1
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _pick_many(items: list[str], title: str = "", *, require_selection: bool = False) -> list[int]:
    """Multi-select menu. Returns list of indices.

    Uses gum → simple-term-menu → numbered prompt (best available).
    """
    def prompt_for_selection() -> None:
        console.print("[yellow]Select at least one item, or cancel to go back.[/yellow]")

    if not _is_interactive():
        for i, item in enumerate(items, 1):
            console.print(f"  {i}) {item}")
        while True:
            answer = typer.prompt(title or "Select (e.g. 1 or 1,2; blank cancels)", default="")
            parsed = _parse_multi_select(answer, len(items))
            if parsed is not None:
                return parsed
            console.print(
                f"[red]Enter one or more numbers 1-{len(items)}, separated by commas or spaces.[/red]",
            )

    if _has_gum():
        import subprocess
        while True:
            cmd = [
                "gum", "choose", "--no-limit",
                "--cursor.foreground", "6",
                "--selected.foreground", "2", "--selected.bold",
            ]
            if title:
                cmd.extend(["--header", title])
            cmd.extend(items)
            r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            _restore_terminal()
            if r.returncode != 0:
                return []  # cancelled or nothing selected → back
            if not r.stdout.strip():
                if require_selection:
                    prompt_for_selection()
                    continue
                return []
            selected_lines = {ln.strip() for ln in r.stdout.strip().split("\n")}
            selected_indices = [i for i, item in enumerate(items) if item.strip() in selected_lines]
            if selected_indices or not require_selection:
                return selected_indices
            prompt_for_selection()

    try:
        while True:
            from simple_term_menu import TerminalMenu
            menu = TerminalMenu(
                items, title=title,
                multi_select=True,
                show_multi_select_hint=True,
                menu_cursor="❯ ",
                menu_cursor_style=("fg_cyan", "bold"),
                multi_select_cursor_style=("fg_green", "bold"),
                menu_highlight_style=("standout",),
                cycle_cursor=True,
                status_bar="  ↑↓ navigate · x/space toggle · enter confirm · q cancel",
                status_bar_style=("fg_gray",),
            )
            result = menu.show()
            _restore_terminal()
            if result is None:
                return []
            selected_indices = list(result) if isinstance(result, tuple) else [result]
            if selected_indices or not require_selection:
                return selected_indices
            prompt_for_selection()
    except Exception:
        _restore_terminal()
        pass

    # Final fallback: comma-separated prompt
    for i, item in enumerate(items, 1):
        console.print(f"  {i}) {item}")
    while True:
        answer = typer.prompt(title or "Select (e.g. 1 or 1,2; blank cancels)", default="")
        parsed = _parse_multi_select(answer, len(items))
        if parsed is not None:
            return parsed
        console.print(
            f"[red]Enter one or more numbers 1-{len(items)}, separated by commas or spaces.[/red]",
        )


@app.command()
def add(model_id: str = typer.Argument(None, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-27B-FP8)")):
    """Search and download a model from HuggingFace."""
    from huggingface_hub import snapshot_download, HfApi
    from vserve.config import cfg as _cfg

    models_dir = _cfg().models_dir
    api = HfApi()

    # Check if argument is an existing repo
    if model_id and "/" in model_id:
        try:
            exists = api.repo_exists(model_id, repo_type="model")
        except Exception as e:
            console.print(f"[red]HuggingFace error:[/red] {e}")
            raise typer.Exit(1)
        if exists:
            _download_model(model_id, models_dir, snapshot_download, api)
            return
        # Not found — fall through to keyword search

    # Search loop
    query = model_id or ""
    results: list = []
    while True:
        if not query:
            try:
                console.print()
                query = input("  Search HuggingFace (Ctrl-C to quit): ")
            except (KeyboardInterrupt, EOFError):
                console.print()
                return
            if not query:
                return

        # Only re-search if we don't already have results for this query
        if not results:
            console.print(f"[dim]Searching '{query}'...[/dim]")
            try:
                results = list(api.list_models(search=query, sort="downloads", limit=20))
            except Exception as e:
                console.print(f"[red]HuggingFace error:[/red] {e}")
                query = ""
                continue

            if not results:
                console.print(f"[yellow]No results for '{query}'.[/yellow]")
                query = ""
                continue

        lines = []
        for m in results:
            dl = f"{m.downloads:,}" if m.downloads else "?"
            lines.append(f"{m.id}  ({dl} downloads)")

        idx = _pick(lines, title=f"Results for '{query}':")
        if idx is None:
            console.clear()
            query = ""
            results = []
            continue

        downloaded = _download_model(results[idx].id, models_dir, snapshot_download, api)
        if not downloaded:
            console.clear()
            continue  # backed out of variant picker — re-show results
        # Offer to download another
        if not typer.confirm("\n  Download another model?", default=False):
            return
        console.clear()
        query = ""
        results = []


def _pick_variants(variants: list) -> list:
    """Interactive variant selection. Returns list of selected Variant objects."""
    from vserve.variants import format_variant_line

    # Single variant — auto-select, no picker needed
    if len(variants) == 1:
        console.print(f"  {format_variant_line(variants[0], index=1)}")
        return variants

    lines = [format_variant_line(v, index=i) for i, v in enumerate(variants, 1)]
    indices = _pick_many(
        lines,
        title="Select variant(s) — x/space to toggle, enter to confirm:",
        require_selection=True,
    )
    return [variants[i] for i in indices]


def _materialize_subdirectory_variant(
    local_dir: pathlib.Path,
    *,
    model_name: str,
    selected_variants: list,
    shared: dict[str, int],
) -> pathlib.Path:
    """Expose a single subdirectory HF variant as its own runnable model root."""
    materialized = _materialize_subdirectory_variants(
        local_dir,
        model_name=model_name,
        selected_variants=selected_variants,
        shared=shared,
    )
    return materialized[0] if materialized else local_dir


def _root_has_top_level_weights(path: pathlib.Path) -> bool:
    from vserve.model_files import iter_top_level_files_with_suffix

    return (
        bool(iter_top_level_files_with_suffix(path, ".safetensors"))
        or bool(iter_top_level_files_with_suffix(path, ".bin"))
        or bool(iter_top_level_files_with_suffix(path, ".gguf"))
    )


def _variant_common_prefix(files: list[str]) -> str | None:
    split_paths = [pathlib.PurePosixPath(name).parts for name in files]
    if not split_paths or not all(len(parts) > 1 for parts in split_paths):
        return None
    prefix = split_paths[0][0]
    if not all(parts[0] == prefix for parts in split_paths):
        return None
    return prefix


def _materialize_subdirectory_variants(
    local_dir: pathlib.Path,
    *,
    model_name: str,
    selected_variants: list,
    shared: dict[str, int],
) -> list[pathlib.Path]:
    """Expose each selected subdirectory HF variant as a separate runnable model root."""
    if not selected_variants:
        return [local_dir]
    if len(selected_variants) == 1 and _root_has_top_level_weights(local_dir):
        ignore = local_dir / ".vserve-ignore"
        if ignore.exists():
            ignore.unlink()
        return [local_dir]

    import re
    import shutil

    def link_or_copy(src: pathlib.Path, dest: pathlib.Path) -> None:
        if not src.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
        except OSError:
            return
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)

    roots: list[pathlib.Path] = []
    materialized_any = False
    materialize_top_level = len(selected_variants) > 1
    for variant in selected_variants:
        variant_files = list(getattr(variant, "files", {}).keys())
        prefix = _variant_common_prefix(variant_files)
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(getattr(variant, "label", prefix or "variant"))).strip("-") or (prefix or "variant")
        if prefix is None and not materialize_top_level:
            roots.append(local_dir)
            continue

        materialized = local_dir.parent / f"{model_name}-{safe_label}"
        materialized.mkdir(parents=True, exist_ok=True)
        materialized_any = True

        for name in shared:
            rel = pathlib.PurePosixPath(name)
            dest_rel = (
                pathlib.PurePosixPath(*rel.parts[1:])
                if prefix is not None and rel.parts and rel.parts[0] == prefix and len(rel.parts) > 1
                else rel
            )
            link_or_copy(local_dir.joinpath(*rel.parts), materialized.joinpath(*dest_rel.parts))

        for name in variant_files:
            rel = pathlib.PurePosixPath(name)
            stripped = pathlib.PurePosixPath(*rel.parts[1:]) if prefix is not None else rel
            link_or_copy(local_dir.joinpath(*rel.parts), materialized.joinpath(*stripped.parts))

        roots.append(materialized)

    ignore = local_dir / ".vserve-ignore"
    if materialized_any and (materialize_top_level or not _root_has_top_level_weights(local_dir)):
        ignore.write_text("materialized variants live in sibling model roots\n")
    elif ignore.exists() and _root_has_top_level_weights(local_dir):
        ignore.unlink()
    return roots or [local_dir]


def _safe_variant_label(label: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "variant"


def _gguf_variant_root(base_dir: pathlib.Path, *, model_name: str, variant) -> pathlib.Path:
    label = _safe_variant_label(str(getattr(variant, "label", "variant")))
    suffix = label if label.lower() not in model_name.lower() else "variant"
    return base_dir.parent / f"{model_name}-{suffix}"


def _variant_contains_gguf(variant) -> bool:
    from vserve.model_files import is_gguf_name

    return any(is_gguf_name(filename) for filename in getattr(variant, "files", {}))


def _expected_download_roots(
    local_dir: pathlib.Path,
    *,
    model_name: str,
    selected_variants: list,
    is_gguf_download: bool,
) -> list[pathlib.Path]:
    if is_gguf_download:
        return [
            _gguf_variant_root(local_dir, model_name=model_name, variant=variant)
            for variant in selected_variants
        ]
    return [local_dir]


def _download_roots_ready(roots: list[pathlib.Path]) -> bool:
    if not roots:
        return False
    from vserve.model_files import iter_recursive_weight_files

    for root in roots:
        try:
            if not root.exists() or not iter_recursive_weight_files(root):
                return False
        except OSError:
            return False
    return True


def _strip_downloaded_file_prefix(downloaded: pathlib.Path, root: pathlib.Path, filename: str) -> pathlib.Path:
    """Move a downloaded repo subpath to the runnable root with only its basename."""
    dest = root / pathlib.PurePosixPath(filename).name
    if downloaded.resolve(strict=False) == dest.resolve(strict=False):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    import shutil

    shutil.move(str(downloaded), str(dest))
    parent = downloaded.parent
    while parent != root and root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return dest


def _clear_stale_gguf_files(root: pathlib.Path) -> None:
    if not root.exists():
        return
    from vserve.model_files import iter_recursive_files_with_suffix

    for stale in iter_recursive_files_with_suffix(root, ".gguf"):
        try:
            stale.unlink()
        except OSError:
            pass


def _download_model(model_id: str, models_dir: "pathlib.Path", snapshot_download: object, api: object) -> bool:
    """Download a single model by its HuggingFace ID. Returns True if download happened."""
    from vserve.variants import fetch_repo_variants, _format_bytes

    parts = model_id.split("/")
    if len(parts) != 2:
        console.print(f"[red]Invalid model ID '{model_id}'.[/red] Expected: provider/model-name")
        raise typer.Exit(1)

    provider, model_name = parts

    # Determine destination — will be updated after variant selection for GGUF
    local_dir = models_dir / provider / model_name
    is_gguf_download = False

    if local_dir.exists() and any(local_dir.iterdir()):
        console.print(f"[yellow]{model_id} already exists at {local_dir}[/yellow]")
        if not typer.confirm("Re-download?", default=False):
            return True  # not a back-navigation, user chose to keep existing

    # --- Variant picker ---
    console.print(f"\n[bold]{model_id}[/bold]\n")
    console.print("[dim]Fetching file list...[/dim]")

    try:
        variants, shared = fetch_repo_variants(model_id, api)
    except Exception as e:
        console.print(f"[red]Failed to fetch file list:[/red] {e}")
        return False

    if not variants:
        console.print("[yellow]No weight files found in this repo.[/yellow]")
        return False

    # Show shared files summary
    shared_size = sum(shared.values())
    shared_names = ", ".join(list(shared.keys())[:3])
    if len(shared) > 3:
        shared_names += ", ..."
    console.print(f"  Shared: {shared_names} ({_format_bytes(shared_size)}, {len(shared)} files)\n")

    # Selection (variants shown inside _pick_variants via _pick_many)
    selected_variants = _pick_variants(variants)
    if not selected_variants:
        console.clear()
        return False  # user backed out — navigate back

    selected_gguf_flags = [_variant_contains_gguf(v) for v in selected_variants]
    if any(selected_gguf_flags) and not all(selected_gguf_flags):
        console.print("[red]Cannot download GGUF and non-GGUF variants into one runnable model root.[/red]")
        console.print("  Choose GGUF variants for llama.cpp, or choose safetensors/bin variants for vLLM.")
        raise typer.Exit(1)

    # Confirmation
    total = sum(v.total_bytes for v in selected_variants) + shared_size
    console.print(f"\n  Total download: [bold]{_format_bytes(total)}[/bold]")
    if not typer.confirm("  Download?", default=True):
        return False  # user declined — navigate back

    # Build allow_patterns
    allow_files: list[str] = list(shared.keys())
    for v in selected_variants:
        allow_files.extend(v.files.keys())

    # Check if this is a GGUF download — route to llama-cpp models dir
    is_gguf_download = all(selected_gguf_flags)
    if is_gguf_download:
        from vserve.config import cfg as _cfg2
        lc_root = _cfg2().llamacpp_root
        if lc_root:
            local_dir = lc_root / "models" / provider / model_name
        else:
            console.print("[yellow]llama.cpp not configured — downloading GGUF to default models dir[/yellow]")

    console.print(f"\n[bold]Downloading[/bold] {model_id}")
    console.print(f"  To: {local_dir}\n")
    downloaded_dirs: list[pathlib.Path] = []

    lock_name = f"download-{provider}--{model_name}"
    lock = VserveLock(lock_name, f"downloading {model_id}")
    try:
        lock.acquire()
    except PermissionError as exc:
        console.print(Panel(
            f"[bold]{exc}[/bold]",
            title=f"[red]vserve {__version__}: permission denied[/red]",
            border_style="red",
        ))
        raise typer.Exit(1) from None
    except LockHeld as exc:
        import os
        me = os.environ.get("USER", "?")
        if exc.info and exc.info.user != me:
            notify_user(
                exc.info.user,
                f"{me} is also waiting for {model_id} download",
            )
        console.print(f"[yellow]{model_id} is being downloaded by {exc.info.user if exc.info else 'another user'}.[/yellow]")
        console.print("[dim]Waiting for it to finish...[/dim]")
        from vserve.lock import wait_for_release
        if not wait_for_release(lock_name, timeout=7200):
            console.print("[red]Timed out waiting for download.[/red]")
            raise typer.Exit(1)
        ready_roots = _expected_download_roots(
            local_dir,
            model_name=model_name,
            selected_variants=selected_variants,
            is_gguf_download=is_gguf_download,
        )
        if _download_roots_ready(ready_roots):
            try:
                from vserve.models import detect_model
                info = detect_model(ready_roots[0])
            except Exception:
                info = None
            if info:
                console.print(f"\n[green]{info.full_name} is ready[/green] (downloaded by {exc.info.user if exc.info else 'another user'})")
            else:
                console.print(f"\n[green]{model_id} is ready.[/green]")
            return True
        console.print("[yellow]Download seems to have failed. Retrying...[/yellow]")
        # Reacquire the lock for the retry
        try:
            lock.acquire()
        except LockHeld:
            console.print("[yellow]Another download started. Exiting.[/yellow]")
            return True
        except PermissionError as exc:
            console.print(Panel(
                f"[bold]{exc}[/bold]",
                title=f"[red]vserve {__version__}: permission denied[/red]",
                border_style="red",
            ))
            raise typer.Exit(1) from None

    try:
        if is_gguf_download:
            from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
            from vserve.model_files import is_gguf_name
            for variant in selected_variants:
                variant_dir = _gguf_variant_root(
                    local_dir,
                    model_name=model_name,
                    variant=variant,
                )
                variant_dir.mkdir(parents=True, exist_ok=True)
                _clear_stale_gguf_files(variant_dir)
                downloaded_dirs.append(variant_dir)
                for filename in variant.files:
                    if is_gguf_name(filename):
                        downloaded_path = pathlib.Path(
                            hf_hub_download(repo_id=model_id, filename=filename, local_dir=variant_dir)
                        )
                        _strip_downloaded_file_prefix(downloaded_path, variant_dir, filename)
                # Also grab tokenizer_config.json and shared non-GGUF files in each runnable root.
                for filename in ["tokenizer_config.json", *shared.keys()]:
                    if is_gguf_name(filename):
                        continue
                    try:
                        downloaded_path = pathlib.Path(
                            hf_hub_download(repo_id=model_id, filename=filename, local_dir=variant_dir)
                        )
                        if "/" in filename:
                            _strip_downloaded_file_prefix(downloaded_path, variant_dir, filename)
                    except Exception:
                        pass
        else:
            snapshot_download(  # type: ignore[operator]
                repo_id=model_id,
                local_dir=local_dir,
                allow_patterns=allow_files,
            )
            materialized_dirs = _materialize_subdirectory_variants(
                local_dir,
                model_name=model_name,
                selected_variants=selected_variants,
                shared=shared,
            )
            downloaded_dirs.extend(materialized_dirs)
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        lock.release()

    for downloaded_dir in downloaded_dirs or [local_dir]:
        try:
            from vserve.models import detect_model
            info = detect_model(downloaded_dir)
        except Exception:
            info = None
        if info:
            console.print(f"\n[green]Downloaded {info.full_name}[/green]")
            console.print(f"  Weight files: {info.model_size_gb:.1f} GB")
            console.print(f"  Quant: {info.quant_method or 'none'}")
            if info.max_position_embeddings:
                console.print(f"  Context: {info.max_position_embeddings:,} tokens")

            _auto_tune_downloaded_model(info)
        else:
            console.print(f"\n[green]Downloaded to {downloaded_dir}[/green]")
    return True


def _auto_tune_downloaded_model(info: ModelInfo) -> None:
    """Analytically tune after download without starting a serving backend."""
    try:
        from vserve.backends import get_backend
        from vserve.gpu import get_gpu_info, resolve_gpu_memory_utilization
        from vserve.config import cfg as _cfg3, write_limits
        from vserve.runtime import build_tuning_fingerprint

        backend = get_backend(info)
        gpu = get_gpu_info()
        gpu_mem_util = resolve_gpu_memory_utilization(gpu.vram_total_gb, config=_cfg3())
        console.print(f"\n[dim]Auto-tuning for {gpu.name}...[/dim]")
        limits_data = backend.tune(info, gpu, gpu_mem_util=gpu_mem_util)
        runtime_info = None
        runtime_info_fn = getattr(backend, "runtime_info", None)
        if callable(runtime_info_fn):
            try:
                runtime_info = runtime_info_fn()
            except Exception:
                runtime_info = None
        limits_data["backend"] = backend.name
        limits_data["fingerprint"] = build_tuning_fingerprint(
            model_info=info,
            gpu=gpu,
            backend=backend.name,
            gpu_mem_util=gpu_mem_util,
            runtime_info=runtime_info,
        )
        lim_path = limits_path(info.provider, info.model_name)
        write_limits(lim_path, limits_data)
        console.print(f"[green]Tuned.[/green] Ready: vserve run {info.model_name}")
        if backend.name in {"vllm", "llamacpp"}:
            console.print(f"[dim]Benchmark later: vserve tune {info.model_name} --bench[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Auto-tune skipped: {e}[/yellow]")
        console.print(f"  Run: vserve tune {info.model_name}")


def _model_rm_impl(model: str | None, force: bool = False):
    """Shared implementation for model rm / model remove."""
    import shutil

    if model is None:
        all_models = _all_models()
        if not all_models:
            console.print("[dim]No models to remove.[/dim]")
            return
        name_w = max(len(m.full_name) for m in all_models)
        size_w = max(len(f"{m.model_size_gb} GB") for m in all_models)
        items = [
            f"{m.full_name:<{name_w}}  {f'{m.model_size_gb} GB':>{size_w}}"
            for m in all_models
        ]
        idx = _pick(items, title="Remove which model?")
        if idx is None:
            return
        m = all_models[idx]
    else:
        m = _resolve_model(model)

    size_gb = m.model_size_gb
    console.print(f"\n  [bold]{m.full_name}[/bold]")
    console.print(f"  Path: {m.path}")
    console.print(f"  Size: {size_gb} GB\n")

    if not force:
        if not typer.confirm(f"  Delete {m.full_name}?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Delete model directory
    shutil.rmtree(m.path)
    console.print(f"[green]Deleted {m.full_name}[/green] ({size_gb} GB)")

    # Clean up cached limits
    lim = limits_path(m.provider, m.model_name)
    if lim.exists():
        lim.unlink()
        console.print("  [dim]Removed limits cache[/dim]")

    # Clean up profile configs
    prof = profile_path(m.provider, m.model_name, "custom")
    if prof.exists():
        prof.unlink()
        console.print("  [dim]Removed profile config[/dim]")

    # Remove empty parent dir (provider dir) if it's now empty
    parent = m.path.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


@app.command()
def rm(
    model_terms: list[str] = typer.Argument(None, help="Model name terms (fuzzy match)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a downloaded model."""
    _model_rm_impl(_join_model_terms(model_terms), force=force)


@app.command(hidden=True)
def remove(
    model_terms: list[str] = typer.Argument(None, help="Model name terms (fuzzy match)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a downloaded model (alias for rm)."""
    _model_rm_impl(_join_model_terms(model_terms), force=force)


@app.command()
def tune(
    model_terms: list[str] = typer.Argument(None, help="Model name terms (fuzzy match)"),
    all_models: bool = typer.Option(False, "--all", help="Tune all downloaded models"),
    recalc: bool = typer.Option(False, "--recalc", help="Force recalculation even if cached"),
    gpu_util: float = typer.Option(None, "--gpu-util", help="GPU memory utilization (0.5-0.99, auto if omitted)"),
    bench: bool = typer.Option(False, "--bench", help="Run a bounded backend micro-benchmark after tuning"),
    bench_seconds: int = typer.Option(120, "--bench-seconds", help="Maximum benchmark seconds per model (30-600)"),
    bench_startup_seconds: int = typer.Option(300, "--bench-startup-seconds", help="Maximum backend startup seconds per benchmark candidate (60-900)"),
    bench_candidates: int = typer.Option(1, "--bench-candidates", help="Number of recommended profiles to benchmark (1-3)"),
    bench_requests: int = typer.Option(8, "--bench-requests", help="Requests per benchmark candidate (1-32)"),
):
    """Calculate context and concurrency limits for a model."""
    if gpu_util is not None and not 0.5 <= gpu_util <= 0.99:
        console.print("[red]--gpu-util must be between 0.5 and 0.99[/red]")
        raise typer.Exit(1)
    if not 30 <= bench_seconds <= 600:
        console.print("[red]--bench-seconds must be between 30 and 600[/red]")
        raise typer.Exit(1)
    if not 60 <= bench_startup_seconds <= 900:
        console.print("[red]--bench-startup-seconds must be between 60 and 900[/red]")
        raise typer.Exit(1)
    if not 1 <= bench_candidates <= 3:
        console.print("[red]--bench-candidates must be between 1 and 3[/red]")
        raise typer.Exit(1)
    if not 1 <= bench_requests <= 32:
        console.print("[red]--bench-requests must be between 1 and 32[/red]")
        raise typer.Exit(1)

    from vserve.gpu import get_gpu_info, compute_gpu_memory_utilization, resolve_gpu_memory_utilization
    from vserve.config import limits_cache_matches, write_limits
    from vserve.runtime import build_tuning_fingerprint

    # Resolve which models to tune
    model = _join_model_terms(model_terms)
    if all_models:
        models_to_tune = _all_models()
    elif model:
        m = _resolve_model(model)
        models_to_tune = [m]
    else:
        all_m = _all_models()
        if not all_m:
            console.print("[red]No models found.[/red] Run: vserve add")
            raise typer.Exit(1)
        items = [m.full_name for m in all_m] + ["All models"]
        idx = _pick(items, title="Which model?")
        if idx is None:
            raise typer.Exit(0)
        if idx == len(all_m):
            models_to_tune = all_m
        else:
            models_to_tune = [all_m[idx]]

    gpu = get_gpu_info()
    if gpu_util is None:
        from vserve.config import cfg as _cfg, save_config, reset_config
        _c = _cfg()
        if _c.gpu_memory_utilization is not None or _c.gpu_overhead_gb is not None:
            gpu_util = resolve_gpu_memory_utilization(gpu.vram_total_gb, config=_c)
        else:
            gpu_util = resolve_gpu_memory_utilization(gpu.vram_total_gb, config=_c)
            # Interactive menu only when user runs bare `vserve tune` (no model arg)
            if model is None and not all_models:
                console.print(f"\n[bold]GPU memory reservation[/bold]  [dim]({gpu.vram_total_gb:.0f} GB total, current: {gpu_util:.0%})[/dim]")
                console.print("  1) Auto (3.5 GB overhead) — use for this run")
                console.print("  2) Set GPU usage % (save to config)")
                console.print("  3) Set overhead in GB (save to config)")
                choice = typer.prompt("\n  Choice", default="1")
                if choice == "2":
                    val = typer.prompt("  GPU usage %", default=f"{gpu_util:.2f}")
                    gpu_util = max(0.5, min(0.99, float(val)))
                    _c.gpu_memory_utilization = gpu_util
                    _c.gpu_overhead_gb = None
                    save_config(_c)
                    reset_config()
                    console.print(f"  [green]Saved gpu_memory_utilization={gpu_util:.2%}[/green]")
                elif choice == "3":
                    val = typer.prompt("  Overhead GB", default="3.5")
                    overhead = max(0.5, float(val))
                    gpu_util = compute_gpu_memory_utilization(gpu.vram_total_gb, overhead)
                    _c.gpu_overhead_gb = overhead
                    _c.gpu_memory_utilization = None
                    save_config(_c)
                    reset_config()
                    console.print(f"  [green]Saved gpu_overhead_gb={overhead} GB[/green]")

    console.print(f"\n[bold]vserve tune[/bold]  [dim]{gpu.name} ({gpu.vram_total_gb:.0f} GB, util {gpu_util:.0%})[/dim]\n")

    for m in models_to_tune:
        from vserve.backends import get_backend
        try:
            backend = get_backend(m)
        except ValueError:
            console.print(f"[bold]{m.full_name}[/bold]  [red]No backend available[/red]")
            continue

        runtime_info = None
        runtime_info_fn = getattr(backend, "runtime_info", None)
        if callable(runtime_info_fn):
            try:
                runtime_info = runtime_info_fn()
            except Exception:
                runtime_info = None
        fingerprint = build_tuning_fingerprint(
            model_info=m,
            gpu=gpu,
            backend=backend.name,
            gpu_mem_util=gpu_util,
            runtime_info=runtime_info,
        )

        limits_data: dict | None = None
        used_cache = False

        # Check cached limits against the full versioned fingerprint.
        lim_path = limits_path(m.provider, m.model_name)
        if not recalc:
            existing = read_limits(lim_path)
            if existing is not None and limits_cache_matches(existing, backend=backend.name, fingerprint=fingerprint):
                limits_data = existing
                used_cache = True
                if not bench:
                    console.print(f"[bold]{m.full_name}[/bold]  [dim](cached — use --recalc to refresh)[/dim]")
                    _print_limits_table(existing, m)
                    continue

        if limits_data is None:
            # For vLLM backend, check architecture fields
            if backend.name == "vllm" and (m.num_kv_heads is None or m.head_dim is None or m.num_layers is None):
                console.print(f"[bold]{m.full_name}[/bold]")
                console.print("  [red]Missing architecture fields in config.json[/red]")
                console.print(f"  num_kv_heads={m.num_kv_heads}, head_dim={m.head_dim}, num_layers={m.num_layers}")
                continue

            try:
                limits_data = backend.tune(m, gpu, gpu_mem_util=gpu_util)
            except Exception as e:
                console.print(f"[bold]{m.full_name}[/bold]  [red]{e}[/red]")
                continue

        limits_data["backend"] = backend.name
        limits_data["fingerprint"] = fingerprint
        if bench:
            console.print(f"[bold]{m.full_name}[/bold]  [dim]({'cached + ' if used_cache else ''}{backend.display_name} benchmark)[/dim]")
            limits_data["benchmark_results"] = _run_tuning_benchmarks(
                m,
                backend,
                limits_data,
                gpu_mem_util=gpu_util,
                bench_seconds=bench_seconds,
                bench_candidates=bench_candidates,
                bench_requests=bench_requests,
                bench_startup_seconds=bench_startup_seconds,
            )
        write_limits(lim_path, limits_data)
        if not bench:
            console.print(f"[bold]{m.full_name}[/bold]  [dim]({m.model_size_gb} GB, {backend.display_name})[/dim]")
        _print_limits_table(limits_data, m)
        tp = limits_data.get("tool_call_parser")
        rp = limits_data.get("reasoning_parser")
        supports = limits_data.get("supports_tools")
        if tp or rp:
            parts = []
            if tp:
                parts.append(f"tools=[green]{tp}[/green]")
            if rp:
                parts.append(f"reasoning=[green]{rp}[/green]")
            console.print(f"  Capabilities: {' '.join(parts)}")
        elif supports:
            # llama.cpp: tools/reasoning supported via --jinja, no parser name needed
            caps = ["tool calling"]
            if limits_data.get("supports_reasoning"):
                caps.append("reasoning")
            console.print(f"  Capabilities: [green]{' + '.join(caps)} (--jinja)[/green]")
        else:
            from vserve.tools import supports_tools as _supports_tools
            if _supports_tools(m.path):
                console.print("  Capabilities: [yellow]tool markers found but parser unknown[/yellow]")
                console.print("                use --tools --tool-parser <parser> with vserve run")
            else:
                console.print("  Capabilities: [dim]no tool calling detected[/dim]")
        console.print(f"  [green]Saved to {lim_path}[/green]")

        # Offer pre-caching for vLLM (flashinfer JIT compilation)
        if backend.name == "vllm" and not model and not all_models:
            # Only offer pre-cache in interactive mode (bare `vserve tune`)
            try:
                from vserve.config import cfg as _cfg2
                fi_cache = _cfg2().vllm_root / ".cache" / "flashinfer"
                needs_precache = not fi_cache.is_dir() or not any(fi_cache.rglob("*.so"))
            except Exception:
                needs_precache = False
            if needs_precache:
                console.print("  [yellow]First vLLM start will compile flashinfer kernels (~5-10 min).[/yellow]")
                if typer.confirm("  Pre-cache now? (starts and stops vLLM once)", default=False):
                    console.print("  [dim]Pre-caching — this will take a few minutes...[/dim]")
                    lock = _lock_or_exit("gpu", f"pre-caching {backend.display_name}")
                    started_here = False
                    try:
                        from vserve.config import write_profile_yaml
                        # Build a minimal config just for pre-caching
                        precache_cfg = backend.build_config(m, {
                            "context": 4096, "kv_dtype": "auto", "slots": 1,
                            "batched_tokens": None, "gpu_mem_util": gpu_util,
                            "port": 8888, "tools": False, "tool_parser": None,
                            "reasoning_parser": None,
                        })
                        cfg_path = profile_path(m.provider, m.model_name, "precache")
                        write_profile_yaml(cfg_path, precache_cfg, comment="vserve tune — pre-cache")
                        _session_or_exit()
                        try:
                            if backend.is_running():
                                console.print("  [yellow]vLLM is already running — skipping pre-cache warmup.[/yellow]")
                                continue
                        except Exception as e:
                            console.print(f"  [yellow]Could not determine backend state for pre-cache: {e}[/yellow]")
                            continue
                        backend.start(cfg_path)
                        started_here = True
                        write_session(f"{m.full_name} precache")
                        import time
                        from urllib.request import urlopen
                        health = backend.health_url(8888)
                        for i in range(150):
                            time.sleep(2)
                            try:
                                resp = urlopen(health, timeout=2)
                                if resp.status == 200:
                                    console.print("  [green]Pre-cache complete.[/green]")
                                    break
                            except Exception:
                                pass
                            if i > 5 and not backend.is_running():
                                console.print("  [yellow]Pre-cache failed — will compile on first real start.[/yellow]")
                                break
                            if i > 0 and i % 15 == 0:
                                console.print(f"  [dim]compiling... ({i * 2}s)[/dim]")
                    except Exception as e:
                        console.print(f"  [yellow]Pre-cache failed: {e}[/yellow]")
                    finally:
                        if started_here:
                            try:
                                backend.stop()
                            except Exception as e:
                                console.print(f"  [yellow]Could not stop pre-cache backend cleanly: {e}[/yellow]")
                            clear_session()
                        lock.release()
        console.print()


def _benchmark_candidate_names(limits_data: dict, max_candidates: int) -> list[str]:
    recommendations = limits_data.get("recommendations")
    if not isinstance(recommendations, dict):
        return []
    ordered = ["interactivity", "balanced", "throughput"]
    return [
        name for name in ordered
        if isinstance(recommendations.get(name), dict)
    ][:max_candidates]


def _llamacpp_benchmark_candidates(m: ModelInfo, limits_data: dict, max_candidates: int) -> list[dict]:
    limits = limits_data.get("limits")
    if not isinstance(limits, dict):
        return []
    working: list[tuple[int, int]] = []
    for ctx_str, entry in limits.items():
        try:
            ctx = int(str(ctx_str))
        except ValueError:
            continue
        slots = _llamacpp_slots_from_limits_entry(entry)
        if slots is not None:
            working.append((ctx, int(slots)))
    if not working:
        return []
    working.sort(key=lambda item: item[0])
    n_gpu_layers = limits_data.get("n_gpu_layers")
    if not isinstance(n_gpu_layers, int) or isinstance(n_gpu_layers, bool):
        n_gpu_layers = limits_data.get("num_layers")
    if not isinstance(n_gpu_layers, int) or isinstance(n_gpu_layers, bool):
        n_gpu_layers = 0
    is_embedding = bool(limits_data.get("is_embedding") or m.is_embedding)
    pooling = limits_data.get("pooling") if isinstance(limits_data.get("pooling"), str) else None
    candidates: list[dict] = []
    for index, (ctx, slots) in enumerate(working[:max_candidates], start=1):
        candidates.append({
            "profile": "interactive" if index == 1 else f"context-{ctx}",
            "context": ctx,
            "parallel": min(max(1, slots), 4),
            "n_gpu_layers": n_gpu_layers,
            "embedding": is_embedding,
            "pooling": pooling,
        })
    return candidates


def _exception_is_startup_timeout(exc: Exception) -> bool:
    text = str(exc).lower()
    return "health" in text or "timeout" in text or "timed out" in text


def _measurement_succeeded(result: dict) -> bool:
    measurement = result.get("measurement")
    if not isinstance(measurement, dict):
        return False
    if measurement.get("status") != "ok":
        return False
    completed = measurement.get("requests_completed")
    return isinstance(completed, int) and completed > 0


def _wait_backend_stopped(backend, *, timeout_s: int) -> bool:
    import time

    deadline = time.monotonic() + max(1, timeout_s)
    while time.monotonic() < deadline:
        try:
            if not backend.is_running():
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _run_tuning_benchmarks(
    m: ModelInfo,
    backend,
    limits_data: dict,
    *,
    gpu_mem_util: float,
    bench_seconds: int,
    bench_candidates: int,
    bench_requests: int,
    bench_startup_seconds: int = 300,
) -> dict:
    """Start the serving backend briefly and measure recommended profiles within hard bounds."""
    from datetime import datetime, timezone
    import json
    from vserve.bench import run_openai_completion_benchmark, run_openai_embedding_benchmark
    from vserve.config import profile_path as _profile_path, write_profile_yaml

    started_at = datetime.now(timezone.utc).isoformat()
    backend_name = getattr(backend, "name", None)
    if backend_name not in {"vllm", "llamacpp"}:
        return {
            "status": "skipped",
            "reason": "benchmarks are currently implemented for vLLM and llama.cpp only",
            "benchmarked_at": started_at,
        }
    try:
        if backend.is_running():
            return {
                "status": "skipped",
                "reason": f"{backend.display_name} is already running",
                "benchmarked_at": started_at,
            }
    except Exception:
        return {
            "status": "skipped",
            "reason": "could not determine backend running state",
            "benchmarked_at": started_at,
        }

    results: list[dict] = []
    if backend_name == "vllm":
        candidate_names = _benchmark_candidate_names(limits_data, bench_candidates)
        recommendations = limits_data.get("recommendations")
        if not candidate_names or not isinstance(recommendations, dict):
            return {
                "status": "skipped",
                "reason": "no vLLM recommendations available to benchmark",
                "benchmarked_at": started_at,
            }
        candidates: list[dict] = []
        for name in candidate_names:
            rec = recommendations.get(name)
            if not isinstance(rec, dict):
                continue
            context = rec.get("context")
            kv_dtype = rec.get("kv_cache_dtype")
            max_num_seqs = rec.get("max_num_seqs")
            if not isinstance(context, int) or not isinstance(kv_dtype, str) or not isinstance(max_num_seqs, int):
                continue
            candidates.append({
                "profile": name,
                "context": context,
                "choices": {
                    "context": context,
                    "kv_dtype": kv_dtype,
                    "slots": min(max_num_seqs, 4),
                    "batched_tokens": rec.get("max_num_batched_tokens"),
                    "gpu_mem_util": gpu_mem_util,
                    "port": 8888,
                    "tools": False,
                    "tool_parser": None,
                    "reasoning_parser": None,
                    "performance_mode": rec.get("performance_mode"),
                    "optimization_level": rec.get("optimization_level"),
                    "block_size": rec.get("block_size"),
                    "enable_prefix_caching": True,
                },
                "summary": {
                    "profile": name,
                    "context": context,
                    "kv_cache_dtype": kv_dtype,
                    "max_num_seqs": min(max_num_seqs, 4),
                    "max_num_batched_tokens": rec.get("max_num_batched_tokens"),
                },
                "config_kind": "yaml",
            })
    else:
        candidates = []
        for candidate in _llamacpp_benchmark_candidates(m, limits_data, bench_candidates):
            choices = {
                "context": candidate["context"],
                "n_gpu_layers": candidate["n_gpu_layers"],
                "parallel": candidate["parallel"],
                "port": 8888,
                "tools": False,
                "embedding": candidate["embedding"],
                "pooling": candidate["pooling"],
            }
            candidates.append({
                "profile": candidate["profile"],
                "context": candidate["context"],
                "choices": choices,
                "summary": {
                    "profile": candidate["profile"],
                    "context": candidate["context"],
                    "parallel": candidate["parallel"],
                    "n_gpu_layers": candidate["n_gpu_layers"],
                },
                "config_kind": "json",
            })
        if not candidates:
            return {
                "status": "skipped",
                "reason": "no llama.cpp capacity limits available to benchmark",
                "benchmarked_at": started_at,
            }

    measurement_timeout_seconds = min(float(bench_seconds), 45.0)

    for candidate in candidates:
        name = str(candidate["profile"])
        choices = candidate["choices"]
        assert isinstance(choices, dict)
        port = 8888
        port_value = choices.get("port")
        if isinstance(port_value, int) and not isinstance(port_value, bool):
            port = port_value
        cfg = backend.build_config(m, choices)
        if candidate["config_kind"] == "yaml":
            cfg_path = _profile_path(m.provider, m.model_name, f"bench-{name}")
            write_profile_yaml(cfg_path, cfg, comment=f"vserve tune --bench — {name}")
        else:
            cfg_path = backend.root_dir / "configs" / "models" / f"{m.provider}--{m.model_name}.bench-{name}.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        candidate_result = dict(candidate["summary"])
        started_here = False
        try:
            started_here = True
            _launch_backend(
                backend,
                cfg_path,
                f"{m.full_name} bench {name}",
                non_interactive=True,
                replace=False,
                health_timeout_s=bench_startup_seconds,
            )
            is_embedding = bool(choices.get("embedding") or m.is_embedding)
            benchmark_fn = run_openai_embedding_benchmark if is_embedding else run_openai_completion_benchmark
            measurement = benchmark_fn(
                f"http://localhost:{port}",
                model=str(m.path),
                request_count=bench_requests,
                timeout_s=measurement_timeout_seconds,
            )
            candidate_result["measurement"] = measurement
        except Exception as exc:
            candidate_result["status"] = "startup_timeout" if _exception_is_startup_timeout(exc) else "error"
            candidate_result["error"] = str(exc)
        finally:
            if started_here:
                try:
                    backend.stop(non_interactive=True)
                except Exception as exc:
                    candidate_result["stop_error"] = str(exc)
                if not _wait_backend_stopped(backend, timeout_s=min(bench_startup_seconds, 120)):
                    candidate_result["stop_wait_timeout"] = True
                clear_session()
        results.append(candidate_result)

    if any(_measurement_succeeded(result) for result in results):
        status = "ok"
    elif any(result.get("status") == "startup_timeout" for result in results):
        status = "startup_timeout"
    elif results:
        status = "measurement_error"
    else:
        status = "skipped"
    return {
        "status": status,
        "benchmarked_at": started_at,
        "max_seconds": bench_seconds,
        "measurement_timeout_seconds": measurement_timeout_seconds,
        "startup_timeout_seconds": bench_startup_seconds,
        "candidate_limit": bench_candidates,
        "request_count": bench_requests,
        "results": results,
    }


def _print_benchmark_summary(limits_data: dict) -> None:
    benchmark_results = limits_data.get("benchmark_results")
    if not isinstance(benchmark_results, dict):
        return
    status = benchmark_results.get("status", "unknown")
    result_parts = []
    raw_results = benchmark_results.get("results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            profile = item.get("profile")
            measurement = item.get("measurement")
            if isinstance(profile, str) and isinstance(measurement, dict):
                tps = measurement.get("tokens_per_second")
                if tps is None:
                    tps = measurement.get("items_per_second")
                    rate_label = "items/s"
                else:
                    rate_label = "tok/s"
                p95 = measurement.get("p95_latency_ms")
                result_parts.append(f"{profile}: {tps} {rate_label}, p95 {p95} ms")
    suffix = f" ({'; '.join(result_parts)})" if result_parts else ""
    console.print(f"  [dim]Benchmark: {status}{suffix}[/dim]")


def _print_limits_table(limits_data: dict, m: "ModelInfo") -> None:
    """Print the context × concurrency limit table.

    Both backends emit a 2D `{ctx: {dtype: slots}}` matrix today. llama.cpp
    additionally exposes GPU layer offload, MoE expert-CPU offload, and a
    recommended KV dtype; vLLM exposes scheduler profile recommendations.
    """
    avail = limits_data.get("available_kv_gb")
    if avail is not None:
        console.print(f"  [dim]KV cache: {avail} GB available[/dim]")

    backend = limits_data.get("backend")
    limits = limits_data.get("limits", {})
    legacy_flat = any(isinstance(v, (int, type(None))) for v in limits.values())

    if backend == "llamacpp":
        n_layers = limits_data.get("n_gpu_layers")
        num_layers = limits_data.get("num_layers")
        if n_layers is not None and num_layers:
            offload = "full" if n_layers >= num_layers else f"{n_layers}/{num_layers} layers"
            console.print(f"  [dim]GPU offload: {offload}[/dim]")

    if legacy_flat:
        # Pre-0.5.8 llama.cpp cache shape: {"4096": 8, "8192": 4, ...}
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Context", style="bold")
        table.add_column("Parallel slots", justify="right")
        for ctx_str in sorted(limits, key=int):
            entry = limits[ctx_str]
            ctx_val = int(ctx_str)
            slot_str = f"{entry} slots" if entry else "[dim]OOM[/dim]"
            table.add_row(f"{ctx_val:,}", slot_str)
        console.print(table)
    else:
        # 2D matrix: {"4096": {"f16": 89, "q8_0": 178, ...}, ...}
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Context", style="bold")
        dtype_order = _vllm_limit_dtype_order(limits_data, limits)
        for dtype in dtype_order:
            table.add_column(_vllm_kv_label(dtype), justify="right")

        for ctx_str in sorted(limits, key=int):
            entry = limits[ctx_str]
            ctx_val = int(ctx_str)
            choices = _vllm_limits_entry(entry)
            row = [f"{ctx_val:,}"]
            for dtype in dtype_order:
                value = choices.get(dtype)
                row.append(f"{value} slots" if value else "[dim]OOM[/dim]")
            table.add_row(*row)
        console.print(table)

        if backend == "vllm":
            recommendations = limits_data.get("recommendations")
            if isinstance(recommendations, dict) and recommendations:
                rendered = []
                for name in ("interactivity", "balanced", "throughput"):
                    rec = recommendations.get(name)
                    if not isinstance(rec, dict):
                        continue
                    ctx = rec.get("context")
                    seqs = rec.get("max_num_seqs")
                    kv = rec.get("kv_cache_dtype")
                    bt = rec.get("max_num_batched_tokens")
                    if ctx and seqs and kv and bt:
                        rendered.append(f"{name}: {int(ctx):,}/{kv}/{seqs} seqs/{int(bt):,} batch")
                if rendered:
                    console.print(f"  [dim]Profiles: {'; '.join(rendered)}[/dim]")
        elif backend == "llamacpp":
            rec_kv = limits_data.get("recommended_kv_dtype")
            if rec_kv:
                console.print(
                    f"  [dim]Recommended KV: -ctk {rec_kv} -ctv {rec_kv} "
                    "(symmetric pair → fused Flash-Attention path)[/dim]"
                )
            moe = limits_data.get("moe")
            if isinstance(moe, dict) and moe.get("is_moe"):
                _print_llamacpp_moe_block(moe, dtype_order)

    _print_benchmark_summary(limits_data)
    console.print()


def _print_llamacpp_moe_block(moe: dict, dtype_order: list[str]) -> None:
    """Render the MoE expert-CPU-offload (`-ot`) info block under the matrix."""
    expert_count = moe.get("expert_count")
    expert_used = moe.get("expert_used_count")
    freed = moe.get("estimated_gpu_freed_gb")
    gpu_resident = moe.get("estimated_gpu_resident_gb")
    pattern = moe.get("ot_pattern")
    header_bits = [f"MoE: {expert_count} experts"]
    if expert_used:
        header_bits.append(f"{expert_used} active per token")
    if freed is not None:
        header_bits.append(f"-ot would free ~{freed} GB")
    if gpu_resident is not None:
        header_bits.append(f"GPU resident ~{gpu_resident} GB")
    console.print(f"  [dim]{' · '.join(header_bits)}[/dim]")
    note = moe.get("note")
    if note:
        console.print(f"  [dim]({note})[/dim]")
        return
    limits_with_ot = moe.get("limits_with_ot")
    if not isinstance(limits_with_ot, dict) or not limits_with_ot:
        return
    console.print(f"  [dim]With -ot \"{pattern}\":[/dim]")
    ot_table = Table(show_header=True, box=None, padding=(0, 2))
    ot_table.add_column("Context", style="bold")
    for dtype in dtype_order:
        ot_table.add_column(_vllm_kv_label(dtype), justify="right")
    for ctx_str in sorted(limits_with_ot, key=int):
        row_entry = limits_with_ot[ctx_str]
        choices = _vllm_limits_entry(row_entry)
        row = [f"{int(ctx_str):,}"]
        for dtype in dtype_order:
            value = choices.get(dtype)
            row.append(f"{value} slots" if value else "[dim]OOM[/dim]")
        ot_table.add_row(*row)
    console.print(ot_table)


def _wait_for_health(
    *,
    health_url: str,
    timeout_s: int,
    poll_s: int = 3,
    log_tail_fn,
    service_running_fn,
    sleep_fn=None,
    urlopen_fn=None,
):
    """Poll health_url until 200 / service dies / timeout, deduping log output.

    Returns one of: "ready", "stopped", "timeout". The caller renders the
    final-state banner so output is byte-stable across changes here.

    Dedup rule: the journal tail is printed only when its hash changes, so the
    user sees each fresh block once instead of every 3 s. Between unchanged
    polls a single "." (no newline) prints to confirm liveness. A
    "still starting (Ns)" line flushes every 30 s.
    """
    import hashlib
    import time
    from urllib.request import urlopen as _default_urlopen

    sleep_fn = sleep_fn or time.sleep
    urlopen_fn = urlopen_fn or _default_urlopen

    iterations = max(1, (timeout_s + poll_s - 1) // poll_s)
    last_tail_hash: str | None = None
    dots_printed = 0

    for i in range(iterations):
        sleep_fn(poll_s)
        elapsed_s = (i + 1) * poll_s

        try:
            with urlopen_fn(health_url, timeout=2) as resp:
                if resp.status == 200:
                    if dots_printed:
                        console.print()
                    return "ready"
        except Exception:
            pass

        tail = log_tail_fn()
        if tail:
            tail_hash = hashlib.sha256(tail.encode("utf-8", errors="replace")).hexdigest()
            if tail_hash != last_tail_hash:
                if dots_printed:
                    console.print()
                    dots_printed = 0
                console.print(f"  [dim]Latest service logs ({elapsed_s}s):[/dim]")
                for line in tail.splitlines():
                    console.print(f"    [dim]{line}[/dim]")
                last_tail_hash = tail_hash
            else:
                console.print(".", end="")
                dots_printed += 1

        if elapsed_s > 10 and service_running_fn() is False:
            if dots_printed:
                console.print()
            return "stopped"

        if elapsed_s % 30 == 0:
            if dots_printed:
                console.print()
                dots_printed = 0
            console.print(f"  [dim]still starting... ({elapsed_s}s)[/dim]")

    if dots_printed:
        console.print()
    return "timeout"


def _launch_backend(
    backend,
    cfg_path: "pathlib.Path",
    label: str,
    *,
    non_interactive: bool = False,
    replace: bool = False,
    health_timeout_s: int | None = None,
) -> None:
    """Stop any running backend, start with given config, wait for health."""
    import json
    import subprocess
    import time
    from vserve.backends import _BACKENDS, probe_running_backends
    from vserve.config import read_profile_yaml

    lock = _lock_or_exit("gpu", f"starting {backend.display_name} ({label})")

    # Read config (YAML for vLLM, JSON for llama.cpp)
    needs_precache = False
    resolved_health_timeout_s = health_timeout_s or 300
    health_poll_s = 3
    try:
        def _service_running_state() -> bool | None:
            try:
                return backend.is_running()
            except Exception:
                return None

        def _recent_service_log_tail() -> str | None:
            try:
                result = subprocess.run(
                    [
                        "journalctl",
                        "-u",
                        backend.service_name,
                        "--no-pager",
                        "-n",
                        "5",
                        "-o",
                        "cat",
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=1,
                )
            except Exception:
                return None
            if result.returncode != 0:
                return None
            tail = result.stdout.strip()
            return tail or None

        if str(cfg_path).endswith(".json"):
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception as exc:
                console.print(f"[red]Launch config unreadable:[/red] {exc}")
                console.print(f"  Config: {cfg_path}")
                raise typer.Exit(1)
            if not isinstance(cfg, dict):
                console.print("[red]Launch config unreadable:[/red] expected a JSON object")
                console.print(f"  Config: {cfg_path}")
                raise typer.Exit(1)
        else:
            try:
                cfg = read_profile_yaml(cfg_path) or {}
            except ValueError as exc:
                console.print(f"[red]Launch config unreadable:[/red] {exc}")
                console.print(f"  Config: {cfg_path}")
                raise typer.Exit(1)

        _session_or_exit(fail_on_probe_uncertainty=True)

        running_backends, probe_failed = probe_running_backends()
        if not running_backends and probe_failed:
            console.print("[red]Could not determine whether another backend is already running.[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)

        if running_backends:
            stop_candidates = list(running_backends)
            running_names = ", ".join(candidate.display_name for candidate in running_backends)
            same_backend_only = (
                len(running_backends) == 1
                and getattr(running_backends[0], "name", None) == backend.name
            )
            if probe_failed:
                stop_candidates = list(_BACKENDS)
                running_names = ", ".join(candidate.display_name for candidate in stop_candidates)
                console.print("[yellow]Backend state is uncertain; stopping all known backends before restart.[/yellow]")
                prompt = f"Stop {running_names} and start {backend.display_name}?"
            elif same_backend_only:
                console.print(f"[yellow]{backend.display_name} is already running.[/yellow]")
                prompt = "Stop and restart?"
            else:
                console.print(f"[yellow]Running backend(s): {running_names}.[/yellow]")
                prompt = f"Stop {running_names} and start {backend.display_name}?"
            if non_interactive and not replace:
                console.print(f"[red]{running_names} is already running.[/red]")
                console.print("  Re-run with [cyan]--replace[/cyan] to stop it and start this profile.")
                raise typer.Exit(1)
            if not replace:
                if not typer.confirm(prompt, default=False):
                    return
            for candidate in stop_candidates:
                try:
                    candidate.stop(non_interactive=non_interactive)
                except Exception as e:
                    console.print(f"[red]{e}[/red]")
                    console.print(f"  Check: sudo journalctl -u {candidate.service_name} --no-pager -n 20")
                    raise typer.Exit(1)
            for _ in range(15):
                time.sleep(2)
                remaining_backends, remaining_probe_failed = probe_running_backends()
                if not remaining_backends and not remaining_probe_failed:
                    break
            else:
                console.print("[red]Timed out waiting for running backends to stop before restart.[/red]")
                console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
                raise typer.Exit(1)

        # llama.cpp profiles record both ctx_size (the -c value = total context
        # across all slots) and ctx_per_slot (per-slot window the user asked
        # for). Display the per-slot number so the line matches the tune output.
        ctx = (
            cfg.get("max-model-len")
            or cfg.get("ctx_per_slot")
            or cfg.get("ctx_size")
            or cfg.get("ctx-size", "?")
        )
        ctx_d = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        launch_port = cfg.get("port", 8888)
        console.print(f"\n[bold]Starting[/bold] with [cyan]{label}[/cyan] ({backend.display_name})")
        console.print(f"  context: {ctx_d}")

        if backend.name == "vllm":
            from vserve.config import cfg as _cfg
            fi_cache = _cfg().vllm_root / ".cache" / "flashinfer"
            needs_precache = not fi_cache.is_dir() or not list(fi_cache.glob("**/*.so"))
            if needs_precache and health_timeout_s is None:
                resolved_health_timeout_s = 600
                console.print("  [yellow]First run — compiling kernels (~5-10 min).[/yellow]")

        try:
            backend.start(cfg_path, non_interactive=non_interactive)
        except RuntimeError as e:
            _record_backend_manifest(backend, cfg_path, label=label, status="failed", port=launch_port, error=str(e))
            console.print(f"[red]{e}[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)
        except Exception as e:
            _record_backend_manifest(backend, cfg_path, label=label, status="failed", port=launch_port, error=str(e))
            console.print(f"[red]Failed to start {backend.display_name}: {e}[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)

        write_session(label)

        port = launch_port
        health = backend.health_url(port)

        console.print(f"  [dim]Waiting for {health} ...[/dim]")
        outcome = _wait_for_health(
            health_url=health,
            timeout_s=resolved_health_timeout_s,
            poll_s=health_poll_s,
            log_tail_fn=_recent_service_log_tail,
            service_running_fn=_service_running_state,
        )
        if outcome == "ready":
            _record_backend_manifest(backend, cfg_path, label=label, status="ready", port=port)
            console.print(f"\n[bold green]{backend.display_name} is running[/bold green] at http://localhost:{port}/v1")
            console.print(f"  Config: {cfg_path}")
            console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f\n")
            return
        if outcome == "stopped":
            clear_session()
            _record_backend_manifest(
                backend,
                cfg_path,
                label=label,
                status="failed",
                port=port,
                error="service stopped before health endpoint became ready",
            )
            console.print(f"[red]Service stopped unexpectedly.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
            raise typer.Exit(1)

        service_running = _service_running_state()

        if service_running is False:
            clear_session()
            _record_backend_manifest(
                backend,
                cfg_path,
                label=label,
                status="failed",
                port=port,
                error="timed out waiting for health endpoint",
            )
            console.print(f"[red]Timed out waiting for health endpoint.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
            raise typer.Exit(1)
        if service_running is True:
            _record_backend_manifest(backend, cfg_path, label=label, status="warming", port=port)
            console.print("[yellow]Health endpoint is still warming up, but the service remains active.[/yellow]")
            if needs_precache:
                console.print("  [yellow]First run kernel compilation may still be in progress.[/yellow]")
            if non_interactive or not _is_interactive():
                console.print("  [yellow]Non-interactive run requires the API health check to pass.[/yellow]")
                console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f")
                console.print(f"  Health: {health}")
                raise typer.Exit(1)
            console.print(f"  Returning while startup finishes in the background for {label}.")
            console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f")
            console.print(f"  Health: {health}")
            return
        _record_backend_manifest(
            backend,
            cfg_path,
            label=label,
            status="failed",
            port=port,
            error="service state could not be confirmed",
        )
        console.print(f"[red]Timed out waiting for health endpoint, and service state could not be confirmed.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
        raise typer.Exit(1)
    finally:
        lock.release()



def _custom_config(
    m: ModelInfo,
    backend,
    *,
    tools: bool = False,
    tool_parser: str | None = None,
    trust_remote_code: bool = False,
    runtime_info=None,
) -> "pathlib.Path":
    """Guide user through parameter selection, delegate config building to backend."""
    if backend.name == "llamacpp":
        return _custom_config_llamacpp(m, backend, tools=tools)
    return _custom_config_vllm(
        m,
        backend,
        tools=tools,
        tool_parser=tool_parser,
        trust_remote_code=trust_remote_code,
        runtime_info=runtime_info,
    )


def _scripted_config(
    m: ModelInfo,
    backend,
    *,
    context: int | None,
    slots: int | None,
    kv_cache_dtype: str | None,
    batched_tokens: int | None,
    gpu_util: float | None,
    port: int,
    tools: bool,
    tool_parser: str | None,
    reasoning_parser: str | None,
    gpu_layers: int | None,
    embedding: bool,
    pooling: str | None,
    save_profile: str | None,
    trust_remote_code: bool,
    runtime_info=None,
    # llama.cpp-specific knobs (ignored by vLLM)
    llamacpp_kv_cache_k: str | None = None,
    llamacpp_kv_cache_v: str | None = None,
    llamacpp_batch_size: int | None = None,
    llamacpp_ubatch_size: int | None = None,
    llamacpp_override_tensors: list[str] | None = None,
    llamacpp_no_moe_offload: bool = False,
) -> "pathlib.Path":
    """Build a launch config from CLI flags without prompting."""
    from vserve import config as config_module
    from vserve.gpu import get_gpu_info, resolve_gpu_memory_utilization

    if save_profile is not None:
        save_profile = _validate_profile_name(save_profile)

    gpu = get_gpu_info()
    effective_gpu_util = resolve_gpu_memory_utilization(
        gpu.vram_total_gb,
        requested=gpu_util,
        config=config_module.cfg(),
    )
    need_tuned_defaults = context is None or slots is None
    if backend.name == "vllm":
        need_tuned_defaults = need_tuned_defaults or kv_cache_dtype is None
    if backend.name == "llamacpp":
        need_tuned_defaults = need_tuned_defaults or gpu_layers is None
    limits_data = _ensure_scripted_limits(
        m,
        backend,
        gpu=gpu,
        gpu_mem_util=effective_gpu_util,
        required=need_tuned_defaults,
        runtime_info=runtime_info,
    )

    if backend.name == "llamacpp":
        if need_tuned_defaults:
            chosen_context, chosen_slots, chosen_layers, chosen_k, chosen_v = _choose_llamacpp_scripted_defaults(
                m,
                limits_data,
                context=context,
                slots=slots,
                gpu_layers=gpu_layers,
                kv_cache_k=llamacpp_kv_cache_k,
                kv_cache_v=llamacpp_kv_cache_v,
            )
        else:
            assert context is not None
            assert slots is not None
            assert gpu_layers is not None
            chosen_context, chosen_slots, chosen_layers = context, slots, gpu_layers
            chosen_k = llamacpp_kv_cache_k or limits_data.get("recommended_kv_dtype") or "f16"
            chosen_v = llamacpp_kv_cache_v or chosen_k

        # Warn on asymmetric K/V — fused FA path falls back to a slow
        # non-fused implementation when types don't match.
        if chosen_k != chosen_v:
            console.print(
                f"[yellow]Warning:[/yellow] asymmetric KV dtypes "
                f"(-ctk {chosen_k} -ctv {chosen_v}) disable the fused "
                "Flash-Attention path. Performance will degrade."
            )

        # MoE expert CPU offload (`-ot ".ffn_.*_exps.=CPU"`). Auto-include
        # when limits_data flags MoE and the user did not opt out.
        override_tensors: list[str] = list(llamacpp_override_tensors or [])
        moe_info = limits_data.get("moe")
        if (
            not llamacpp_no_moe_offload
            and not llamacpp_override_tensors
            and isinstance(moe_info, dict)
            and moe_info.get("is_moe")
            and moe_info.get("ot_pattern")
        ):
            override_tensors.append(str(moe_info["ot_pattern"]))
            freed = moe_info.get("estimated_gpu_freed_gb")
            if freed:
                console.print(
                    f"  [dim]MoE detected: defaulting to "
                    f"`-ot \"{moe_info['ot_pattern']}\"` "
                    f"(frees ~{freed} GB VRAM). Pass --no-moe-offload to opt out.[/dim]"
                )

        choices = {
            "context": chosen_context,
            "n_gpu_layers": chosen_layers,
            "parallel": chosen_slots,
            "port": port,
            "tools": tools,
            "embedding": embedding,
            "pooling": pooling,
            "kv_cache_k": chosen_k,
            "kv_cache_v": chosen_v,
            "override_tensors": override_tensors,
            "batch_size": llamacpp_batch_size,
            "ubatch_size": llamacpp_ubatch_size,
        }
        cfg = backend.build_config(m, choices)
        profile_name = save_profile or "custom"
        cfg_path = backend.root_dir / "configs" / "models" / f"{m.provider}--{m.model_name}.{profile_name}.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        return cfg_path

    if need_tuned_defaults:
        chosen_context, chosen_kv, chosen_slots, recommended_scheduler = _choose_vllm_scripted_defaults(
            m,
            limits_data,
            context=context,
            slots=slots,
            kv_cache_dtype=kv_cache_dtype,
        )
    else:
        assert context is not None
        assert slots is not None
        assert kv_cache_dtype is not None
        chosen_context, chosen_kv, chosen_slots = context, kv_cache_dtype, slots
        recommended_scheduler = {}
    tool_info: dict = {}
    resolved_parser = tool_parser
    if tools or tool_parser:
        if resolved_parser is not None:
            if not _runtime_parser_supported(backend, resolved_parser, kind="tool"):
                console.print(f"[red]vLLM tool parser '{resolved_parser}' is not supported by the configured runtime.[/red]")
                raise typer.Exit(1)
        else:
            cached_parser = limits_data.get("tool_call_parser")
            if isinstance(cached_parser, str) and _runtime_parser_supported(backend, cached_parser, kind="tool"):
                resolved_parser = cached_parser
        if not resolved_parser:
            tool_info = _detect_capabilities_for_scripted_run(m, backend)
            detected_parser = tool_info.get("tool_call_parser")
            if isinstance(detected_parser, str) and _runtime_parser_supported(backend, detected_parser, kind="tool"):
                resolved_parser = detected_parser
        if not resolved_parser:
            console.print("[red]Tool calling was requested, but no valid vLLM tool parser was found.[/red]")
            console.print("  Re-run with [cyan]--tool-parser <name>[/cyan] after checking the installed vLLM runtime.")
            raise typer.Exit(1)

    resolved_reasoning = reasoning_parser
    if resolved_reasoning is not None:
        if not _runtime_parser_supported(backend, resolved_reasoning, kind="reasoning"):
            console.print(f"[red]vLLM reasoning parser '{resolved_reasoning}' is not supported by the configured runtime.[/red]")
            raise typer.Exit(1)
    else:
        cached_reasoning = limits_data.get("reasoning_parser")
        if isinstance(cached_reasoning, str) and _runtime_parser_supported(backend, cached_reasoning, kind="reasoning"):
            resolved_reasoning = cached_reasoning
    if not resolved_reasoning:
        if not tool_info:
            tool_info = _detect_capabilities_for_scripted_run(m, backend)
        detected_reasoning = tool_info.get("reasoning_parser")
        if isinstance(detected_reasoning, str) and _runtime_parser_supported(backend, detected_reasoning, kind="reasoning"):
            resolved_reasoning = detected_reasoning

    vllm_choices: dict[str, object] = {
        "context": chosen_context,
        "kv_dtype": chosen_kv,
        "slots": chosen_slots,
        "batched_tokens": batched_tokens if batched_tokens is not None else recommended_scheduler.get("max_num_batched_tokens"),
        "gpu_mem_util": effective_gpu_util,
        "port": port,
        "tools": bool(tools or tool_parser) and bool(resolved_parser),
        "tool_parser": resolved_parser if (tools or tool_parser) else None,
        "reasoning_parser": resolved_reasoning,
        "trust_remote_code": trust_remote_code,
        "performance_mode": recommended_scheduler.get("performance_mode"),
        "optimization_level": recommended_scheduler.get("optimization_level"),
        "block_size": recommended_scheduler.get("block_size"),
        "enable_prefix_caching": True,
    }
    cfg = backend.build_config(m, vllm_choices)
    profile_name = save_profile or "custom"
    cfg_path = config_module.profile_path(m.provider, m.model_name, profile_name)
    config_module.write_profile_yaml(cfg_path, cfg, comment=f"vserve run — {profile_name} profile")
    return cfg_path


def _infer_backend_from_profile(path: pathlib.Path) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            from vserve.config import try_read_profile_yaml

            data = try_read_profile_yaml(path) or {}
        elif suffix == ".json":
            import json

            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
    except Exception:
        data = {}
    backend_name = data.get("backend") if isinstance(data, dict) else None
    if isinstance(backend_name, str) and backend_name:
        return backend_name
    if suffix in {".yaml", ".yml"}:
        return "vllm"
    if suffix == ".json":
        return "llamacpp"
    return None


def _profile_roots() -> list[pathlib.Path]:
    from vserve.backends import _BACKENDS
    from vserve.config import cfg as _cfg

    roots = [_cfg().configs_dir]
    for backend in _BACKENDS:
        roots.append(backend.root_dir / "configs" / "models")
    return roots


def _path_is_under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _is_known_profile_file(path: pathlib.Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in {".yaml", ".yml", ".json"} and any(_path_is_under(path, root) for root in _profile_roots())


def _resolve_profile_path(
    profile: str,
    *,
    backend=None,
    model: ModelInfo | None = None,
    allow_external_path: bool = False,
) -> tuple[pathlib.Path, str | None]:
    candidate = pathlib.Path(profile).expanduser()
    if candidate.exists():
        if allow_external_path or _is_known_profile_file(candidate):
            return candidate, _infer_backend_from_profile(candidate)
        console.print(f"[red]{candidate} is not a vserve profile.[/red]")
        console.print("  Use [cyan]vserve profile list[/cyan] to see removable profiles.")
        raise typer.Exit(1)
    if model is not None:
        from vserve import config as config_module
        if backend is not None and backend.name == "llamacpp":
            path = backend.root_dir / "configs" / "models" / f"{model.provider}--{model.model_name}.{profile}.json"
        elif backend is not None:
            path = config_module.profile_path(model.provider, model.model_name, profile)
        else:
            path = config_module.profile_path(model.provider, model.model_name, profile)
        if path.exists():
            return path, _infer_backend_from_profile(path) or getattr(backend, "name", None)
    matches = []
    for path in _profile_files():
        if backend is not None and _infer_backend_from_profile(path) != backend.name:
            continue
        if path.stem.endswith(f".{profile}") or path.name == profile:
            matches.append(path)
    matches = sorted(matches)
    if len(matches) == 1:
        path = matches[0]
        return path, _infer_backend_from_profile(path)
    if len(matches) > 1:
        console.print(f"[red]Profile '{profile}' matches multiple configs; include the model name or path.[/red]")
        for match in matches:
            console.print(f"  {match}")
        raise typer.Exit(1)
    console.print(f"[red]Profile '{profile}' not found.[/red]")
    raise typer.Exit(1)


def _profile_files() -> list[pathlib.Path]:
    from vserve.backends import _BACKENDS
    from vserve.config import cfg as _cfg

    paths = list(_cfg().configs_dir.glob("*.yaml"))
    paths.extend(_cfg().configs_dir.glob("*.yml"))
    for backend in _BACKENDS:
        paths.extend((backend.root_dir / "configs" / "models").glob("*.json"))
    return sorted({path for path in paths if _is_known_profile_file(path)})


@profile_app.command("list")
def profile_list():
    """List saved serving profiles."""
    files = _profile_files()
    if not files:
        console.print("[dim]No saved profiles.[/dim]")
        return
    for path in files:
        console.print(str(path))


@profile_app.command("show")
def profile_show(profile: str = typer.Argument(..., help="Profile name or path")):
    """Show a saved serving profile."""
    path, _backend_name = _resolve_profile_path(profile, allow_external_path=False)
    console.print(path.read_text())


@profile_app.command("rm")
def profile_rm(
    profile: str = typer.Argument(..., help="Profile name or path"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a saved serving profile."""
    path, _backend_name = _resolve_profile_path(profile, allow_external_path=False)
    if not force and not typer.confirm(f"Delete {path}?", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return
    path.unlink()
    console.print(f"[green]Deleted[/green] {path}")


def _llamacpp_slots_from_limits_entry(entry: object) -> int | None:
    if isinstance(entry, dict):
        values = [v for v in entry.values() if v is not None]
        return max(values) if values else None
    if entry is None or isinstance(entry, bool) or not isinstance(entry, int):
        return None
    return entry


def _vllm_limits_entry(entry: object) -> dict[str, int | None]:
    if isinstance(entry, dict):
        cleaned: dict[str, int | None] = {}
        for key, value in entry.items():
            if not isinstance(key, str):
                continue
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                cleaned[key] = value
        return cleaned
    if entry is None or isinstance(entry, bool) or not isinstance(entry, int):
        return {}
    return {"auto": entry}


def _vllm_limit_dtype_order(limits_data: dict, limits: dict) -> list[str]:
    dtypes_obj = limits_data.get("kv_cache_dtypes")
    if isinstance(dtypes_obj, dict):
        ordered = [key for key in dtypes_obj if isinstance(key, str)]
        if ordered:
            return ordered
    seen: list[str] = []
    for _ctx, entry in sorted(limits.items(), key=lambda item: int(str(item[0]))):
        for dtype in _vllm_limits_entry(entry):
            if dtype not in seen:
                seen.append(dtype)
    preferred = ["auto", "fp8"]
    return [dtype for dtype in preferred if dtype in seen] + [
        dtype for dtype in seen if dtype not in preferred
    ]


def _vllm_kv_label(dtype: str) -> str:
    labels = {
        "auto": "Auto KV",
        "fp8": "FP8 KV",
        "fp8_e4m3": "FP8 e4m3",
        "fp8_e5m2": "FP8 e5m2",
        "turboquant_k8v4": "TQ k8v4",
        "turboquant_4bit_nc": "TQ 4bit",
        "turboquant_k3v4_nc": "TQ k3v4",
        "turboquant_3bit_nc": "TQ 3bit",
        # llama.cpp K/V dtypes — labels are intentionally symmetric since
        # fused Flash-Attention requires K and V to match.
        "f16": "F16 KV",
        "bf16": "BF16 KV",
        "f32": "F32 KV",
        "q8_0": "Q8 KV",
        "q5_1": "Q5_1 KV",
        "q5_0": "Q5_0 KV",
        "q4_1": "Q4_1 KV",
        "q4_0": "Q4_0 KV",
        "iq4_nl": "IQ4_NL KV",
    }
    return labels.get(dtype, dtype)


_VLLM_AUTOMATIC_KV_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc")


def _build_current_tuning_fingerprint(m: ModelInfo, backend, *, gpu, gpu_mem_util: float, runtime_info=None) -> dict:
    from vserve.runtime import build_tuning_fingerprint

    if runtime_info is None:
        runtime_info_fn = getattr(backend, "runtime_info", None)
    else:
        runtime_info_fn = None
    if callable(runtime_info_fn):
        try:
            runtime_info = runtime_info_fn()
        except Exception:
            runtime_info = None
    return build_tuning_fingerprint(
        model_info=m,
        gpu=gpu,
        backend=backend.name,
        gpu_mem_util=gpu_mem_util,
        runtime_info=runtime_info,
    )


def _ensure_scripted_limits(m: ModelInfo, backend, *, gpu, gpu_mem_util: float, required: bool, runtime_info=None) -> dict:
    """Return cached limits for non-interactive defaults, tuning once if needed."""
    from vserve import config as config_module

    lim_path = limits_path(m.provider, m.model_name)
    fingerprint = _build_current_tuning_fingerprint(
        m,
        backend,
        gpu=gpu,
        gpu_mem_util=gpu_mem_util,
        runtime_info=runtime_info,
    )
    lim = read_limits(lim_path)
    if isinstance(lim, dict) and config_module.limits_cache_matches(lim, backend=backend.name, fingerprint=fingerprint):
        return lim
    if not required:
        return {"backend": backend.name, "limits": {}}
    lim = backend.tune(m, gpu, gpu_mem_util=gpu_mem_util)
    if isinstance(lim, dict):
        lim.setdefault("backend", backend.name)
        lim["fingerprint"] = fingerprint
        config_module.write_limits(lim_path, lim)
        return lim
    raise RuntimeError(f"{backend.display_name} did not return valid tuning limits")


def _choose_vllm_scripted_defaults(
    m: ModelInfo,
    limits_data: dict,
    *,
    context: int | None,
    slots: int | None,
    kv_cache_dtype: str | None,
) -> tuple[int, str, int, dict]:
    limits = limits_data.get("limits", {})
    limits = limits if isinstance(limits, dict) else {}
    recommendation: dict = {}
    if context is None and slots is None and kv_cache_dtype is None:
        recommendations = limits_data.get("recommendations")
        if isinstance(recommendations, dict):
            balanced = recommendations.get("balanced")
            if isinstance(balanced, dict):
                rec_context = balanced.get("context")
                rec_kv = balanced.get("kv_cache_dtype")
                rec_slots = balanced.get("max_num_seqs")
                if (
                    isinstance(rec_context, int)
                    and isinstance(rec_kv, str)
                    and isinstance(rec_slots, int)
                    and _vllm_limits_entry(limits.get(str(rec_context), {})).get(rec_kv) is not None
                ):
                    return rec_context, rec_kv, rec_slots, dict(balanced)

    working_contexts: list[int] = []
    for ctx_str, entry in limits.items():
        try:
            ctx = int(str(ctx_str))
        except ValueError:
            continue
        choices = _vllm_limits_entry(entry)
        if kv_cache_dtype is not None:
            if choices.get(kv_cache_dtype) is not None:
                working_contexts.append(ctx)
        elif any(choices.get(dtype) is not None for dtype in _VLLM_AUTOMATIC_KV_DTYPES):
            working_contexts.append(ctx)

    if not working_contexts:
        raise ValueError(f"No tuned vLLM capacity is available for {m.full_name}; run vserve tune {m.full_name}")
    if context is not None and context not in working_contexts:
        raise ValueError(f"No tuned vLLM capacity for context {context}; run vserve tune {m.full_name} --recalc")
    chosen_context = context or max(working_contexts)
    ctx_entry = _vllm_limits_entry(limits.get(str(chosen_context), {}))
    if kv_cache_dtype is not None:
        if ctx_entry.get(kv_cache_dtype) is None:
            raise ValueError(f"No tuned vLLM capacity for context {chosen_context} with KV dtype {kv_cache_dtype}")
        chosen_kv = kv_cache_dtype
    else:
        candidate_kv = next((dtype for dtype in _VLLM_AUTOMATIC_KV_DTYPES if ctx_entry.get(dtype) is not None), None)
        if candidate_kv is None:
            raise ValueError(f"No tuned vLLM capacity for context {chosen_context}")
        chosen_kv = candidate_kv

    if slots is not None:
        chosen_slots = slots
    else:
        tuned_slots = ctx_entry.get(chosen_kv)
        if tuned_slots is None:
            raise ValueError(f"No tuned vLLM slot count for context {chosen_context} with KV dtype {chosen_kv}")
        chosen_slots = int(tuned_slots)
    return chosen_context, chosen_kv, chosen_slots, recommendation


def _choose_llamacpp_scripted_defaults(
    m: ModelInfo,
    limits_data: dict,
    *,
    context: int | None,
    slots: int | None,
    gpu_layers: int | None,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
) -> tuple[int, int, int, str, str]:
    """Pick (context, slots, n_gpu_layers, kv_cache_k, kv_cache_v) for a
    scripted llama.cpp run. KV dtype defaults to the tuner's recommendation
    (f16 unless q8_0 strictly fits more slots — see
    LlamaCppBackend._recommended_kv_dtype). The fused Flash-Attention path
    requires K and V to match, so a single dtype is picked unless the user
    explicitly passes asymmetric K and V values.
    """
    limits = limits_data.get("limits", {})
    limits = limits if isinstance(limits, dict) else {}
    working_contexts: list[int] = []
    for ctx_str, entry in limits.items():
        try:
            ctx = int(str(ctx_str))
        except ValueError:
            continue
        if _llamacpp_slots_from_limits_entry(entry) is not None:
            working_contexts.append(ctx)
    if not working_contexts:
        raise ValueError(f"No tuned llama.cpp capacity is available for {m.full_name}; run vserve tune {m.full_name}")
    if context is not None and context not in working_contexts:
        raise ValueError(f"No tuned llama.cpp capacity for context {context}; run vserve tune {m.full_name} --recalc")
    chosen_context = context or max(working_contexts)

    recommended_kv = limits_data.get("recommended_kv_dtype") or "f16"
    chosen_k = kv_cache_k or recommended_kv
    chosen_v = kv_cache_v or kv_cache_k or recommended_kv

    if slots is not None:
        chosen_slots = slots
    else:
        entry = limits.get(str(chosen_context))
        if isinstance(entry, dict):
            tuned_slots: int | None = entry.get(chosen_k)
            if tuned_slots is None:
                # Fall back to f16 row then to max across dtypes.
                tuned_slots = entry.get("f16")
            if tuned_slots is None:
                tuned_slots = _llamacpp_slots_from_limits_entry(entry)
        else:
            tuned_slots = _llamacpp_slots_from_limits_entry(entry)
        if tuned_slots is None:
            raise ValueError(f"No tuned llama.cpp slot count for context {chosen_context}")
        chosen_slots = int(tuned_slots)

    tuned_layers = limits_data.get("n_gpu_layers")
    if gpu_layers is not None:
        chosen_layers = gpu_layers
    else:
        if not isinstance(tuned_layers, int):
            raise ValueError(f"No tuned llama.cpp GPU layer count for {m.full_name}; run vserve tune {m.full_name}")
        chosen_layers = tuned_layers
    return chosen_context, chosen_slots, chosen_layers, chosen_k, chosen_v


def _detect_capabilities_for_scripted_run(m: ModelInfo, backend) -> dict:
    detect_fn = getattr(backend, "detect_tools", None)
    if not callable(detect_fn):
        return {}
    info = detect_fn(m.path)
    return info if isinstance(info, dict) else {}


def _runtime_parser_supported(backend, parser: str | None, *, kind: str) -> bool:
    if not parser:
        return False
    attr = "available_tool_parsers" if kind == "tool" else "available_reasoning_parsers"
    available_fn = getattr(backend, attr, None)
    if not callable(available_fn):
        return False
    try:
        available = available_fn()
    except Exception:
        return False
    return isinstance(available, set) and parser in available


def _custom_config_llamacpp(m: ModelInfo, backend, *, tools: bool = False) -> "pathlib.Path":
    """Interactive config wizard for llama.cpp models."""
    import json
    from vserve.config import read_limits, write_limits as _write_limits, limits_cache_matches
    from vserve.gpu import get_gpu_info, resolve_gpu_memory_utilization
    from vserve.config import cfg as _cfg

    gpu = get_gpu_info()
    gpu_mem_util = resolve_gpu_memory_utilization(gpu.vram_total_gb, config=_cfg())

    lim_path_ = limits_path(m.provider, m.model_name)
    fingerprint = _build_current_tuning_fingerprint(m, backend, gpu=gpu, gpu_mem_util=gpu_mem_util)
    lim = read_limits(lim_path_)
    if not limits_cache_matches(lim, backend=backend.name, fingerprint=fingerprint):
        console.print(f"[dim]  Auto-tuning {m.model_name}...[/dim]")
        lim = backend.tune(m, gpu, gpu_mem_util=gpu_mem_util)
        if not isinstance(lim, dict):
            console.print(f"[red]{backend.display_name} did not return valid tuning limits.[/red]")
            raise typer.Exit(1)
        lim.setdefault("backend", backend.name)
        lim["fingerprint"] = fingerprint
        _write_limits(lim_path_, lim)
        console.print("[green]  Tuned.[/green]")
    if not isinstance(lim, dict):
        console.print(f"[red]No valid llama.cpp tuning limits found for {m.full_name}.[/red]")
        raise typer.Exit(1)

    n_gpu_layers = lim.get("n_gpu_layers", 0)
    num_layers = lim.get("num_layers", 0)
    full_offload = lim.get("full_offload", True)
    limits_obj = lim.get("limits", {})
    limits = limits_obj if isinstance(limits_obj, dict) else {}

    # 1. Context window
    working_ctxs = sorted(
        int(str(c))
        for c, v in limits.items()
        if _llamacpp_slots_from_limits_entry(v) is not None
    )
    if not working_ctxs:
        console.print("[red]No working configs found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold] (llama.cpp)\n")
    ctx_items = []
    for ctx in working_ctxs:
        slots = _llamacpp_slots_from_limits_entry(limits.get(str(ctx)))
        slot_str = f"({slots} slots)" if slots else ""
        ctx_items.append(f"{ctx // 1024}k  {slot_str}")
    console.print("  [bold]1. Context window[/bold]")
    ctx_idx = _pick(ctx_items, title="  Context:")
    if ctx_idx is None:
        raise typer.Exit(0)
    chosen_ctx = working_ctxs[ctx_idx]

    # 2. GPU layers
    if not full_offload:
        console.print(f"\n  [bold]2. GPU layers[/bold] (max: {num_layers}, fits: {n_gpu_layers})")
        console.print(f"     [yellow]Model partially fits — {n_gpu_layers}/{num_layers} layers on GPU[/yellow]")
        layers_str = typer.prompt(f"  Layers [1-{num_layers}]", default=str(n_gpu_layers))
        try:
            n_gpu_layers = min(max(1, int(layers_str)), num_layers)
        except ValueError:
            pass
    else:
        console.print(f"\n  [dim]GPU layers: {n_gpu_layers}/{num_layers} (all on GPU)[/dim]")

    # 3. Parallel slots
    max_parallel = _llamacpp_slots_from_limits_entry(limits.get(str(chosen_ctx), 1)) or 1
    console.print(f"\n  [bold]3. Parallel slots[/bold] (max: {max_parallel})")
    par_str = typer.prompt(f"  Slots [1-{max_parallel}]", default=str(max_parallel))
    try:
        chosen_parallel = min(max(1, int(par_str)), max_parallel)
    except ValueError:
        chosen_parallel = max_parallel

    # 4. Embedding or tool calling
    is_embedding = lim.get("is_embedding", False) or m.is_embedding
    chosen_pooling: str | None = None

    if is_embedding:
        pooling_options = ["mean", "cls", "last"]
        default_pooling = lim.get("pooling", "mean")
        default_idx = pooling_options.index(default_pooling) if default_pooling in pooling_options else 0
        pooling_labels = [
            "mean  — average all tokens (Nomic, E5, Jina, Qwen)",
            "cls   — [CLS] token only (BGE, BERT-style)",
            "last  — last token (decoder-based embeddings)",
        ]
        console.print("\n  [bold]4. Pooling strategy[/bold]")
        pool_idx = _pick(pooling_labels, title="  Pooling:")
        if pool_idx is None:
            pool_idx = default_idx
        chosen_pooling = pooling_options[pool_idx]
    else:
        # Use cached limits first, fall back to live detection
        supports_tools = lim.get("supports_tools", False)
        supports_reasoning = lim.get("supports_reasoning", False)
        if not supports_tools and not supports_reasoning:
            tool_info = backend.detect_tools(m.path)
            supports_tools = tool_info.get("supports_tools", False)
            supports_reasoning = tool_info.get("supports_reasoning", False)

        if supports_tools or supports_reasoning:
            caps = []
            if supports_tools:
                caps.append("tool calling")
            if supports_reasoning:
                caps.append("reasoning")
            console.print(f"\n  [bold]4. Capabilities[/bold] ({' + '.join(caps)} via --jinja)")
            if not tools:
                tools = typer.confirm("     Enable? (--jinja)", default=False)
            else:
                console.print("     [green]Enabled[/green] (--jinja)")

    # Build config via backend
    choices: dict = {
        "context": chosen_ctx,
        "n_gpu_layers": n_gpu_layers,
        "parallel": chosen_parallel,
        "port": 8888,
    }
    if is_embedding:
        choices["embedding"] = True
        choices["pooling"] = chosen_pooling
    else:
        choices["tools"] = tools
    cfg = backend.build_config(m, choices)

    # Summary
    console.print("\n  [bold]Summary[/bold]")
    console.print(f"    Context:     {chosen_ctx // 1024}k")
    console.print(f"    GPU layers:  {n_gpu_layers}/{num_layers}")
    console.print(f"    Parallel:    {chosen_parallel}")
    if is_embedding:
        console.print(f"    Mode:        [green]embedding (--pooling {chosen_pooling})[/green]")
    elif tools:
        console.print("    Tool calling: [green]enabled (--jinja)[/green]")

    if not typer.confirm("\n  Start?", default=True):
        raise typer.Exit(0)

    # Write JSON config for llama-server
    cfg_dir = backend.root_dir / "configs" / "models"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{m.provider}--{m.model_name}.custom.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    console.clear()
    return cfg_path


def _custom_config_vllm(
    m: ModelInfo,
    backend,
    *,
    tools: bool = False,
    tool_parser: str | None = None,
    trust_remote_code: bool = False,
    runtime_info=None,
) -> "pathlib.Path":
    """Interactive config wizard for vLLM models."""
    from vserve.config import cfg as _cfg, limits_cache_matches, read_limits, write_profile_yaml
    from vserve.gpu import get_gpu_info, resolve_gpu_memory_utilization
    from vserve.runtime import build_tuning_fingerprint

    gpu = get_gpu_info()
    gpu_mem_util = resolve_gpu_memory_utilization(gpu.vram_total_gb, config=_cfg())
    if runtime_info is None:
        runtime_info_fn = getattr(backend, "runtime_info", None)
    else:
        runtime_info_fn = None
    if callable(runtime_info_fn):
        try:
            runtime_info = runtime_info_fn()
        except Exception:
            runtime_info = None
    fingerprint = build_tuning_fingerprint(
        model_info=m,
        gpu=gpu,
        backend=backend.name,
        gpu_mem_util=gpu_mem_util,
        runtime_info=runtime_info,
    )

    lim_path_ = limits_path(m.provider, m.model_name)
    lim = read_limits(lim_path_)
    if not limits_cache_matches(lim, backend=backend.name, fingerprint=fingerprint):
        if m.num_kv_heads is None or m.head_dim is None or m.num_layers is None:
            console.print("[red]Missing architecture fields in config.json[/red]")
            console.print(f"  num_kv_heads={m.num_kv_heads}, head_dim={m.head_dim}, num_layers={m.num_layers}")
            raise typer.Exit(1)
        console.print(f"[dim]  Auto-tuning {m.model_name}...[/dim]")
        from vserve.config import write_limits as _write_limits
        lim = backend.tune(m, gpu, gpu_mem_util=gpu_mem_util)
        lim["backend"] = backend.name
        lim["fingerprint"] = fingerprint
        _write_limits(lim_path_, lim)
        console.print("[green]  Tuned.[/green]")
    assert lim is not None
    limits = lim.get("limits", {})

    # 1. Context window
    working_ctxs = sorted(
        (int(str(c)) for c, d in limits.items() if any(v is not None for v in _vllm_limits_entry(d).values())),
    )
    if not working_ctxs:
        console.print("[red]No working configs found in probe data.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold] (vLLM)\n")

    # 1. Context window
    ctx_items = [f"{ctx // 1024}k" for ctx in working_ctxs]
    console.print("  [bold]1. Context window[/bold] (max tokens per request)")
    ctx_idx = _pick(ctx_items, title="  Context:")
    if ctx_idx is None:
        raise typer.Exit(0)
    chosen_ctx = working_ctxs[ctx_idx]

    # 2. KV cache dtype
    ctx_entry = _vllm_limits_entry(limits.get(str(chosen_ctx), {}))
    working_kvs = [k for k, v in ctx_entry.items() if v is not None]

    kv_labels = {
        "auto": "auto (bfloat16)",
        "fp8": "fp8 (saves memory, slightly less accurate)",
    }
    kv_items = [f"{kv}  — {kv_labels.get(kv, kv)}" for kv in working_kvs]
    console.print("\n  [bold]2. KV cache dtype[/bold]")
    kv_idx = _pick(kv_items, title="  KV dtype:")
    if kv_idx is None:
        raise typer.Exit(0)
    chosen_kv = working_kvs[kv_idx]

    # 3. Concurrent slots
    max_seqs = int(ctx_entry.get(chosen_kv, 1) or 1)
    console.print(f"\n  [bold]3. Concurrent slots[/bold] (max: {max_seqs})")
    console.print("     How many simultaneous requests?")
    seqs_str = typer.prompt(f"  Slots [1-{max_seqs}]", default=str(max_seqs))
    try:
        chosen_seqs = min(max(1, int(seqs_str)), max_seqs)
    except ValueError:
        chosen_seqs = max_seqs

    # 4. Batched tokens (for throughput tuning)
    bt_options = [
        "auto  — let vLLM decide (good for chat)",
        "2048  — balanced",
        "4096  — high throughput",
        "8192  — maximum throughput (batch processing)",
    ]
    bt_values: list[int | None] = [None, 2048, 4096, 8192]
    console.print("\n  [bold]4. Max batched tokens[/bold]")
    bt_idx = _pick(bt_options, title="  Batched tokens:")
    chosen_bt = bt_values[bt_idx] if bt_idx is not None else None

    # Tool calling & reasoning
    tool_info = backend.detect_tools(m.path)
    resolved_parser: str | None = None
    resolved_reasoning: str | None = None

    if tool_parser:
        resolved_parser = tool_parser
        tools = True
    else:
        resolved_parser = lim.get("tool_call_parser") or tool_info.get("tool_call_parser")

    resolved_reasoning = lim.get("reasoning_parser") or tool_info.get("reasoning_parser")

    if resolved_parser or resolved_reasoning:
        console.print("\n  [bold]5. Capabilities[/bold]")
        if resolved_parser:
            if not tools:
                enable_tools = typer.confirm(
                    f"     Enable tool calling? (parser: {resolved_parser})", default=False,
                )
                tools = enable_tools
            else:
                console.print(f"     Tool calling: [green]{resolved_parser}[/green]")
        if resolved_reasoning:
            if not tools:
                console.print(f"     Reasoning: [dim]{resolved_reasoning} (enable tool calling to activate)[/dim]")
            else:
                console.print(f"     Reasoning:    [green]{resolved_reasoning}[/green]")
    else:
        from vserve.tools import supports_tools as _supports_tools
        if _supports_tools(m.path):
            console.print("\n  [bold]5. Capabilities[/bold]")
            console.print("     [yellow]Tool markers found but parser unknown[/yellow]")
            console.print("     Use --tool-parser <name> to enable")

    # Build config via backend
    enable_tools = tools and bool(resolved_parser)
    choices = {
        "context": chosen_ctx,
        "kv_dtype": chosen_kv,
        "slots": chosen_seqs,
        "batched_tokens": chosen_bt,
        "gpu_mem_util": gpu_mem_util,
        "port": 8888,
        "tools": enable_tools,
        "tool_parser": resolved_parser if enable_tools else None,
        "reasoning_parser": resolved_reasoning if enable_tools else None,
        "trust_remote_code": trust_remote_code,
    }
    cfg = backend.build_config(m, choices)

    # Summary
    console.print("\n  [bold]Summary[/bold]")
    console.print(f"    Context:        {chosen_ctx // 1024}k")
    console.print(f"    KV dtype:       {chosen_kv}")
    console.print(f"    Slots:          {chosen_seqs}")
    console.print(f"    Batched tokens: {chosen_bt or 'auto'}")
    console.print("    Prefix:         always on")
    if tools and resolved_parser:
        cap_parts = [f"tools=[green]{resolved_parser}[/green]"]
        if resolved_reasoning:
            cap_parts.append(f"reasoning=[green]{resolved_reasoning}[/green]")
        console.print(f"    Tool calling:   {' '.join(cap_parts)}")

    if not typer.confirm("\n  Start?", default=True):
        raise typer.Exit(0)

    cfg_path = profile_path(m.provider, m.model_name, "custom")
    write_profile_yaml(cfg_path, cfg, comment="vserve run — custom config")
    console.clear()
    return cfg_path


@app.command()
def run(
    model_terms: list[str] = typer.Argument(None, help="Model name terms (fuzzy match)"),
    tools: bool = typer.Option(False, "--tools", help="Enable tool/function calling"),
    tool_parser: str | None = typer.Option(None, "--tool-parser", help="Override tool-call parser (e.g. hermes, qwen3_coder)"),
    reasoning_parser: str | None = typer.Option(None, "--reasoning-parser", help="Override reasoning parser (e.g. qwen3, deepseek_r1)"),
    backend_name: str | None = typer.Option(None, "--backend", help="Force backend (vllm, llamacpp)"),
    allow_unsupported_runtime: bool = typer.Option(False, "--allow-unsupported-runtime", help="Allow starting outside vserve's pinned runtime range"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Build config from flags/defaults without prompting"),
    replace: bool = typer.Option(False, "--replace", help="For --yes, stop a running backend before starting this profile"),
    profile: str | None = typer.Option(None, "--profile", help="Saved profile name or explicit config path"),
    save_profile: str | None = typer.Option(None, "--save-profile", help="Save generated config under this profile name"),
    context: int | None = typer.Option(None, "--context", help="Context length in tokens"),
    slots: int | None = typer.Option(None, "--slots", help="Concurrent request slots"),
    kv_cache_dtype: str | None = typer.Option(None, "--kv-cache-dtype", help="vLLM KV cache dtype"),
    batched_tokens: int | None = typer.Option(None, "--batched-tokens", help="vLLM max batched tokens"),
    gpu_util: float | None = typer.Option(None, "--gpu-util", help="GPU memory utilization"),
    port: int = typer.Option(8888, "--port", help="Serving port"),
    gpu_layers: int | None = typer.Option(None, "--gpu-layers", help="llama.cpp GPU layers"),
    embedding: bool = typer.Option(False, "--embedding", help="Run llama.cpp in embedding mode"),
    pooling: str | None = typer.Option(None, "--pooling", help="llama.cpp embedding pooling"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Allow vLLM to execute model repository code"),
    kv_cache_k: str | None = typer.Option(None, "--kv-cache-k", help="llama.cpp K-cache dtype (f16, q8_0, q4_1, q4_0, …); fused FA requires K=V"),
    kv_cache_v: str | None = typer.Option(None, "--kv-cache-v", help="llama.cpp V-cache dtype; defaults to --kv-cache-k"),
    batch_size_lc: int | None = typer.Option(None, "--batch-size", help="llama.cpp logical batch size (-b); default 2048"),
    ubatch_size_lc: int | None = typer.Option(None, "--ubatch-size", help="llama.cpp physical batch size (-ub); default 512"),
    override_tensor: list[str] = typer.Option([], "--override-tensor", "-ot", help="llama.cpp --override-tensor pattern (repeatable, e.g. '.ffn_.*_exps.=CPU')"),
    no_moe_offload: bool = typer.Option(False, "--no-moe-offload", help="Disable the auto-applied MoE expert-CPU offload (-ot)"),
):
    """Start serving a model — interactive config picker."""
    from vserve.backends import get_backend, get_backend_by_name

    # Check session lock early — before interactive config
    _session_or_exit()

    query = _join_model_terms(model_terms)
    if save_profile is not None:
        save_profile = _validate_profile_name(save_profile)

    m: ModelInfo | None
    resolved_profile_path: pathlib.Path | None = None
    inferred_profile_backend: str | None = None
    if query is None and profile is None:
        all_models = _all_models()
        if not all_models:
            console.print("[red]No models found.[/red] Run: vserve add")
            raise typer.Exit(1)

        from vserve.backends import _BACKENDS

        name_w = max(len(m.full_name) for m in all_models)
        size_w = max(len(f"{m.model_size_gb} GB") for m in all_models)
        items = []
        for m in all_models:
            lim = read_limits(limits_path(m.provider, m.model_name))
            size = f"{m.model_size_gb} GB"

            # Capability tags
            tags: list[str] = []
            if lim:
                tags.append("tuned")
                tp = lim.get("tool_call_parser") or lim.get("supports_tools")
                rp = lim.get("reasoning_parser") or lim.get("supports_reasoning")
                if tp:
                    tags.append("tools")
                if rp:
                    tags.append("reasoning")
            else:
                # Try live detect for untuned models
                backend_obj = None
                for b in _BACKENDS:
                    if b.can_serve(m):
                        backend_obj = b
                        break
                if backend_obj:
                    tool_info = backend_obj.detect_tools(m.path)
                    if tool_info.get("tool_call_parser") or tool_info.get("supports_tools"):
                        tags.append("tools")
                    if tool_info.get("reasoning_parser") or tool_info.get("supports_reasoning"):
                        tags.append("reasoning")

            if m.is_embedding:
                tags.append("embedding")

            tag_str = "  ".join(tags) if tags else "not tuned"
            items.append(f"{m.full_name:<{name_w}}  {size:>{size_w}}  {tag_str}")

        idx = _pick(items, title="Select a model:")
        if idx is None:
            raise typer.Exit(0)
        m = all_models[idx]
    else:
        m = _resolve_model(query) if query is not None else None
        if profile is not None and m is None:
            resolved_profile_path, inferred_profile_backend = _resolve_profile_path(
                profile,
                backend=None,
                model=None,
                allow_external_path=True,
            )

    if backend_name:
        try:
            backend = get_backend_by_name(backend_name)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
    elif inferred_profile_backend:
        try:
            backend = get_backend_by_name(inferred_profile_backend)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
    else:
        if m is None:
            console.print("[red]A model query is required unless the profile path or content identifies a backend.[/red]")
            console.print("  Use [cyan]--backend vllm[/cyan] for YAML profiles or [cyan]--backend llamacpp[/cyan] for llama.cpp JSON profiles.")
            raise typer.Exit(1)
        backend = get_backend(m)

    if m is not None and not backend.can_serve(m):
        console.print(f"[red]{backend.display_name} cannot serve {m.full_name}.[/red]")
        console.print(f"  {_backend_format_guidance(backend.name)}")
        raise typer.Exit(1)

    # Hot path: prefer the cached runtime info and skip `pip check`. The
    # diagnostic commands (`vserve doctor`, `vserve runtime check vllm`) still
    # invoke the full probe with pip check.
    runtime_info = _check_backend_runtime_or_exit(
        backend,
        allow_unsupported_runtime=allow_unsupported_runtime,
        prefer_cache=True,
    )

    if tool_parser and not tools:
        tools = True
    if profile:
        if resolved_profile_path is None:
            resolved_profile_path, _profile_backend = _resolve_profile_path(
                profile,
                backend=backend,
                model=m,
                allow_external_path=True,
            )
        cfg_path = resolved_profile_path
    elif yes or any(
        value is not None
        for value in (
            context, slots, kv_cache_dtype, batched_tokens, gpu_util, save_profile,
            reasoning_parser, gpu_layers, pooling,
            kv_cache_k, kv_cache_v, batch_size_lc, ubatch_size_lc,
        )
    ) or embedding or override_tensor or no_moe_offload:
        if m is None:
            console.print("[red]A model query is required when building a profile from flags.[/red]")
            raise typer.Exit(1)
        cfg_path = _scripted_config(
            m,
            backend,
            context=context,
            slots=slots,
            kv_cache_dtype=kv_cache_dtype,
            batched_tokens=batched_tokens,
            gpu_util=gpu_util,
            port=port,
            tools=tools,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            gpu_layers=gpu_layers,
            embedding=embedding,
            pooling=pooling,
            save_profile=save_profile,
            trust_remote_code=trust_remote_code,
            runtime_info=runtime_info,
            llamacpp_kv_cache_k=kv_cache_k,
            llamacpp_kv_cache_v=kv_cache_v,
            llamacpp_batch_size=batch_size_lc,
            llamacpp_ubatch_size=ubatch_size_lc,
            llamacpp_override_tensors=list(override_tensor) if override_tensor else None,
            llamacpp_no_moe_offload=no_moe_offload,
        )
    else:
        if m is None:
            console.print("[red]A model query is required for interactive configuration.[/red]")
            raise typer.Exit(1)
        cfg_path = _custom_config(
            m,
            backend,
            tools=tools,
            tool_parser=tool_parser,
            trust_remote_code=trust_remote_code,
            runtime_info=runtime_info,
        )
    _launch_backend(
        backend,
        cfg_path,
        m.model_name if m is not None else (profile or "profile"),
        non_interactive=yes,
        replace=replace,
    )


@app.command()
def stop():
    """Stop the inference server."""
    from vserve.backends import _BACKENDS, probe_running_backends

    _session_or_exit(fail_on_probe_uncertainty=False, allow_unknown_owner=True)
    # If the warning above fired (no marker existed), claim the orphan now so
    # the TOCTOU re-checks below pass silently and a future stop/run from this
    # user resolves cleanly.
    if read_session() is None:
        try:
            write_session("orphan-claim")
        except Exception:
            pass

    running_backends, probe_failed = probe_running_backends()
    if not running_backends:
        if probe_failed:
            from vserve.backends import _BACKENDS

            lock = _lock_or_exit("gpu", "stopping backend (probe uncertain)")
            try:
                _session_or_exit(
                    fail_on_probe_uncertainty=False,
                    allow_unknown_owner=True,
                    quiet=True,
                )
                fallback_stop_errors: list[str] = []
                for candidate in _BACKENDS:
                    try:
                        candidate.stop()
                    except Exception as exc:
                        fallback_stop_errors.append(f"{candidate.display_name}: {exc}")
                confirmed_backends, confirmed_probe_failed = probe_running_backends()
                if confirmed_backends:
                    remaining_names = ", ".join(candidate.display_name for candidate in confirmed_backends)
                    console.print(f"[red]{remaining_names} still appear to be running after fallback stop attempts.[/red]")
                    for detail in fallback_stop_errors:
                        console.print(f"  {detail}")
                    raise typer.Exit(1)
                if confirmed_probe_failed:
                    console.print("[red]Could not verify backend state after fallback stop attempts.[/red]")
                    for detail in fallback_stop_errors:
                        console.print(f"  {detail}")
                    raise typer.Exit(1)
                clear_session()
                console.print("[yellow]Backend state was uncertain; issued stop requests to all known backends.[/yellow]")
                for detail in fallback_stop_errors:
                    console.print(f"  [dim]{detail}[/dim]")
                console.print("[green]Stop request completed.[/green]")
                return
            finally:
                lock.release()
        clear_session()
        console.print("[dim]No server is running.[/dim]")
        return

    stop_candidates = list(_BACKENDS) if probe_failed else list(running_backends)
    running_names = ", ".join(candidate.display_name for candidate in stop_candidates)
    lock = _lock_or_exit("gpu", f"stopping {running_names}")
    try:
        _session_or_exit(
            fail_on_probe_uncertainty=False,
            allow_unknown_owner=True,
            quiet=True,
        )  # re-check under flock (TOCTOU)

        stop_errors: list[str] = []
        for candidate in stop_candidates:
            try:
                candidate.stop()
            except Exception as e:
                stop_errors.append(f"{candidate.display_name}: {e}")

        confirmed_backends, confirmed_probe_failed = probe_running_backends()
        if confirmed_backends:
            remaining_names = ", ".join(candidate.display_name for candidate in confirmed_backends)
            console.print(f"[red]{remaining_names} still appear to be running after stop.[/red]")
            for detail in stop_errors:
                console.print(f"  {detail}")
            raise typer.Exit(1)
        if confirmed_probe_failed:
            console.print("[red]Could not verify backend state after stop.[/red]")
            for detail in stop_errors:
                console.print(f"  {detail}")
            raise typer.Exit(1)
        clear_session()
        if stop_errors:
            for detail in stop_errors:
                console.print(f"  [dim]{detail}[/dim]")
        console.print(f"[green]Stopped: {running_names}.[/green]")
    finally:
        lock.release()


@app.command()
def fan(
    mode: str = typer.Argument(None, help="auto | <30-100> | off"),
):
    """GPU fan control — auto curve, fixed speed, or off."""
    from pathlib import Path

    import vserve.fan as _fan
    from vserve.fan import read_state
    from vserve.gpu import get_fan_speed
    import os
    import signal as sig
    import time

    def _sudo_reexec() -> None:
        """Re-execute this command under sudo if not already root."""
        if os.geteuid() != 0:
            import sys
            os.execvp("sudo", ("sudo", *sys.argv))

    _fan._resolve_paths()
    PID_PATH = _fan.PID_PATH
    STATE_PATH = _fan.STATE_PATH

    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _daemon_pid() -> int | None:
        if not PID_PATH.exists():
            return None
        try:
            pid = int(PID_PATH.read_text().strip())
        except (ValueError, OSError):
            _safe_unlink(PID_PATH)
            _safe_unlink(STATE_PATH)
            return None
        if not _pid_alive(pid):
            _safe_unlink(PID_PATH)
            _safe_unlink(STATE_PATH)
            return None
        try:
            # Verify it's actually our daemon, not a recycled PID.
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except PermissionError:
            return pid
        except OSError:
            _safe_unlink(PID_PATH)
            _safe_unlink(STATE_PATH)
            return None
        if "vserve.fan" not in cmdline:
            _safe_unlink(PID_PATH)
            _safe_unlink(STATE_PATH)
            return None
        return pid

    def _stop_daemon() -> bool:
        pid = _daemon_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, sig.SIGTERM)
        except ProcessLookupError:
            _safe_unlink(PID_PATH)
            _safe_unlink(STATE_PATH)
            return True
        except PermissionError:
            console.print(f"[red]Permission denied stopping fan daemon pid {pid}.[/red]")
            raise typer.Exit(1) from None
        for _ in range(50):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)
        if _pid_alive(pid):
            console.print(f"[red]Fan daemon pid {pid} did not stop within 5 seconds.[/red]")
            raise typer.Exit(1)
        _safe_unlink(PID_PATH)
        _safe_unlink(STATE_PATH)
        return True

    def _start_daemon(qs: int, qe: int, qm: int) -> None:
        _stop_daemon()
        # Wait for old daemon to release its fan lock (cleanup in finally block)
        from vserve.lock import wait_for_release
        if not wait_for_release("fan", timeout=5.0):
            console.print("[red]Old fan daemon did not release lock in time.[/red]")
            raise typer.Exit(1)
        import subprocess
        import sys
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "vserve.fan",
                "--quiet-start", str(qs),
                "--quiet-end", str(qe),
                "--quiet-max", str(qm),
            ],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        if proc.poll() is not None:
            from vserve.config import cfg as _cfg
            console.print(f"[red]Fan daemon failed to start.[/red] Check {_cfg().logs_dir / 'vserve-fan.log'}")
            raise typer.Exit(1)
        console.print(f"[green]Fan daemon started[/green] (pid {proc.pid})")

    def _start_fixed_daemon(speed: int) -> None:
        _stop_daemon()
        from vserve.lock import wait_for_release
        if not wait_for_release("fan", timeout=5.0):
            console.print("[red]Old fan daemon did not release lock in time.[/red]")
            raise typer.Exit(1)
        import subprocess
        import sys
        proc = subprocess.Popen(
            [sys.executable, "-m", "vserve.fan", "--fixed", str(speed)],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        if proc.poll() is not None:
            from vserve.config import cfg as _cfg
            console.print(f"[red]Fan daemon failed to start.[/red] Check {_cfg().logs_dir / 'vserve-fan.log'}")
            raise typer.Exit(1)
        console.print(f"[green]Fan held at {speed}%[/green] (pid {proc.pid})")

    def _is_orphaned_fan() -> bool:
        """Check if fan daemon crashed, leaving stale PID/state files."""
        if _daemon_pid() is not None:
            return False  # daemon is alive
        _fan._resolve_paths()
        return _fan.PID_PATH.exists() or _fan.STATE_PATH.exists()

    def _clean_orphaned_fan() -> None:
        """Remove stale PID/state files from crashed daemon."""
        _fan._resolve_paths()
        _safe_unlink(_fan.PID_PATH)
        _safe_unlink(_fan.STATE_PATH)
        from vserve.gpu import restore_fan_auto
        try:
            restore_fan_auto()
            console.print("[green]Fan auto control restored.[/green]")
        except Exception:
            console.print("[yellow]Could not restore auto fan control via NVML.[/yellow]")

    def _show_status() -> None:
        try:
            speed: int | str = get_fan_speed()
        except Exception:
            speed = "?"
        pid = _daemon_pid()
        state = read_state() if pid else None
        try:
            from vserve.fan import _get_gpu_temp
            temp = _get_gpu_temp()
        except Exception:
            temp = None

        temp_str = f"{temp}°C" if temp is not None else "?"
        speed_str = f"{speed}%" if isinstance(speed, int) else "?"
        console.print(f"\n[bold]GPU Fan[/bold]  {speed_str} @ {temp_str}")

        if _is_orphaned_fan():
            console.print("  [red]Warning: fan daemon crashed — fan may be stuck in manual mode[/red]")
            console.print("  [red]Run: sudo vserve fan off[/red]")
        elif state and "fixed" in state:
            console.print(f"  [green]Fixed at {state['fixed']}%[/green] (daemon holding)")
        elif state:
            qs, qe, qm = state["quiet_start"], state["quiet_end"], state["quiet_max"]
            console.print(f"  [green]Auto curve[/green] — quiet {qs:02d}:00-{qe:02d}:00 (max {qm}%), otherwise 100%")
        elif pid:
            console.print("  [green]Daemon running[/green]")
        else:
            console.print("  [dim]No daemon — NVIDIA auto[/dim]")

    # --- Direct mode (non-interactive) ---
    if mode == "auto":
        _sudo_reexec()
        _start_daemon(9, 18, 60)
        return
    if mode == "off":
        if _daemon_pid() or _is_orphaned_fan():
            _sudo_reexec()
        if _stop_daemon():
            console.print("[green]Fan daemon stopped[/green], auto control restored.")
        elif _is_orphaned_fan():
            _clean_orphaned_fan()
        else:
            console.print("[dim]No fan daemon running.[/dim]")
        return
    if mode is not None:
        try:
            percent = int(mode)
        except ValueError:
            console.print(f"[red]Unknown mode '{mode}'.[/red] Use: auto, off, or 30-100")
            raise typer.Exit(1)
        if not 30 <= percent <= 100:
            console.print("[red]Fan speed must be 30-100.[/red]")
            raise typer.Exit(1)
        _sudo_reexec()
        _start_fixed_daemon(percent)
        return

    # --- Interactive mode ---
    _sudo_reexec()
    _show_status()

    pid = _daemon_pid()
    state = read_state() if pid else None

    options: list[tuple[str, str]] = []
    if pid:
        options.append(("curve", "Change quiet hours / fan cap"))
        options.append(("fixed", "Fixed speed"))
        options.append(("off", "Off — restore NVIDIA auto"))
    else:
        options.append(("curve", "Auto curve (temp-based with quiet hours)"))
        options.append(("fixed", "Fixed speed"))

    console.print()
    descs = [desc for _, desc in options]
    idx = _pick(descs, title="Fan mode:")
    if idx is None:
        return
    action = options[idx][0]

    if action == "off":
        _stop_daemon()
        console.print("[green]Fan daemon stopped[/green], auto control restored.")
        return

    if action == "fixed":
        val = typer.prompt("Fan speed % (30-100)", type=int)
        if not 30 <= val <= 100:
            console.print(f"[red]Fan speed must be 30-100, got {val}.[/red]")
            raise typer.Exit(1)
        _start_fixed_daemon(val)
        return

    # action == "curve"
    defaults = state if state and "fixed" not in state else {"quiet_start": 9, "quiet_end": 18, "quiet_max": 60}
    qs = defaults["quiet_start"]
    qe = defaults["quiet_end"]
    qm = defaults["quiet_max"]

    console.print(f"\n  Current: quiet {qs:02d}:00-{qe:02d}:00, max {qm}%")
    qs = typer.prompt("  Quiet start hour (0-23)", type=int, default=qs)
    qe = typer.prompt("  Quiet end hour (0-23)", type=int, default=qe)
    qm = typer.prompt("  Quiet max fan % (30-100)", type=int, default=qm)

    if not 0 <= qs <= 23:
        console.print(f"[red]Quiet start hour must be 0-23, got {qs}.[/red]")
        raise typer.Exit(1)
    if not 0 <= qe <= 23:
        console.print(f"[red]Quiet end hour must be 0-23, got {qe}.[/red]")
        raise typer.Exit(1)
    if not 30 <= qm <= 100:
        console.print(f"[red]Quiet max fan must be 30-100, got {qm}.[/red]")
        raise typer.Exit(1)

    _start_daemon(qs, qe, qm)
    console.print(f"  Quiet {qs:02d}:00-{qe:02d}:00 (max {qm}%), otherwise 100%")


@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show current serving status."""
    from vserve.backends import _BACKENDS
    from vserve.config import active_yaml_path, read_active_manifest, try_read_profile_yaml

    import json as _json

    def _backend_manifest(backend) -> dict | None:
        manifest_path_fn = getattr(backend, "active_manifest_path", None)
        if not callable(manifest_path_fn):
            return None
        try:
            manifest_path = manifest_path_fn()
            if not isinstance(manifest_path, (str, pathlib.Path)):
                return None
            return read_active_manifest(pathlib.Path(manifest_path))
        except Exception:
            return None

    def _read_backend_config(backend, manifest: dict | None) -> tuple[dict, pathlib.Path | None, pathlib.Path | None]:
        cfg_data: dict = {}
        config_source: pathlib.Path | None = None
        invalid_source: pathlib.Path | None = None

        manifest_config = manifest.get("config_path") if isinstance(manifest, dict) else None
        if isinstance(manifest_config, str) and manifest_config:
            candidate = pathlib.Path(manifest_config)
            if candidate.exists():
                config_source = candidate
                try:
                    if candidate.suffix.lower() == ".json":
                        data = _json.loads(candidate.read_text())
                        if isinstance(data, dict):
                            return data, candidate, None
                        return {}, candidate, candidate
                    data = try_read_profile_yaml(candidate)
                    if data:
                        return data, candidate, None
                    return {}, candidate, candidate
                except Exception:
                    return {}, candidate, candidate

        if backend.name == "llamacpp":
            active_config_path_fn = getattr(backend, "_active_config_path", None)
            active_config_path = active_config_path_fn() if callable(active_config_path_fn) else None
            active_json = (
                pathlib.Path(active_config_path).with_suffix(".json")
                if isinstance(active_config_path, (str, pathlib.Path))
                else None
            )
            if active_json is not None and active_json.exists():
                try:
                    data = _json.loads(active_json.read_text())
                    if isinstance(data, dict):
                        cfg_data = data
                        config_source = active_json
                    else:
                        invalid_source = active_json
                        config_source = active_json
                except Exception:
                    invalid_source = active_json
                    config_source = active_json
        else:
            active = active_yaml_path()
            active_is_symlink = active.is_symlink()
            if active_is_symlink or _safe_path_exists(active):
                resolved = _safe_resolve_path(active) if active_is_symlink else active
                config_source = resolved or active
                data = try_read_profile_yaml(active)
                if data:
                    cfg_data = data
                else:
                    invalid_source = config_source
        return cfg_data, config_source, invalid_source

    backend_entries: list[dict] = []
    running_backends = []
    probe_errors: list[str] = []
    backend_details: dict[str, tuple[dict, pathlib.Path | None, pathlib.Path | None, dict | None]] = {}
    for b in _BACKENDS:
        probe_error = None
        try:
            running = bool(b.is_running())
        except Exception as exc:
            running = None
            probe_error = f"{b.display_name}: {exc}"
            probe_errors.append(probe_error)
        manifest = _backend_manifest(b)
        cfg_data, config_source, invalid_source = _read_backend_config(b, manifest)
        backend_details[b.name] = (cfg_data, config_source, invalid_source, manifest)
        port = cfg_data.get("port") if cfg_data else None
        if port is None and isinstance(manifest, dict):
            port = manifest.get("port")
        endpoint = None
        health_url = None
        if isinstance(port, int):
            endpoint = f"http://localhost:{port}/v1"
            try:
                candidate_health_url = b.health_url(port)
                health_url = candidate_health_url if isinstance(candidate_health_url, str) else None
            except Exception:
                health_url = None
        entry = {
            "name": b.name,
            "display_name": b.display_name,
            "running": running,
            "probe_error": probe_error,
            "config_source": str(config_source) if config_source else None,
            "config_readable": bool(cfg_data),
            "config_invalid": str(invalid_source) if invalid_source else None,
            "active_manifest": manifest,
            "port": port,
            "endpoint": endpoint,
            "health_url": health_url,
        }
        if cfg_data:
            model_path = cfg_data.get("model")
            entry["model"] = model_path
            entry["model_name"] = (
                str(model_path).split("/")[-1] if "/" in str(model_path) else model_path
            )
        backend_entries.append(entry)
        if running is True:
            running_backends.append(b)

    if not running_backends:
        if json_output:
            typer.echo(_json.dumps({
                "running": False,
                "running_backends": [],
                "multiple_active": len(running_backends) > 1,
                "uncertain": bool(probe_errors),
                "probe_errors": probe_errors,
                "backends": backend_entries,
                "next_action": "resolve probe errors" if probe_errors else "vserve run",
            }, sort_keys=True))
            if probe_errors:
                raise typer.Exit(1)
            return
        if probe_errors:
            console.print("[red]Could not determine server state.[/red]")
            for detail in probe_errors:
                console.print(f"  {detail}")
            raise typer.Exit(1)
        console.print("[dim]No server is running.[/dim] Start with: vserve run")
        return

    running_backend = running_backends[0]
    cfg, config_source, invalid_source, manifest = backend_details.get(running_backend.name, ({}, None, None, None))
    if not cfg:
        if json_output:
            primary_entry = next((entry for entry in backend_entries if entry["name"] == running_backend.name), {})
            typer.echo(_json.dumps({
                "running": True,
                "backend": running_backend.name,
                "running_backends": [
                    {"name": b.name, "display_name": b.display_name}
                    for b in running_backends
                ],
                "multiple_active": len(running_backends) > 1,
                "uncertain": bool(probe_errors),
                "probe_errors": probe_errors,
                "backends": backend_entries,
                "config_source": primary_entry.get("config_source"),
                "config_readable": False,
                "active_manifest": manifest,
            }, sort_keys=True))
            return
        if invalid_source is not None:
            console.print(f"[bold green]{running_backend.display_name} is running[/bold green] (active config unreadable)")
            console.print(f"  [dim]{invalid_source}[/dim]\n")
            return
        console.print(f"[bold green]{running_backend.display_name} is running[/bold green] (no active config found)")
        return

    model_path = cfg.get("model", "?")
    model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path
    port = cfg.get("port", 8888)
    manifest_state = manifest.get("status") if isinstance(manifest, dict) else None

    def _running_next_action() -> str:
        if len(running_backends) > 1:
            return "stop extra active backend"
        if manifest_state in {"starting", "warming"}:
            return "wait for health"
        if manifest_state == "failed":
            return "inspect failed launch"
        if manifest_state and manifest_state != "ready":
            return "check backend state"
        return "ready"

    if json_output:
        typer.echo(_json.dumps({
            "running": True,
            "backend": running_backend.name,
            "display_name": running_backend.display_name,
            "running_backends": [
                {"name": b.name, "display_name": b.display_name}
                for b in running_backends
            ],
            "multiple_active": len(running_backends) > 1,
            "uncertain": bool(probe_errors),
            "probe_errors": probe_errors,
            "backends": backend_entries,
            "model": model_path,
            "model_name": model_name,
            "port": port,
            "endpoint": f"http://localhost:{port}/v1",
            "config_source": str(config_source) if config_source else None,
            "config_readable": True,
            "active_manifest": manifest,
            "next_action": _running_next_action(),
        }, sort_keys=True))
        return

    console.print(f"\n[bold green]{running_backend.display_name} is running[/bold green]")
    if len(running_backends) > 1:
        names = ", ".join(backend.display_name for backend in running_backends)
        console.print(f"  [bold red]Multiple active backends[/bold red]  {names}")
        console.print("  [yellow]Run vserve stop, then start one backend again.[/yellow]")
    if manifest:
        if manifest_state and manifest_state != "ready":
            console.print(f"  [bold]State[/bold]      {manifest_state}")
    console.print(f"  [bold]Model[/bold]      {model_name}")
    console.print(f"  [bold]Endpoint[/bold]   http://localhost:{port}/v1")
    console.print()

    console.print("  [bold]Serving config[/bold]")

    if running_backend.name == "llamacpp":
        # Prefer ctx_per_slot (vserve 0.5.9+) so the displayed value matches
        # the per-slot window the user picked. ctx_size is the llama-server
        # `-c` value, which is total ctx across all slots.
        ctx = cfg.get("ctx_per_slot") or cfg.get("ctx_size", "?")
        ctx_display = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"    Context window:    {ctx_display}  (per-slot)")
        console.print(f"    Concurrent slots:  {cfg.get('parallel', '?')}  (max in-flight requests)")
        ngl = cfg.get("n_gpu_layers")
        if ngl is not None:
            console.print(f"    GPU layers:        {ngl}")
        if cfg.get("flash_attn"):
            console.print("    Flash attention:   on")
        ctk = cfg.get("cache_type_k")
        ctv = cfg.get("cache_type_v")
        if ctk or ctv:
            console.print(f"    KV cache dtype:    K={ctk or 'f16'} V={ctv or 'f16'}")
        if cfg.get("override_tensors"):
            ot = ", ".join(cfg["override_tensors"])
            console.print(f"    Override tensors:  {ot}")
        if cfg.get("embedding"):
            pooling = cfg.get("pooling", "mean")
            console.print(f"    Mode:              embedding (pooling: {pooling})")
        if cfg.get("jinja"):
            console.print("    Tool calling:      enabled (--jinja)")
    else:
        # vLLM
        ctx = cfg.get("max-model-len", "?")
        ctx_display = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"    Context window:    {ctx_display}")
        console.print(f"    Concurrent slots:  {cfg.get('max-num-seqs', '?')}  (max in-flight requests)")
        console.print(f"    KV cache dtype:    {cfg.get('kv-cache-dtype', 'auto')}")
        if cfg.get("enable-prefix-caching"):
            console.print("    Prefix caching:    enabled")
        bt = cfg.get("max-num-batched-tokens")
        if bt:
            console.print(f"    Batched tokens:    {bt}")
        gpu_util = cfg.get("gpu-memory-utilization")
        if isinstance(gpu_util, float):
            console.print(f"    GPU memory:        {gpu_util:.1%}")

    # GPU memory bar (shared for both backends)
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().split("\n")[0]
        used, total = [int(x.strip()) for x in out.split(",")]
        pct = used * 100 // total if total else 0
        bar_w = 20
        filled = pct * bar_w // 100
        bar = "[green]" + "█" * filled + "[/green][dim]" + "░" * (bar_w - filled) + "[/dim]"
        console.print(f"\n  [bold]GPU[/bold]        {bar}  {used / 1024:.1f} / {total / 1024:.1f} GB ({pct}%)")
    except Exception:
        pass

    if config_source:
        console.print(f"\n  [dim]{config_source}[/dim]\n")


@app.command()
def init():
    """Set up vserve — scan the system, write config, optionally install login banner."""
    import shutil
    import subprocess
    from pathlib import Path as _Path

    from vserve.config import (
        CONFIG_FILE, save_config,
        _discover_vllm_root, _discover_cuda_home, _discover_service, _discover_port,
        _build_config, reset_config, find_systemd_unit_path,
    )

    def _ok(msg: str) -> None:
        console.print(f"  [green]{msg}[/green]")

    def _warn(msg: str) -> None:
        console.print(f"  [yellow]{msg}[/yellow]")

    def _fail(msg: str) -> None:
        console.print(f"  [red]{msg}[/red]")

    console.print("\n[bold]vserve setup[/bold]\n")
    console.print("[dim]Scanning your system...[/dim]\n")

    # ── GPU ──
    try:
        from vserve.gpu import get_gpu_info
        gpu = get_gpu_info()
        _ok(f"GPU         {gpu.name} ({gpu.vram_total_gb:.0f} GB)")
        _ok(f"Driver      {gpu.driver}")
        _ok(f"CUDA        {gpu.cuda}")
    except Exception:
        _fail("GPU         nvidia-smi not found or no GPU detected")
        _fail("            vserve requires an NVIDIA GPU with drivers installed")
        _fail("            Install drivers: https://www.nvidia.com/drivers")

    # ── CUDA toolkit ──
    cuda = _discover_cuda_home()
    nvcc = shutil.which("nvcc")
    if nvcc:
        _ok(f"nvcc        {cuda}")
    else:
        _fail("nvcc        not on PATH — needed for first-run JIT compilation")
        _fail("            Install: sudo apt install nvidia-cuda-toolkit")
        _warn(f"            using fallback: {cuda}")

    # Discover optional backends before printing backend-specific guidance.
    detected_vllm_root = _discover_vllm_root()
    root = detected_vllm_root or _Path("/opt/vllm")
    llamacpp_root = None
    llamacpp_bin = shutil.which("llama-server")
    llamacpp_candidate = _Path("/opt/llama-cpp")
    if (llamacpp_candidate / "bin" / "llama-server").exists():
        llamacpp_root = llamacpp_candidate
    elif llamacpp_bin:
        llamacpp_root = _Path(llamacpp_bin).resolve().parent.parent

    # ── vLLM ──
    if detected_vllm_root:
        # Check vLLM version
        vllm_bin = root / "venv" / "bin" / "vllm"
        if not vllm_bin.exists():
            vllm_bin = root / ".venv" / "bin" / "vllm"
        try:
            r = subprocess.run(
                [str(vllm_bin), "--version"],
                capture_output=True, text=True, timeout=10,
            )
            ver = r.stdout.strip() or r.stderr.strip()
            _ok(f"vLLM        {ver} ({root})")
        except Exception:
            _ok(f"vLLM        found at {root}")
    elif llamacpp_root:
        _warn("vLLM        not found (optional — needed for safetensors models)")
    else:
        _warn("vLLM        not found (install it for safetensors models)")
        _warn("            See: https://docs.vllm.ai/en/latest/getting_started/installation.html")

    # ── llama.cpp ──
    if llamacpp_root == llamacpp_candidate:
        _ok(f"llama.cpp   found at {llamacpp_candidate}")
    elif llamacpp_bin:
        _ok(f"llama.cpp   {llamacpp_bin}")
    elif detected_vllm_root:
        _warn("llama.cpp   not found (optional — for GGUF models)")
    else:
        _warn("llama.cpp   not found (install it for GGUF models)")

    if not detected_vllm_root and not llamacpp_root:
        _warn("Backends    no serving backend detected yet")
        _warn("            Install vLLM and/or llama.cpp, then rerun vserve init")

    # ── Models ──
    if detected_vllm_root:
        models_dir = root / "models"
        if models_dir.is_dir():
            models = list(models_dir.glob("*/*/config.json"))
            _ok(f"Models      {len(models)} found at {models_dir}")
        else:
            _warn(f"Models      {models_dir} does not exist")
    if llamacpp_root:
        lc_models = llamacpp_root / "models"
        if lc_models.is_dir():
            from vserve.model_files import iter_recursive_files_with_suffix
            gguf_count = len(iter_recursive_files_with_suffix(lc_models, ".gguf"))
            _ok(f"GGUF models {gguf_count} found at {lc_models}")
        else:
            _warn(f"GGUF models {lc_models} does not exist")

    # ── systemd (vLLM) ──
    svc_name, svc_user = _discover_service()
    if detected_vllm_root:
        svc_path = find_systemd_unit_path(svc_name)
        if svc_path is not None:
            _ok(f"systemd     {svc_name}.service (user: {svc_user})")
        else:
            _fail("systemd     no vLLM systemd service found (required for vLLM runs)")
            _fail(f"            Create /etc/systemd/system/{svc_name}.service with:")
            _fail("            [Service] ExecStart=/opt/vllm/venv/bin/vllm serve ...")
    else:
        _warn("systemd     vLLM service not checked (vLLM not installed)")

    # ── systemd (llama.cpp) ──
    if llamacpp_root:
        lc_svc_name = "llama-cpp"
        lc_svc_user = "llama-cpp"
        lc_svc_path = find_systemd_unit_path(lc_svc_name)
        if lc_svc_path is not None:
            _ok(f"systemd     {lc_svc_name}.service")
        else:
            _warn(f"systemd     no {lc_svc_name}.service found")
            _warn(f"            Create /etc/systemd/system/{lc_svc_name}.service with:")
            _warn(f"            [Service] ExecStart={llamacpp_root}/configs/active.sh")

        # Service user
        try:
            r = subprocess.run(["id", lc_svc_user], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                _ok(f"user        {lc_svc_user} exists")
            else:
                _warn(f"user        {lc_svc_user} not found")
                _warn(f"            sudo useradd -r -s /usr/sbin/nologin -g llm {lc_svc_user}")
        except Exception:
            pass

    # ── Port ──
    port = _discover_port(root)
    _ok(f"Port        {port}")

    # ── NVML fan control ──
    try:
        from vserve.gpu import get_fan_count

        fan_count = get_fan_count()
        if fan_count > 0:
            _ok(f"Fan control {fan_count} fan(s) exposed via NVML (root required)")
        else:
            _warn("Fan control no controllable fans exposed via NVML")
    except Exception:
        _warn("Fan control NVML unavailable")

    # ── gum (interactive UI) ──
    gum_available = _has_gum()
    if gum_available:
        _ok("gum         installed (enhanced interactive menus)")
    else:
        _warn("gum         not installed (using built-in menus)")

    # ── Write config ──
    console.print()
    if CONFIG_FILE.exists():
        console.print(f"  [dim]Config exists at {CONFIG_FILE}[/dim]")
        if not typer.confirm("  Overwrite config?", default=True):
            console.print()
        else:
            config = _build_config(root, cuda, svc_name, svc_user, port, llamacpp_root=llamacpp_root)
            save_config(config)
            reset_config()
            _ok(f"Config written to {CONFIG_FILE}")
    else:
        config = _build_config(root, cuda, svc_name, svc_user, port, llamacpp_root=llamacpp_root)
        save_config(config)
        reset_config()
        _ok(f"Config written to {CONFIG_FILE}")

    # ── Welcome banner ──
    banner_src = _Path(__file__).parent / "welcome.sh"
    banner_dest = CONFIG_FILE.parent / "welcome.sh"
    _marker = "# vserve login banner"

    # Detect shell rc file
    import os
    _shell = _Path(os.environ.get("SHELL", "/bin/bash")).name
    if _shell == "zsh":
        _rc = _Path.home() / ".zshrc"
    elif _shell == "fish":
        _rc = _Path.home() / ".config" / "fish" / "config.fish"
    else:
        _rc = _Path.home() / ".bashrc"

    _source_line = f'[ -f "{banner_dest}" ] && source "{banner_dest}"'
    _rc_has_marker = _rc.exists() and _marker in _rc.read_text()
    has_banner = banner_dest.exists() and _rc_has_marker

    if _shell == "fish":
        console.print("  [yellow]Login banner not supported for fish shell.[/yellow]")
    elif not gum_available:
        if has_banner:
            console.print("  [yellow]Login banner is installed but inactive until gum is installed.[/yellow]")
        else:
            console.print("  [dim]Login banner unavailable until gum is installed.[/dim]")
    elif has_banner:
        console.print(f"  [dim]Login banner already installed ({_rc.name} → {banner_dest})[/dim]")
        if typer.confirm("  Reinstall login banner?", default=False):
            import shutil
            shutil.copy2(banner_src, banner_dest)
            _ok(f"Banner updated at {banner_dest}")
    else:
        console.print()
        console.print("  [bold]Login banner[/bold]")
        console.print("  Shows GPU status, model, and commands on every SSH login.")
        console.print("  Requires: gum")
        if typer.confirm("  Install login banner?", default=True):
            import shutil
            shutil.copy2(banner_src, banner_dest)
            if not _rc_has_marker:
                with open(_rc, "a") as f:
                    f.write(f"\n{_marker}\n{_source_line}\n")
            _ok(f"Banner installed ({_rc.name} → {banner_dest})")
        else:
            console.print("  [dim]Skipped.[/dim]")

    # ── Next steps ──
    console.print()
    console.print("[bold]Next steps[/bold]")
    if not detected_vllm_root and not llamacpp_root:
        console.print("  Install vLLM and/or llama.cpp, then rerun vserve init")
    console.print("  vserve add      Search & download a model")
    console.print("  vserve doctor   Full system check")
    console.print("  vserve          Dashboard")
    console.print()


def _doctor_summary_label(warn_count: int, fail_count: int) -> str:
    if fail_count == 0 and warn_count == 0:
        return "All clear"
    if fail_count == 0:
        return f"{warn_count} warning(s) found"
    if warn_count == 0:
        return f"{fail_count} issue(s) found"
    return f"{fail_count} issue(s) and {warn_count} warning(s) found"


@app.command()
def doctor(
    strict: bool = typer.Option(False, "--strict", help="Exit nonzero if any check fails"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Check system readiness."""
    import json
    import os
    import subprocess
    import socket
    from pathlib import Path

    from vserve.config import (
        VLLM_BIN,
        VLLM_ROOT,
        active_yaml_path,
        try_read_profile_yaml,
        LOGS_DIR,
        find_systemd_unit_path,
        unit_uses_environment_file,
    )
    from vserve.backends import running_backend as _running_backend

    ok_count = 0
    warn_count = 0
    fail_count = 0
    checks: list[dict[str, str]] = []

    def _emit(*args, **kwargs) -> None:
        if not json_output:
            console.print(*args, **kwargs)

    def _ok(msg: str) -> None:
        nonlocal ok_count
        checks.append({"status": "ok", "message": msg, "fix": ""})
        _emit(f"  [green]OK[/green]    {msg}")
        ok_count += 1

    def _fail(msg: str, fix: str = "") -> None:
        nonlocal fail_count
        checks.append({"status": "fail", "message": msg, "fix": fix})
        _emit(f"  [red]FAIL[/red]  {msg}")
        if fix:
            _emit(f"          Fix: {fix}")
        fail_count += 1

    def _warn(msg: str, fix: str = "") -> None:
        nonlocal warn_count
        checks.append({"status": "warn", "message": msg, "fix": fix})
        _emit(f"  [yellow]WARN[/yellow]  {msg}")
        if fix:
            _emit(f"          Fix: {fix}")
        warn_count += 1

    def _fail_or_warn(required: bool, msg: str, fix: str = "") -> None:
        if required:
            _fail(msg, fix)
        else:
            _warn(msg, fix)

    _emit("\n[bold]vserve doctor[/bold]\n")

    from vserve.config import cfg as _cfg
    _c = _cfg()
    vllm_runtime_present = VLLM_BIN.exists()
    vllm_required = vllm_runtime_present or _c.llamacpp_root is None

    # -- Environment --
    _emit("  [bold]Environment[/bold]")

    # nvcc
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5,
                           env={**os.environ, "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", "")})
        if r.returncode == 0:
            nvcc_lines = [ln for ln in r.stdout.splitlines() if "release" in ln]
            _ok(f"nvcc {nvcc_lines[0].split('release')[-1].strip().rstrip(',') if nvcc_lines else 'found'}")
        else:
            _fail("nvcc not working", "Install CUDA toolkit or check /usr/local/cuda/bin/nvcc")
    except Exception:
        _fail("nvcc not found", "Install: sudo apt install nvidia-cuda-toolkit  OR  https://developer.nvidia.com/cuda-downloads")

    # vLLM
    if vllm_required:
        try:
            r = subprocess.run([str(VLLM_BIN), "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ver = r.stdout.strip() or r.stderr.strip() or "found"
                _ok(f"vLLM {ver}")
            else:
                details = r.stderr.strip() or r.stdout.strip()
                _fail(f"vLLM not working at {VLLM_BIN}", details or "Check the vLLM installation and environment")
        except Exception:
            _fail(f"vLLM not found at {VLLM_BIN}")
    else:
        _warn("vLLM not configured (skipped — llama.cpp-only setup)")

    # GPU
    try:
        from vserve.gpu import get_gpu_info
        gpu = get_gpu_info()
        mem_used = 0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5)
            mem_used = int(out.decode().strip().split("\n")[0])
        except Exception:
            pass
        _ok(f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, {mem_used} MiB used)")
    except Exception:
        _fail("GPU not accessible", "Install NVIDIA drivers: https://www.nvidia.com/drivers  then check: nvidia-smi")

    # -- Backends --
    _emit("\n  [bold]Backends[/bold]")
    from vserve.backends import _BACKENDS
    for b in _BACKENDS:
        for desc, check_fn in b.doctor_checks():
            try:
                if check_fn():
                    _ok(desc)
                else:
                    _warn(desc)
            except Exception:
                _warn(f"{desc} (check error)")

    # -- Per-backend checks --
    # ── vLLM ──
    _emit("\n  [bold]vLLM[/bold]")

    # Runtime compatibility
    if not vllm_required:
        _warn("vLLM checks skipped (llama.cpp-only setup)")
    else:
        try:
            from vserve.runtime import check_vllm_compatibility, collect_vllm_runtime_info

            runtime_info = collect_vllm_runtime_info(_c)
            runtime_check = check_vllm_compatibility(runtime_info)
            if runtime_check.supported:
                _ok(f"Runtime supported ({runtime_info.vllm_version}, {runtime_check.range})")
            else:
                _fail("Unsupported vLLM runtime", "; ".join(runtime_check.errors))
            for warning in runtime_check.warnings:
                _warn(warning)
        except Exception as exc:
            _warn("Could not complete vLLM runtime compatibility check", str(exc))

    # Service user
    try:
        r = subprocess.run(["id", _c.service_user], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok(f"Service user '{_c.service_user}' exists")
        else:
            _fail_or_warn(vllm_required, f"Service user '{_c.service_user}' not found")
    except Exception:
        _fail_or_warn(vllm_required, "Cannot check service user")

    # systemd unit
    svc_path = find_systemd_unit_path(_c.service_name)
    if svc_path is not None:
        try:
            svc_content = svc_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f"Could not read systemd unit: {svc_path.name}", str(exc))
        else:
            unit_failed = False
            if "ProtectSystem=strict" in svc_content:
                _fail_or_warn(
                    vllm_required,
                    "systemd unit has ProtectSystem=strict",
                    "Remove it — breaks nvcc JIT compilation",
                )
                unit_failed = vllm_required
            if "TimeoutStartSec" not in svc_content:
                _warn("No TimeoutStartSec in service — default 90s may be too short for JIT",
                      "Add TimeoutStartSec=600")
            env_path = VLLM_ROOT / "configs" / ".env"
            if not unit_uses_environment_file(svc_content, env_path):
                _fail_or_warn(
                    vllm_required,
                    "systemd unit does not load vserve configs/.env",
                    f"Add EnvironmentFile={env_path} so CUDA_VISIBLE_DEVICES is enforced",
                )
                unit_failed = vllm_required
            if not unit_failed:
                _ok("systemd unit configured correctly")
    else:
        _fail_or_warn(
            vllm_required,
            f"No systemd unit found for {_c.service_name}.service",
            "Create one: https://docs.vllm.ai  or see vserve docs/troubleshooting.md",
        )

    # .env
    env_path = VLLM_ROOT / "configs" / ".env"
    if env_path.exists():
        try:
            env_content = env_path.read_text()
            missing = [v for v in ["CUDA_HOME", "TMPDIR", "VLLM_RPC_BASE_PATH", "CUDA_VISIBLE_DEVICES"] if v not in env_content]
            if missing:
                _warn(f".env missing: {', '.join(missing)}")
            else:
                _ok(".env has required variables")
        except PermissionError:
            _ok(".env exists (not readable — OK, contains secrets)")
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f".env unreadable: {env_path}", str(exc))
    else:
        _fail_or_warn(vllm_required, f"No .env at {env_path}")

    # Models directory
    vllm_models = VLLM_ROOT / "models"
    if vllm_models.exists():
        try:
            vllm_mc = sum(1 for p in vllm_models.glob("*/*/config.json"))
        except OSError as exc:
            _warn(f"Models dir unreadable: {vllm_models}", str(exc))
        else:
            _ok(f"Models dir: {vllm_models} ({vllm_mc} models)")
    else:
        _warn(f"Models dir {vllm_models} does not exist")

    # Caches
    cache_checks = [
        (VLLM_ROOT / ".cache" / "flashinfer", "FlashInfer JIT"),
        (VLLM_ROOT / ".cache" / "vllm" / "torch_compile_cache", "torch.compile"),
    ]
    for cdir, label in cache_checks:
        if not cdir.exists():
            _warn(f"{label} cache missing — first start will JIT compile (2-10 min)")
            continue
        try:
            files = []
            for f in cdir.rglob("*"):
                try:
                    if f.is_file():
                        files.append(f)
                except OSError:
                    continue
        except OSError as exc:
            _warn(f"{label} cache unreadable", str(exc))
            continue
        if not files:
            _warn(f"{label} cache dir exists but is empty — first start may be slow")
            continue
        size_bytes = 0
        for f in files:
            try:
                size_bytes += f.stat().st_size
            except OSError:
                continue
        _ok(f"{label} cache ({size_bytes / (1024 * 1024):.0f} MB)")

    # Active config
    active = active_yaml_path()
    active_is_symlink = active.is_symlink()
    if active_is_symlink:
        target = _safe_resolve_path(active)
        if target is None:
            _warn("active.yaml symlink is unreadable or recursive")
        elif _safe_path_exists(target):
            cfg_data = try_read_profile_yaml(active)
            if cfg_data is None:
                _warn(f"active.yaml unreadable: {target.name}")
            else:
                model_path = cfg_data.get("model", "")
                if isinstance(model_path, (str, os.PathLike)) and model_path and Path(model_path).exists():
                    _ok(f"active.yaml → {target.name}")
                else:
                    _fail_or_warn(vllm_required, f"active.yaml model path missing: {model_path}")
        else:
            _fail_or_warn(vllm_required, f"active.yaml → broken symlink: {target}")
    elif _safe_path_exists(active):
        _ok("active.yaml exists (not a symlink)")
    else:
        _ok("No active.yaml (clean — will be created on vserve run)")

    # TMPDIR
    tmp_dir = VLLM_ROOT / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        _ok(f"TMPDIR at {tmp_dir}")
        try:
            sockets = list(tmp_dir.rglob("*"))
            stale = [s for s in sockets if s.is_socket()]
        except OSError as exc:
            _warn(f"Could not inspect TMPDIR sockets: {tmp_dir}", str(exc))
        else:
            if len(stale) > 10:
                _warn(f"{len(stale)} stale sockets in {tmp_dir}", f"sudo find {tmp_dir} -type s -delete")
    else:
        _fail_or_warn(
            vllm_required,
            f"TMPDIR {tmp_dir} does not exist",
            f"sudo mkdir -p {tmp_dir} && sudo chown vllm:llm {tmp_dir}",
        )

    # ── llama.cpp ──
    _emit("\n  [bold]llama.cpp[/bold]")

    lc_root = _c.llamacpp_root
    if lc_root is None:
        _warn("llama.cpp not configured (run vserve init to detect)")
    else:
        _ok(f"Root: {lc_root}")

        # Binary
        lc_bin = lc_root / "bin" / "llama-server"
        if lc_bin.exists():
            try:
                r = subprocess.run([str(lc_bin), "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    ver_line = ""
                    for ln in (r.stdout + r.stderr).splitlines():
                        if "version" in ln.lower() or ln.startswith("b"):
                            ver_line = ln.strip()
                            break
                    _ok(f"llama-server {ver_line}" if ver_line else f"llama-server at {lc_bin}")
                else:
                    details = r.stderr.strip() or r.stdout.strip()
                    _fail(f"llama-server at {lc_bin} is not working",
                          details or "Build or reinstall llama.cpp")
            except Exception:
                _fail(f"llama-server at {lc_bin} is not working")
        elif __import__("shutil").which("llama-server"):
            _ok("llama-server found on PATH")
        else:
            _fail("llama-server not found",
                  "Build: cmake -B build -DGGML_CUDA=ON && cmake --build build -t llama-server")

        # Service user
        lc_user = _c.llamacpp_service_user
        try:
            r = subprocess.run(["id", lc_user], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                _ok(f"Service user '{lc_user}' exists")
            else:
                _fail(f"Service user '{lc_user}' not found",
                      f"sudo useradd -r -s /usr/sbin/nologin -g llm {lc_user}")
        except Exception:
            _fail("Cannot check llama-cpp service user")

        # systemd unit (deep inspection, matching vLLM)
        lc_svc = _c.llamacpp_service_name
        lc_svc_path = find_systemd_unit_path(lc_svc)
        if lc_svc_path is not None:
            try:
                lc_svc_content = lc_svc_path.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                _warn(f"Could not read {lc_svc}.service", str(exc))
            else:
                issues = []
                active_sh = lc_root / "configs" / "active.sh"
                active_sh_has_cuda = False
                try:
                    active_sh_has_cuda = active_sh.exists() and "CUDA_VISIBLE_DEVICES" in active_sh.read_text()
                except (OSError, UnicodeDecodeError):
                    active_sh_has_cuda = False
                if "CUDA_VISIBLE_DEVICES" not in lc_svc_content and not active_sh_has_cuda:
                    issues.append("missing CUDA_VISIBLE_DEVICES (may use wrong GPU)")
                if "TimeoutStartSec" not in lc_svc_content:
                    issues.append("no TimeoutStartSec (large models need time)")
                if issues:
                    _warn(f"{lc_svc}.service: {'; '.join(issues)}")
                else:
                    _ok(f"{lc_svc}.service unit configured correctly")
        else:
            _fail(f"No systemd unit found for {lc_svc}.service",
                  f"Create /etc/systemd/system/{lc_svc}.service with ExecStart={lc_root}/configs/active.sh")

        # Models directory
        lc_models = lc_root / "models"
        if lc_models.exists():
            try:
                from vserve.model_files import iter_recursive_files_with_suffix
                model_count = len(iter_recursive_files_with_suffix(lc_models, ".gguf"))
            except OSError as exc:
                _warn(f"Models dir unreadable: {lc_models}", str(exc))
            else:
                _ok(f"Models dir: {lc_models} ({model_count} GGUF files)")
        else:
            _warn(f"Models dir {lc_models} does not exist",
                  f"sudo mkdir -p {lc_models} && sudo chown {lc_user}:llm {lc_models}")

        # Configs directory + active config
        lc_configs = lc_root / "configs"
        if lc_configs.exists():
            active_sh = lc_configs / "active.sh"
            active_json = lc_configs / "active.json"
            if active_sh.exists():
                _ok("active.sh launch script present")
            else:
                _ok(f"Configs dir: {lc_configs} (no active config yet)")
            if active_json.exists():
                try:
                    import json as _json2
                    lc_cfg = _json2.loads(active_json.read_text())
                    lc_model = lc_cfg.get("model", "")
                    if isinstance(lc_model, (str, os.PathLike)) and lc_model and Path(lc_model).exists():
                        _ok(f"active.json → {Path(lc_model).name}")
                    elif lc_model:
                        _fail(f"active.json model path missing: {lc_model}")
                except Exception:
                    _warn("active.json exists but unreadable")
        else:
            _warn(f"Configs dir {lc_configs} does not exist",
                  f"sudo mkdir -p {lc_configs} && sudo chown {lc_user}:llm {lc_configs}")

        # gguf package
        try:
            import gguf  # type: ignore[import-not-found, import-untyped]  # noqa: F401
            _ok("gguf package installed")
        except ImportError:
            _warn("gguf package not installed — needed for GGUF metadata reading",
                  "pip install 'vserve\\[llamacpp]'")

    # -- Shared --
    _emit("\n  [bold]Shared[/bold]")

    def _port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _health_ok(backend, port: int) -> bool:
        try:
            from urllib.request import urlopen
            with urlopen(backend.health_url(port), timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _active_backend_port(backend) -> int:
        try:
            if backend.name == "llamacpp":
                active_config_path_fn = getattr(backend, "_active_config_path", None)
                active_json = active_config_path_fn().with_suffix(".json") if callable(active_config_path_fn) else None
                if active_json is not None and active_json.exists():
                    data = json.loads(active_json.read_text())
                    if isinstance(data, dict):
                        return int(data.get("port", _c.port))
            active_cfg = try_read_profile_yaml(active_yaml_path()) or {}
            return int(active_cfg.get("port", _c.port))
        except Exception:
            return int(_c.port)

    # Port
    _port = _c.port
    port_in_use = _port_open(_port)
    active_backend = _running_backend()
    if active_backend is not None:
        active_port = _active_backend_port(active_backend)
        active_port_open = _port_open(active_port)
        active_health_ok = _health_ok(active_backend, active_port)
        if active_port_open or active_health_ok:
            _ok(f"{active_backend.display_name} serving on port {active_port}")
            if active_port != _port:
                _warn(f"Configured default port is {_port}, active backend uses {active_port}")
        else:
            _fail(
                f"{active_backend.display_name} appears active but port {active_port} is not open",
                f"Check: sudo journalctl -u {active_backend.service_name} --no-pager -n 50",
            )
    elif port_in_use:
        _fail(f"Port {_port} in use but no backend running — something else is bound",
              f"Check: sudo lsof -i :{_port}  then stop the other service or change port in ~/.config/vserve/config.yaml")
    else:
        _ok(f"Port {_port} available")

    # Log size
    log_file = LOGS_DIR / "vllm.log"
    if log_file.exists():
        try:
            size_mb = log_file.stat().st_size / (1024 * 1024)
        except OSError as exc:
            _warn(f"Could not read log metadata: {log_file}", str(exc))
        else:
            if size_mb > 100:
                _warn(f"vllm.log is {size_mb:.0f} MB", "Consider truncating or adding log rotation")
            else:
                _ok(f"vllm.log ({size_mb:.0f} MB)")

    # Multi-user messaging
    import grp
    try:
        tty_members = grp.getgrnam("tty").gr_mem
        import os
        me = os.environ.get("USER", "")
        if me in tty_members:
            _ok("tty group (terminal messaging between users)")
        else:
            _warn(f"'{me}' not in tty group — vserve can't DM other users",
                  f"sudo usermod -aG tty {me}  (then re-login)")
    except KeyError:
        _warn("tty group not found")

    # Models
    all_models = _all_models()
    probed = sum(1 for m in all_models if read_limits(limits_path(m.provider, m.model_name)))
    _ok(f"{len(all_models)} models downloaded, {probed} probed")

    summary = {"ok": ok_count, "warn": warn_count, "fail": fail_count}
    if json_output:
        typer.echo(json.dumps({"summary": summary, "checks": checks}, sort_keys=True))
    else:
        console.print(
            f"\n  [bold]{_doctor_summary_label(warn_count, fail_count)}[/bold]  "
            f"({ok_count} ok, {warn_count} warn, {fail_count} fail)\n",
        )
    if strict and fail_count:
        raise typer.Exit(1)


# ── Cache management ──

@cache_app.command("clean")
def cache_clean(
    all_caches: bool = typer.Option(False, "--all", help="Also clean flashinfer, torch, vllm caches"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be removed"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean stale sockets and optionally JIT caches."""
    import subprocess
    from vserve.backends import probe_running_backends
    from vserve.config import cfg as _cfg

    running_backends, probe_failed = probe_running_backends()
    if running_backends:
        names = ", ".join(b.display_name for b in running_backends)
        console.print(f"[red]Refusing to clean caches while {names} is running.[/red]")
        console.print("  Run: [cyan]vserve stop[/cyan]")
        raise typer.Exit(1)
    if probe_failed:
        console.print("[red]Refusing to clean caches because backend state is uncertain.[/red]")
        console.print("  Run: [cyan]vserve status[/cyan] and resolve probe errors first.")
        raise typer.Exit(1)

    _c = _cfg()
    root = _c.vllm_root
    def _unsafe_cache_root_reason(path: pathlib.Path) -> str | None:
        try:
            if path.exists():
                if path.is_symlink():
                    return "root is a symlink"
                if not path.is_dir():
                    return "root is not a directory"
                resolved = path.resolve(strict=True)
            else:
                resolved = path.resolve(strict=False)
        except OSError as exc:
            return str(exc)
        dangerous = {pathlib.Path("/").resolve(), pathlib.Path.home().resolve()}
        if resolved in dangerous:
            return f"dangerous root {resolved}"
        return None

    root_problem = _unsafe_cache_root_reason(root)
    if root_problem:
        console.print(f"[red]Refusing to clean caches under unsafe vLLM root: {root} ({root_problem}).[/red]")
        raise typer.Exit(1)

    total_freed = 0
    failures: list[str] = []
    lock = _lock_or_exit("cache", "cleaning backend caches")

    try:
        def _stale_sockets(tmp_dir: pathlib.Path) -> list[pathlib.Path]:
            try:
                return [path for path in tmp_dir.rglob("*") if path.is_socket()]
            except OSError:
                return []

        def _du_mb(path: pathlib.Path) -> int | None:
            commands = [["du", "-sm", str(path)], ["sudo", "-n", "du", "-sm", str(path)]]
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                except Exception:
                    continue
                if result.returncode != 0:
                    continue
                try:
                    return int(result.stdout.split()[0])
                except (ValueError, IndexError):
                    continue
            return None

        def _unsafe_descendant_reason(path: pathlib.Path, root: pathlib.Path) -> str | None:
            import stat

            try:
                relative = path.relative_to(root)
            except ValueError:
                return f"{path} is outside {root}"
            current = root
            for part in relative.parts:
                current = current / part
                try:
                    st = current.lstat()
                except FileNotFoundError:
                    break
                except OSError as exc:
                    return str(exc)
                if stat.S_ISLNK(st.st_mode):
                    return f"{current} is a symlink"
                if current != path and not stat.S_ISDIR(st.st_mode):
                    return f"{current} is not a directory"
            return None

        running_backends, probe_failed = probe_running_backends()
        if running_backends:
            names = ", ".join(b.display_name for b in running_backends)
            console.print(f"[red]Refusing to clean caches while {names} is running.[/red]")
            console.print("  Run: [cyan]vserve stop[/cyan]")
            raise typer.Exit(1)
        if probe_failed:
            console.print("[red]Refusing to clean caches because backend state is uncertain.[/red]")
            console.print("  Run: [cyan]vserve status[/cyan] and resolve probe errors first.")
            raise typer.Exit(1)

        # Always: stale sockets in tmp
        tmp_dir = root / "tmp"
        if tmp_dir.is_symlink():
            console.print(f"  [red]Refusing to clean symlinked tmp dir: {tmp_dir}[/red]")
            failures.append(str(tmp_dir))
            sockets = []
        elif tmp_dir.exists():
            sockets = _stale_sockets(tmp_dir)
            if dry_run:
                console.print(f"  [cyan]Would clean {len(sockets)} stale sockets from {tmp_dir}[/cyan]", soft_wrap=True)
            elif sockets:
                sudo = ["sudo", "-n"] if yes else ["sudo"]
                result = subprocess.run(
                    [*sudo, "find", str(tmp_dir), "-type", "s", "-delete"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    console.print(f"  [green]Cleaned {len(sockets)} stale sockets from {tmp_dir}[/green]")
                else:
                    console.print(f"  [red]Failed to clean sockets: {result.stderr.strip()}[/red]")
                    failures.append(f"sockets in {tmp_dir}")
            else:
                console.print(f"  [dim]No stale sockets in {tmp_dir}[/dim]")

        if all_caches:
            cache_dirs = [
                (root / ".cache" / "flashinfer", "FlashInfer JIT"),
                (root / ".cache" / "torch_extensions", "torch compile"),
                (root / ".cache" / "vllm", "vLLM"),
            ]
            for cache_dir, label in cache_dirs:
                unsafe_reason = _unsafe_descendant_reason(cache_dir, root)
                if unsafe_reason:
                    console.print(f"  [red]Refusing to clean unsafe cache path for {label}: {unsafe_reason}[/red]")
                    failures.append(label)
            if failures:
                raise typer.Exit(1)
            existing = [(cache_dir, label) for cache_dir, label in cache_dirs if cache_dir.exists()]
            planned_total = 0
            sizes: dict[pathlib.Path, int | None] = {}
            for cache_dir, label in cache_dirs:
                if cache_dir.exists():
                    size_mb = _du_mb(cache_dir)
                    sizes[cache_dir] = size_mb
                    if size_mb is not None:
                        planned_total += size_mb
                    if dry_run:
                        size_label = f"{size_mb} MB" if size_mb is not None else "unknown size"
                        console.print(f"  [cyan]Would clean {label} cache ({size_label})[/cyan]")
                elif dry_run:
                    console.print(f"  [dim]{label} cache not found[/dim]")

            if dry_run:
                unknown = any(size is None for size in sizes.values())
                suffix = " plus unknown-size caches" if unknown else ""
                console.print(f"\n  [bold]Would free approximately {planned_total} MB total{suffix}[/bold]")
                return
            if existing and not yes:
                unknown = any(sizes.get(cache_dir) is None for cache_dir, _label in existing)
                size_label = f"{planned_total} MB"
                if unknown:
                    size_label += " plus unknown-size caches"
                if not typer.confirm(f"Delete {len(existing)} cache directories ({size_label})?", default=False):
                    console.print("[dim]Cancelled.[/dim]")
                    return
            for cache_dir, label in cache_dirs:
                if cache_dir.exists():
                    if cache_dir.is_symlink():
                        console.print(f"  [red]Refusing to clean symlinked {label} cache: {cache_dir}[/red]")
                        failures.append(label)
                        continue
                    sudo = ["sudo", "-n"] if yes else ["sudo"]
                    result = subprocess.run(
                        [*sudo, "rm", "-rf", str(cache_dir)],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        size_mb = sizes.get(cache_dir, 0)
                        if size_mb is not None:
                            total_freed += size_mb
                        size_label = f"{size_mb} MB" if size_mb is not None else "unknown size"
                        console.print(f"  [green]Cleaned {label} cache ({size_label})[/green]")
                    else:
                        console.print(f"  [red]Failed to clean {label}: {result.stderr.strip()}[/red]")
                        failures.append(label)
                else:
                    console.print(f"  [dim]{label} cache not found[/dim]")

            if total_freed > 0:
                console.print(f"\n  [bold]Freed {total_freed} MB total[/bold]")
                console.print("  [yellow]Next vserve run will recompile kernels (~5-10 min)[/yellow]")
        elif dry_run:
            console.print("  [bold]Dry run complete.[/bold]")
        if failures:
            raise typer.Exit(1)
    finally:
        lock.release()
