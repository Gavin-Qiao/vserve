"""CLI-level tests for the llama.cpp recipe-module flags on `vserve run`.

0.6.1 added backend-side support for prompt caching (`cache_reuse`,
`cram_mb`, `slot_save_path`, `swa_full`), MoE offload (`n_cpu_moe`), and
reasoning-budget (`reasoning_budget`). 0.6.2 wires the corresponding
CLI flags so users can reach those features from `vserve run`.

These tests verify the flag → choices-dict plumbing — the backend's
behavior with each key is covered by `tests/test_llamacpp.py`.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from _helpers import strip_ansi
from vserve.cli import app

runner = CliRunner()


def _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path):
    """Wire up a `vserve run --yes` happy-path against a llama.cpp model.

    Returns the backend mock so tests can assert on call_args.
    """
    from vserve.config import LIMITS_SCHEMA_VERSION
    from vserve.models import detect_model

    model = detect_model(fake_gguf_model_dir)
    backend = mocker.Mock()
    backend.name = "llamacpp"
    backend.display_name = "llama.cpp"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    backend.available_tool_parsers.return_value = set()
    backend.available_reasoning_parsers.return_value = set()
    backend.root_dir = tmp_path
    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)

    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.cli.read_limits", return_value={
        "schema_version": LIMITS_SCHEMA_VERSION,
        "backend": "llamacpp",
        "limits": {"4096": {"f16": 2}, "8192": {"f16": 4}},
        "recommended_kv_dtype": "f16",
        "n_gpu_layers": 99,
    })
    mocker.patch(
        "vserve.gpu.get_gpu_info",
        return_value=mocker.Mock(name="GPU", vram_total_gb=48.0),
    )
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.json")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._launch_backend")
    return backend


class TestRunHelpShowsLlamacppFlags:
    """Verify each new flag appears in `vserve run --help` so users discover them.

    0.6.3 parametrize cleanup: was 6 separate methods doing identical
    `assert flag in stdout` checks — now one parametrized test.
    """

    @pytest.mark.parametrize("flag", [
        "--cache-reuse", "--cram-mb", "--slot-save-path",
        "--swa-full", "--n-cpu-moe", "--reasoning-budget", "--thinking",
    ])
    def test_help_lists_flag(self, flag):
        result = runner.invoke(app, ["run", "--help"])
        assert flag in strip_ansi(result.stdout), f"{flag} missing from --help output"


class TestPromptCacheFlagsThreadIntoChoices:
    """0.6.3 parametrize cleanup: was 9 separate methods following the same
    "invoke with flag, assert choices[key] == value" pattern."""

    @pytest.mark.parametrize("flag,value,choices_key,expected", [
        ("--cache-reuse",     "256",   "cache_reuse",      256),
        ("--cram-mb",         "1024",  "cram_mb",          1024),
        ("--n-cpu-moe",       "16",    "n_cpu_moe",        16),
        ("--n-cpu-moe",       "99",    "n_cpu_moe",        99),
        ("--reasoning-budget", "2048", "reasoning_budget", 2048),
    ])
    def test_int_valued_flag_threads_through(
        self, mocker, fake_gguf_model_dir, tmp_path, flag, value, choices_key, expected,
    ):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes", flag, value])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices[choices_key] == expected

    def test_slot_save_path_threads_through(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        slot_dir = str(tmp_path / "slots")
        result = runner.invoke(app, ["run", "testmodel", "--yes", "--slot-save-path", slot_dir])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["slot_save_path"] == slot_dir

    def test_swa_full_threads_through(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes", "--swa-full"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["swa_full"] is True

    def test_swa_full_defaults_to_false_when_absent(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        # When not passed, choices either omits the key or sets it falsy.
        assert not choices.get("swa_full")


class TestAllPromptCacheFlagsTogether:
    def test_all_six_flags_thread_through(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(
            app,
            [
                "run", "testmodel", "--yes",
                "--cache-reuse", "256",
                "--cram-mb", "1024",
                "--slot-save-path", str(tmp_path / "slots"),
                "--swa-full",
                "--n-cpu-moe", "8",
                "--reasoning-budget", "4096",
            ],
        )
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["cache_reuse"] == 256
        assert choices["cram_mb"] == 1024
        assert choices["slot_save_path"] == str(tmp_path / "slots")
        assert choices["swa_full"] is True
        assert choices["n_cpu_moe"] == 8
        assert choices["reasoning_budget"] == 4096


class TestFlagsAbsentMeansNotInChoices:
    """When a flag isn't passed, the corresponding key should be absent or
    None — never a synthesised default. (Backend.build_config makes the
    default decision, not the CLI plumbing.)"""

    def test_default_run_has_none_for_new_keys(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        # Each key is either absent or None / falsy — never a hard-coded default.
        for key in ("cache_reuse", "cram_mb", "slot_save_path", "reasoning_budget", "n_cpu_moe"):
            assert choices.get(key) is None, f"{key} should default to None, got {choices.get(key)!r}"


# --- 0.6.2 Task C: --thinking flag (cross-backend) ---


def _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path):
    """Wire up a `vserve run --yes` happy-path against a vLLM model."""
    from vserve.config import LIMITS_SCHEMA_VERSION
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    backend.available_tool_parsers.return_value = set()
    backend.available_reasoning_parsers.return_value = set()
    backend.root_dir = tmp_path
    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)

    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.cli.read_limits", return_value={
        "schema_version": LIMITS_SCHEMA_VERSION,
        "backend": "vllm",
        "limits": {"4096": {"auto": 2}, "8192": {"auto": 4, "fp8": 8}},
    })
    mocker.patch(
        "vserve.gpu.get_gpu_info",
        return_value=mocker.Mock(name="GPU", vram_total_gb=48.0),
    )
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._launch_backend")
    return backend


class TestThinkingFlagHelp:
    def test_help_lists_thinking(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--thinking" in strip_ansi(result.stdout)


class TestThinkingFlagLlamacpp:
    def test_thinking_on_threads_through(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes", "--thinking"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["thinking"] is True

    def test_no_thinking_threads_through(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes", "--no-thinking"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["thinking"] is False

    def test_default_thinking_is_none(self, mocker, fake_gguf_model_dir, tmp_path):
        backend = _setup_llamacpp_yes_path(mocker, fake_gguf_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        # When --thinking is absent, no key is forced — backend default applies
        # (matches sampling_defaults registry behavior).
        assert choices.get("thinking") is None


class TestThinkingFlagVllm:
    def test_thinking_on_threads_through(self, mocker, fake_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes", "--thinking"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["thinking"] is True

    def test_no_thinking_threads_through(self, mocker, fake_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", "--yes", "--no-thinking"])
        assert result.exit_code == 0, result.stdout
        choices = backend.build_config.call_args.args[1]
        assert choices["thinking"] is False
