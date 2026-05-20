"""Tests for the `vserve bench` subcommand.

`vserve bench` is the user-facing entry point for the streaming benchmark
that 0.6.1 item P added to `bench.py`. It auto-discovers the running
backend (vLLM or llama.cpp), reads its config to find the served-model
name and port, runs `run_streaming_benchmark`, and prints percentiles.

These tests verify the CLI plumbing — the streaming benchmark itself is
covered by `tests/test_bench.py`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from vserve.bench import BenchResult
from vserve.cli import app

runner = CliRunner()


def _stub_bench_result(**overrides):
    base = dict(
        ttft_ms_p50=80.0,
        ttft_ms_p99=120.0,
        tpot_ms_p50=22.0,
        tpot_ms_p99=35.0,
        itl_ms_p99=40.0,
        throughput_tokens_per_sec=45.6,
        throughput_requests_per_sec=1.2,
        e2e_p99_ms=2100.0,
        requests_completed=8,
        requests_total=8,
        errors=[],
        total_seconds=10.0,
    )
    base.update(overrides)
    return BenchResult(**base)


def _patch_running_backend(mocker, *, port=8888, model_path="/models/test", served="m-test"):
    """Patch the backend detection so `vserve bench` thinks a backend is up.

    Returns the running-backend mock so the test can assert on it.
    """
    backend = MagicMock()
    backend.name = "llamacpp"
    backend.display_name = "llama.cpp"
    backend.is_running.return_value = True
    backend.active_manifest_path.return_value = "/tmp/active.json"
    backend._active_config_path.return_value = "/tmp/active.cfg"

    # Other backend (vllm) reports not-running so detection picks llamacpp.
    other = MagicMock()
    other.name = "vllm"
    other.display_name = "vLLM"
    other.is_running.return_value = False

    mocker.patch("vserve.backends._BACKENDS", [backend, other])
    mocker.patch(
        "vserve.config.read_active_manifest",
        return_value={"port": port, "config_path": "/tmp/cfg.json"},
    )

    # The bench command reuses _resolve_probe_model_name to get the served name.
    mocker.patch("vserve.cli._resolve_probe_model_name", return_value=served)

    # Pretend the cfg is loadable from the manifest path.
    cfg = {"model": model_path, "port": port}
    import json as _json
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.read_text", return_value=_json.dumps(cfg))
    return backend


class TestBenchCommandHelp:
    def test_bench_command_registered(self):
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "bench" in result.stdout.lower()

    def test_bench_help_shows_duration_concurrency(self):
        result = runner.invoke(app, ["bench", "--help"])
        assert result.exit_code == 0
        assert "--duration-s" in result.stdout
        assert "--concurrency" in result.stdout


class TestBenchCommandWithRunningBackend:
    def test_bench_calls_streaming_benchmark_with_defaults(self, mocker):
        _patch_running_backend(mocker)
        stub = mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(),
        )
        # Skip perf-cache writes for simplicity.
        mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench"])
        assert result.exit_code == 0, result.stdout
        assert stub.called
        kwargs = stub.call_args.kwargs
        # Defaults: 30s duration, concurrency 1, max-tokens 256.
        assert kwargs["duration_s"] == 30.0
        assert kwargs["concurrency"] == 1
        assert kwargs["max_tokens"] == 256

    @pytest.mark.parametrize("flag,value,expected_kwarg,expected_value", [
        ("--duration-s",     "5",    "duration_s",     5.0),
        ("--concurrency",    "4",    "concurrency",    4),
        ("--max-tokens",     "128",  "max_tokens",     128),
        ("--max-latency-ms", "5000", "max_latency_ms", 5000.0),
    ])
    def test_bench_respects_flag(
        self, mocker, flag, value, expected_kwarg, expected_value,
    ):
        """0.6.3 parametrize cleanup: was 4 separate methods following the
        same "invoke with flag, assert kwarg matches" pattern."""
        _patch_running_backend(mocker)
        stub = mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(),
        )
        mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench", flag, value])
        assert result.exit_code == 0, result.stdout
        assert stub.call_args.kwargs[expected_kwarg] == expected_value


class TestBenchCommandOutput:
    def test_prints_ttft_tpot_e2e_percentiles(self, mocker):
        _patch_running_backend(mocker)
        mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(
                ttft_ms_p50=85.0, ttft_ms_p99=130.0,
                tpot_ms_p50=21.0, tpot_ms_p99=40.0,
                e2e_p99_ms=2500.0,
                throughput_tokens_per_sec=50.5,
            ),
        )
        mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench"])
        assert result.exit_code == 0
        # TTFT, TPOT, E2E, tok/s all visible.
        assert "TTFT" in result.stdout
        assert "TPOT" in result.stdout
        assert "tok/s" in result.stdout

    def test_prints_error_summary_when_requests_fail(self, mocker):
        _patch_running_backend(mocker)
        mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(
                requests_completed=0, requests_total=4,
                errors=["connection refused", "connection refused"],
                throughput_tokens_per_sec=0.0,
                throughput_requests_per_sec=0.0,
                ttft_ms_p50=None, ttft_ms_p99=None,
                tpot_ms_p50=None, tpot_ms_p99=None,
                itl_ms_p99=None, e2e_p99_ms=None,
            ),
        )
        mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench"])
        assert "error" in result.stdout.lower()

    def test_json_flag_emits_machine_readable(self, mocker):
        _patch_running_backend(mocker)
        mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(throughput_tokens_per_sec=42.0),
        )
        mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench", "--json"])
        assert result.exit_code == 0
        import json as _json
        # Find the JSON object in stdout — there should be exactly one.
        # Rich may add ANSI codes; strip and parse.
        text = result.stdout.strip()
        # The JSON should be the entire output (no panel formatting).
        data = _json.loads(text)
        assert data["throughput_tokens_per_sec"] == 42.0
        assert "ttft_ms_p50" in data


class TestBenchCommandErrors:
    def test_errors_when_no_backend_running(self, mocker):
        backend1 = MagicMock(name="vllm")
        backend1.name = "vllm"
        backend1.is_running.return_value = False
        backend2 = MagicMock(name="llamacpp")
        backend2.name = "llamacpp"
        backend2.is_running.return_value = False
        mocker.patch("vserve.backends._BACKENDS", [backend1, backend2])

        result = runner.invoke(app, ["bench"])
        # Soft-fail with informative message (non-zero exit).
        assert result.exit_code != 0
        assert "no" in result.stdout.lower() or "not running" in result.stdout.lower()

    def test_errors_when_streaming_benchmark_throws(self, mocker):
        _patch_running_backend(mocker)
        mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            side_effect=RuntimeError("benchmark blew up"),
        )

        result = runner.invoke(app, ["bench"])
        assert result.exit_code != 0
        assert "blew up" in result.stdout or "error" in result.stdout.lower()


class TestBenchCommandPerfCacheWrite:
    def test_bench_writes_result_to_perf_cache_by_default(self, mocker):
        _patch_running_backend(mocker)
        result_stub = _stub_bench_result(throughput_tokens_per_sec=44.0)
        mocker.patch("vserve.cli.run_streaming_benchmark", return_value=result_stub)
        write_stub = mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench"])
        assert result.exit_code == 0
        assert write_stub.called

    def test_bench_skips_perf_cache_with_no_cache_flag(self, mocker):
        _patch_running_backend(mocker)
        mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(),
        )
        write_stub = mocker.patch("vserve.cli._write_bench_to_perf_cache", return_value=None)

        result = runner.invoke(app, ["bench", "--no-cache"])
        assert result.exit_code == 0
        assert not write_stub.called

    def test_bench_perf_cache_failure_does_not_fail_command(self, mocker):
        _patch_running_backend(mocker)
        mocker.patch(
            "vserve.cli.run_streaming_benchmark",
            return_value=_stub_bench_result(),
        )
        # Perf-cache write raises — command should still succeed.
        mocker.patch(
            "vserve.cli._write_bench_to_perf_cache",
            side_effect=RuntimeError("cache write failed"),
        )

        result = runner.invoke(app, ["bench"])
        # Command exits cleanly; we still got the benchmark numbers.
        assert result.exit_code == 0
