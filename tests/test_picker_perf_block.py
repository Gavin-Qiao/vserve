"""Tests for the picker-side perf-cache annotation (item T).

`_print_measured_cells_block` reads the perf cache and annotates the
limits matrix. These tests verify the rendering layer — does it print
anything when the cache has hits, stay silent when it doesn't, prefer
the most-recent measurement when there are multiple?
"""

from __future__ import annotations

from unittest.mock import Mock


def _make_model(tmp_path):
    """Architecture-light ModelInfo for picker tests."""
    from vserve.models import ModelInfo
    p = tmp_path / "m"
    p.mkdir()
    return ModelInfo(
        path=p, provider="u", model_name="m", architecture="LlamaForCausalLM",
        model_type="llama", quant_method=None,
        max_position_embeddings=131072, is_moe=False, model_size_gb=1.0,
    )


def _seed_cache_entry(cache_dir, *, model_path, gpu_uuid="gpu-x",
                     backend="vllm", build_id="vllm-test",
                     context=8192, kv_dtype="fp8", slots=4,
                     decode_tps=78.0, measured_at="2026-05-19T00:00:00+00:00",
                     config_hash="abc"):
    """Persist one PerfEntry into the cache for the picker to find."""
    from vserve.perf_cache import PerfEntry, write_entry
    entry = PerfEntry(
        model_path=model_path, backend=backend, gpu_uuid=gpu_uuid,
        build_id=build_id, driver="d", config_hash=config_hash,
        context=context, kv_dtype=kv_dtype, slots=slots,
        decode_tps_p50=decode_tps, measured_at=measured_at,
    )
    write_entry(entry, directory=cache_dir)


class TestPrintMeasuredCellsBlock:
    def _patch_gpu(self, mocker):
        class FakeGpu:
            index = 0
            name = "RTX PRO 5000"
            driver = "595"
        mocker.patch("vserve.gpu.get_gpu_info", return_value=FakeGpu())
        return FakeGpu

    def _patch_build_id(self, mocker, build_id="vllm-test"):
        mocker.patch("vserve.cli._build_id_for_backend", return_value=build_id)

    def test_renders_when_cache_has_hit(self, tmp_path, capsys, mocker):
        from vserve.cli import _print_measured_cells_block
        from vserve.backends.vllm import VllmBackend
        self._patch_gpu(mocker)
        self._patch_build_id(mocker)
        mocker.patch("vserve.perf_cache.cache_dir", return_value=tmp_path)
        m = _make_model(tmp_path)
        _seed_cache_entry(
            tmp_path, model_path=str(m.path), gpu_uuid="idx-0-RTX PRO 5000",
            context=8192, kv_dtype="fp8", decode_tps=78.0,
        )
        limits_data = {"backend": "vllm"}
        # Stub _BACKENDS so the helper finds something to ask build_id of.
        mocker.patch("vserve.backends._BACKENDS", [VllmBackend()])
        _print_measured_cells_block(
            limits_data, m, "vllm", ["auto", "fp8"], {"8192": {"fp8": 4}},
        )
        out = capsys.readouterr().out
        assert "78 t/s" in out

    def test_silent_when_no_cache_hits(self, tmp_path, capsys, mocker):
        from vserve.cli import _print_measured_cells_block
        from vserve.backends.vllm import VllmBackend
        self._patch_gpu(mocker)
        self._patch_build_id(mocker, build_id="vllm-no-data")
        mocker.patch("vserve.perf_cache.cache_dir", return_value=tmp_path)
        m = _make_model(tmp_path)
        mocker.patch("vserve.backends._BACKENDS", [VllmBackend()])
        _print_measured_cells_block(
            {"backend": "vllm"}, m, "vllm", ["auto", "fp8"], {"8192": {"fp8": 4}},
        )
        out = capsys.readouterr().out
        assert "t/s" not in out
        assert "Measured" not in out

    def test_prefers_most_recent_when_multiple_measurements(self, tmp_path, capsys, mocker):
        from vserve.cli import _print_measured_cells_block
        from vserve.backends.vllm import VllmBackend
        self._patch_gpu(mocker)
        self._patch_build_id(mocker)
        mocker.patch("vserve.perf_cache.cache_dir", return_value=tmp_path)
        m = _make_model(tmp_path)
        # Old measurement.
        _seed_cache_entry(
            tmp_path, model_path=str(m.path), gpu_uuid="idx-0-RTX PRO 5000",
            context=8192, kv_dtype="fp8", decode_tps=40.0,
            measured_at="2025-01-01T00:00:00+00:00", config_hash="cfg-old",
        )
        # Newer measurement at same cell.
        _seed_cache_entry(
            tmp_path, model_path=str(m.path), gpu_uuid="idx-0-RTX PRO 5000",
            context=8192, kv_dtype="fp8", decode_tps=78.0,
            measured_at="2026-05-19T12:00:00+00:00", config_hash="cfg-new",
        )
        mocker.patch("vserve.backends._BACKENDS", [VllmBackend()])
        _print_measured_cells_block(
            {"backend": "vllm"}, m, "vllm", ["auto", "fp8"], {"8192": {"fp8": 4}},
        )
        out = capsys.readouterr().out
        # Newer value shown, not the older one.
        assert "78 t/s" in out
        assert "40 t/s" not in out

    def test_filters_stale_build_id(self, tmp_path, capsys, mocker):
        """Entries from a different build don't appear in the matrix."""
        from vserve.cli import _print_measured_cells_block
        from vserve.backends.vllm import VllmBackend
        self._patch_gpu(mocker)
        self._patch_build_id(mocker, build_id="vllm-current")
        mocker.patch("vserve.perf_cache.cache_dir", return_value=tmp_path)
        m = _make_model(tmp_path)
        _seed_cache_entry(
            tmp_path, model_path=str(m.path), gpu_uuid="idx-0-RTX PRO 5000",
            build_id="vllm-different",  # not current
            context=8192, kv_dtype="fp8", decode_tps=99.0,
        )
        mocker.patch("vserve.backends._BACKENDS", [VllmBackend()])
        _print_measured_cells_block(
            {"backend": "vllm"}, m, "vllm", ["auto", "fp8"], {"8192": {"fp8": 4}},
        )
        out = capsys.readouterr().out
        assert "99 t/s" not in out

    def test_soft_fails_when_perf_cache_unavailable(self, tmp_path, capsys, mocker):
        """If get_gpu_info raises, the picker keeps rendering (silent skip)."""
        from vserve.cli import _print_measured_cells_block
        mocker.patch("vserve.gpu.get_gpu_info", side_effect=RuntimeError("no GPU"))
        m = _make_model(tmp_path)
        _print_measured_cells_block(
            {"backend": "vllm"}, m, "vllm", ["auto", "fp8"], {"8192": {"fp8": 4}},
        )
        out = capsys.readouterr().out
        # No tracebacks, no error message — just silent.
        assert "Traceback" not in out

    def test_unsupported_backend_is_silent(self, tmp_path, capsys, mocker):
        from vserve.cli import _print_measured_cells_block
        m = _make_model(tmp_path)
        _print_measured_cells_block(
            {"backend": "something_unknown"}, m, "something_unknown",
            ["auto"], {"8192": {"auto": 4}},
        )
        out = capsys.readouterr().out
        assert "t/s" not in out


class TestPrintStatusInferenceProbe:
    """Item T's status-side probe: live tok/s + cached baseline."""

    def test_shows_live_decode_when_probe_succeeds(self, capsys, mocker):
        from vserve.bench import BenchResult
        from vserve.cli import _print_status_inference_probe
        backend = Mock()
        backend.name = "vllm"
        cfg = {"served-model-name": ["m"], "model": "/m"}
        mocker.patch(
            "vserve.bench.run_streaming_benchmark",
            return_value=BenchResult(
                ttft_ms_p50=200, ttft_ms_p99=210,
                tpot_ms_p50=12, tpot_ms_p99=15,
                itl_ms_p99=13,
                throughput_tokens_per_sec=88.5,
                throughput_requests_per_sec=1.0,
                e2e_p99_ms=1500, requests_completed=3, requests_total=3,
                errors=[], total_seconds=3.0,
            ),
        )
        # No cached baseline.
        mocker.patch("vserve.gpu.get_gpu_info", side_effect=Exception)
        _print_status_inference_probe(backend, cfg, 8888)
        out = capsys.readouterr().out
        assert "Decode:" in out
        assert "88.5" in out
        assert "TTFT 200" in out

    def test_silent_when_probe_fails_and_no_cache(self, capsys, mocker):
        from vserve.cli import _print_status_inference_probe
        backend = Mock()
        backend.name = "vllm"
        cfg = {"served-model-name": ["m"]}
        mocker.patch("vserve.bench.run_streaming_benchmark",
                     side_effect=RuntimeError("connection refused"))
        mocker.patch("vserve.gpu.get_gpu_info", side_effect=Exception)
        _print_status_inference_probe(backend, cfg, 8888)
        out = capsys.readouterr().out
        # Nothing printed because both live + cache failed.
        assert "Decode" not in out

    def test_shows_cached_baseline_alongside_live(self, capsys, mocker, tmp_path):
        from vserve.bench import BenchResult
        from vserve.cli import _print_status_inference_probe
        from vserve.perf_cache import PerfEntry, write_entry

        backend = Mock()
        backend.name = "vllm"
        cfg = {"served-model-name": ["m"], "model": "/m",
               "max-model-len": 8192, "max-num-seqs": 4, "kv-cache-dtype": "fp8"}
        mocker.patch(
            "vserve.bench.run_streaming_benchmark",
            return_value=BenchResult(
                ttft_ms_p50=200, ttft_ms_p99=210,
                tpot_ms_p50=12, tpot_ms_p99=15,
                itl_ms_p99=13,
                throughput_tokens_per_sec=70.0,
                throughput_requests_per_sec=1.0,
                e2e_p99_ms=1500, requests_completed=3, requests_total=3,
                errors=[], total_seconds=3.0,
            ),
        )

        class FakeGpu:
            index = 0
            name = "RTX"
            driver = "595"
        mocker.patch("vserve.gpu.get_gpu_info", return_value=FakeGpu())
        mocker.patch("vserve.cli._build_id_for_backend", return_value="vllm-test")
        mocker.patch("vserve.perf_cache.cache_dir", return_value=tmp_path)

        # Seed a cache entry that matches our cfg.
        from vserve.perf_cache import config_hash_from_cfg
        entry = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="idx-0-RTX",
            build_id="vllm-test", driver="595",
            config_hash=config_hash_from_cfg(cfg, "vllm"),
            context=8192, kv_dtype="fp8", slots=4,
            decode_tps_p50=82.4, measured_at="2026-05-19T12:00:00+00:00",
        )
        write_entry(entry, directory=tmp_path)
        _print_status_inference_probe(backend, cfg, 8888)
        out = capsys.readouterr().out
        assert "Decode:" in out
        assert "70.0" in out  # live
        assert "Launch baseline" in out
        assert "82.4" in out  # cached
