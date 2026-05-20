"""Tests for the performance cache (item — picker-measured cells)."""

from __future__ import annotations


class TestCacheKey:
    def test_key_changes_with_model(self):
        from vserve.perf_cache import cache_key
        a = cache_key(model_path="/m/a", gpu_uuid="gpu-x", backend="vllm",
                     build_id="vllm-0.21", config_hash="abc")
        b = cache_key(model_path="/m/b", gpu_uuid="gpu-x", backend="vllm",
                     build_id="vllm-0.21", config_hash="abc")
        assert a != b

    def test_key_changes_with_build(self):
        from vserve.perf_cache import cache_key
        a = cache_key(model_path="/m", gpu_uuid="g", backend="vllm",
                     build_id="vllm-0.21", config_hash="abc")
        b = cache_key(model_path="/m", gpu_uuid="g", backend="vllm",
                     build_id="vllm-0.22", config_hash="abc")
        assert a != b

    def test_key_stable_across_calls(self):
        from vserve.perf_cache import cache_key
        a = cache_key(model_path="/m", gpu_uuid="g", backend="vllm",
                     build_id="v", config_hash="abc")
        b = cache_key(model_path="/m", gpu_uuid="g", backend="vllm",
                     build_id="v", config_hash="abc")
        assert a == b


class TestConfigHash:
    def test_vllm_skips_irrelevant_fields(self):
        from vserve.perf_cache import config_hash_from_cfg
        # port shouldn't affect throughput
        a = config_hash_from_cfg({"max-model-len": 8192, "port": 8888}, "vllm")
        b = config_hash_from_cfg({"max-model-len": 8192, "port": 9999}, "vllm")
        assert a == b

    def test_vllm_picks_up_relevant_fields(self):
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg({"max-num-seqs": 4}, "vllm")
        b = config_hash_from_cfg({"max-num-seqs": 8}, "vllm")
        assert a != b

    def test_llamacpp_uses_its_keys(self):
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg({"ctx_size": 4096, "parallel": 1}, "llamacpp")
        b = config_hash_from_cfg({"ctx_size": 4096, "parallel": 4}, "llamacpp")
        assert a != b


class TestWriteRead:
    def test_roundtrip(self, tmp_path):
        from vserve.perf_cache import PerfEntry, read_entry, write_entry
        entry = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g",
            build_id="vllm-0.21", driver="595", config_hash="abc",
            context=8192, kv_dtype="fp8", slots=4,
            decode_tps_p50=72.5, ttft_ms_p50=180.0, sample_count=3,
        )
        path = write_entry(entry, directory=tmp_path)
        assert path.exists()
        out = read_entry(path)
        assert out is not None
        assert out.decode_tps_p50 == 72.5
        assert out.model_path == "/m"

    def test_read_bad_file_returns_none(self, tmp_path):
        from vserve.perf_cache import read_entry
        f = tmp_path / "bad.json"
        f.write_text("not json")
        assert read_entry(f) is None

    def test_atomic_write_replaces_existing(self, tmp_path):
        from vserve.perf_cache import PerfEntry, read_entry, write_entry
        e1 = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g",
            build_id="v", driver="d", config_hash="abc",
            context=8192, kv_dtype="fp8", slots=4,
            decode_tps_p50=50,
        )
        p1 = write_entry(e1, directory=tmp_path)
        # Overwrite with newer value
        e2 = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g",
            build_id="v", driver="d", config_hash="abc",
            context=8192, kv_dtype="fp8", slots=4,
            decode_tps_p50=80,
        )
        p2 = write_entry(e2, directory=tmp_path)
        assert p1 == p2
        out = read_entry(p2)
        assert out is not None
        assert out.decode_tps_p50 == 80


class TestLookupForPicker:
    def test_returns_matching_entries(self, tmp_path):
        from vserve.perf_cache import PerfEntry, lookup_for_picker, write_entry
        e1 = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g", build_id="v",
            driver="d", config_hash="cfg1",
            context=8192, kv_dtype="fp8", slots=4, decode_tps_p50=72,
        )
        e2 = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g", build_id="v",
            driver="d", config_hash="cfg2",
            context=16384, kv_dtype="fp8", slots=2, decode_tps_p50=45,
        )
        # Another model — should NOT be returned.
        e3 = PerfEntry(
            model_path="/other", backend="vllm", gpu_uuid="g", build_id="v",
            driver="d", config_hash="cfg1",
            context=8192, kv_dtype="fp8", slots=4, decode_tps_p50=200,
        )
        for e in (e1, e2, e3):
            write_entry(e, directory=tmp_path)
        out = lookup_for_picker(
            model_path="/m", gpu_uuid="g", backend="vllm",
            build_id="v", directory=tmp_path,
        )
        contexts = sorted(e.context for e in out)
        assert contexts == [8192, 16384]

    def test_filters_stale_build_id(self, tmp_path):
        from vserve.perf_cache import PerfEntry, lookup_for_picker, write_entry
        e_old = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g",
            build_id="vllm-0.20-deadbee",  # different build
            driver="d", config_hash="cfg1",
            context=8192, kv_dtype="fp8", slots=4, decode_tps_p50=99,
        )
        write_entry(e_old, directory=tmp_path)
        out = lookup_for_picker(
            model_path="/m", gpu_uuid="g", backend="vllm",
            build_id="vllm-0.21-abc1234",  # current build
            directory=tmp_path,
        )
        assert out == []

    def test_lookup_one_returns_specific_entry(self, tmp_path):
        from vserve.perf_cache import (
            PerfEntry, lookup_one, write_entry,
        )
        e = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g", build_id="v",
            driver="d", config_hash="exact-cfg",
            context=8192, kv_dtype="fp8", slots=4, decode_tps_p50=72,
        )
        write_entry(e, directory=tmp_path)
        found = lookup_one(
            model_path="/m", gpu_uuid="g", backend="vllm", build_id="v",
            config_hash="exact-cfg", directory=tmp_path,
        )
        assert found is not None
        assert found.decode_tps_p50 == 72

    def test_lookup_one_returns_none_on_miss(self, tmp_path):
        from vserve.perf_cache import lookup_one
        assert lookup_one(
            model_path="/m", gpu_uuid="g", backend="vllm", build_id="v",
            config_hash="nope", directory=tmp_path,
        ) is None


class TestBuildIdHelpers:
    def test_vllm_build_id_with_commit(self):
        from vserve.perf_cache import vllm_build_id
        class FakeRuntime:
            version = "0.21.0"
            commit = "abc1234deadbeef"
        assert vllm_build_id(FakeRuntime()) == "vllm-0.21.0-abc1234"

    def test_vllm_build_id_unknown(self):
        from vserve.perf_cache import vllm_build_id
        assert vllm_build_id(None) == "unknown"

    def test_llamacpp_build_id_with_number_and_commit(self):
        from vserve.perf_cache import llamacpp_build_id
        class FakeBuild:
            build_number = 9222
            commit = "abc1234deadbeef"
        assert llamacpp_build_id(FakeBuild()) == "llamacpp-b9222-abc1234"

    def test_llamacpp_build_id_unknown(self):
        from vserve.perf_cache import llamacpp_build_id
        assert llamacpp_build_id(None) == "llamacpp-unknown"

    def test_gpu_uuid_falls_back_to_index_when_no_uuid(self):
        from vserve.perf_cache import gpu_uuid_or_index
        class FakeGpu:
            index = 0
            name = "RTX PRO 5000"
        assert "RTX PRO 5000" in gpu_uuid_or_index(FakeGpu())


class TestLaunchTimeMeasurement:
    """Wire-through: after _launch_backend reports ready, the measurement
    helper runs a streaming probe and persists a PerfEntry."""

    def test_measure_and_cache_writes_entry(self, mocker, tmp_path):
        from unittest.mock import MagicMock
        from vserve.cli import _measure_and_cache
        from vserve.bench import BenchResult

        # Fake backend handle.
        backend = MagicMock()
        backend.name = "vllm"
        backend.runtime_info = MagicMock(return_value=MagicMock(version="0.21.0", commit="abc123def"))

        cfg = {
            "model": "/m/Gemma-4",
            "served-model-name": ["nvidia/Gemma-4", "gemma-4"],
            "max-model-len": 8192,
            "max-num-seqs": 4,
            "kv-cache-dtype": "fp8",
        }

        mocker.patch(
            "vserve.bench.run_streaming_benchmark",
            return_value=BenchResult(
                ttft_ms_p50=180.0, ttft_ms_p99=210.0,
                tpot_ms_p50=12.0, tpot_ms_p99=15.0,
                itl_ms_p99=14.0, throughput_tokens_per_sec=80.5,
                throughput_requests_per_sec=1.0,
                e2e_p99_ms=2000.0, requests_completed=3,
                requests_total=3, errors=[], total_seconds=5.0,
            ),
        )

        class FakeGpu:
            index = 0
            name = "RTX PRO 5000"
            driver = "595.58.03"
        mocker.patch("vserve.gpu.get_gpu_info", return_value=FakeGpu())
        mocker.patch("vserve.perf_cache.cache_dir", return_value=tmp_path)

        entry = _measure_and_cache(backend, cfg, "/some/path.yaml", 8888)
        assert entry is not None
        assert entry.decode_tps_p50 == 80.5
        assert entry.ttft_ms_p50 == 180.0
        # Persisted to the temp cache dir.
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

    def test_measure_returns_none_on_zero_requests(self, mocker):
        from unittest.mock import MagicMock
        from vserve.cli import _measure_and_cache
        from vserve.bench import BenchResult

        backend = MagicMock()
        backend.name = "vllm"
        backend.runtime_info = MagicMock(return_value=None)
        cfg = {"model": "/m", "served-model-name": ["m"],
               "max-model-len": 8192, "max-num-seqs": 4, "kv-cache-dtype": "fp8"}
        mocker.patch(
            "vserve.bench.run_streaming_benchmark",
            return_value=BenchResult(
                ttft_ms_p50=None, ttft_ms_p99=None,
                tpot_ms_p50=None, tpot_ms_p99=None,
                itl_ms_p99=None, throughput_tokens_per_sec=0.0,
                throughput_requests_per_sec=0.0,
                e2e_p99_ms=None, requests_completed=0,
                requests_total=0, errors=["timeout"], total_seconds=5.0,
            ),
        )
        assert _measure_and_cache(backend, cfg, "/x.yaml", 8888) is None


class TestResolveProbeModelName:
    def test_vllm_picks_first_alias(self):
        from unittest.mock import MagicMock
        from vserve.cli import _resolve_probe_model_name
        backend = MagicMock()
        backend.name = "vllm"
        cfg = {"served-model-name": ["nvidia/Gemma-4", "gemma-4"]}
        assert _resolve_probe_model_name(backend, cfg) == "nvidia/Gemma-4"

    def test_vllm_falls_back_to_model_path(self):
        from unittest.mock import MagicMock
        from vserve.cli import _resolve_probe_model_name
        backend = MagicMock()
        backend.name = "vllm"
        cfg = {"model": "/opt/vllm/models/x"}
        assert _resolve_probe_model_name(backend, cfg) == "/opt/vllm/models/x"

    def test_llamacpp_uses_constant_name(self):
        from unittest.mock import MagicMock
        from vserve.cli import _resolve_probe_model_name
        backend = MagicMock()
        backend.name = "llamacpp"
        assert _resolve_probe_model_name(backend, {}) == "llamacpp"


class TestCacheRobustness:
    """Edge cases: corrupted entries, concurrent writes, lookup misses."""

    def test_lookup_for_picker_skips_unparseable_files(self, tmp_path):
        """Cache dir with a corrupted JSON file shouldn't poison the lookup."""
        from vserve.perf_cache import PerfEntry, lookup_for_picker, write_entry
        # Write one good entry and one corrupted file.
        good = PerfEntry(
            model_path="/m", backend="vllm", gpu_uuid="g", build_id="v",
            driver="d", config_hash="abc",
            context=8192, kv_dtype="fp8", slots=4, decode_tps_p50=72,
        )
        write_entry(good, directory=tmp_path)
        (tmp_path / "corrupted.json").write_text("not valid json {{{")
        out = lookup_for_picker(
            model_path="/m", gpu_uuid="g", backend="vllm",
            build_id="v", directory=tmp_path,
        )
        # The good entry comes through; the corrupted file is silently skipped.
        assert len(out) == 1
        assert out[0].decode_tps_p50 == 72

    def test_lookup_for_picker_empty_cache(self, tmp_path):
        from vserve.perf_cache import lookup_for_picker
        assert lookup_for_picker(
            model_path="/m", gpu_uuid="g", backend="vllm",
            build_id="v", directory=tmp_path,
        ) == []

    def test_lookup_for_picker_missing_directory(self, tmp_path):
        from vserve.perf_cache import lookup_for_picker
        nonexistent = tmp_path / "no-such-dir"
        assert lookup_for_picker(
            model_path="/m", gpu_uuid="g", backend="vllm",
            build_id="v", directory=nonexistent,
        ) == []

    def test_lookup_one_overwrite_returns_most_recent(self, tmp_path):
        """Successive writes with the same key replace, not duplicate."""
        from vserve.perf_cache import PerfEntry, lookup_one, write_entry
        for tps in (40.0, 60.0, 80.0):
            e = PerfEntry(
                model_path="/m", backend="vllm", gpu_uuid="g", build_id="v",
                driver="d", config_hash="abc",
                context=8192, kv_dtype="fp8", slots=4, decode_tps_p50=tps,
            )
            write_entry(e, directory=tmp_path)
        # Only one file persists; lookup_one returns the latest tps.
        found = lookup_one(
            model_path="/m", gpu_uuid="g", backend="vllm", build_id="v",
            config_hash="abc", directory=tmp_path,
        )
        assert found is not None
        assert found.decode_tps_p50 == 80.0
        # Only one cache file with that key.
        assert len(list(tmp_path.glob("*.json"))) == 1


class TestConfigHashStability:
    """Same config dict produces the same hash regardless of dict ordering."""

    def test_dict_key_order_irrelevant_for_vllm(self):
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg(
            {"max-model-len": 8192, "max-num-seqs": 4, "kv-cache-dtype": "fp8"},
            "vllm",
        )
        b = config_hash_from_cfg(
            {"kv-cache-dtype": "fp8", "max-num-seqs": 4, "max-model-len": 8192},
            "vllm",
        )
        assert a == b

    def test_dict_key_order_irrelevant_for_llamacpp(self):
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg(
            {"ctx_size": 32768, "parallel": 4, "cache_type_k": "q8_0"},
            "llamacpp",
        )
        b = config_hash_from_cfg(
            {"parallel": 4, "cache_type_k": "q8_0", "ctx_size": 32768},
            "llamacpp",
        )
        assert a == b

    def test_changing_quantization_changes_hash(self):
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg({"quantization": "fp8"}, "vllm")
        b = config_hash_from_cfg({"quantization": "nvfp4"}, "vllm")
        assert a != b

    def test_changing_attention_backend_changes_hash(self):
        """Item R: attention-config switch must invalidate the cache."""
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg({"attention-config": {"backend": "FLASHMLA"}}, "vllm")
        b = config_hash_from_cfg({"attention-config": {"backend": "TRITON_ATTN"}}, "vllm")
        assert a != b

    def test_overriding_n_cpu_moe_changes_hash(self):
        """Item H: switching to ncmoe changes throughput characteristics."""
        from vserve.perf_cache import config_hash_from_cfg
        a = config_hash_from_cfg({"ctx_size": 4096, "n_cpu_moe": 0}, "llamacpp")
        b = config_hash_from_cfg({"ctx_size": 4096, "n_cpu_moe": 99}, "llamacpp")
        assert a != b
