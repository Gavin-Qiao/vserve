"""Tests for the llama.cpp backend."""

import struct
from unittest.mock import Mock

from vserve.backends.llamacpp import LlamaCppBackend


GGUF_UINT32 = 4
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value_type: int, value) -> bytes:
    if value_type == GGUF_UINT32:
        return struct.pack("<I", int(value))
    if value_type == GGUF_STRING:
        return _gguf_string(str(value))
    if value_type == GGUF_BOOL:
        return struct.pack("<?", bool(value))
    if value_type == GGUF_ARRAY:
        element_type, items = value
        payload = struct.pack("<IQ", int(element_type), len(items))
        for item in items:
            payload += _gguf_value(element_type, item)
        return payload
    raise AssertionError(f"unsupported test GGUF value type: {value_type}")


def _write_minimal_gguf(path, entries: dict[str, tuple[int, object]]) -> None:
    payload = b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries))
    for key, (value_type, value) in entries.items():
        payload += _gguf_string(key)
        payload += struct.pack("<I", value_type)
        payload += _gguf_value(value_type, value)
    path.write_bytes(payload)


class TestLlamaCppCanServe:
    def test_can_serve_gguf(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        assert b.can_serve(m) is True

    def test_cannot_serve_safetensors(self, fake_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)
        assert b.can_serve(m) is False


class TestLlamaCppIdentity:
    def test_name(self):
        b = LlamaCppBackend()
        assert b.name == "llamacpp"
        assert b.display_name == "llama.cpp"
        assert b.service_name == "llama-cpp"
        assert b.service_user == "llama-cpp"

    def test_configured_service_identity(self, mocker):
        b = LlamaCppBackend()
        mocker.patch(
            "vserve.config.cfg",
            return_value=Mock(
                llamacpp_service_name="custom-llama",
                llamacpp_service_user="svc-llama",
                llamacpp_root=None,
            ),
        )

        assert b.service_name == "custom-llama"
        assert b.service_user == "svc-llama"


class TestLlamaCppHealthUrl:
    def test_health_url(self):
        b = LlamaCppBackend()
        assert b.health_url(8888) == "http://localhost:8888/health"
        assert b.health_url(9999) == "http://localhost:9999/health"


class TestLlamaCppBuildConfig:
    def test_basic_config(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 8192,
            "n_gpu_layers": 35,
            "parallel": 4,
            "port": 8888,
            "tools": False,
        }
        cfg = b.build_config(m, choices)
        # ctx_size is the llama-server -c value (total ctx across slots).
        # Per-slot context (what the user asked for) lives in ctx_per_slot.
        assert cfg["ctx_size"] == 8192 * 4
        assert cfg["ctx_per_slot"] == 8192
        assert cfg["n_gpu_layers"] == 35
        assert cfg["parallel"] == 4
        assert cfg["flash_attn"] is True
        assert "cont_batching" not in cfg
        assert "cont-batching" not in cfg
        assert "jinja" not in cfg

    def test_config_with_tools(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 8192,
            "n_gpu_layers": 35,
            "parallel": 4,
            "port": 8888,
            "tools": True,
        }
        cfg = b.build_config(m, choices)
        assert cfg["jinja"] is True

    def test_config_model_path_is_gguf_file(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}
        cfg = b.build_config(m, choices)
        assert cfg["model"].endswith(".gguf")

    def test_config_model_path_accepts_uppercase_gguf_extension(self, tmp_path):
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "Model-Q4_K_M.GGUF").write_bytes(b"\0")
        from vserve.models import detect_model
        m = detect_model(model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}
        cfg = b.build_config(m, choices)

        assert cfg["model"].endswith("Model-Q4_K_M.GGUF")

    def test_config_selects_one_coherent_split_shard_set(self, tmp_path):
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        for name, size in {
            "Model-Q4_K_M-00001-of-00002.gguf": 1,
            "Model-Q4_K_M-00002-of-00002.gguf": 1,
            "Model-Q8_0-00001-of-00002.gguf": 3,
            "Model-Q8_0-00002-of-00002.gguf": 3,
        }.items():
            (model_dir / name).write_bytes(b"\0" * size)
        from vserve.models import detect_model
        m = detect_model(model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}
        cfg = b.build_config(m, choices)

        assert cfg["model"].endswith("Model-Q8_0-00001-of-00002.gguf")

    def test_config_rejects_incomplete_split_shard_set(self, tmp_path):
        import pytest

        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0")
        from vserve.models import detect_model
        m = detect_model(model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}

        with pytest.raises(ValueError, match="Incomplete split GGUF"):
            b.build_config(m, choices)

    def test_config_embedding_mode(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 512,
            "n_gpu_layers": 10,
            "parallel": 8,
            "port": 8888,
            "embedding": True,
            "pooling": "mean",
        }
        cfg = b.build_config(m, choices)
        assert cfg["embedding"] is True
        assert cfg["pooling"] == "mean"
        assert "jinja" not in cfg  # no tool calling in embedding mode

    def test_config_embedding_cls_pooling(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 512,
            "n_gpu_layers": 10,
            "parallel": 1,
            "port": 8888,
            "embedding": True,
            "pooling": "cls",
        }
        cfg = b.build_config(m, choices)
        assert cfg["pooling"] == "cls"


class TestLlamaCppQuant:
    def test_quant_flag_always_empty(self):
        b = LlamaCppBackend()
        assert b.quant_flag("gptq") == ""
        assert b.quant_flag(None) == ""
        assert b.quant_flag("fp8") == ""


class TestLlamaCppDetectTools:
    def test_detect_tools_with_template(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        result = b.detect_tools(fake_gguf_model_dir)
        assert "supports_tools" in result
        assert result["supports_tools"] is True

    def test_detect_tools_no_template(self, tmp_path):
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "u" / "m"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"\0" * 100)
        result = b.detect_tools(model_dir)
        assert result["supports_tools"] is False


class TestLlamaCppMetadata:
    def test_read_gguf_metadata_without_optional_package(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        gguf_path = tmp_path / "model.gguf"
        _write_minimal_gguf(
            gguf_path,
            {
                "general.architecture": (GGUF_STRING, "llama"),
                "llama.block_count": (GGUF_UINT32, 32),
                "llama.context_length": (GGUF_UINT32, 262144),
                "llama.embedding_length": (GGUF_UINT32, 4096),
                "llama.attention.head_count": (GGUF_UINT32, 32),
                "llama.attention.head_count_kv": (GGUF_UINT32, 8),
                "llama.attention.key_length": (GGUF_UINT32, 128),
                "llama.attention.value_length": (GGUF_UINT32, 128),
                "tokenizer.ggml.pooling_type": (GGUF_UINT32, 1),
            },
        )

        metadata = LlamaCppBackend()._read_gguf_metadata(gguf_path)

        assert metadata["arch"] == "llama"
        assert metadata["num_layers"] == 32
        assert metadata["max_context"] == 262144
        assert metadata["num_kv_heads"] == 8
        assert metadata["key_length"] == 128
        assert metadata["value_length"] == 128
        assert metadata["pooling"] == "mean"

    def test_read_gguf_metadata_accepts_layerwise_kv_heads(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        gguf_path = tmp_path / "model.gguf"
        _write_minimal_gguf(
            gguf_path,
            {
                "general.architecture": (GGUF_STRING, "gemma4"),
                "gemma4.block_count": (GGUF_UINT32, 3),
                "gemma4.context_length": (GGUF_UINT32, 262144),
                "gemma4.embedding_length": (GGUF_UINT32, 2816),
                "gemma4.attention.head_count": (GGUF_UINT32, 16),
                "gemma4.attention.head_count_kv": (GGUF_ARRAY, (GGUF_UINT32, [8, 2, 8])),
                "gemma4.attention.key_length": (GGUF_UINT32, 512),
                "gemma4.attention.value_length": (GGUF_UINT32, 512),
            },
        )

        metadata = LlamaCppBackend()._read_gguf_metadata(gguf_path)

        assert metadata["num_layers"] == 3
        assert metadata["max_context"] == 262144
        assert metadata["num_kv_heads"] == [8, 2, 8]
        assert metadata["key_length"] == 512
        assert metadata["value_length"] == 512

    def test_read_gguf_metadata_includes_hybrid_attention_fields(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        gguf_path = tmp_path / "model.gguf"
        _write_minimal_gguf(
            gguf_path,
            {
                "general.architecture": (GGUF_STRING, "qwen35"),
                "qwen35.block_count": (GGUF_UINT32, 64),
                "qwen35.context_length": (GGUF_UINT32, 262144),
                "qwen35.embedding_length": (GGUF_UINT32, 5120),
                "qwen35.attention.head_count": (GGUF_UINT32, 24),
                "qwen35.attention.head_count_kv": (GGUF_UINT32, 4),
                "qwen35.attention.key_length": (GGUF_UINT32, 256),
                "qwen35.attention.value_length": (GGUF_UINT32, 256),
                "qwen35.ssm.conv_kernel": (GGUF_UINT32, 4),
                "qwen35.ssm.inner_size": (GGUF_UINT32, 2560),
                "qwen35.ssm.state_size": (GGUF_UINT32, 128),
                "qwen35.ssm.time_step_rank": (GGUF_UINT32, 128),
                "qwen35.ssm.group_count": (GGUF_UINT32, 4),
                "qwen35.full_attention_interval": (GGUF_UINT32, 4),
            },
        )

        metadata = LlamaCppBackend()._read_gguf_metadata(gguf_path)

        assert metadata["full_attention_interval"] == 4
        assert metadata["ssm_conv_kernel"] == 4
        assert metadata["ssm_inner_size"] == 2560
        assert metadata["ssm_state_size"] == 128
        assert metadata["ssm_group_count"] == 4

    def test_read_gguf_metadata_includes_swa_fields(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        gguf_path = tmp_path / "model.gguf"
        _write_minimal_gguf(
            gguf_path,
            {
                "general.architecture": (GGUF_STRING, "gemma4"),
                "gemma4.block_count": (GGUF_UINT32, 6),
                "gemma4.context_length": (GGUF_UINT32, 262144),
                "gemma4.embedding_length": (GGUF_UINT32, 2816),
                "gemma4.attention.head_count": (GGUF_UINT32, 16),
                "gemma4.attention.head_count_kv": (GGUF_ARRAY, (GGUF_UINT32, [8, 8, 8, 8, 8, 2])),
                "gemma4.attention.key_length": (GGUF_UINT32, 512),
                "gemma4.attention.value_length": (GGUF_UINT32, 512),
                "gemma4.attention.sliding_window": (GGUF_UINT32, 1024),
                "gemma4.attention.sliding_window_pattern": (GGUF_ARRAY, (GGUF_BOOL, [True, True, True, True, True, False])),
                "gemma4.attention.key_length_swa": (GGUF_UINT32, 256),
                "gemma4.attention.value_length_swa": (GGUF_UINT32, 256),
            },
        )

        metadata = LlamaCppBackend()._read_gguf_metadata(gguf_path)

        assert metadata["sliding_window"] == 1024
        assert metadata["sliding_window_pattern"] == [1, 1, 1, 1, 1, 0]
        assert metadata["key_length_swa"] == 256
        assert metadata["value_length_swa"] == 256


class TestLlamaCppTune:
    def _mock_metadata(self, mocker):
        mocker.patch.object(
            LlamaCppBackend, "_read_gguf_metadata",
            return_value={
                "arch": "llama",
                "num_layers": 32,
                "max_context": 8192,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
        )

    def test_tune_basic(self, fake_gguf_model_dir, mocker):
        """Tune produces expected output structure."""
        self._mock_metadata(mocker)
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert "n_gpu_layers" in result
        assert "limits" in result
        assert "model_path" in result
        assert "supports_tools" in result
        assert result["backend"] == "llamacpp"

    def test_tune_estimates_when_gguf_reader_is_missing(self, fake_gguf_model_dir, mocker):
        """Missing optional gguf reader should not block useful estimated tuning."""
        mocker.patch.dict("sys.modules", {"gguf": None})
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)

        assert result["backend"] == "llamacpp"
        assert result["metadata_estimated"] is True
        assert result["limits"]

    def test_tune_uses_builtin_gguf_context_when_reader_is_missing(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        model_dir = tmp_path / "models" / "provider" / "LongContext-GGUF"
        model_dir.mkdir(parents=True)
        _write_minimal_gguf(
            model_dir / "LongContext-Q4_K_M.gguf",
            {
                "general.architecture": (GGUF_STRING, "llama"),
                "llama.block_count": (GGUF_UINT32, 32),
                "llama.context_length": (GGUF_UINT32, 262144),
                "llama.embedding_length": (GGUF_UINT32, 4096),
                "llama.attention.head_count": (GGUF_UINT32, 32),
                "llama.attention.head_count_kv": (GGUF_UINT32, 8),
                "llama.attention.key_length": (GGUF_UINT32, 128),
                "llama.attention.value_length": (GGUF_UINT32, 128),
            },
        )
        from vserve.models import detect_model
        m = detect_model(model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = LlamaCppBackend().tune(m, gpu, gpu_mem_util=0.90)

        assert result["max_context"] == 262144
        assert "metadata_estimated" not in result
        assert "262144" in result["limits"]

    def test_tune_qwen35_counts_only_full_attention_layers_for_context_capacity(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        model_dir = tmp_path / "models" / "unsloth" / "Qwen3.6-27B-GGUF-Q4_K_XL"
        model_dir.mkdir(parents=True)
        gguf_path = model_dir / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
        _write_minimal_gguf(
            gguf_path,
            {
                "general.architecture": (GGUF_STRING, "qwen35"),
                "qwen35.block_count": (GGUF_UINT32, 64),
                "qwen35.context_length": (GGUF_UINT32, 262144),
                "qwen35.embedding_length": (GGUF_UINT32, 5120),
                "qwen35.attention.head_count": (GGUF_UINT32, 24),
                "qwen35.attention.head_count_kv": (GGUF_UINT32, 4),
                "qwen35.attention.key_length": (GGUF_UINT32, 256),
                "qwen35.attention.value_length": (GGUF_UINT32, 256),
                "qwen35.full_attention_interval": (GGUF_UINT32, 4),
            },
        )
        with gguf_path.open("r+b") as f:
            f.truncate(16 * 1024**3)
        from vserve.models import detect_model
        m = detect_model(model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = LlamaCppBackend().tune(m, gpu, gpu_mem_util=0.96)

        # 2D matrix (since 0.5.8): assert the f16 column matches the historical
        # single-int answer. q8_0/q4_0/q4_1 columns are bonus capacity.
        assert result["limits"]["262144"]["f16"] == 1

    def test_tune_gemma4_caps_swa_cache_by_sliding_window(self, tmp_path, mocker):
        mocker.patch.dict("sys.modules", {"gguf": None})
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B-it-GGUF"
        model_dir.mkdir(parents=True)
        gguf_path = model_dir / "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
        _write_minimal_gguf(
            gguf_path,
            {
                "general.architecture": (GGUF_STRING, "gemma4"),
                "gemma4.block_count": (GGUF_UINT32, 30),
                "gemma4.context_length": (GGUF_UINT32, 262144),
                "gemma4.embedding_length": (GGUF_UINT32, 2816),
                "gemma4.attention.head_count": (GGUF_UINT32, 16),
                "gemma4.attention.head_count_kv": (GGUF_ARRAY, (GGUF_UINT32, [8, 8, 8, 8, 8, 2] * 5)),
                "gemma4.attention.key_length": (GGUF_UINT32, 512),
                "gemma4.attention.value_length": (GGUF_UINT32, 512),
                "gemma4.attention.sliding_window": (GGUF_UINT32, 1024),
                "gemma4.attention.sliding_window_pattern": (GGUF_ARRAY, (GGUF_UINT32, [1, 1, 1, 1, 1, 0] * 5)),
                "gemma4.attention.key_length_swa": (GGUF_UINT32, 256),
                "gemma4.attention.value_length_swa": (GGUF_UINT32, 256),
            },
        )
        with gguf_path.open("r+b") as f:
            f.truncate(12 * 1024**3)
        from vserve.models import detect_model
        m = detect_model(model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = LlamaCppBackend().tune(m, gpu, gpu_mem_util=0.96)

        assert result["limits"]["262144"]["f16"] == 6

    def test_llamacpp_memory_counts_recurrent_state_bytes(self):
        metadata = {
            "num_layers": 4,
            "num_kv_heads": 2,
            "key_length": 8,
            "value_length": 8,
            "full_attention_interval": 2,
            "ssm_conv_kernel": 4,
            "ssm_inner_size": 16,
            "ssm_state_size": 8,
            "ssm_group_count": 2,
        }

        total = LlamaCppBackend._llamacpp_kv_cache_bytes(metadata, context=128, parallel=3)

        attention_bytes = 2 * (2 * (8 + 8) * 2 * 128 * 3)
        recurrent_per_layer = ((4 - 1) * (16 + 2 * 2 * 8) + 8 * 16) * 4 * 3
        assert total == attention_bytes + 2 * recurrent_per_layer

    def test_llamacpp_kv_cache_bytes_q8_halves_attention_bytes(self):
        """q8_0 K and V should roughly halve KV memory vs f16 (recurrent state
        stays fp32-sized in llama.cpp and is unaffected by --cache-type-k/v)."""
        metadata = {
            "num_layers": 2,
            "num_kv_heads": 4,
            "key_length": 64,
            "value_length": 64,
        }
        f16_bytes = LlamaCppBackend._llamacpp_kv_cache_bytes(
            metadata, context=4096, parallel=1, k_dtype="f16", v_dtype="f16"
        )
        q8_bytes = LlamaCppBackend._llamacpp_kv_cache_bytes(
            metadata, context=4096, parallel=1, k_dtype="q8_0", v_dtype="q8_0"
        )
        # q8_0 is 1.0625 bytes/elem vs f16's 2.0 → 53.125% of f16 cost.
        assert 0.50 < q8_bytes / f16_bytes < 0.55

    def test_llamacpp_kv_cache_bytes_q4_smaller_than_q8(self):
        metadata = {
            "num_layers": 2,
            "num_kv_heads": 4,
            "key_length": 64,
            "value_length": 64,
        }
        q8 = LlamaCppBackend._llamacpp_kv_cache_bytes(
            metadata, context=4096, parallel=1, k_dtype="q8_0", v_dtype="q8_0"
        )
        q4 = LlamaCppBackend._llamacpp_kv_cache_bytes(
            metadata, context=4096, parallel=1, k_dtype="q4_0", v_dtype="q4_0"
        )
        assert q4 < q8

    def test_tune_emits_kv_dtype_matrix(self, fake_gguf_model_dir, mocker):
        """Tune output should contain a {ctx: {dtype: slots}} matrix."""
        self._mock_metadata(mocker)
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        limits = result["limits"]
        # Every row is a dict with at least f16 and q8_0 columns.
        for ctx_str, row in limits.items():
            assert isinstance(row, dict), f"row at {ctx_str} should be a dict"
            assert "f16" in row
            assert "q8_0" in row
        # The kv_cache_dtypes profile block is exposed for the renderer.
        assert "kv_cache_dtypes" in result
        assert set(result["kv_cache_dtypes"].keys()) >= {"f16", "q8_0", "q4_0", "q4_1"}

    def test_tune_recommends_q8_when_strictly_more_slots(self, fake_gguf_model_dir, mocker):
        self._mock_metadata(mocker)
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        gpu = Mock()
        gpu.vram_total_gb = 48.0
        result = b.tune(m, gpu, gpu_mem_util=0.90)
        # q8_0 has strictly smaller bytes/element than f16, so for any
        # model that fits at f16, q8_0 will also fit and the recommendation
        # should land on q8_0.
        assert result["recommended_kv_dtype"] in ("q8_0", "f16")

    def test_tune_full_offload_with_tiny_model(self, fake_gguf_model_dir, mocker):
        """Tiny model fully fits on GPU."""
        self._mock_metadata(mocker)
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert result["full_offload"] is True

    def test_tune_uses_one_coherent_split_shard_set(self, tmp_path, mocker):
        """Multiple split GGUF variants in one dir should not be summed together."""
        self._mock_metadata(mocker)
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        for name, size_gb in {
            "Model-Q4_K_M-00001-of-00002.gguf": 1,
            "Model-Q4_K_M-00002-of-00002.gguf": 1,
            "Model-Q8_0-00001-of-00002.gguf": 3,
            "Model-Q8_0-00002-of-00002.gguf": 3,
        }.items():
            path = model_dir / name
            with path.open("wb") as f:
                f.truncate(size_gb * 1024**3)
        from vserve.models import detect_model
        m = detect_model(model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)

        assert result["model_size_gb"] == 6.0


class TestLlamaCppLifecycle:
    def test_start_calls_systemctl(self, mocker, tmp_path):
        b = LlamaCppBackend()
        mock_run = mocker.patch("vserve.backends.llamacpp.subprocess.run",
                                return_value=Mock(returncode=0, stdout="", stderr=""))
        mocker.patch("vserve.backends.llamacpp.shutil.copy2")

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{"model": "test"}')

        # Mock active config path
        active = tmp_path / "configs" / "active.json"
        active.parent.mkdir(parents=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)

        b.start(cfg_path)
        calls = [c for c in mock_run.call_args_list if "start" in str(c)]
        assert len(calls) >= 1

    def test_start_rejects_invalid_json_config(self, tmp_path):
        import pytest

        b = LlamaCppBackend()
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{not json")

        with pytest.raises(RuntimeError, match="Invalid llama.cpp config"):
            b.start(cfg_path)

    def test_stop_calls_systemctl(self, mocker):
        b = LlamaCppBackend()
        mock_run = mocker.patch("vserve.backends.llamacpp.subprocess.run",
                                return_value=Mock(returncode=0, stdout="", stderr=""))
        b.stop()
        assert mock_run.call_args[0][0][-1] == "llama-cpp"

    def test_is_running(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="active", stderr=""))
        assert b.is_running() is True

    def test_is_running_raises_on_service_error(self, mocker):
        import pytest

        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=1, stdout="", stderr="dbus error"),
        )
        with pytest.raises(RuntimeError, match="systemctl is-active"):
            b.is_running()

    def test_is_running_missing_unit_is_false(self, mocker):
        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=1, stdout="", stderr="Unit llama-cpp.service could not be found."),
        )
        assert b.is_running() is False

    def test_is_running_activating_is_uncertain(self, mocker):
        import pytest

        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=3, stdout="activating", stderr=""),
        )
        with pytest.raises(RuntimeError, match="is transitional"):
            b.is_running()

    def test_start_uses_configured_service_name(self, mocker, tmp_path):
        b = LlamaCppBackend()
        mock_run = mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch(
            "vserve.config.cfg",
            return_value=Mock(
                llamacpp_service_name="custom-llama",
                llamacpp_service_user="svc-llama",
                llamacpp_root=None,
            ),
        )

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{"model": "test"}')
        active = tmp_path / "configs" / "active.json"
        active.parent.mkdir(parents=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)

        b.start(cfg_path)

        start_calls = [call for call in mock_run.call_args_list if "start" in call.args[0]]
        assert start_calls
        assert start_calls[-1].args[0][-1] == "custom-llama"

    def test_is_not_running(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=3, stdout="inactive", stderr=""))
        assert b.is_running() is False


class TestLlamaCppFindEntrypoint:
    def test_find_on_path(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.shutil.which", return_value="/usr/bin/llama-server")
        # Mock root_dir to a nonexistent path
        mocker.patch.object(type(b), "root_dir", new_callable=lambda: property(lambda self: __import__("pathlib").Path("/nonexistent")))
        result = b.find_entrypoint()
        assert result is not None

    def test_not_found(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.shutil.which", return_value=None)
        mocker.patch.object(type(b), "root_dir", new_callable=lambda: property(lambda self: __import__("pathlib").Path("/nonexistent")))
        assert b.find_entrypoint() is None


class TestLlamaCppRuntimeInfo:
    def test_runtime_info_calls_llama_server_version(self, mocker, tmp_path):
        b = LlamaCppBackend()
        exe = tmp_path / "llama-server"
        exe.write_text("")
        mocker.patch.object(b, "find_entrypoint", return_value=exe)
        run = mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=0, stdout="llama-server 2026\n", stderr=""),
        )

        info = b.runtime_info()

        assert info["llama_server_version"] == "llama-server 2026"
        run.assert_called_once_with([str(exe), "--version"], capture_output=True, text=True, timeout=10)

    def test_compatibility_fails_without_entrypoint(self, mocker):
        b = LlamaCppBackend()
        mocker.patch.object(b, "find_entrypoint", return_value=None)

        result = b.compatibility()

        assert result["supported"] is False
        assert "entrypoint not found" in result["errors"][0]


class TestLlamaCppDoctorChecks:
    def test_returns_callables(self):
        b = LlamaCppBackend()
        checks = b.doctor_checks()
        assert len(checks) >= 2
        for desc, fn in checks:
            assert isinstance(desc, str)
            assert callable(fn)


class TestLlamaCppStartFailure:
    def test_start_raises_on_failure(self, mocker, tmp_path):
        import pytest
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=1, stdout="", stderr="Unit not found"))
        mocker.patch("vserve.backends.llamacpp.shutil.copy2")
        active = tmp_path / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{}')
        with pytest.raises(RuntimeError, match="systemctl start"):
            b.start(cfg_path)

    def test_start_failure_rolls_back_active_links_and_writes_manifest(self, mocker, tmp_path):
        import json
        import pytest

        from vserve.config import read_active_manifest

        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=1, stdout="", stderr="Unit not found"),
        )
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        active = tmp_path / "configs" / "active.sh"
        active.parent.mkdir(parents=True)
        previous_script = tmp_path / "configs" / "models" / "previous.sh"
        previous_json = tmp_path / "configs" / "models" / "previous.json"
        previous_script.parent.mkdir(parents=True)
        previous_script.write_text("#!/bin/bash\n")
        previous_json.write_text("{}")
        active.symlink_to(previous_script)
        active_json = active.with_suffix(".json")
        active_json.symlink_to(previous_json)
        manifest_path = tmp_path / "run" / "active-manifest.json"

        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "active_manifest_path", return_value=manifest_path)

        cfg_path = tmp_path / "configs" / "models" / "next.json"
        cfg_path.write_text(json.dumps({"model": "test.gguf", "port": 8888}))

        with pytest.raises(RuntimeError, match="systemctl start"):
            b.start(cfg_path)

        assert active.resolve() == previous_script.resolve()
        assert active_json.resolve() == previous_json.resolve()
        manifest = read_active_manifest(manifest_path)
        assert manifest is not None
        assert manifest["status"] == "failed"
        assert "Unit not found" in manifest["error"]


class TestLlamaCppLaunchScript:
    def test_script_content(self, mocker, tmp_path):
        """Launch script has correct flags and is shell-safe."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/llama-cpp/models/test/model.gguf",
            "host": "0.0.0.0",
            "port": 8888,
            "ctx_size": 8192,
            "n_gpu_layers": 32,
            "parallel": 4,
            "flash_attn": True,
            "jinja": True,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert script.startswith("#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=0\nexec ")
        assert "-m /opt/llama-cpp/models/test/model.gguf" in script
        assert "-c 8192" in script
        assert "-ngl 32" in script
        assert "-np 4" in script
        assert "-fa on" in script
        assert "--jinja" in script
        assert "--host 0.0.0.0" in script
        assert "--port 8888" in script

    def test_script_exports_configured_gpu_index(self, mocker, tmp_path):
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))
        mocker.patch("vserve.config.cfg", return_value=Mock(gpu_index=2))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "model": "/m/model.gguf",
            "host": "0.0.0.0",
            "port": 8888,
            "ctx_size": 4096,
            "n_gpu_layers": 10,
            "parallel": 1,
        }))

        b.start(cfg_path)

        script = active.read_text()
        assert "export CUDA_VISIBLE_DEVICES=2\n" in script

    def test_script_quoting_spaces(self, mocker, tmp_path):
        """Model paths with spaces are properly quoted in script."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/models/My Model Dir/model file.gguf",
            "host": "0.0.0.0",
            "port": 8888,
            "ctx_size": 4096,
            "n_gpu_layers": 10,
            "parallel": 1,
            "flash_attn": True,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        # shlex.join should quote the path with spaces using single quotes
        assert "My Model Dir" in script
        assert "model file.gguf" in script
        # The path should be single-quoted by shlex
        assert "'/opt/models/My Model Dir/model file.gguf'" in script


class TestLlamaCppEmbedding:
    def _mock_metadata(self, mocker, pooling=None):
        meta = {
            "arch": "nomic-bert",
            "num_layers": 12,
            "max_context": 8192,
            "num_kv_heads": 12,
            "head_dim": 64,
            "pooling": pooling,
        }
        mocker.patch.object(LlamaCppBackend, "_read_gguf_metadata", return_value=meta)

    def test_tune_embedding_model(self, fake_embedding_model_dir, mocker):
        """tune() returns is_embedding and pooling for embedding models."""
        self._mock_metadata(mocker, pooling="mean")
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_embedding_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert result["is_embedding"] is True
        assert result["pooling"] == "mean"
        assert result["supports_tools"] is False

    def test_tune_embedding_guesses_pooling(self, fake_embedding_model_dir, mocker):
        """tune() guesses pooling when GGUF metadata lacks it."""
        self._mock_metadata(mocker, pooling=None)
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_embedding_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert result["is_embedding"] is True
        assert result["pooling"] == "mean"  # nomic → mean

    def test_tune_non_embedding_has_no_embedding_key(self, fake_gguf_model_dir, mocker):
        """tune() for non-embedding models has no is_embedding key."""
        mocker.patch.object(LlamaCppBackend, "_read_gguf_metadata", return_value={
            "arch": "llama", "num_layers": 32, "max_context": 8192,
            "num_kv_heads": 8, "head_dim": 128, "pooling": None,
        })
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert "is_embedding" not in result
        assert "pooling" not in result

    def test_build_config_embedding_no_jinja(self, fake_embedding_model_dir):
        """Embedding config has --embedding but not --jinja."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_embedding_model_dir)

        choices = {
            "context": 512, "n_gpu_layers": 12, "parallel": 8,
            "port": 8888, "embedding": True, "pooling": "mean",
        }
        cfg = b.build_config(m, choices)
        assert cfg["embedding"] is True
        assert cfg["pooling"] == "mean"
        assert "jinja" not in cfg

    def test_start_script_embedding_flags(self, mocker, tmp_path):
        """Launch script includes --embedding and --pooling flags."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/llama-cpp/models/nomic/embed.gguf",
            "host": "0.0.0.0", "port": 8888,
            "ctx_size": 8192, "n_gpu_layers": 12, "parallel": 8,
            "flash_attn": True,
            "embedding": True, "pooling": "mean",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert "--embedding" in script
        assert "--pooling mean" in script
        assert "--jinja" not in script

    def test_start_script_cls_pooling(self, mocker, tmp_path):
        """Launch script respects cls pooling for BGE models."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/llama-cpp/models/bge/model.gguf",
            "host": "0.0.0.0", "port": 8888,
            "ctx_size": 512, "n_gpu_layers": 12, "parallel": 1,
            "flash_attn": True,
            "embedding": True, "pooling": "cls",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert "--pooling cls" in script

    def test_start_script_no_pooling_when_absent(self, mocker, tmp_path):
        """No --pooling flag when pooling is not set."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/m/model.gguf", "host": "0.0.0.0", "port": 8888,
            "ctx_size": 4096, "n_gpu_layers": 10, "parallel": 1,
            "flash_attn": True,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert "--pooling" not in script
        assert "--embedding" not in script

    def test_build_config_no_tools_no_embedding(self, fake_gguf_model_dir):
        """Config with neither tools nor embedding has no jinja or embedding flags."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
        }
        cfg = b.build_config(m, choices)
        assert "jinja" not in cfg
        assert "embedding" not in cfg
        assert "pooling" not in cfg

    def test_build_config_emits_kv_cache_dtypes(self, fake_gguf_model_dir):
        """When kv_cache_k/v are set, build_config records them in the JSON."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
            "kv_cache_k": "q8_0", "kv_cache_v": "q8_0",
        })
        assert cfg["cache_type_k"] == "q8_0"
        assert cfg["cache_type_v"] == "q8_0"

    def test_build_config_ctx_size_is_per_slot_times_parallel(self, fake_gguf_model_dir):
        """Per-slot context × parallel = ctx_size (the -c value llama-server expects).

        llama-server's -c is total KV-cache size across slots; per-slot
        window = -c / -np. vserve's user-facing 'context' is per-slot, so
        build_config has to multiply before writing the launch JSON.
        """
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 32768, "n_gpu_layers": 30, "parallel": 4,
            "port": 8888, "tools": False,
        })
        assert cfg["ctx_size"] == 32768 * 4  # 131072 — what llama-server's -c sees
        assert cfg["ctx_per_slot"] == 32768  # what the user asked for
        assert cfg["parallel"] == 4

    def test_build_config_single_slot_ctx_size_equals_context(self, fake_gguf_model_dir):
        """With parallel=1, ctx_size == per-slot context."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 8192, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
        })
        assert cfg["ctx_size"] == 8192
        assert cfg["ctx_per_slot"] == 8192

    def test_build_config_emits_batch_and_ubatch(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
            "batch_size": 4096, "ubatch_size": 512,
        })
        assert cfg["batch_size"] == 4096
        assert cfg["ubatch_size"] == 512

    def test_build_config_records_override_tensors(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
            "override_tensors": [".ffn_.*_exps.=CPU"],
        })
        assert cfg["override_tensors"] == [".ffn_.*_exps.=CPU"]

    def test_start_emits_ctk_ctv_b_ub_and_ot(self, fake_gguf_model_dir, tmp_path, mocker):
        """The generated launch script contains every new flag in canonical form."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 8192, "n_gpu_layers": 30, "parallel": 4,
            "port": 8888, "tools": True,
            "kv_cache_k": "q8_0", "kv_cache_v": "q8_0",
            "batch_size": 4096, "ubatch_size": 512,
            "override_tensors": [".ffn_.*_exps.=CPU"],
        })
        cfg_path = tmp_path / "test.json"
        import json
        cfg_path.write_text(json.dumps(cfg))
        active = tmp_path / "active.sh"
        active.parent.mkdir(parents=True, exist_ok=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "_assert_unit_safe_for_privileged_action")
        run = mocker.patch("vserve.backends.llamacpp.subprocess.run",
                           return_value=Mock(returncode=0, stdout="", stderr=""))

        b.start(cfg_path)

        script = active.read_text()
        assert "-ctk q8_0" in script
        assert "-ctv q8_0" in script
        assert "-b 4096" in script
        assert "-ub 512" in script
        assert "-ot '.ffn_.*_exps.=CPU'" in script or '-ot ".ffn_.*_exps.=CPU"' in script
        # L7: --no-mmap is auto-added whenever -ot is set so the new llama.cpp
        # binary doesn't emit its "mmap + tensor override" perf warning.
        assert "--no-mmap" in script
        # Context fix: per-slot 8192 × 4 slots = 32768 in the -c arg, so each
        # slot ends up with the 8192 window the caller actually asked for.
        assert "-c 32768" in script
        assert "-np 4" in script
        assert run.called

    def test_start_omits_no_mmap_when_no_override_tensors(self, fake_gguf_model_dir, tmp_path, mocker):
        """--no-mmap is only auto-added when -ot is in play."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
        })
        cfg_path = tmp_path / "test.json"
        import json
        cfg_path.write_text(json.dumps(cfg))
        active = tmp_path / "active.sh"
        active.parent.mkdir(parents=True, exist_ok=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "_assert_unit_safe_for_privileged_action")
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        b.start(cfg_path)

        script = active.read_text()
        assert "--no-mmap" not in script

    def test_start_respects_explicit_mmap_true(self, fake_gguf_model_dir, tmp_path, mocker):
        """Setting mmap=True in the cfg suppresses the auto --no-mmap even when -ot is on."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        # build_config doesn't expose mmap directly today — write it into the
        # JSON to simulate a user manually editing the profile.
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
            "override_tensors": [".ffn_.*_exps.=CPU"],
        })
        cfg["mmap"] = True
        cfg_path = tmp_path / "test.json"
        import json
        cfg_path.write_text(json.dumps(cfg))
        active = tmp_path / "active.sh"
        active.parent.mkdir(parents=True, exist_ok=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "_assert_unit_safe_for_privileged_action")
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        b.start(cfg_path)

        script = active.read_text()
        assert "--no-mmap" not in script
        # -ot still emitted
        assert "-ot" in script

    def test_is_unsloth_ud_detects_UD_prefix(self, tmp_path):
        """ModelInfo.is_unsloth_ud is True only when filename contains -UD-."""
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "unsloth" / "Qwen3-GGUF-Q4_K_XL"
        model_dir.mkdir(parents=True)
        (model_dir / "Qwen3-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="Qwen3-GGUF-Q4_K_XL",
            architecture="qwen3", model_type="gguf", quant_method=None,
            max_position_embeddings=0, is_moe=False, model_size_gb=1.0, is_gguf=True,
        )
        assert m.is_unsloth_ud is True

    def test_is_unsloth_ud_false_for_plain_quant(self, tmp_path):
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "unsloth" / "Qwen3-GGUF-Q4_K_M"
        model_dir.mkdir(parents=True)
        (model_dir / "Qwen3-Q4_K_M.gguf").write_bytes(b"GGUF")
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="Qwen3-GGUF-Q4_K_M",
            architecture="qwen3", model_type="gguf", quant_method=None,
            max_position_embeddings=0, is_moe=False, model_size_gb=1.0, is_gguf=True,
        )
        assert m.is_unsloth_ud is False

    def test_is_unsloth_ud_false_for_non_unsloth_provider(self, tmp_path):
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "other" / "Qwen3-UD-Q4_K_XL"
        model_dir.mkdir(parents=True)
        (model_dir / "Qwen3-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
        m = ModelInfo(
            path=model_dir, provider="other", model_name="Qwen3-UD-Q4_K_XL",
            architecture="qwen3", model_type="gguf", quant_method=None,
            max_position_embeddings=0, is_moe=False, model_size_gb=1.0, is_gguf=True,
        )
        assert m.is_unsloth_ud is False

    def test_guess_pooling_case_insensitive(self):
        assert LlamaCppBackend._guess_pooling("BGE-Large-EN-v1.5") == "cls"
        assert LlamaCppBackend._guess_pooling("NOMIC-EMBED-TEXT") == "mean"
        assert LlamaCppBackend._guess_pooling("Jina-Reranker-v2") == "rank"
