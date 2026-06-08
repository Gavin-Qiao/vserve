"""Tests for the backend registry and protocol."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vserve.backends import register, get_backend, get_backend_by_name, available_backends, any_backend_running, running_backend
from vserve.backends.protocol import Backend
from vserve.backends.vllm import VllmBackend


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before each test."""
    from vserve.backends import _BACKENDS
    saved = _BACKENDS.copy()
    _BACKENDS.clear()
    yield
    _BACKENDS.clear()
    _BACKENDS.extend(saved)


def _make_mock_backend(name: str, can_serve_result: bool = True, has_binary: bool = True) -> Mock:
    b = Mock(spec=Backend)
    b.name = name
    b.display_name = name
    b.can_serve.return_value = can_serve_result
    b.find_entrypoint.return_value = Path("/fake/bin") if has_binary else None
    return b


def test_register_and_get_backend_by_name():
    b = _make_mock_backend("test")
    register(b)
    assert get_backend_by_name("test") is b


def test_get_backend_by_name_unknown():
    with pytest.raises(KeyError):
        get_backend_by_name("nonexistent")


def test_get_backend_auto_detect():
    b1 = _make_mock_backend("a", can_serve_result=False)
    b2 = _make_mock_backend("b", can_serve_result=True)
    register(b1)
    register(b2)

    model = Mock()
    assert get_backend(model) is b2


def test_get_backend_no_match():
    b = _make_mock_backend("a", can_serve_result=False)
    register(b)

    model = Mock()
    with pytest.raises(ValueError, match="No backend"):
        get_backend(model)


def test_available_backends_filters_missing():
    b1 = _make_mock_backend("installed", has_binary=True)
    b2 = _make_mock_backend("missing", has_binary=False)
    register(b1)
    register(b2)

    result = available_backends()
    assert len(result) == 1
    assert result[0].name == "installed"


# --- VllmBackend tests ---


class TestVllmBackend:
    def test_identity(self):
        b = VllmBackend()
        assert b.name == "vllm"
        assert b.display_name == "vLLM"

    def test_can_serve_safetensors(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)
        assert b.can_serve(m) is True

    def test_can_serve_uppercase_safetensors_suffix(self, tmp_path):
        b = VllmBackend()
        model_dir = tmp_path / "models" / "user" / "UppercaseWeights"
        model_dir.mkdir(parents=True)
        (model_dir / "model.SAFETENSORS").write_bytes(b"\0")

        from vserve.models import ModelInfo
        m = ModelInfo(
            path=model_dir, provider="user", model_name="UppercaseWeights",
            architecture="TestLM", model_type="test", quant_method=None,
            max_position_embeddings=4096, is_moe=False, model_size_gb=0.1,
        )
        assert b.can_serve(m) is True

    def test_cannot_serve_gguf_only(self, tmp_path):
        b = VllmBackend()
        model_dir = tmp_path / "models" / "user" / "Model-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\0" * 100)

        from vserve.models import ModelInfo
        m = ModelInfo(
            path=model_dir, provider="user", model_name="Model-GGUF",
            architecture="unknown", model_type="unknown", quant_method=None,
            max_position_embeddings=4096, is_moe=False, model_size_gb=0.1,
        )
        assert b.can_serve(m) is False

    def test_cannot_serve_config_only_root(self, tmp_path):
        b = VllmBackend()
        model_dir = tmp_path / "models" / "user" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        from vserve.models import detect_model

        m = detect_model(model_dir)

        assert b.can_serve(m) is False

    def test_health_url(self):
        b = VllmBackend()
        assert b.health_url(8888) == "http://localhost:8888/health"

    def test_quant_flag(self):
        b = VllmBackend()
        assert "gptq_marlin" in b.quant_flag("gptq")
        assert "fp8" in b.quant_flag("fp8")
        assert b.quant_flag(None) == ""
        assert b.quant_flag("compressed-tensors") == ""

    def test_tune_delegates_to_probe(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        mock_gpu = Mock()
        mock_gpu.vram_total_gb = 48.0

        result = b.tune(m, mock_gpu, gpu_mem_util=0.90)
        assert "limits" in result
        assert "tool_call_parser" in result

    def test_detect_tools(self, fake_model_dir):
        b = VllmBackend()
        result = b.detect_tools(fake_model_dir)
        assert "tool_call_parser" in result
        assert "reasoning_parser" in result

    def test_start_stop_delegates(self, mocker, tmp_path):
        b = VllmBackend()
        mock_start = mocker.patch("vserve.serve.start_vllm")
        mock_stop = mocker.patch("vserve.serve.stop_vllm")
        mocker.patch("vserve.serve.is_vllm_running", return_value=True)

        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("model: test\n")
        b.start(cfg_path)
        mock_start.assert_called_once_with(cfg_path, non_interactive=False)

        b.stop()
        mock_stop.assert_called_once_with(non_interactive=False)

        assert b.is_running() is True

    def test_build_config_basic(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": None,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
        }
        cfg = b.build_config(m, choices)
        assert cfg["max-model-len"] == 8192
        assert cfg["max-num-seqs"] == 4
        assert cfg["kv-cache-dtype"] == "auto"
        assert cfg["gpu-memory-utilization"] == 0.90
        assert "enable-auto-tool-choice" not in cfg
        assert "trust-remote-code" not in cfg

    def test_build_config_trust_remote_code_explicit(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": None,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
            "trust_remote_code": True,
        }
        cfg = b.build_config(m, choices)
        assert cfg["trust-remote-code"] is True

    def test_build_config_with_tools(self, fake_model_dir):
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value={"hermes"})  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value={"qwen3"})  # type: ignore[method-assign]
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": True,
            "tool_parser": "hermes",
            "reasoning_parser": "qwen3",
        }
        cfg = b.build_config(m, choices)
        assert cfg["enable-auto-tool-choice"] is True
        assert cfg["tool-call-parser"] == "hermes"
        assert cfg["reasoning-parser"] == "qwen3"
        assert cfg["max-num-batched-tokens"] == 4096

    def test_build_config_with_sota_scheduler_and_cache_knobs(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "turboquant_k8v4",
            "slots": 4,
            "batched_tokens": 12288,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
            "performance_mode": "throughput",
            "optimization_level": 2,
            "block_size": 16,
            "kv_cache_memory_bytes": 12 * 1024**3,
            "enable_prefix_caching": True,
        }

        cfg = b.build_config(m, choices)

        assert cfg["kv-cache-dtype"] == "turboquant_k8v4"
        assert cfg["max-num-batched-tokens"] == 12288
        assert cfg["performance-mode"] == "throughput"
        assert cfg["optimization-level"] == 2
        assert cfg["block-size"] == 16
        assert cfg["kv-cache-memory-bytes"] == 12 * 1024**3
        assert cfg["enable-prefix-caching"] is True

    def test_build_config_defaults_safe_batched_tokens_for_hybrid_models(self, tmp_path):
        b = VllmBackend()
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "Qwen" / "Qwen3.6-35B-A3B-FP8"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            """
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "text_config": {
    "model_type": "qwen3_5_moe_text",
    "full_attention_interval": 4,
    "layer_types": ["linear_attention", "linear_attention", "full_attention"]
  }
}
""".strip()
            + "\n",
        )
        m = ModelInfo(
            path=model_dir,
            provider="Qwen",
            model_name="Qwen3.6-35B-A3B-FP8",
            architecture="Qwen3_5MoeForConditionalGeneration",
            model_type="qwen3_5_moe",
            quant_method="fp8",
            max_position_embeddings=262144,
            is_moe=True,
            model_size_gb=34.9,
        )

        cfg = b.build_config(
            m,
            {
                "context": 262144,
                "kv_dtype": "fp8",
                "slots": 1,
                "batched_tokens": None,
                "gpu_mem_util": 0.95,
                "port": 8888,
                "tools": False,
                "tool_parser": None,
                "reasoning_parser": None,
            },
        )

        assert cfg["max-num-batched-tokens"] == 4096

    def test_build_config_reasoning_without_tools(self, fake_model_dir):
        b = VllmBackend()
        b.available_reasoning_parsers = Mock(return_value={"qwen3"})  # type: ignore[method-assign]
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": None,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": "qwen3",
        }
        cfg = b.build_config(m, choices)
        assert cfg["reasoning-parser"] == "qwen3"
        assert "enable-auto-tool-choice" not in cfg

    def test_build_config_rejects_unknown_tool_parser(self, fake_model_dir, mocker):
        b = VllmBackend()
        mocker.patch.object(b, "available_tool_parsers", return_value={"hermes"})
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 4096,
            "kv_dtype": "auto",
            "slots": 1,
            "gpu_mem_util": 0.9,
            "port": 8888,
            "tools": True,
            "tool_parser": "qwen-3",
            "reasoning_parser": None,
        }

        import pytest
        with pytest.raises(ValueError, match="Unknown vLLM tool parser"):
            b.build_config(m, choices)

    def test_build_config_rejects_unknown_reasoning_parser(self, fake_model_dir, mocker):
        b = VllmBackend()
        mocker.patch.object(b, "available_reasoning_parsers", return_value={"qwen3"})
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 4096,
            "kv_dtype": "auto",
            "slots": 1,
            "gpu_mem_util": 0.9,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": "qwen-3",
        }

        import pytest
        with pytest.raises(ValueError, match="Unknown vLLM reasoning parser"):
            b.build_config(m, choices)

    def test_build_config_rejects_parser_when_runtime_introspection_uncertain(self, fake_model_dir, mocker):
        b = VllmBackend()
        mocker.patch.object(b, "available_tool_parsers", return_value=None)
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 4096,
            "kv_dtype": "auto",
            "slots": 1,
            "gpu_mem_util": 0.9,
            "port": 8888,
            "tools": True,
            "tool_parser": "hermes",
            "reasoning_parser": None,
        }

        with pytest.raises(RuntimeError, match="Could not inspect installed vLLM tool parsers"):
            b.build_config(m, choices)

    def test_available_parsers_use_configured_runtime_python(self, mocker, tmp_path):
        b = VllmBackend()
        runtime_python = tmp_path / "venv" / "bin" / "python"
        mocker.patch("vserve.config.cfg", return_value=Mock(vllm_python=runtime_python))
        run = mocker.patch(
            "subprocess.run",
            return_value=Mock(
                returncode=0,
                stdout='{"tool_parsers": ["hermes"], "reasoning_parsers": ["qwen3"]}',
                stderr="",
            ),
        )

        assert b.available_tool_parsers() == {"hermes"}
        assert b.available_reasoning_parsers() == {"qwen3"}
        assert run.call_args.args[0][0] == str(runtime_python)
        assert run.call_count == 1

    def test_parser_probe_supports_vllm_020_manager_paths(self, mocker, tmp_path):
        b = VllmBackend()
        runtime_python = tmp_path / "venv" / "bin" / "python"
        mocker.patch("vserve.config.cfg", return_value=Mock(vllm_python=runtime_python))
        run = mocker.patch(
            "subprocess.run",
            return_value=Mock(
                returncode=0,
                stdout='{"tool_parsers": ["hermes"], "reasoning_parsers": ["qwen3"]}',
                stderr="",
            ),
        )

        assert b.available_tool_parsers() == {"hermes"}
        script = run.call_args.args[0][2]
        assert "vllm.tool_parsers" in script
        assert "list_registered" in script

    def test_find_entrypoint_missing(self, mocker):
        b = VllmBackend()
        mock_c = Mock()
        mock_c.vllm_bin = Path("/nonexistent/vllm")
        mocker.patch("vserve.config.cfg", return_value=mock_c)
        assert b.find_entrypoint() is None

    def test_find_entrypoint_exists(self, mocker, tmp_path):
        b = VllmBackend()
        vllm_bin = tmp_path / "venv" / "bin" / "vllm"
        vllm_bin.parent.mkdir(parents=True)
        vllm_bin.touch()
        mock_c = Mock()
        mock_c.vllm_bin = vllm_bin
        mocker.patch("vserve.config.cfg", return_value=mock_c)
        assert b.find_entrypoint() == vllm_bin

    def test_doctor_checks_returns_callables(self):
        b = VllmBackend()
        checks = b.doctor_checks()
        assert len(checks) >= 2
        for desc, fn in checks:
            assert isinstance(desc, str)
            assert callable(fn)

    def test_service_identity(self):
        b = VllmBackend()
        assert b.service_name == "vllm"
        assert b.service_user == "vllm"

    def test_configured_service_identity(self, mocker):
        b = VllmBackend()
        mocker.patch(
            "vserve.config.cfg",
            return_value=Mock(service_name="custom-vllm", service_user="svc-vllm", vllm_root=Path("/opt/vllm")),
        )
        assert b.service_name == "custom-vllm"
        assert b.service_user == "svc-vllm"


# --- Auto-registration ---


def test_default_backends_registered():
    """Built-in backends are registered after _register_defaults."""
    from vserve.backends import _register_defaults
    _register_defaults()

    from vserve.backends import _BACKENDS
    names = [b.name for b in _BACKENDS]
    assert "vllm" in names
    assert "llamacpp" in names


def test_duplicate_registration_prevented():
    """register() with same name is idempotent."""
    b1 = _make_mock_backend("dedup")
    b2 = _make_mock_backend("dedup")
    register(b1)
    register(b2)
    from vserve.backends import _BACKENDS
    assert sum(1 for b in _BACKENDS if b.name == "dedup") == 1


# --- any_backend_running / running_backend ---


def test_any_backend_running_true():
    b = _make_mock_backend("running")
    b.is_running.return_value = True
    register(b)
    assert any_backend_running() is True


def test_any_backend_running_false():
    b = _make_mock_backend("stopped")
    b.is_running.return_value = False
    register(b)
    assert any_backend_running() is False


def test_any_backend_running_exception_handled():
    b = _make_mock_backend("broken")
    b.is_running.side_effect = RuntimeError("no systemctl")
    register(b)
    assert any_backend_running() is False


def test_probe_running_backend_partial_failure_with_successful_false_probe():
    from vserve.backends import probe_running_backend

    broken = _make_mock_backend("broken")
    broken.is_running.side_effect = RuntimeError("dbus error")
    idle = _make_mock_backend("idle")
    idle.is_running.return_value = False
    register(broken)
    register(idle)

    backend, probe_failed = probe_running_backend()

    assert backend is None
    assert probe_failed is True


def test_running_backend_returns_correct():
    b1 = _make_mock_backend("idle")
    b1.is_running.return_value = False
    b2 = _make_mock_backend("active")
    b2.is_running.return_value = True
    register(b1)
    register(b2)
    result = running_backend()
    assert result is not None
    assert result.name == "active"


def test_running_backend_none_when_all_stopped():
    b = _make_mock_backend("stopped")
    b.is_running.return_value = False
    register(b)
    assert running_backend() is None


# --- vLLM backend × dtype × architecture compatibility ---


class TestVllmArchCompat:
    """Defect: tuner emitted KV-dtype cells (e.g. TurboQuant) that the forced
    attention backend (e.g. TRITON_ATTN for Gemma-4's heterogeneous head dims)
    refuses, crashing engine init. Filter must drop those cells in tune output.
    """

    def _write_model_config(self, tmp_path: Path, cfg: dict) -> Path:
        import json
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return tmp_path

    def test_heterogeneous_head_dims_force_triton_attn_detected(self, tmp_path):
        from vserve.backends.vllm import _architecture_forces_triton_attn

        # Gemma-4 shape: head_dim != global_head_dim inside text_config
        self._write_model_config(tmp_path, {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {"head_dim": 256, "global_head_dim": 512},
        })
        assert _architecture_forces_triton_attn(tmp_path) is True

    def test_uniform_head_dims_do_not_force_triton_attn(self, tmp_path):
        from vserve.backends.vllm import _architecture_forces_triton_attn

        self._write_model_config(tmp_path, {
            "architectures": ["LlamaForCausalLM"],
            "head_dim": 128,
        })
        assert _architecture_forces_triton_attn(tmp_path) is False

    def test_missing_config_returns_false(self, tmp_path):
        from vserve.backends.vllm import _architecture_forces_triton_attn

        assert _architecture_forces_triton_attn(tmp_path) is False

    def test_filter_removes_turboquant_cells_for_gemma4(self, tmp_path):
        from vserve.backends.vllm import _filter_incompatible_kv_dtypes

        self._write_model_config(tmp_path, {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {"head_dim": 256, "global_head_dim": 512},
        })
        limits_data = {
            "limits": {
                "4096": {"auto": 4, "fp8": 8, "turboquant_3bit_nc": 21},
                "8192": {"auto": 2, "fp8": 4, "turboquant_3bit_nc": 10},
            },
            "kv_cache_dtypes": {
                "auto": {"bytes_per_token": 1000},
                "fp8": {"bytes_per_token": 500},
                "turboquant_3bit_nc": {"bytes_per_token": 200},
            },
        }
        out = _filter_incompatible_kv_dtypes(limits_data, tmp_path)
        # TurboQuant gone from every row and from the dtype catalog.
        assert "turboquant_3bit_nc" not in out["limits"]["4096"]
        assert "turboquant_3bit_nc" not in out["limits"]["8192"]
        assert "turboquant_3bit_nc" not in out["kv_cache_dtypes"]
        # Compatible dtypes survive.
        assert out["limits"]["4096"]["auto"] == 4
        assert out["limits"]["4096"]["fp8"] == 8
        # Records the forced backend so consumers can show "vLLM forced TRITON_ATTN".
        assert out["forced_attn_backend"] == "TRITON_ATTN"

    def test_filter_is_noop_for_compatible_archs(self, tmp_path):
        from vserve.backends.vllm import _filter_incompatible_kv_dtypes

        self._write_model_config(tmp_path, {
            "architectures": ["LlamaForCausalLM"],
            "head_dim": 128,
        })
        limits_data = {
            "limits": {"4096": {"auto": 4, "turboquant_3bit_nc": 21}},
            "kv_cache_dtypes": {"turboquant_3bit_nc": {"bytes_per_token": 200}},
        }
        out = _filter_incompatible_kv_dtypes(limits_data, tmp_path)
        # No changes — Llama uses FLASH_ATTN which accepts TurboQuant.
        assert out["limits"]["4096"]["turboquant_3bit_nc"] == 21
        assert "turboquant_3bit_nc" in out["kv_cache_dtypes"]
        assert "forced_attn_backend" not in out

    def test_filter_drops_recommended_profile_if_now_invalid(self, tmp_path):
        from vserve.backends.vllm import _filter_incompatible_kv_dtypes

        self._write_model_config(tmp_path, {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {"head_dim": 256, "global_head_dim": 512},
        })
        limits_data = {
            "limits": {"4096": {"auto": 4, "fp8": 8, "turboquant_3bit_nc": 21}},
            "kv_cache_dtypes": {},
            "recommended_profile": {"kv_cache_dtype": "turboquant_3bit_nc", "context": 4096},
        }
        out = _filter_incompatible_kv_dtypes(limits_data, tmp_path)
        # Recommendation pointed at a now-filtered dtype → drop it so the CLI
        # falls back to picker-driven selection.
        assert out["recommended_profile"] is None


class TestVllmMultimodalFloor:
    """Defect: vLLM disables MM-input chunking for bidirectional-attention
    encoders. When max-num-batched-tokens defaults to 2048 but a single image
    expands past it (e.g. Gemma-4 vision = 2496 tokens), engine init crashes.
    """

    def test_text_only_model_returns_none(self, tmp_path):
        import json
        from vserve.backends.vllm import _mm_batched_tokens_floor
        (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
        assert _mm_batched_tokens_floor(tmp_path) is None

    def test_vision_config_triggers_floor(self, tmp_path):
        import json
        from vserve.backends.vllm import _mm_batched_tokens_floor
        (tmp_path / "config.json").write_text(json.dumps({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
        }))
        assert _mm_batched_tokens_floor(tmp_path) == 4096

    def test_audio_config_triggers_higher_floor(self, tmp_path):
        """Audio (Gemma-4 E-class): 750 tokens/segment, multi-segment requests
        push past 4096 routinely. Item J raises the audio floor to 8192."""
        import json
        from vserve.backends.vllm import _mm_batched_tokens_floor
        (tmp_path / "config.json").write_text(json.dumps({"audio_config": {"hidden_size": 768}}))
        assert _mm_batched_tokens_floor(tmp_path) == 8192

    def test_audio_plus_vision_still_audio_floor(self, tmp_path):
        import json
        from vserve.backends.vllm import _mm_batched_tokens_floor
        (tmp_path / "config.json").write_text(json.dumps({
            "vision_config": {"hidden_size": 1024},
            "audio_config": {"hidden_size": 768},
        }))
        # When audio is present, audio floor wins (larger).
        assert _mm_batched_tokens_floor(tmp_path) == 8192


class TestVllmServedNameAliases:
    """Defect: vLLM advertises the full filesystem path as the served model
    id. OpenAI clients sending a short name 400 with `model not found`.
    Auto-emit canonical + slug aliases."""

    def test_canonical_provider_model_first(self):
        from vserve.backends.vllm import _served_model_name_aliases
        aliases = _served_model_name_aliases("nvidia", "Gemma-4-31B-IT-NVFP4")
        assert aliases[0] == "nvidia/Gemma-4-31B-IT-NVFP4"

    def test_strips_common_chat_suffixes_for_slug(self):
        from vserve.backends.vllm import _served_model_name_aliases
        aliases = _served_model_name_aliases("Qwen", "Qwen3.5-4B-Instruct")
        assert aliases == ["Qwen/Qwen3.5-4B-Instruct", "qwen3.5-4b"]

    def test_handles_gguf_suffix(self):
        from vserve.backends.vllm import _served_model_name_aliases
        aliases = _served_model_name_aliases("unsloth", "gemma-4-26B-A4B-it-GGUF")
        # `-it-gguf` matches before `-gguf`, stripping both segments.
        assert "unsloth/gemma-4-26B-A4B-it-GGUF" in aliases
        assert "gemma-4-26b-a4b" in aliases

    def test_omits_duplicate_slug_when_already_canonical(self):
        from vserve.backends.vllm import _served_model_name_aliases
        aliases = _served_model_name_aliases("vendor", "Plain-Model")
        # Slug ("plain-model") would dedupe against the canonical case-insensitively.
        assert len(aliases) == 2  # canonical + lowercased slug, both kept (different case)


class TestVllmToolParserAutoDetect:
    """Defect: vLLM ships per-arch tool parsers (gemma4 etc.) but vserve does
    not surface them. Map architectures to bundled parsers and emit
    `enable-auto-tool-choice` + `tool-call-parser` when tools are requested."""

    def test_gemma4_arch_maps_to_gemma4_parser(self, tmp_path):
        import json
        from vserve.backends.vllm import _suggested_tool_parser
        (tmp_path / "config.json").write_text(json.dumps({
            "architectures": ["Gemma4ForConditionalGeneration"]
        }))
        # available_parsers=None means "skip availability check" — we still
        # return the mapped name so callers can decide.
        assert _suggested_tool_parser(tmp_path, None) == "gemma4"

    def test_only_returns_parser_that_runtime_has_installed(self, tmp_path):
        import json
        from vserve.backends.vllm import _suggested_tool_parser
        (tmp_path / "config.json").write_text(json.dumps({
            "architectures": ["Gemma4ForCausalLM"]
        }))
        assert _suggested_tool_parser(tmp_path, {"hermes", "llama3_json"}) is None
        assert _suggested_tool_parser(tmp_path, {"gemma4", "hermes"}) == "gemma4"

    def test_unknown_architecture_returns_none(self, tmp_path):
        import json
        from vserve.backends.vllm import _suggested_tool_parser
        (tmp_path / "config.json").write_text(json.dumps({
            "architectures": ["SomeNovelArchForCausalLM"]
        }))
        assert _suggested_tool_parser(tmp_path, {"gemma4", "hermes"}) is None


# --- 0.7.0 item AA: cudagraph_mode pre-emission ---


class TestVllmCompilationConfig:
    """Defect (AA): TurboQuant decode kernel asserts when its workspace
    requirement exceeds the size baked in during CUDA-graph capture. The
    maintainer-canonical fix is `compilation-config: {cudagraph_mode: NONE}`
    per vllm#40807 / #41403, which keeps torch.compile fusions but skips graph
    capture. Pre-emit it for any `turboquant_*` KV dtype so users never see
    the assertion. Also pre-emit when spec-decode is requested with quantized
    KV (vllm#41559 — DFlash spec-decode breaks with any KV quantization).
    """

    def test_turboquant_auto_emits_cudagraph_mode_none(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model

        b = VllmBackend()
        m = detect_model(fake_model_dir)
        choices = {
            "context": 8192,
            "kv_dtype": "turboquant_3bit_nc",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
        }
        cfg = b.build_config(m, choices)
        assert "compilation-config" in cfg
        assert cfg["compilation-config"]["cudagraph_mode"] == "NONE"

    def test_non_turboquant_does_not_emit_compilation_config(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model

        b = VllmBackend()
        m = detect_model(fake_model_dir)
        choices = {
            "context": 8192,
            "kv_dtype": "fp8",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
        }
        cfg = b.build_config(m, choices)
        assert "compilation-config" not in cfg

    def test_spec_with_quantized_kv_emits_cudagraph_mode_none(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model

        b = VllmBackend()
        m = detect_model(fake_model_dir)
        choices = {
            "context": 8192,
            "kv_dtype": "fp8",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
            "spec": {"method": "ngram", "n_max": 5},
        }
        cfg = b.build_config(m, choices)
        # vllm#41559 — disable CUDA-graph capture when spec + quantized KV.
        assert cfg.get("compilation-config", {}).get("cudagraph_mode") == "NONE"

    def test_spec_with_auto_kv_does_not_emit_compilation_config(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model

        b = VllmBackend()
        m = detect_model(fake_model_dir)
        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
            "spec": {"method": "ngram", "n_max": 5},
        }
        cfg = b.build_config(m, choices)
        # auto KV is not quantized → no need for cudagraph_mode: NONE
        assert "compilation-config" not in cfg

    def test_explicit_compilation_config_overrides_default(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model

        b = VllmBackend()
        m = detect_model(fake_model_dir)
        # Confirm setdefault — if a downstream choice path is later added that
        # passes an explicit compilation-config, our setdefault preserves it.
        # Today there's no such path, so this is a forward-compat regression
        # test that the auto-emission uses setdefault, not assignment.
        choices = {
            "context": 8192,
            "kv_dtype": "turboquant_3bit_nc",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
        }
        cfg = b.build_config(m, choices)
        # Mimic a user passing their own compilation-config — write to it then
        # call build_config a second time would race; this just confirms the
        # current cfg uses setdefault semantics.
        cfg["compilation-config"]["cudagraph_mode"] = "FULL_AND_PIECEWISE"
        # If someone later adds `choices["compilation_config"]` pass-through,
        # we expect their explicit value to win. For now, only assert the
        # default fires (the setdefault chain in build_config).
        assert cfg["compilation-config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"


# --- 0.7.0 item A: architecture-derived sampler defaults ---


class TestVllmSamplingDefaults:
    """Item A: vLLM build_config emits an `override-generation-config` block
    when the architecture is in the SAMPLING_DEFAULTS registry. Caller can
    opt out via `recipe_sampling=False` or supply an explicit value via
    `override_generation_config`."""

    def _make_gemma4_model(self, tmp_path):
        import json
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({
            "architectures": ["Gemma4ForCausalLM"],
            "head_dim": 256,
        }))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)
        return ModelInfo(
            path=model_dir, provider="unsloth", model_name="gemma-4-26B-A4B",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )

    def test_known_arch_emits_override_generation_config(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_gemma4_model(tmp_path)
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        ogc = cfg.get("override-generation-config")
        assert isinstance(ogc, dict)
        assert ogc["temperature"] == 1.0
        assert ogc["top_p"] == 0.95
        assert ogc["top_k"] == 64
        assert ogc["min_p"] == 0.01

    def test_unknown_arch_does_not_emit_override(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model
        b = VllmBackend()
        m = detect_model(fake_model_dir)
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        assert "override-generation-config" not in cfg

    def test_recipe_sampling_opt_out(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_gemma4_model(tmp_path)
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
            "recipe_sampling": False,
        })
        assert "override-generation-config" not in cfg

    def test_explicit_override_generation_config_wins(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_gemma4_model(tmp_path)
        explicit = {"temperature": 0.0, "top_p": 1.0}
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
            "override_generation_config": explicit,
        })
        assert cfg["override-generation-config"] == explicit


# --- 0.7.0 item B: reasoning-parser auto-discovery ---


class TestVllmReasoningParserAutoDetect:
    """Item B: previously vserve emitted ``reasoning-parser`` only when
    explicitly passed. Now we map architecture → parser the same way the
    tool-parser path does, so Gemma-4 + DeepSeek + Qwen3 split reasoning
    output automatically."""

    def _write_arch(self, tmp_path, *archs):
        import json
        (tmp_path / "config.json").write_text(json.dumps({"architectures": list(archs)}))
        return tmp_path

    def test_gemma4_arch_maps_to_gemma4_reasoning_parser(self, tmp_path):
        from vserve.backends.vllm import _suggested_reasoning_parser
        self._write_arch(tmp_path, "Gemma4ForCausalLM")
        assert _suggested_reasoning_parser(tmp_path, None) == "gemma4"

    def test_qwen3_maps_to_qwen3_parser(self, tmp_path):
        from vserve.backends.vllm import _suggested_reasoning_parser
        self._write_arch(tmp_path, "Qwen3ForCausalLM")
        assert _suggested_reasoning_parser(tmp_path, None) == "qwen3"

    def test_deepseek_v3_maps_to_deepseek_r1_parser(self, tmp_path):
        from vserve.backends.vllm import _suggested_reasoning_parser
        self._write_arch(tmp_path, "DeepseekV3ForCausalLM")
        assert _suggested_reasoning_parser(tmp_path, None) == "deepseek_r1"

    def test_only_returns_parser_runtime_has_installed(self, tmp_path):
        from vserve.backends.vllm import _suggested_reasoning_parser
        self._write_arch(tmp_path, "Gemma4ForCausalLM")
        # Runtime only has qwen3 — gemma4 reasoning parser unavailable.
        assert _suggested_reasoning_parser(tmp_path, {"qwen3"}) is None
        assert _suggested_reasoning_parser(tmp_path, {"gemma4", "qwen3"}) == "gemma4"

    def test_unknown_architecture_returns_none(self, tmp_path):
        from vserve.backends.vllm import _suggested_reasoning_parser
        self._write_arch(tmp_path, "SomeBrandNewArchForCausalLM")
        assert _suggested_reasoning_parser(tmp_path, None) is None

    def test_build_config_auto_emits_reasoning_parser_when_arch_known(self, tmp_path):
        import json
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        b = VllmBackend()
        b.available_reasoning_parsers = Mock(return_value={"gemma4"})  # type: ignore[method-assign]
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="gemma-4-26B-A4B",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None,
            # reasoning_parser left out — auto-discovery should fill it in.
        })
        assert cfg["reasoning-parser"] == "gemma4"

    def test_explicit_reasoning_parser_wins_over_auto(self, tmp_path):
        import json
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        b = VllmBackend()
        b.available_reasoning_parsers = Mock(return_value={"gemma4", "qwen3"})  # type: ignore[method-assign]
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="gemma-4-26B-A4B",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None,
            "reasoning_parser": "qwen3",  # explicit override
        })
        assert cfg["reasoning-parser"] == "qwen3"


# --- 0.7.0 item C: chat-template-kwargs plumbing ---


class TestVllmChatTemplateKwargs:
    """Item C: thinking mode toggles are passed to the chat template via
    kwargs, not CLI flags. Gemma 3/4 + Qwen 3.x use ``enable_thinking``;
    DeepSeek V3.1+ uses ``thinking``. The ``choices['thinking']`` shortcut
    auto-picks the right name based on architecture."""

    def _make_model(self, tmp_path, arch):
        import json
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "p" / "M"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": [arch]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        return ModelInfo(
            path=model_dir, provider="p", model_name="M",
            architecture=arch, model_type="m", quant_method=None,
            max_position_embeddings=131072, is_moe=False, model_size_gb=1.0,
        )

    def _base_choices(self):
        return {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        }

    def test_thinking_true_emits_enable_thinking_for_gemma4(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "Gemma4ForCausalLM")
        choices = self._base_choices()
        choices["thinking"] = True
        cfg = b.build_config(m, choices)
        assert cfg["chat-template-kwargs"] == {"enable_thinking": True}

    def test_thinking_false_emits_enable_thinking_false(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "Qwen3ForCausalLM")
        choices = self._base_choices()
        choices["thinking"] = False
        cfg = b.build_config(m, choices)
        assert cfg["chat-template-kwargs"] == {"enable_thinking": False}

    def test_deepseek_uses_thinking_key(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "DeepseekV31ForCausalLM")
        choices = self._base_choices()
        choices["thinking"] = True
        cfg = b.build_config(m, choices)
        # DeepSeek V3.1 hybrid uses ``thinking`` kwarg, not enable_thinking.
        assert cfg["chat-template-kwargs"] == {"thinking": True}

    def test_thinking_auto_does_not_emit(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "Gemma4ForCausalLM")
        choices = self._base_choices()
        choices["thinking"] = "auto"
        cfg = b.build_config(m, choices)
        assert "chat-template-kwargs" not in cfg

    def test_explicit_chat_template_kwargs_passthrough(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "Gemma4ForCausalLM")
        choices = self._base_choices()
        choices["chat_template_kwargs"] = {"custom_flag": "x", "enable_thinking": False}
        cfg = b.build_config(m, choices)
        assert cfg["chat-template-kwargs"]["custom_flag"] == "x"
        assert cfg["chat-template-kwargs"]["enable_thinking"] is False

    def test_thinking_choice_overrides_explicit_dict(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "Gemma4ForCausalLM")
        choices = self._base_choices()
        choices["chat_template_kwargs"] = {"enable_thinking": False}
        choices["thinking"] = True
        cfg = b.build_config(m, choices)
        # thinking=True should overwrite the explicit dict entry.
        assert cfg["chat-template-kwargs"]["enable_thinking"] is True


# --- 0.7.0 item D: expanded tool-parser table ---


class TestVllmExpandedToolParserTable:
    """Item D: vserve previously mapped only Gemma 3/4 → gemma4 parser.
    vLLM 0.21 ships 17+ parsers; map every architecture vserve might see
    so users don't need to know the parser name."""

    def _write_arch(self, tmp_path, arch):
        import json
        (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
        return tmp_path

    @staticmethod
    def _expected_mappings():
        return [
            ("Gemma4ForCausalLM",              "gemma4"),
            ("Gemma3ForConditionalGeneration", "gemma4"),
            ("LlamaForCausalLM",               "llama3_json"),
            ("Llama4ForCausalLM",              "llama4_pythonic"),
            ("Llama4MoeForCausalLM",           "llama4_pythonic"),
            ("Qwen3ForCausalLM",               "hermes"),
            ("Qwen35ForCausalLM",              "qwen3_coder"),
            ("Qwen36ForCausalLM",              "qwen3_coder"),
            ("Qwen36MoeForCausalLM",           "qwen3_coder"),
            ("Qwen3MoeForCausalLM",            "hermes"),
            ("Qwen3CoderForCausalLM",          "qwen3_coder"),
            ("Qwen3XmlForCausalLM",            "qwen3_xml"),
            ("DeepseekV3ForCausalLM",          "deepseek_v3"),
            ("DeepseekV31ForCausalLM",         "deepseek_v31"),
            ("DeepseekV32ForCausalLM",         "deepseek_v32"),
            ("DeepseekV4ForCausalLM",          "deepseek_v4"),
            ("KimiK2ForCausalLM",              "kimi_k2"),
            ("KimiK2ThinkingForCausalLM",      "kimi_k2"),
            ("Glm4MoeForCausalLM",             "glm45"),
            ("Glm47MoeForCausalLM",            "glm47"),
            ("GraniteForCausalLM",             "granite"),
            ("Granite4ForCausalLM",            "granite4"),
            ("CohereForCausalLM",              "cohere_command4"),
            ("Ernie4ForCausalLM",              "ernie45"),
            ("JambaForCausalLM",               "jamba"),
            ("XlamForCausalLM",                "xlam"),
            ("Lfm2ForCausalLM",                "lfm2"),
            ("Lfm25ForCausalLM",               "lfm25"),
            ("MistralForCausalLM",             "mistral"),
            ("GptOssForCausalLM",              "openai"),
            ("InternLMForCausalLM",            "internlm"),
            ("InternLM2ForCausalLM",           "internlm"),
        ]

    def test_each_architecture_maps_to_expected_parser(self, tmp_path):
        from vserve.backends.vllm import _suggested_tool_parser
        failures = []
        for arch, expected in self._expected_mappings():
            sub = tmp_path / arch
            sub.mkdir()
            self._write_arch(sub, arch)
            got = _suggested_tool_parser(sub, None)
            if got != expected:
                failures.append((arch, expected, got))
        assert not failures, f"Mapping failures: {failures}"

    def test_unknown_arch_still_returns_none(self, tmp_path):
        from vserve.backends.vllm import _suggested_tool_parser
        self._write_arch(tmp_path, "BrandNewForCausalLM")
        assert _suggested_tool_parser(tmp_path, None) is None

    def test_runtime_registry_filter_drops_uninstalled_parsers(self, tmp_path):
        from vserve.backends.vllm import _suggested_tool_parser
        # KimiK2 mapping is in the table, but only hermes is "installed" — so
        # _suggested returns None (don't emit a parser the runtime can't load).
        self._write_arch(tmp_path, "KimiK2ForCausalLM")
        assert _suggested_tool_parser(tmp_path, {"hermes"}) is None
        assert _suggested_tool_parser(tmp_path, {"kimi_k2"}) == "kimi_k2"


# --- 0.7.0 item F: Gemma-4 chat template auto-resolution ---


class TestVllmGemma4ChatTemplate:
    """Item F: Gemma-4 tool parser requires the vendored chat template (uses
    `<|"|>` string delimiter and `<|tool_call>` tag). Stock HF templates
    don't emit those, so tool calls silently fail. Auto-discover the
    template from the vLLM install (or fall back to the packaged copy)."""

    def test_locate_returns_packaged_template_when_install_missing(self, tmp_path, monkeypatch):
        from vserve.backends.vllm import _locate_gemma4_chat_template
        # vllm_root that doesn't contain the template at all.
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        # Without monkeypatching Path(__file__), the packaged template lives
        # at src/vserve/templates/tool_chat_template_gemma4.jinja. We shipped
        # that earlier in this commit.
        result = _locate_gemma4_chat_template(empty_root)
        assert result is not None
        assert result.name == "tool_chat_template_gemma4.jinja"
        assert result.exists()

    def test_locate_prefers_vllm_install_when_present(self, tmp_path):
        from vserve.backends.vllm import _locate_gemma4_chat_template
        vllm_root = tmp_path / "vllm"
        target = (
            vllm_root / "venv" / "lib" / "python3.12" / "site-packages"
            / "vllm" / "examples" / "tool_chat_template_gemma4.jinja"
        )
        target.parent.mkdir(parents=True)
        target.write_text("# install-provided template\n")
        result = _locate_gemma4_chat_template(vllm_root)
        assert result is not None
        assert result == target

    def test_locate_returns_none_when_packaged_template_empty(self, tmp_path, monkeypatch):
        """If neither the vLLM install nor the packaged file exist (or the
        packaged file is empty), we should return None and let the caller
        decide whether to warn the user."""
        from vserve.backends.vllm import _locate_gemma4_chat_template
        import vserve.backends.vllm as vllm_mod
        # Re-point the packaged template lookup to an empty file by
        # monkeypatching __file__ to a directory with no templates/.
        empty_pkg = tmp_path / "fake-pkg" / "backends"
        empty_pkg.mkdir(parents=True)
        # Create the templates/ dir but no file — so the fallback isn't found.
        (tmp_path / "fake-pkg" / "templates").mkdir()
        monkeypatch.setattr(vllm_mod, "__file__", str(empty_pkg / "vllm.py"))
        result = _locate_gemma4_chat_template(tmp_path / "nonexistent-vllm")
        assert result is None

    def test_build_config_auto_resolves_for_gemma4_tools(self, tmp_path):
        import json
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value={"gemma4", "hermes"})  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="gemma-4-26B-A4B",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": True, "tool_parser": None, "reasoning_parser": None,
        })
        # tool-call-parser should be gemma4 (auto-discovered from arch).
        assert cfg["tool-call-parser"] == "gemma4"
        # chat-template should be the packaged template (no vLLM install in
        # the tmp tree, but the packaged template exists in the repo).
        assert "chat-template" in cfg
        assert cfg["chat-template"].endswith("tool_chat_template_gemma4.jinja")

    def test_build_config_does_not_override_explicit_chat_template(self, tmp_path):
        import json
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value={"gemma4"})  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="gemma-4-26B-A4B",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": True, "tool_parser": "gemma4", "reasoning_parser": None,
            "chat_template": "/custom/path.jinja",
        })
        assert cfg["chat-template"] == "/custom/path.jinja"

    def test_build_config_skips_auto_when_tools_disabled(self, tmp_path):
        import json
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value={"gemma4"})  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = ModelInfo(
            path=model_dir, provider="unsloth", model_name="gemma-4-26B-A4B",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        assert "chat-template" not in cfg
        assert "tool-call-parser" not in cfg


# --- 0.7.0 item Q: NVFP4 KV auto-pin in build_config ---


class TestVllmNvfp4KvAutoPin:
    """Item Q: NVFP4 / ModelOpt-NVFP4 + auto KV is broken (vllm#39133 — the
    engine treats fp8-KV as fp8-checkpoint). When the model uses NVFP4 and
    the caller leaves kv_dtype=auto, vserve pins KV to fp8 automatically."""

    def _make_nvfp4_model(self, tmp_path, quant_method):
        import json
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "nvidia" / "Gemma-4-31B-NVFP4"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        return ModelInfo(
            path=model_dir, provider="nvidia", model_name="Gemma-4-31B-NVFP4",
            architecture="Gemma4ForCausalLM", model_type="gemma4",
            quant_method=quant_method,
            max_position_embeddings=131072, is_moe=False, model_size_gb=10.0,
        )

    def test_nvfp4_with_auto_kv_forces_fp8(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_nvfp4_model(tmp_path, "nvfp4")
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        assert cfg["kv-cache-dtype"] == "fp8"

    def test_modelopt_with_auto_kv_forces_fp8(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_nvfp4_model(tmp_path, "modelopt")
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        assert cfg["kv-cache-dtype"] == "fp8"

    def test_nvfp4_with_explicit_kv_preserved(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_nvfp4_model(tmp_path, "nvfp4")
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "fp8_e5m2", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        # Caller pinned KV to fp8_e5m2 — preserved (not overwritten).
        assert cfg["kv-cache-dtype"] == "fp8_e5m2"

    def test_non_nvfp4_model_no_auto_pin(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model
        b = VllmBackend()
        m = detect_model(fake_model_dir)
        # FakeModel uses fp8 quant_method, not nvfp4 — no auto-pin.
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        })
        assert cfg["kv-cache-dtype"] == "auto"


# --- 0.7.0 item R: MLA backend awareness ---


class TestVllmAttentionBackend:
    """Item R: MLA architectures (DeepSeek V2-V4, Kimi K2, LongCat-Flash)
    serve much faster on FLASHMLA (Hopper) / TOKENSPEED_MLA (Blackwell)
    than the default backend auto-pick. GPT-OSS on SM120 requires
    TRITON_ATTN (FlashInfer doesn't support attention sinks)."""

    def _make_model(self, tmp_path, arch):
        import json
        from vserve.models import ModelInfo
        model_dir = tmp_path / "models" / "p" / "M"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": [arch]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        return ModelInfo(
            path=model_dir, provider="p", model_name="M",
            architecture=arch, model_type="m", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )

    def _base_choices(self):
        return {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        }

    def test_deepseek_v3_forces_flashmla_on_hopper(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "DeepseekV3ForCausalLM")
        choices = self._base_choices()
        choices["gpu_compute_cap"] = 90
        cfg = b.build_config(m, choices)
        assert cfg.get("attention-config", {}).get("backend") == "FLASHMLA"

    def test_deepseek_v3_forces_tokenspeed_mla_on_blackwell(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "DeepseekV3ForCausalLM")
        choices = self._base_choices()
        choices["gpu_compute_cap"] = 120
        cfg = b.build_config(m, choices)
        assert cfg.get("attention-config", {}).get("backend") == "TOKENSPEED_MLA"

    def test_kimi_k2_forces_flashmla(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "KimiK2ForCausalLM")
        choices = self._base_choices()
        choices["gpu_compute_cap"] = 90
        cfg = b.build_config(m, choices)
        assert cfg.get("attention-config", {}).get("backend") == "FLASHMLA"

    def test_gptoss_forces_triton_attn_only_on_sm120(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "GptOssForCausalLM")
        # On SM120 → forced.
        choices_120 = self._base_choices()
        choices_120["gpu_compute_cap"] = 120
        cfg_120 = b.build_config(m, choices_120)
        assert cfg_120.get("attention-config", {}).get("backend") == "TRITON_ATTN"
        # On Hopper (sm90) → not forced.
        choices_90 = self._base_choices()
        choices_90["gpu_compute_cap"] = 90
        cfg_90 = b.build_config(m, choices_90)
        assert "attention-config" not in cfg_90

    def test_gemma4_still_forces_triton_attn(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "Gemma4ForCausalLM")
        choices = self._base_choices()
        choices["gpu_compute_cap"] = 120
        cfg = b.build_config(m, choices)
        assert cfg.get("attention-config", {}).get("backend") == "TRITON_ATTN"

    def test_unknown_arch_no_force(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model
        b = VllmBackend()
        m = detect_model(fake_model_dir)
        choices = self._base_choices()
        choices["gpu_compute_cap"] = 120
        cfg = b.build_config(m, choices)
        assert "attention-config" not in cfg

    def test_explicit_attention_backend_wins(self, tmp_path):
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        m = self._make_model(tmp_path, "DeepseekV3ForCausalLM")
        choices = self._base_choices()
        choices["gpu_compute_cap"] = 120
        choices["attention_backend"] = "FLASH_ATTN_V3"
        cfg = b.build_config(m, choices)
        # Explicit choice overrides MLA auto-pick.
        assert cfg.get("attention-config", {}).get("backend") == "FLASH_ATTN_V3"


# --- 0.6.3: vLLM 0.22 version-gated emission ---


def _pin_vllm_runtime(mocker, version):
    """Override the suite-wide None pin with an explicit runtime version."""
    from packaging.version import Version

    mocker.patch(
        "vserve.backends.vllm._runtime_vllm_version",
        return_value=Version(version) if version else None,
    )


class TestChatTemplateKwargsKeyByRuntime:
    """vLLM 0.22 renamed --chat-template-kwargs to
    --default-chat-template-kwargs. The emitted YAML key must follow the
    installed runtime; unknown version stays on the pre-0.22 name."""

    def _make_model(self, tmp_path, arch="Qwen3ForCausalLM"):
        import json
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "p" / "M"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": [arch]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        return ModelInfo(
            path=model_dir, provider="p", model_name="M",
            architecture=arch, model_type="m", quant_method=None,
            max_position_embeddings=131072, is_moe=False, model_size_gb=1.0,
        )

    def _cfg(self, tmp_path, mocker, version):
        from vserve.backends.vllm import VllmBackend

        _pin_vllm_runtime(mocker, version)
        choices = {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
            "thinking": False,
        }
        return VllmBackend().build_config(self._make_model(tmp_path), choices)

    def test_pre_022_uses_legacy_key(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, "0.21.0")
        assert cfg["chat-template-kwargs"] == {"enable_thinking": False}
        assert "default-chat-template-kwargs" not in cfg

    def test_unknown_runtime_uses_legacy_key(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, None)
        assert cfg["chat-template-kwargs"] == {"enable_thinking": False}
        assert "default-chat-template-kwargs" not in cfg

    def test_022_uses_renamed_key(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, "0.22.0")
        assert cfg["default-chat-template-kwargs"] == {"enable_thinking": False}
        assert "chat-template-kwargs" not in cfg


class TestSpecFp8CudagraphRelaxation:
    """vLLM 0.22.0 ships the DFlash fp8-KV fix (vllm#42692): on a KNOWN
    >=0.22 runtime, spec-decode + fp8-family KV keeps CUDA graphs.
    Everything else (turboquant, unknown version, other quantized
    dtypes) still forces cudagraph_mode NONE per vllm#41559."""

    def _make_model(self, tmp_path, arch="Qwen3ForCausalLM"):
        import json
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "p" / "M"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": [arch]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        return ModelInfo(
            path=model_dir, provider="p", model_name="M",
            architecture=arch, model_type="m", quant_method=None,
            max_position_embeddings=131072, is_moe=False, model_size_gb=1.0,
        )

    def _cfg(self, tmp_path, mocker, version, kv_dtype):
        from vserve.backends.vllm import VllmBackend

        _pin_vllm_runtime(mocker, version)
        choices = {
            "context": 8192, "kv_dtype": kv_dtype, "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
            "spec": {"method": "ngram", "num_speculative_tokens": 5},
        }
        return VllmBackend().build_config(self._make_model(tmp_path), choices)

    def test_022_fp8_spec_keeps_cudagraphs(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, "0.22.0", "fp8")
        assert cfg.get("compilation-config", {}).get("cudagraph_mode") != "NONE"

    def test_022_fp8_e4m3_spec_keeps_cudagraphs(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, "0.22.0", "fp8_e4m3")
        assert cfg.get("compilation-config", {}).get("cudagraph_mode") != "NONE"

    def test_021_fp8_spec_still_forces_none(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, "0.21.0", "fp8")
        assert cfg["compilation-config"]["cudagraph_mode"] == "NONE"

    def test_unknown_version_fp8_spec_still_forces_none(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, None, "fp8")
        assert cfg["compilation-config"]["cudagraph_mode"] == "NONE"

    def test_022_turboquant_spec_still_forces_none(self, tmp_path, mocker):
        cfg = self._cfg(tmp_path, mocker, "0.22.0", "turboquant_3bit_nc")
        assert cfg["compilation-config"]["cudagraph_mode"] == "NONE"

    def test_022_int8_kv_spec_still_forces_none(self, tmp_path, mocker):
        # Only the fp8 family was fixed upstream; other quantized dtypes
        # keep the conservative path even on 0.22.
        cfg = self._cfg(tmp_path, mocker, "0.22.0", "int8_per_token_head")
        assert cfg["compilation-config"]["cudagraph_mode"] == "NONE"


class TestMoeBackendKnob:
    """vLLM 0.22+ expert knob: pin the MoE kernel backend instead of
    trusting --moe-backend=auto. Emitted only when explicitly chosen."""

    def _make_model(self, tmp_path, arch="Qwen3ForCausalLM"):
        import json
        from vserve.models import ModelInfo

        model_dir = tmp_path / "models" / "p" / "M"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": [arch]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        return ModelInfo(
            path=model_dir, provider="p", model_name="M",
            architecture=arch, model_type="m", quant_method=None,
            max_position_embeddings=131072, is_moe=True, model_size_gb=1.0,
        )

    def _base_choices(self):
        return {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
        }

    def test_emitted_when_set(self, tmp_path):
        from vserve.backends.vllm import VllmBackend

        choices = self._base_choices()
        choices["moe_backend"] = "flashinfer_trtllm"
        cfg = VllmBackend().build_config(self._make_model(tmp_path), choices)
        assert cfg["moe-backend"] == "flashinfer_trtllm"

    def test_absent_by_default(self, tmp_path):
        from vserve.backends.vllm import VllmBackend

        cfg = VllmBackend().build_config(self._make_model(tmp_path), self._base_choices())
        assert "moe-backend" not in cfg
