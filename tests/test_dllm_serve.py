"""Block-diffusion (dLLM) serve path + crash-loop backstop.

Covers the DiffusionGemma serve recipe vserve emits (V2 model runner env,
trust-remote-code, diffusion sampler, TRITON_ATTN, runai_streamer, capped
concurrency/util, no AR-only knobs) and the launch-flow guard that stops a
crash-looping unit instead of leaving it to spin.
"""
import json


def _make_dllm_model(tmp_path, *, quant="nvfp4"):
    from vserve.models import ModelInfo

    d = tmp_path / "models" / "RedHatAI" / "diffusiongemma-26B-A4B-it-NVFP4"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": "diffusion_gemma",
        "text_config": {"model_type": "diffusion_gemma_text"},
    }))
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    m = ModelInfo(
        path=d, provider="RedHatAI", model_name="diffusiongemma-26B-A4B-it-NVFP4",
        architecture="DiffusionGemmaForBlockDiffusion", model_type="diffusion_gemma",
        quant_method=quant, max_position_embeddings=262144, is_moe=True, model_size_gb=14.0,
    )
    return m


def _dllm_choices(**over):
    base = {
        "context": 16384, "kv_dtype": "auto", "slots": 32,
        "batched_tokens": 8192, "gpu_mem_util": 0.85, "port": 8888,
        "tools": False, "tool_parser": None, "reasoning_parser": None,
    }
    base.update(over)
    return base


def _backend():
    from unittest.mock import Mock
    from vserve.backends.vllm import VllmBackend
    b = VllmBackend()
    b.available_tool_parsers = Mock(return_value={"gemma4"})        # type: ignore[method-assign]
    b.available_reasoning_parsers = Mock(return_value={"gemma4"})   # type: ignore[method-assign]
    return b


class TestBlockDiffusionDetection:
    def test_detects_by_model_type(self, tmp_path):
        from vserve.backends.vllm import _is_block_diffusion
        assert _is_block_diffusion(_make_dllm_model(tmp_path).path) is True

    def test_detects_by_architecture(self, tmp_path):
        from vserve.backends.vllm import _is_block_diffusion
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["FooDiffusionForBlockDiffusion"]}))
        assert _is_block_diffusion(d) is True

    def test_negative_for_causal_lm(self, tmp_path):
        from vserve.backends.vllm import _is_block_diffusion
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}))
        assert _is_block_diffusion(d) is False


class TestDllmBuildConfig:
    def test_emits_v2_runner_recipe(self, tmp_path):
        from vserve.backends.vllm import _DLLM_MAX_NUM_SEQS, _DLLM_GPU_MEM_UTIL
        b = _backend()
        cfg = b.build_config(_make_dllm_model(tmp_path), _dllm_choices(gpu_compute_cap=120))
        assert cfg["trust-remote-code"] is True
        assert cfg["load-format"] == "runai_streamer"
        assert cfg["model-loader-extra-config"]["memory_limit"] > 0
        assert cfg["hf-overrides"]["diffusion_sampler"] == "entropy_bound"
        assert cfg["hf-overrides"]["diffusion_entropy_bound"] == 0.1
        assert cfg["attention-config"]["backend"] == "TRITON_ATTN"
        # concurrency + utilization capped for the diffusion-state VRAM
        assert cfg["max-num-seqs"] == _DLLM_MAX_NUM_SEQS      # slots=32 capped
        assert cfg["gpu-memory-utilization"] == _DLLM_GPU_MEM_UTIL  # 0.85 capped down

    def test_skips_ar_only_knobs(self, tmp_path):
        # NVFP4 would normally force fp8 KV; ngram spec would normally be emitted.
        from vserve.recipes.spec_decode import SpecConfig
        b = _backend()
        cfg = b.build_config(
            _make_dllm_model(tmp_path),
            _dllm_choices(gpu_compute_cap=120, spec=SpecConfig(method="ngram", n_max=5, n_min=1)),
        )
        assert cfg["kv-cache-dtype"] == "auto"          # NOT forced to fp8
        assert "speculative-config" not in cfg          # ngram suppressed for dLLM

    def test_user_lower_util_is_respected(self, tmp_path):
        from vserve.backends.vllm import _DLLM_GPU_MEM_UTIL
        b = _backend()
        cfg = b.build_config(_make_dllm_model(tmp_path), _dllm_choices(gpu_compute_cap=120, gpu_mem_util=0.5))
        assert cfg["gpu-memory-utilization"] == 0.5      # below the cap → kept
        assert _DLLM_GPU_MEM_UTIL == 0.70


class TestModelRuntimeEnvs:
    def test_v2_runner_env_for_dllm(self, tmp_path):
        from vserve.serve import _resolve_model_runtime_envs
        cfg_path = tmp_path / "active.yaml"
        cfg_path.write_text(f"model: {_make_dllm_model(tmp_path).path}\n")
        assert _resolve_model_runtime_envs(cfg_path) == {"VLLM_USE_V2_MODEL_RUNNER": "1"}

    def test_no_env_for_causal_lm(self, tmp_path):
        from vserve.serve import _resolve_model_runtime_envs
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        cfg_path = tmp_path / "active.yaml"
        cfg_path.write_text(f"model: {d}\n")
        assert _resolve_model_runtime_envs(cfg_path) == {}


class TestCrashLoopGuard:
    def _wait(self, *, restart_seq, service_running=True, timeout_s=30):
        from vserve import cli
        # urlopen always fails so health never goes 200
        def urlopen(url, timeout=2):
            raise OSError("connection refused")
        seq = list(restart_seq)
        def restart_count():
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return cli._wait_for_health(
            health_url="http://x/health", timeout_s=timeout_s, poll_s=1,
            log_tail_fn=lambda: "", service_running_fn=lambda: service_running,
            sleep_fn=lambda s: None, urlopen_fn=urlopen,
            restart_count_fn=restart_count, baseline_restarts=0,
        )

    def test_climbing_restart_count_is_crashloop(self):
        # NRestarts goes 0 -> 1: engine died and was auto-restarted -> crash-loop.
        assert self._wait(restart_seq=[1]) == "crashloop"

    def test_stable_restart_count_is_not_crashloop(self):
        # A genuinely-warming engine keeps its PID; NRestarts stays at baseline.
        assert self._wait(restart_seq=[0], timeout_s=5) == "timeout"


class TestPerModelServiceResolution:
    """vserve drives a separate dLLM service so dLLMs run on a newer runtime
    while the pinned-stable service stays untouched."""

    def _cfg(self, mocker, tmp_path, *, dllm_service_name, model_path):
        from unittest.mock import Mock
        active = tmp_path / "active.yaml"
        if model_path is not None:
            active.write_text(f"model: {model_path}\n")
        c = Mock()
        c.service_name = "vllm"
        c.dllm_service_name = dllm_service_name
        c.active_yaml = active
        mocker.patch("vserve.serve.cfg", return_value=c)
        return c

    def test_dllm_model_resolves_to_nightly_service(self, mocker, tmp_path):
        from vserve.serve import _resolve_vllm_service
        m = _make_dllm_model(tmp_path)
        self._cfg(mocker, tmp_path, dllm_service_name="vllm-nightly", model_path=m.path)
        assert _resolve_vllm_service() == "vllm-nightly"

    def test_ar_model_resolves_to_stable_service(self, mocker, tmp_path):
        from vserve.serve import _resolve_vllm_service
        ar = tmp_path / "ar"
        ar.mkdir()
        (ar / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        self._cfg(mocker, tmp_path, dllm_service_name="vllm-nightly", model_path=ar)
        assert _resolve_vllm_service() == "vllm"

    def test_no_dllm_service_configured_stays_stable(self, mocker, tmp_path):
        from vserve.serve import _resolve_vllm_service
        m = _make_dllm_model(tmp_path)
        self._cfg(mocker, tmp_path, dllm_service_name=None, model_path=m.path)
        assert _resolve_vllm_service() == "vllm"

    def test_configured_services_lists_both(self, mocker, tmp_path):
        from vserve.serve import _configured_vllm_services
        self._cfg(mocker, tmp_path, dllm_service_name="vllm-nightly", model_path=None)
        assert _configured_vllm_services() == ["vllm", "vllm-nightly"]

    def test_configured_services_stable_only(self, mocker, tmp_path):
        from vserve.serve import _configured_vllm_services
        self._cfg(mocker, tmp_path, dllm_service_name=None, model_path=None)
        assert _configured_vllm_services() == ["vllm"]
