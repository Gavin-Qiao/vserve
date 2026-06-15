import json
from pathlib import Path

from vserve.runtime import (
    SUPPORTED_VLLM_RANGE,
    DETECTOR_SCHEMA_VERSION,
    RuntimeInfo,
    build_tuning_fingerprint,
    check_vllm_compatibility,
    collect_vllm_runtime_info,
    upgrade_vllm_stable,
)


def test_check_vllm_compatibility_accepts_stable_020():
    info = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.20.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    result = check_vllm_compatibility(info)

    assert result.supported is True
    assert result.range == SUPPORTED_VLLM_RANGE
    assert result.errors == []
    assert any("0.20.0" in message for message in result.messages)


def test_check_vllm_compatibility_rejects_beta_and_wrong_minor():
    beta = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.20.0rc1",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )
    older = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.19.2",
        torch_version="2.10.0",
        torch_cuda="12.8",
        transformers_version="4.56.0",
        huggingface_hub_version="0.36.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    assert check_vllm_compatibility(beta).supported is False
    assert "pre-release" in " ".join(check_vllm_compatibility(beta).errors)
    assert check_vllm_compatibility(older).supported is False
    assert ">=0.20,<0.23" in " ".join(check_vllm_compatibility(older).errors)


def test_check_vllm_compatibility_accepts_stable_021():
    info = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.21.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    result = check_vllm_compatibility(info)

    assert result.supported is True
    assert result.range == SUPPORTED_VLLM_RANGE
    assert result.errors == []
    assert any("0.21.0" in message for message in result.messages)


def test_check_vllm_compatibility_rejects_023():
    too_new = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.23.0",
        torch_version="2.12.0",
        torch_cuda="13.1",
        transformers_version="5.7.0",
        huggingface_hub_version="1.13.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    result = check_vllm_compatibility(too_new)

    assert result.supported is False
    assert ">=0.20,<0.23" in " ".join(result.errors)


def test_collect_vllm_runtime_info_uses_vllm_python(mocker, tmp_path):
    vllm_bin = tmp_path / "venv" / "bin" / "vllm"
    vllm_python = tmp_path / "venv" / "bin" / "python"
    vllm_bin.parent.mkdir(parents=True)
    vllm_bin.touch()
    vllm_python.touch()
    cfg = mocker.Mock(vllm_bin=vllm_bin, vllm_python=vllm_python)

    def fake_run(cmd, *args, **kwargs):
        if cmd == [str(vllm_bin), "--version"]:
            return mocker.Mock(returncode=0, stdout="vLLM 0.20.0\n", stderr="")
        if cmd[:3] == [str(vllm_python), "-c", mocker.ANY]:
            return mocker.Mock(
                returncode=0,
                stdout=(
                    '{"vllm":"0.20.0","torch":"2.11.0+cu130","torch_cuda":"13.0",'
                    '"transformers":"5.6.2","huggingface_hub":"1.12.0"}\n'
                ),
                stderr="",
            )
        if cmd[:3] == [str(vllm_python), "-m", "pip"]:
            return mocker.Mock(returncode=0, stdout="No broken requirements found.\n", stderr="")
        raise AssertionError(cmd)

    mocker.patch("vserve.runtime.subprocess.run", side_effect=fake_run)

    info = collect_vllm_runtime_info(cfg)

    assert info.vllm_version == "0.20.0"
    assert info.torch_version == "2.11.0+cu130"
    assert info.transformers_version == "5.6.2"
    assert info.pip_check_ok is True


def test_upgrade_vllm_stable_force_reinstalls_pinned_stable(mocker, tmp_path):
    vllm_python = tmp_path / "venv" / "bin" / "python"
    vllm_python.parent.mkdir(parents=True)
    vllm_python.touch()
    cfg = mocker.Mock(vllm_python=vllm_python)
    run = mocker.patch("vserve.runtime.subprocess.run", return_value=mocker.Mock(returncode=0, stdout="", stderr=""))
    invalidate = mocker.patch("vserve.runtime.invalidate_vllm_runtime_cache")

    upgrade_vllm_stable(cfg)

    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert cmd[:4] == [str(vllm_python), "-m", "pip", "install"]
    assert "--force-reinstall" in cmd
    assert "vllm==0.21.0" in cmd
    invalidate.assert_called_once()


def _build_venv(tmp_path):
    """Make a fake vLLM venv layout that satisfies the cache key probe."""
    vllm_bin = tmp_path / "venv" / "bin" / "vllm"
    vllm_python = tmp_path / "venv" / "bin" / "python"
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    vllm_bin.parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    vllm_bin.touch()
    vllm_python.touch()
    return vllm_bin, vllm_python


def test_collect_vllm_runtime_info_returns_cached_when_prefer_cache_and_key_matches(mocker, tmp_path):
    from vserve.runtime import (
        _vllm_runtime_cache_key,
        _write_vllm_runtime_cache,
        RuntimeInfo,
    )

    vllm_bin, vllm_python = _build_venv(tmp_path)
    cfg = mocker.Mock(vllm_bin=vllm_bin, vllm_python=vllm_python)
    cache_path = tmp_path / "cache.json"

    key = _vllm_runtime_cache_key(vllm_bin, vllm_python)
    assert key is not None
    cached = RuntimeInfo(
        backend="vllm",
        executable=vllm_bin,
        python=vllm_python,
        vllm_version="0.21.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=None,
        pip_check_output="",
    )
    _write_vllm_runtime_cache(cache_path, cache_key=key, info=cached)
    run = mocker.patch("vserve.runtime.subprocess.run")

    info = collect_vllm_runtime_info(
        cfg, prefer_cache=True, with_pip_check=False, cache_path=cache_path
    )

    assert info.vllm_version == "0.21.0"
    assert info.torch_version == "2.11.0+cu130"
    run.assert_not_called()


def test_collect_vllm_runtime_info_repopulates_cache_on_key_drift(mocker, tmp_path):
    from vserve.runtime import _write_vllm_runtime_cache, RuntimeInfo

    vllm_bin, vllm_python = _build_venv(tmp_path)
    cfg = mocker.Mock(vllm_bin=vllm_bin, vllm_python=vllm_python)
    cache_path = tmp_path / "cache.json"

    stale = RuntimeInfo(
        backend="vllm",
        executable=vllm_bin,
        python=vllm_python,
        vllm_version="0.19.0",
        torch_version="2.10.0",
        torch_cuda="12.8",
        transformers_version="4.56.0",
        huggingface_hub_version="0.36.0",
        pip_check_ok=None,
        pip_check_output="",
    )
    _write_vllm_runtime_cache(cache_path, cache_key="stale-key", info=stale)

    def fake_run(cmd, *args, **kwargs):
        if cmd == [str(vllm_bin), "--version"]:
            return mocker.Mock(returncode=0, stdout="vLLM 0.21.0\n", stderr="")
        if cmd[:3] == [str(vllm_python), "-c", mocker.ANY]:
            return mocker.Mock(
                returncode=0,
                stdout='{"vllm":"0.21.0","torch":"2.11.0+cu130","torch_cuda":"13.0","transformers":"5.6.2","huggingface_hub":"1.12.0"}\n',
                stderr="",
            )
        raise AssertionError(cmd)

    mocker.patch("vserve.runtime.subprocess.run", side_effect=fake_run)

    info = collect_vllm_runtime_info(
        cfg, prefer_cache=True, with_pip_check=False, cache_path=cache_path
    )

    assert info.vllm_version == "0.21.0"
    # Cache rewritten with fresh data.
    rewritten = json.loads(cache_path.read_text())
    assert rewritten["vllm_version"] == "0.21.0"
    assert rewritten["cache_key"] != "stale-key"


def test_collect_vllm_runtime_info_does_not_run_pip_check_on_cheap_path(mocker, tmp_path):
    vllm_bin, vllm_python = _build_venv(tmp_path)
    cfg = mocker.Mock(vllm_bin=vllm_bin, vllm_python=vllm_python)
    cache_path = tmp_path / "cache.json"

    pip_calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        if cmd[:3] == [str(vllm_python), "-m", "pip"]:
            pip_calls.append(list(cmd))
            return mocker.Mock(returncode=0, stdout="", stderr="")
        if cmd == [str(vllm_bin), "--version"]:
            return mocker.Mock(returncode=0, stdout="vLLM 0.21.0\n", stderr="")
        if cmd[:3] == [str(vllm_python), "-c", mocker.ANY]:
            return mocker.Mock(
                returncode=0,
                stdout='{"vllm":"0.21.0","torch":"2.11.0","torch_cuda":"13.0","transformers":"5.6.2","huggingface_hub":"1.12.0"}\n',
                stderr="",
            )
        raise AssertionError(cmd)

    mocker.patch("vserve.runtime.subprocess.run", side_effect=fake_run)

    info = collect_vllm_runtime_info(
        cfg, prefer_cache=True, with_pip_check=False, cache_path=cache_path
    )

    assert info.vllm_version == "0.21.0"
    assert info.pip_check_ok is None
    assert pip_calls == []


def test_invalidate_vllm_runtime_cache_removes_file(tmp_path):
    from vserve.runtime import invalidate_vllm_runtime_cache

    path = tmp_path / "cache.json"
    path.write_text('{"schema_version":1}')
    assert path.exists()

    invalidate_vllm_runtime_cache(path)

    assert not path.exists()
    # Calling again is a no-op.
    invalidate_vllm_runtime_cache(path)


def test_build_tuning_fingerprint_includes_template_detector_runtime_and_files(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights-v1")
    (model_dir / "tokenizer_config.json").write_text('{"chat_template": "hello {{ tools }}"}')
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=4096,
        is_moe=False,
        model_size_gb=1.0,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()
    runtime = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.20.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
    )

    fp1 = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="vllm",
        gpu_mem_util=0.91,
        runtime_info=runtime,
    )
    (model_dir / "tokenizer_config.json").write_text('{"chat_template": "changed {{ tools }}"}')
    fp2 = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="vllm",
        gpu_mem_util=0.91,
        runtime_info=runtime,
    )

    assert fp1["detector_schema_version"] == DETECTOR_SCHEMA_VERSION
    assert fp1["tokenizer_template_hash"] != fp2["tokenizer_template_hash"]
    assert fp1["vllm_version"] == "0.20.0"
    assert fp1["model_file_identity"][0]["path"] == "model.safetensors"


def test_build_tuning_fingerprint_accepts_runtime_identity(tmp_path):
    """Regression: the llama.cpp backend passes a RuntimeIdentity (0.6.3 changed
    it from a dict). build_tuning_fingerprint called ``.fingerprint()`` — a
    method only RuntimeInfo had — raising ``AttributeError: 'RuntimeIdentity'
    object has no attribute 'fingerprint'`` and breaking every uncached GGUF
    tune (and run)."""
    from vserve.models import ModelInfo
    from vserve.backends.protocol import RuntimeIdentity

    model_dir = tmp_path / "models" / "provider" / "Model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights-v1")
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model",
        architecture="gguf",
        model_type="gguf",
        quant_method=None,
        max_position_embeddings=4096,
        is_moe=False,
        model_size_gb=1.0,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()
    rid = RuntimeIdentity(
        backend="llamacpp",
        executable=Path("/opt/llama-cpp/llama-server"),
        version="b1234",
    )

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="llamacpp",
        gpu_mem_util=0.92,
        runtime_info=rid,
    )
    assert fp["runtime_version"] == "b1234"
    assert str(fp["runtime_executable"]).endswith("llama-server")


def test_build_tuning_fingerprint_includes_gguf_metadata_hash(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF-metadata-v1" + b"\0" * 128)
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model-GGUF",
        architecture="gguf",
        model_type="gguf",
        quant_method=None,
        max_position_embeddings=0,
        is_moe=False,
        model_size_gb=1.0,
        is_gguf=True,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="llamacpp",
        gpu_mem_util=0.91,
    )

    assert fp["gguf_metadata_hash"]


def test_build_tuning_fingerprint_includes_uppercase_gguf_metadata_hash(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model-Q4_K_M.GGUF"
    gguf.write_bytes(b"GGUF-metadata-v1" + b"\0" * 128)
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model-GGUF",
        architecture="gguf",
        model_type="gguf",
        quant_method=None,
        max_position_embeddings=0,
        is_moe=False,
        model_size_gb=1.0,
        is_gguf=True,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="llamacpp",
        gpu_mem_util=0.91,
    )

    assert fp["gguf_metadata_hash"]


def test_build_tuning_fingerprint_includes_gpu_index(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model",
        architecture="TestLM",
        model_type="test",
        quant_method=None,
        max_position_embeddings=4096,
        is_moe=False,
        model_size_gb=1.0,
    )
    gpu = type("GPU", (), {
        "name": "GPU 1",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
        "index": 1,
    })()

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="vllm",
        gpu_mem_util=0.91,
    )

    assert fp["gpu_index"] == 1


def test_build_tuning_fingerprint_includes_llamacpp_runtime_identity(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model-GGUF",
        architecture="gguf",
        model_type="gguf",
        quant_method=None,
        max_position_embeddings=0,
        is_moe=False,
        model_size_gb=1.0,
        is_gguf=True,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="llamacpp",
        gpu_mem_util=0.91,
        runtime_info={
            "backend": "llamacpp",
            "executable": "/opt/llama-cpp/bin/llama-server",
            "llama_server_version": "llama-server 2026",
        },
    )

    assert fp["llama_server_version"] == "llama-server 2026"
    assert fp["runtime_executable"] == "/opt/llama-cpp/bin/llama-server"


# --- 0.6.3: vLLM 0.22 support ---


def test_check_vllm_compatibility_accepts_022():
    info = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.22.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    result = check_vllm_compatibility(info)

    assert result.supported is True, result.errors
    assert any("0.22.0" in message for message in result.messages)


class TestInstalledVllmVersion:
    """installed_vllm_version is the single probe behind every 0.22
    emission gate — None must mean "behave pre-0.22"."""

    def _info(self, version):
        return RuntimeInfo(
            backend="vllm", executable=None, python=None, vllm_version=version
        )

    def test_parses_version(self, mocker):
        import vserve.runtime as rt

        mocker.patch.object(
            rt, "collect_vllm_runtime_info", return_value=self._info("0.22.0")
        )
        from packaging.version import Version

        assert rt.installed_vllm_version() == Version("0.22.0")

    def test_none_when_unknown(self, mocker):
        import vserve.runtime as rt

        mocker.patch.object(
            rt, "collect_vllm_runtime_info", return_value=self._info(None)
        )
        assert rt.installed_vllm_version() is None

    def test_none_when_unparseable(self, mocker):
        import vserve.runtime as rt

        mocker.patch.object(
            rt, "collect_vllm_runtime_info", return_value=self._info("not-a-version")
        )
        assert rt.installed_vllm_version() is None

    def test_uses_fast_cache_path(self, mocker):
        import vserve.runtime as rt

        spy = mocker.patch.object(
            rt, "collect_vllm_runtime_info", return_value=self._info("0.22.0")
        )
        rt.installed_vllm_version()
        kwargs = spy.call_args.kwargs
        assert kwargs.get("prefer_cache") is True
        assert kwargs.get("with_pip_check") is False
