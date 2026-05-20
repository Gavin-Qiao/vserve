from pathlib import Path
import click
import pytest
from typer.testing import CliRunner
from vserve.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM inference manager" in result.output


def test_list_command_exists():
    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    assert "List downloaded models" in result.output


def test_ls_alias_works():
    result = runner.invoke(app, ["ls", "--help"])
    assert result.exit_code == 0


def test_add_command_exists():
    result = runner.invoke(app, ["add", "--help"])
    assert result.exit_code == 0


def test_rm_command_exists():
    result = runner.invoke(app, ["rm", "--help"])
    assert result.exit_code == 0
    assert "Remove a downloaded model" in result.output


def test_tune_command_exists():
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0


def test_run_command_exists():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_run_accepts_tokenized_query_and_noninteractive_flags(mocker, fake_model_dir, tmp_path):
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    backend.build_config.return_value = {
        "model": str(model.path),
        "host": "0.0.0.0",
        "port": 9999,
        "max-model-len": 4096,
        "max-num-seqs": 2,
        "kv-cache-dtype": "auto",
    }
    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0))
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    launch = mocker.patch("vserve.cli._launch_backend")

    result = runner.invoke(
        app,
        [
            "run",
            "testmodel",
            "fp8",
            "--yes",
            "--context",
            "4096",
            "--slots",
            "2",
            "--kv-cache-dtype",
            "auto",
            "--port",
            "9999",
        ],
    )

    assert result.exit_code == 0
    backend.build_config.assert_called_once()
    choices = backend.build_config.call_args.args[1]
    assert choices["context"] == 4096
    assert choices["slots"] == 2
    assert choices["kv_dtype"] == "auto"
    assert choices["port"] == 9999
    launch.assert_called_once()


def test_run_yes_uses_tuned_defaults_and_detected_tool_parser(mocker, fake_model_dir, tmp_path):
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    backend.available_tool_parsers.return_value = {"hermes"}
    backend.available_reasoning_parsers.return_value = {"qwen3"}
    from vserve.config import LIMITS_SCHEMA_VERSION
    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.cli.read_limits", return_value={
        "schema_version": LIMITS_SCHEMA_VERSION,
        "backend": "vllm",
        "limits": {"4096": {"auto": 2}, "8192": {"auto": 4, "fp8": 8}},
        "tool_call_parser": "hermes",
        "reasoning_parser": "qwen3",
    })
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0))
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    launch = mocker.patch("vserve.cli._launch_backend")

    result = runner.invoke(app, ["run", "testmodel", "--yes", "--tools"])

    assert result.exit_code == 0
    choices = backend.build_config.call_args.args[1]
    assert choices["context"] == 8192
    assert choices["slots"] == 4
    assert choices["kv_dtype"] == "auto"
    assert choices["tools"] is True
    assert choices["tool_parser"] == "hermes"
    assert choices["reasoning_parser"] == "qwen3"
    launch.assert_called_once()
    assert launch.call_args.kwargs["non_interactive"] is True


def test_run_yes_auto_tunes_when_limits_missing(mocker, fake_model_dir, tmp_path):
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    backend.tune.return_value = {
        "backend": "vllm",
        "limits": {"4096": {"auto": 2}},
    }
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.cli.read_limits", return_value=None)
    write_limits = mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0))
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._launch_backend")

    result = runner.invoke(app, ["run", "testmodel", "--yes"])

    assert result.exit_code == 0
    backend.tune.assert_called_once()
    write_limits.assert_called_once()


def test_ensure_scripted_limits_retunes_stale_cache(mocker, fake_model_dir):
    from vserve.cli import _ensure_scripted_limits
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.tune.return_value = {"backend": "vllm", "limits": {"8192": {"auto": 4}}}
    gpu = mocker.Mock(name="GPU", vram_total_gb=48.0, driver="drv", cuda="13.0")
    from vserve.config import LIMITS_SCHEMA_VERSION
    mocker.patch("vserve.cli.read_limits", return_value={
        "schema_version": LIMITS_SCHEMA_VERSION,
        "backend": "vllm",
        "fingerprint": {"old": True},
        "limits": {"4096": {"auto": 1}},
    })
    write_limits = mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={"fresh": True})

    result = _ensure_scripted_limits(model, backend, gpu=gpu, gpu_mem_util=0.9, required=True)

    assert result["limits"] == {"8192": {"auto": 4}}
    backend.tune.assert_called_once()
    write_limits.assert_called_once()


def test_ensure_scripted_limits_reuses_runtime_info(mocker, fake_model_dir):
    from vserve.cli import _ensure_scripted_limits
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.runtime_info.side_effect = AssertionError("runtime_info should be reused")
    backend.tune.return_value = {"backend": "vllm", "limits": {"8192": {"auto": 4}}}
    gpu = mocker.Mock(name="GPU", vram_total_gb=48.0, driver="drv", cuda="13.0")
    runtime_info = mocker.Mock()
    mocker.patch("vserve.cli.read_limits", return_value=None)
    build_fingerprint = mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={"fresh": True})
    mocker.patch("vserve.config.write_limits")

    _ensure_scripted_limits(
        model,
        backend,
        gpu=gpu,
        gpu_mem_util=0.9,
        required=True,
        runtime_info=runtime_info,
    )

    backend.runtime_info.assert_not_called()
    assert build_fingerprint.call_args.kwargs["runtime_info"] is runtime_info


def test_custom_vllm_config_reuses_runtime_info(mocker, fake_model_dir, tmp_path):
    from vserve.cli import _custom_config_vllm
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.runtime_info.side_effect = AssertionError("runtime_info should be reused")
    backend.detect_tools.return_value = {}
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    runtime_info = mocker.Mock()
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0, driver="drv", cuda="13.0"))
    mocker.patch("vserve.gpu.resolve_gpu_memory_utilization", return_value=0.9)
    mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={"fresh": True})
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.config.read_limits", return_value={
        "backend": "vllm",
        "limits": {"4096": {"auto": 2}},
    })
    mocker.patch("vserve.cli._pick", side_effect=[0, 0, 0])
    mocker.patch("typer.prompt", return_value="1")
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    write_profile = mocker.patch("vserve.config.write_profile_yaml")

    _custom_config_vllm(model, backend, runtime_info=runtime_info)

    backend.runtime_info.assert_not_called()
    write_profile.assert_called_once()


def test_custom_vllm_config_does_not_validate_reasoning_when_tools_disabled(mocker, fake_model_dir, tmp_path):
    from vserve.cli import _custom_config_vllm
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.runtime_info.side_effect = AssertionError("runtime_info should be reused")
    backend.detect_tools.return_value = {}
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    runtime_info = mocker.Mock()
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0, driver="drv", cuda="13.0"))
    mocker.patch("vserve.gpu.resolve_gpu_memory_utilization", return_value=0.9)
    mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={"fresh": True})
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.config.read_limits", return_value={
        "backend": "vllm",
        "limits": {"4096": {"auto": 2}},
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
    })
    mocker.patch("vserve.cli._pick", side_effect=[0, 0, 0])
    mocker.patch("typer.prompt", return_value="1")
    mocker.patch("typer.confirm", side_effect=[False, True])
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    mocker.patch("vserve.config.write_profile_yaml")

    _custom_config_vllm(model, backend, runtime_info=runtime_info)

    choices = backend.build_config.call_args.args[1]
    assert choices["tools"] is False
    assert choices["tool_parser"] is None
    assert choices["reasoning_parser"] is None


def test_vllm_scripted_defaults_reject_requested_oom_context(fake_model_dir):
    from vserve.cli import _choose_vllm_scripted_defaults
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    limits = {"limits": {"8192": {"auto": None, "fp8": None}}}

    with pytest.raises(ValueError, match="No tuned vLLM capacity"):
        _choose_vllm_scripted_defaults(
            model,
            limits,
            context=8192,
            slots=None,
            kv_cache_dtype=None,
        )


def test_vllm_scripted_defaults_prefers_balanced_recommendation(fake_model_dir):
    from vserve.cli import _choose_vllm_scripted_defaults
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    limits = {
        "limits": {
            "4096": {"auto": 8},
            "8192": {"auto": 4, "fp8": 8},
            "16384": {"auto": None, "fp8": 4},
        },
        "recommendations": {
            "balanced": {
                "context": 8192,
                "kv_cache_dtype": "auto",
                "max_num_seqs": 4,
                "max_num_batched_tokens": 8192,
                "performance_mode": "balanced",
            },
        },
    }

    context, kv_dtype, slots, scheduler = _choose_vllm_scripted_defaults(
        model,
        limits,
        context=None,
        slots=None,
        kv_cache_dtype=None,
    )

    assert context == 8192
    assert kv_dtype == "auto"
    assert slots == 4
    assert scheduler["max_num_batched_tokens"] == 8192
    assert scheduler["performance_mode"] == "balanced"


def test_vllm_scripted_defaults_respects_explicit_context_over_recommendation(fake_model_dir):
    from vserve.cli import _choose_vllm_scripted_defaults
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    limits = {
        "limits": {
            "8192": {"auto": 4},
            "16384": {"fp8": 4},
        },
        "recommendations": {
            "balanced": {
                "context": 8192,
                "kv_cache_dtype": "auto",
                "max_num_seqs": 4,
                "max_num_batched_tokens": 8192,
                "performance_mode": "balanced",
            },
        },
    }

    context, kv_dtype, slots, scheduler = _choose_vllm_scripted_defaults(
        model,
        limits,
        context=16384,
        slots=None,
        kv_cache_dtype=None,
    )

    assert context == 16384
    assert kv_dtype == "fp8"
    assert slots == 4
    assert scheduler == {}


def test_vllm_scripted_defaults_does_not_auto_pick_turboquant(fake_model_dir):
    from vserve.cli import _choose_vllm_scripted_defaults
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    limits = {
        "limits": {
            "8192": {"auto": None, "fp8": 2, "turboquant_k8v4": 6},
            "16384": {"auto": None, "fp8": None, "turboquant_k8v4": 3},
        },
    }

    context, kv_dtype, slots, scheduler = _choose_vllm_scripted_defaults(
        model,
        limits,
        context=None,
        slots=None,
        kv_cache_dtype=None,
    )

    assert context == 8192
    assert kv_dtype == "fp8"
    assert slots == 2
    assert scheduler == {}


def test_vllm_scripted_defaults_allows_explicit_turboquant(fake_model_dir):
    from vserve.cli import _choose_vllm_scripted_defaults
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    limits = {
        "limits": {
            "8192": {"auto": None, "fp8": None, "turboquant_k8v4": 3},
        },
    }

    context, kv_dtype, slots, _scheduler = _choose_vllm_scripted_defaults(
        model,
        limits,
        context=8192,
        slots=None,
        kv_cache_dtype="turboquant_k8v4",
    )

    assert context == 8192
    assert kv_dtype == "turboquant_k8v4"
    assert slots == 3


def test_run_profile_path_infers_backend_without_model(mocker, tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("model: /models/test\nport: 8888\n")
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.compatibility.return_value = mocker.Mock(supported=True)
    mocker.patch("vserve.backends.get_backend_by_name", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    launch = mocker.patch("vserve.cli._launch_backend")

    result = runner.invoke(app, ["run", "--profile", str(profile), "--yes"])

    assert result.exit_code == 0
    launch.assert_called_once()
    assert launch.call_args.args[0] is backend
    assert launch.call_args.args[1] == profile


def test_profile_namespace_exists():
    result = runner.invoke(app, ["profile", "--help"])
    assert result.exit_code == 0
    assert "Manage saved serving profiles" in result.output


def test_profile_rm_refuses_external_path_even_with_force(tmp_path):
    outside = tmp_path / "not-a-profile.yaml"
    outside.write_text("model: nope\n")

    result = runner.invoke(app, ["profile", "rm", str(outside), "--force"])

    assert result.exit_code == 1
    assert outside.exists()
    assert "not a vserve profile" in result.output


def test_profile_show_refuses_external_path(tmp_path):
    outside = tmp_path / "not-a-profile.yaml"
    outside.write_text("model: nope\n")

    result = runner.invoke(app, ["profile", "show", str(outside)])

    assert result.exit_code == 1
    assert "not a vserve profile" in result.output


def test_profile_show_by_name_refuses_symlink_to_external_file(mocker, tmp_path):
    profile_root = tmp_path / "configs"
    profile_root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("model: /secret/model\n")
    (profile_root / "provider--Model.custom.yaml").symlink_to(outside)

    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(configs_dir=profile_root))
    mocker.patch("vserve.backends._BACKENDS", [])

    result = runner.invoke(app, ["profile", "show", "custom"])

    assert result.exit_code == 1
    assert "/secret/model" not in result.output


def test_run_save_profile_rejects_path_traversal(mocker, fake_model_dir):
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")

    result = runner.invoke(app, ["run", "testmodel", "--yes", "--save-profile", "../escape"])

    assert result.exit_code == 1
    assert "Profile name" in result.output
    backend.build_config.assert_not_called()


def test_materialize_subdirectory_variant_as_runnable_root(tmp_path):
    from vserve.cli import _materialize_subdirectory_variant
    from vserve.variants import Variant

    local_dir = tmp_path / "models" / "provider" / "Model"
    (local_dir / "original").mkdir(parents=True)
    (local_dir / "config.json").write_text("{}")
    (local_dir / "tokenizer.json").write_text("{}")
    (local_dir / "original" / "model.safetensors.index.json").write_text("{}")
    (local_dir / "original" / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    variant = Variant(
        label="original",
        files={
            "original/model.safetensors.index.json": 2,
            "original/model-00001-of-00001.safetensors": 7,
        },
    )

    materialized = _materialize_subdirectory_variant(
        local_dir,
        model_name="Model",
        selected_variants=[variant],
        shared={"config.json": 2, "tokenizer.json": 2},
    )

    assert materialized == local_dir.parent / "Model-original"
    assert (materialized / "config.json").exists()
    assert (materialized / "tokenizer.json").exists()
    assert (materialized / "model.safetensors.index.json").exists()
    assert (materialized / "model-00001-of-00001.safetensors").exists()
    assert (local_dir / ".vserve-ignore").exists()


def test_materialize_top_level_variants_as_separate_roots(tmp_path):
    from vserve.cli import _materialize_subdirectory_variants
    from vserve.variants import Variant

    local_dir = tmp_path / "provider" / "Model"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}")
    (local_dir / "tokenizer.json").write_text("{}")
    (local_dir / "model-fp8.safetensors").write_bytes(b"fp8")
    (local_dir / "model-bf16.safetensors").write_bytes(b"bf16")

    roots = _materialize_subdirectory_variants(
        local_dir,
        model_name="Model",
        selected_variants=[
            Variant(label="fp8", files={"model-fp8.safetensors": 3}),
            Variant(label="bf16", files={"model-bf16.safetensors": 4}),
        ],
        shared={"config.json": 2, "tokenizer.json": 2},
    )

    assert roots == [local_dir.parent / "Model-fp8", local_dir.parent / "Model-bf16"]
    assert (roots[0] / "model-fp8.safetensors").exists()
    assert not (roots[0] / "model-bf16.safetensors").exists()
    assert (roots[1] / "model-bf16.safetensors").exists()
    assert not (roots[1] / "model-fp8.safetensors").exists()
    assert (local_dir / ".vserve-ignore").exists()


def test_download_gguf_variants_create_one_root_per_variant(mocker, tmp_path):
    from vserve.cli import _download_model
    from vserve.variants import Variant

    llama_root = tmp_path / "llama"
    variants = [
        Variant(label="Q4_K_M", files={"Model-Q4_K_M.gguf": 4}),
        Variant(label="Q8_0", files={"Model-Q8_0.gguf": 8}),
    ]
    mocker.patch("vserve.variants.fetch_repo_variants", return_value=(variants, {"config.json": 1}))
    mocker.patch("vserve.cli._pick_variants", return_value=variants)
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch(
        "vserve.config.cfg",
        return_value=mocker.Mock(llamacpp_root=llama_root),
    )
    mocker.patch("vserve.models.detect_model", return_value=None)

    def fake_hf_hub_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return str(path)

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download)

    assert _download_model("provider/Model", tmp_path / "models", mocker.Mock(), mocker.Mock()) is True

    assert (llama_root / "models" / "provider" / "Model-Q4_K_M" / "Model-Q4_K_M.gguf").exists()
    assert (llama_root / "models" / "provider" / "Model-Q8_0" / "Model-Q8_0.gguf").exists()
    assert not (llama_root / "models" / "provider" / "Model" / "Model-Q4_K_M.gguf").exists()


def test_download_single_subdir_gguf_materializes_top_level_variant_root(mocker, tmp_path):
    from vserve.cli import _download_model
    from vserve.variants import Variant

    llama_root = tmp_path / "llama"
    variant = Variant(label="Q4_K_M", files={"quant/Model-Q4_K_M.gguf": 4})
    mocker.patch("vserve.variants.fetch_repo_variants", return_value=([variant], {"config.json": 1}))
    mocker.patch("vserve.cli._pick_variants", return_value=[variant])
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(llamacpp_root=llama_root))
    mocker.patch("vserve.models.detect_model", return_value=None)

    def fake_hf_hub_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return str(path)

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download)

    assert _download_model("provider/Model", tmp_path / "models", mocker.Mock(), mocker.Mock()) is True

    assert (llama_root / "models" / "provider" / "Model-Q4_K_M" / "Model-Q4_K_M.gguf").exists()
    assert not (llama_root / "models" / "provider" / "Model-Q4_K_M" / "quant" / "Model-Q4_K_M.gguf").exists()


def test_download_uppercase_gguf_variant_downloads_weight_file(mocker, tmp_path):
    from vserve.cli import _download_model
    from vserve.variants import Variant

    llama_root = tmp_path / "llama"
    variant = Variant(label="Q4_K_M", files={"quant/Model-Q4_K_M.GGUF": 4})
    mocker.patch("vserve.variants.fetch_repo_variants", return_value=([variant], {"config.json": 1}))
    mocker.patch("vserve.cli._pick_variants", return_value=[variant])
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(llamacpp_root=llama_root))
    mocker.patch("vserve.models.detect_model", return_value=None)

    downloaded: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, local_dir):
        downloaded.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return str(path)

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download)

    assert _download_model("provider/Model", tmp_path / "models", mocker.Mock(), mocker.Mock()) is True

    assert "quant/Model-Q4_K_M.GGUF" in downloaded
    assert (llama_root / "models" / "provider" / "Model-Q4_K_M" / "Model-Q4_K_M.GGUF").exists()
    assert not (llama_root / "models" / "provider" / "Model-Q4_K_M" / "quant" / "Model-Q4_K_M.GGUF").exists()


def test_download_wait_for_gguf_lock_checks_variant_roots(mocker, tmp_path):
    from vserve.cli import _download_model
    from vserve.lock import LockHeld, LockInfo
    from vserve.variants import Variant

    llama_root = tmp_path / "llama"
    ready_root = llama_root / "models" / "provider" / "Model-Q4_K_M"
    ready_root.mkdir(parents=True)
    (ready_root / "Model-Q4_K_M.gguf").write_bytes(b"ready")
    variant = Variant(label="Q4_K_M", files={"Model-Q4_K_M.gguf": 4})

    mocker.patch("vserve.variants.fetch_repo_variants", return_value=([variant], {}))
    mocker.patch("vserve.cli._pick_variants", return_value=[variant])
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(llamacpp_root=llama_root))
    mocker.patch("vserve.lock.wait_for_release", return_value=True)
    lock = mocker.Mock()
    lock.acquire.side_effect = [
        LockHeld("download-provider--Model", LockInfo(pid=123, user="alice", since="now", command="download")),
        AssertionError("download should not retry once the GGUF variant root is ready"),
    ]
    mocker.patch("vserve.cli.VserveLock", return_value=lock)

    assert _download_model("provider/Model", tmp_path / "models", mocker.Mock(), mocker.Mock()) is True


def test_download_roots_ready_requires_weight_file(tmp_path):
    from vserve.cli import _download_roots_ready

    root = tmp_path / "models" / "provider" / "Partial"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")

    assert _download_roots_ready([root]) is False

    (root / "model.safetensors").write_bytes(b"weights")

    assert _download_roots_ready([root]) is True


def test_download_rejects_mixed_gguf_and_non_gguf_selection(mocker, tmp_path):
    import typer
    from vserve.cli import _download_model
    from vserve.variants import Variant

    variants = [
        Variant(label="Q4_K_M", files={"Model-Q4_K_M.gguf": 4}),
        Variant(label="fp16", files={"model.safetensors": 8}),
    ]
    mocker.patch("vserve.variants.fetch_repo_variants", return_value=(variants, {"config.json": 1}))
    mocker.patch("vserve.cli._pick_variants", return_value=variants)
    mocker.patch("typer.confirm", return_value=True)

    with pytest.raises(typer.Exit):
        _download_model("provider/Model", tmp_path / "models", mocker.Mock(), mocker.Mock())


def test_auto_tune_downloaded_model_does_not_benchmark_vllm_by_default(mocker, fake_model_dir):
    from vserve.cli import _auto_tune_downloaded_model
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.tune.return_value = {
        "backend": "vllm",
        "limits": {"4096": {"auto": 2}},
        "recommendations": {"balanced": {"context": 4096, "kv_cache_dtype": "auto", "max_num_seqs": 2}},
    }
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0, driver="drv", cuda="13.0"))
    mocker.patch("vserve.gpu.resolve_gpu_memory_utilization", return_value=0.9)
    mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={"fresh": True})
    write_limits = mocker.patch("vserve.config.write_limits")
    bench = mocker.patch("vserve.cli._run_tuning_benchmarks", return_value={"status": "ok", "results": []})

    _auto_tune_downloaded_model(model)

    bench.assert_not_called()
    saved = write_limits.call_args.args[1]
    assert "benchmark_results" not in saved


def test_auto_tune_downloaded_model_does_not_benchmark_llamacpp_by_default(mocker, fake_gguf_model_dir):
    from vserve.cli import _auto_tune_downloaded_model
    from vserve.models import detect_model

    model = detect_model(fake_gguf_model_dir)
    backend = mocker.Mock()
    backend.name = "llamacpp"
    backend.display_name = "llama.cpp"
    backend.tune.return_value = {
        "backend": "llamacpp",
        "limits": {"4096": 4},
        "n_gpu_layers": 32,
    }
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=mocker.Mock(name="GPU", vram_total_gb=48.0, driver="drv", cuda="13.0"))
    mocker.patch("vserve.gpu.resolve_gpu_memory_utilization", return_value=0.9)
    mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={"fresh": True})
    write_limits = mocker.patch("vserve.config.write_limits")
    bench = mocker.patch("vserve.cli._run_tuning_benchmarks", return_value={"status": "ok", "results": []})

    _auto_tune_downloaded_model(model)

    bench.assert_not_called()
    saved = write_limits.call_args.args[1]
    assert "benchmark_results" not in saved


def test_stop_command_exists():
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0


def test_stop_failure_preserves_session(mocker):
    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.is_running.return_value = True
    backend.stop.side_effect = RuntimeError("stop failed")
    lock = mocker.Mock()

    mocker.patch("vserve.backends._BACKENDS", [backend])
    mocker.patch("vserve.cli._session_or_exit")
    clear_session = mocker.patch("vserve.cli.clear_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)

    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 1
    clear_session.assert_not_called()
    lock.release.assert_called_once()



def test_list_runs(mocker):
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vserve.cli.scan_models", return_value=[fake])
    mocker.patch("vserve.cli.read_limits", return_value=None)

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output


def test_add_shows_variant_picker(mocker):
    """add with full repo ID shows variant picker."""
    from vserve.variants import Variant

    mock_api_cls = mocker.patch("huggingface_hub.HfApi")
    mock_api = mock_api_cls.return_value
    mock_api.repo_exists.return_value = True

    mocker.patch("vserve.variants.fetch_repo_variants", return_value=(
        [
            Variant(label="mxfp4", files={
                "model.safetensors.index.json": 50_000,
                "model-00001-of-00001.safetensors": 14_000_000_000,
            }),
        ],
        {"config.json": 1200, "tokenizer.json": 4_000_000},
    ))
    mocker.patch("huggingface_hub.snapshot_download")
    mocker.patch("vserve.cli.VserveLock")
    mocker.patch("vserve.models.detect_model", side_effect=FileNotFoundError)

    result = runner.invoke(app, ["add", "test/model"], input="1\ny\n")

    assert "mxfp4" in result.output
    assert "13.0 GB" in result.output


def test_add_nonexistent_repo_falls_to_search(mocker):
    """add with non-existent repo ID falls through to search."""
    mock_api_cls = mocker.patch("huggingface_hub.HfApi")
    mock_api = mock_api_cls.return_value
    mock_api.repo_exists.return_value = False
    mock_api.list_models.return_value = []

    result = runner.invoke(app, ["add", "nonexistent/model"], input="\n")

    assert "Searching" in result.output


def test_rm_deletes(mocker, tmp_path):
    """rm deletes the model dir and cached limits."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "test" / "Model-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake = ModelInfo(
        path=model_dir,
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.limits_path", return_value=tmp_path / "nonexistent.yaml")
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "nonexistent_profile.yaml")

    result = runner.invoke(app, ["rm", "Model-7B", "--force"])
    assert result.exit_code == 0
    assert "Deleted" in result.output
    assert not model_dir.exists()


def test_rm_cancelled(mocker, tmp_path):
    """rm without --force prompts and can be cancelled."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "test" / "Model-7B"
    model_dir.mkdir(parents=True)

    fake = ModelInfo(
        path=model_dir,
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])

    result = runner.invoke(app, ["rm", "Model-7B"], input="n\n")
    assert model_dir.exists()
    assert "Cancelled" in result.output


def test_remove_alias_works():
    """remove is a hidden alias for rm."""
    result = runner.invoke(app, ["remove", "--help"])
    assert result.exit_code == 0
    assert "alias for rm" in result.output


def test_status_command_exists():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_status_handles_unreadable_active_config(mocker, tmp_path):
    active = tmp_path / "active.yaml"
    active.write_text("not: valid: yaml: {{{{")

    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.return_value = True

    mocker.patch("vserve.backends._BACKENDS", [backend])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "active config unreadable" in result.output
    assert str(active) in result.output


def test_status_handles_non_mapping_active_config(mocker, tmp_path):
    active = tmp_path / "active.yaml"
    active.write_text("- one\n- two\n")

    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.return_value = True

    mocker.patch("vserve.backends._BACKENDS", [backend])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "active config unreadable" in result.output


def test_status_reports_probe_uncertainty(mocker):
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.side_effect = RuntimeError("dbus unavailable")

    mocker.patch("vserve.backends._BACKENDS", [backend])

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Could not determine server state" in result.output
    assert "dbus unavailable" in result.output


def test_status_json_does_not_start_background_update_check(mocker):
    refresh = mocker.patch("vserve.version.background_refresh")
    mocker.patch("vserve.backends._BACKENDS", [])

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    refresh.assert_not_called()


def test_status_json_reports_multiple_active_backends(mocker, tmp_path):
    import json

    active = tmp_path / "active.yaml"
    active.write_text("model: /models/test\nport: 8888\n")

    vllm = mocker.Mock()
    vllm.name = "vllm"
    vllm.display_name = "vLLM"
    vllm.is_running.return_value = True
    vllm.active_manifest_path.return_value = tmp_path / "vllm-manifest.json"

    llama = mocker.Mock()
    llama.name = "llamacpp"
    llama.display_name = "llama.cpp"
    llama.is_running.return_value = True
    llama.active_manifest_path.return_value = tmp_path / "llama-manifest.json"

    mocker.patch("vserve.backends._BACKENDS", [vllm, llama])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)
    mocker.patch("subprocess.check_output", return_value=b"0, 1024\n")

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["multiple_active"] is True
    assert [b["name"] for b in data["running_backends"]] == ["vllm", "llamacpp"]
    assert [b["name"] for b in data["backends"]] == ["vllm", "llamacpp"]
    assert all("running" in b and "probe_error" in b for b in data["backends"])


def test_status_text_warns_about_multiple_active_backends(mocker, tmp_path):
    active = tmp_path / "active.yaml"
    active.write_text("model: /models/test\nport: 8888\n")

    vllm = mocker.Mock()
    vllm.name = "vllm"
    vllm.display_name = "vLLM"
    vllm.is_running.return_value = True
    vllm.active_manifest_path.return_value = tmp_path / "vllm-manifest.json"

    llama = mocker.Mock()
    llama.name = "llamacpp"
    llama.display_name = "llama.cpp"
    llama.is_running.return_value = True
    llama.active_manifest_path.return_value = tmp_path / "llama-manifest.json"

    mocker.patch("vserve.backends._BACKENDS", [vllm, llama])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)
    mocker.patch("subprocess.check_output", return_value=b"0, 1024\n")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Multiple active backends" in result.output
    assert "vLLM, llama.cpp" in result.output


def test_status_json_reports_warming_manifest_as_not_ready(mocker, tmp_path):
    import json

    active = tmp_path / "active.yaml"
    active.write_text("model: /models/test\nport: 8888\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"backend":"vllm","status":"warming","port":8888}\n')

    vllm = mocker.Mock()
    vllm.name = "vllm"
    vllm.display_name = "vLLM"
    vllm.is_running.return_value = True
    vllm.active_manifest_path.return_value = manifest_path
    vllm.health_url.return_value = "http://localhost:8888/health"

    mocker.patch("vserve.backends._BACKENDS", [vllm])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["active_manifest"]["status"] == "warming"
    assert data["next_action"] != "ready"


def test_status_json_reports_active_with_probe_error_uncertainty(mocker, tmp_path):
    import json

    active = tmp_path / "active.yaml"
    active.write_text("model: /models/test\nport: 8888\n")

    vllm = mocker.Mock()
    vllm.name = "vllm"
    vllm.display_name = "vLLM"
    vllm.is_running.return_value = True
    vllm.active_manifest_path.return_value = tmp_path / "manifest.json"

    broken = mocker.Mock()
    broken.name = "llamacpp"
    broken.display_name = "llama.cpp"
    broken.is_running.side_effect = RuntimeError("dbus unavailable")

    mocker.patch("vserve.backends._BACKENDS", [vllm, broken])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)
    mocker.patch("subprocess.check_output", return_value=b"0, 1024\n")

    result = runner.invoke(app, ["status", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["uncertain"] is True
    assert "dbus unavailable" in data["probe_errors"][0]


def test_safe_resolve_path_returns_none_for_symlink_loop(tmp_path):
    from vserve.cli import _safe_resolve_path

    active = tmp_path / "active.yaml"
    target = tmp_path / "profile.yaml"
    active.symlink_to(target)
    target.symlink_to(active)

    assert _safe_resolve_path(active) is None


def test_doctor_command_exists():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0


def test_doctor_json_strict_exits_nonzero_on_failures(mocker, tmp_path):
    import json

    fake_cfg = mocker.Mock(
        service_user="vllm",
        service_name="vllm",
        llamacpp_root=None,
        port=8888,
        logs_dir=tmp_path / "logs",
    )
    fake_cfg.logs_dir.mkdir()
    vllm_bin = tmp_path / "missing-vllm"
    vllm_root = tmp_path / "vllm-root"
    vllm_root.mkdir()

    def fake_run(cmd, *args, **kwargs):
        exe = str(cmd[0])
        if exe == "nvcc":
            return mocker.Mock(returncode=1, stdout="", stderr="missing nvcc")
        if exe == str(vllm_bin):
            return mocker.Mock(returncode=1, stdout="", stderr="missing vllm")
        if exe == "id":
            return mocker.Mock(returncode=1, stdout="", stderr="missing user")
        return mocker.Mock(returncode=0, stdout="", stderr="")

    mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("subprocess.check_output", return_value=b"0\n")
    mocker.patch("vserve.gpu.get_gpu_info", side_effect=RuntimeError("no gpu"))
    mocker.patch("vserve.config.VLLM_BIN", vllm_bin)
    mocker.patch("vserve.config.VLLM_ROOT", vllm_root)
    mocker.patch("vserve.config.LOGS_DIR", fake_cfg.logs_dir)
    mocker.patch("vserve.config.active_yaml_path", return_value=tmp_path / "active.yaml")
    mocker.patch("vserve.config.try_read_profile_yaml", return_value=None)
    mocker.patch("vserve.config.find_systemd_unit_path", return_value=None)
    mocker.patch("vserve.config.cfg", return_value=fake_cfg)
    mocker.patch("vserve.backends.any_backend_running", return_value=False)
    mocker.patch("vserve.backends._BACKENDS", [])
    mocker.patch("vserve.cli._all_models", return_value=[])

    result = runner.invoke(app, ["doctor", "--json", "--strict"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["summary"]["fail"] > 0
    assert any(check["status"] == "fail" for check in data["checks"])


def test_runtime_check_vllm_reports_supported_runtime(mocker):
    info = mocker.Mock(
        backend="vllm",
        vllm_version="0.20.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found.",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
    )
    mocker.patch("vserve.runtime.collect_vllm_runtime_info", return_value=info)

    result = runner.invoke(app, ["runtime", "check", "vllm"])

    assert result.exit_code == 0
    assert "vLLM runtime supported" in result.output
    assert "0.20.0" in result.output


def test_runtime_check_vllm_fails_unsupported_runtime(mocker):
    info = mocker.Mock(
        backend="vllm",
        vllm_version="0.20.0rc1",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found.",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
    )
    mocker.patch("vserve.runtime.collect_vllm_runtime_info", return_value=info)

    result = runner.invoke(app, ["runtime", "check", "vllm"])

    assert result.exit_code == 1
    assert "Unsupported vLLM runtime" in result.output
    assert "pre-release" in result.output


def test_runtime_upgrade_vllm_stable_invokes_runtime_installer(mocker):
    upgrade = mocker.patch("vserve.runtime.upgrade_vllm_stable")
    backend = mocker.Mock()
    backend.is_running.return_value = False
    mocker.patch("vserve.backends.get_backend_by_name", return_value=backend)
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    check = mocker.patch("vserve.runtime.collect_vllm_runtime_info")
    check.return_value = mocker.Mock(
        backend="vllm",
        vllm_version="0.20.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found.",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
    )

    result = runner.invoke(app, ["runtime", "upgrade", "vllm", "--stable"])

    assert result.exit_code == 0
    upgrade.assert_called_once()
    lock.release.assert_called_once()
    assert "vLLM runtime upgraded" in result.output


def test_runtime_upgrade_vllm_stable_refuses_running_service(mocker):
    backend = mocker.Mock()
    backend.is_running.return_value = True
    mocker.patch("vserve.backends.get_backend_by_name", return_value=backend)
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    upgrade = mocker.patch("vserve.runtime.upgrade_vllm_stable")

    result = runner.invoke(app, ["runtime", "upgrade", "vllm", "--stable"])

    assert result.exit_code == 1
    assert "Stop vLLM before upgrading" in result.output
    upgrade.assert_not_called()
    lock.release.assert_called_once()


def test_cache_clean_dry_run_preview_and_refuses_running_backend(mocker):
    backend = mocker.Mock()
    backend.display_name = "vLLM"
    mocker.patch("vserve.backends.probe_running_backends", return_value=([backend], False))

    result = runner.invoke(app, ["cache", "clean", "--dry-run"])

    assert result.exit_code == 1
    assert "Refusing to clean caches while vLLM is running" in result.output


def test_cache_clean_rechecks_backend_state_under_gpu_lock(mocker, tmp_path):
    backend = mocker.Mock()
    backend.display_name = "vLLM"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([], False), ([backend], False)])
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(vllm_root=tmp_path))

    result = runner.invoke(app, ["cache", "clean", "--all", "--yes"])

    assert result.exit_code == 1
    assert "Refusing to clean caches while vLLM is running" in result.output
    lock.release.assert_called_once()


def test_cache_clean_dry_run_counts_recursive_sockets(mocker, tmp_path):
    import socket

    tmp_dir = tmp_path / "tmp" / "nested"
    tmp_dir.mkdir(parents=True)
    sock_path = tmp_dir / "rpc.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(sock_path))
        mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
        mocker.patch("vserve.cli._lock_or_exit", return_value=mocker.Mock())
        mocker.patch("vserve.config.cfg", return_value=mocker.Mock(vllm_root=tmp_path))

        result = runner.invoke(app, ["cache", "clean", "--dry-run"])
    finally:
        sock.close()

    assert result.exit_code == 0
    assert f"Would clean 1 stale sockets from {tmp_path / 'tmp'}" in result.output


def test_cache_clean_refuses_symlinked_vllm_root_before_sudo(mocker, tmp_path):
    outside = tmp_path / "outside"
    (outside / ".cache" / "vllm").mkdir(parents=True)
    root_link = tmp_path / "vllm-link"
    root_link.symlink_to(outside)
    run = mocker.patch("subprocess.run")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
    mocker.patch("vserve.cli._lock_or_exit", return_value=mocker.Mock())
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(vllm_root=root_link))

    result = runner.invoke(app, ["cache", "clean", "--all", "--yes"])

    assert result.exit_code == 1
    assert "Refusing to clean caches under unsafe vLLM root" in result.output
    run.assert_not_called()


def test_cache_clean_refuses_symlinked_cache_parent_before_sudo(mocker, tmp_path):
    outside = tmp_path / "outside-cache"
    (outside / "vllm").mkdir(parents=True)
    cache_parent = tmp_path / ".cache"
    cache_parent.symlink_to(outside)
    run = mocker.patch("subprocess.run")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
    mocker.patch("vserve.cli._lock_or_exit", return_value=mocker.Mock())
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(vllm_root=tmp_path))

    result = runner.invoke(app, ["cache", "clean", "--all", "--yes"])

    assert result.exit_code == 1
    assert "Refusing to clean unsafe cache path" in result.output
    run.assert_not_called()


def test_cache_clean_delete_failure_exits_nonzero(mocker, tmp_path):
    cache_dir = tmp_path / ".cache" / "vllm"
    cache_dir.mkdir(parents=True)
    mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
    mocker.patch("vserve.cli._lock_or_exit", return_value=mocker.Mock())
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(vllm_root=tmp_path))

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["du", "-sm"]:
            return mocker.Mock(returncode=0, stdout="1\tcache\n", stderr="")
        return mocker.Mock(returncode=1, stdout="", stderr="permission denied")

    mocker.patch("subprocess.run", side_effect=fake_run)

    result = runner.invoke(app, ["cache", "clean", "--all", "--yes"])

    assert result.exit_code == 1
    assert "Failed to clean vLLM" in result.output


def test_doctor_handles_invalid_utf8_systemd_unit(mocker, tmp_path):
    unit_path = tmp_path / "vllm.service"
    unit_path.write_bytes(b"\x80\x81")

    fake_cfg = mocker.Mock()
    fake_cfg.service_user = "vllm"
    fake_cfg.service_name = "vllm"
    fake_cfg.llamacpp_root = None
    fake_cfg.port = 8888
    fake_cfg.logs_dir = tmp_path / "logs"
    fake_cfg.logs_dir.mkdir()

    fake_gpu = mocker.Mock()
    fake_gpu.name = "NVIDIA Test GPU"
    fake_gpu.vram_total_gb = 48

    vllm_bin = tmp_path / "vllm"
    vllm_root = tmp_path / "vllm-root"
    vllm_root.mkdir()

    def fake_run(cmd, *args, **kwargs):
        exe = str(cmd[0])
        if exe == "nvcc":
            return mocker.Mock(returncode=0, stdout="Cuda compilation tools, release 13.0, V13.0.0\n", stderr="")
        if exe == str(vllm_bin):
            return mocker.Mock(returncode=0, stdout="vllm 0.0.0\n", stderr="")
        if exe == "id":
            return mocker.Mock(returncode=0, stdout="uid=1000(vllm)\n", stderr="")
        return mocker.Mock(returncode=0, stdout="", stderr="")

    mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("subprocess.check_output", return_value=b"0\n")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.config.VLLM_BIN", vllm_bin)
    mocker.patch("vserve.config.VLLM_ROOT", vllm_root)
    mocker.patch("vserve.config.LOGS_DIR", fake_cfg.logs_dir)
    mocker.patch("vserve.config.active_yaml_path", return_value=tmp_path / "active.yaml")
    mocker.patch("vserve.config.try_read_profile_yaml", return_value=None)
    mocker.patch("vserve.config.find_systemd_unit_path", return_value=unit_path)
    mocker.patch("vserve.config.cfg", return_value=fake_cfg)
    mocker.patch("vserve.backends.any_backend_running", return_value=False)
    mocker.patch("vserve.backends._BACKENDS", [])
    mocker.patch("vserve.cli._all_models", return_value=[])

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Could not read systemd unit" in result.output


def test_doctor_strict_fails_missing_exact_vllm_env_file_even_without_timeout(mocker, tmp_path):
    import json

    unit_path = tmp_path / "vllm.service"
    vllm_root = tmp_path / "vllm-root"
    (vllm_root / "configs" / "models").mkdir(parents=True)
    (vllm_root / "configs" / ".env").write_text(
        "CUDA_HOME=/usr/local/cuda\nTMPDIR=/tmp\nVLLM_RPC_BASE_PATH=/tmp\nCUDA_VISIBLE_DEVICES=0\n"
    )
    (vllm_root / "tmp").mkdir()
    unit_path.write_text("[Service]\nExecStart=/opt/vllm/bin/vllm serve\n")

    fake_cfg = mocker.Mock(
        service_user="vllm",
        service_name="vllm",
        llamacpp_root=None,
        port=8888,
        logs_dir=tmp_path / "logs",
    )
    fake_cfg.logs_dir.mkdir()
    fake_gpu = mocker.Mock(name="GPU", vram_total_gb=48)
    vllm_bin = tmp_path / "vllm"

    def fake_run(cmd, *args, **kwargs):
        exe = str(cmd[0])
        if exe == "nvcc":
            return mocker.Mock(returncode=0, stdout="Cuda compilation tools, release 13.0\n", stderr="")
        if exe == str(vllm_bin):
            return mocker.Mock(returncode=0, stdout="vllm 0.20.0\n", stderr="")
        if exe == "id":
            return mocker.Mock(returncode=0, stdout="uid=1000(vllm)\n", stderr="")
        return mocker.Mock(returncode=0, stdout="", stderr="")

    mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("subprocess.check_output", return_value=b"0\n")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.config.VLLM_BIN", vllm_bin)
    mocker.patch("vserve.config.VLLM_ROOT", vllm_root)
    mocker.patch("vserve.config.LOGS_DIR", fake_cfg.logs_dir)
    mocker.patch("vserve.config.active_yaml_path", return_value=vllm_root / "configs" / "active.yaml")
    mocker.patch("vserve.config.try_read_profile_yaml", return_value=None)
    mocker.patch("vserve.config.find_systemd_unit_path", return_value=unit_path)
    mocker.patch("vserve.config.cfg", return_value=fake_cfg)
    mocker.patch("vserve.runtime.collect_vllm_runtime_info", return_value=mocker.Mock(vllm_version="0.20.0"))
    mocker.patch("vserve.runtime.check_vllm_compatibility", return_value=mocker.Mock(supported=True, range="0.20.x", warnings=[]))
    mocker.patch("vserve.backends._BACKENDS", [])
    mocker.patch("vserve.backends.running_backend", return_value=None)
    mocker.patch("vserve.cli._all_models", return_value=[])

    result = runner.invoke(app, ["doctor", "--json", "--strict"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert any(
        check["status"] == "fail" and "configs/.env" in check["message"]
        for check in data["checks"]
    )


def test_doctor_strict_llamacpp_only_skips_vllm_required_failures(mocker, tmp_path, free_tcp_port):
    import json

    llama_root = tmp_path / "llama"
    (llama_root / "bin").mkdir(parents=True)
    (llama_root / "bin" / "llama-server").write_text("#!/bin/sh\n")
    (llama_root / "models").mkdir()
    (llama_root / "configs").mkdir()
    (llama_root / "configs" / "active.sh").write_text("export CUDA_VISIBLE_DEVICES=0\n")
    unit_path = tmp_path / "llama-cpp.service"
    unit_path.write_text(f"[Service]\nExecStart={llama_root}/configs/active.sh\nTimeoutStartSec=600\n")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    fake_cfg = mocker.Mock(
        service_user="vllm",
        service_name="vllm",
        llamacpp_root=llama_root,
        llamacpp_service_user="llama-cpp",
        llamacpp_service_name="llama-cpp",
        port=free_tcp_port,
        logs_dir=logs_dir,
    )
    fake_gpu = mocker.Mock(name="GPU", vram_total_gb=48)

    def fake_run(cmd, *args, **kwargs):
        exe = str(cmd[0])
        if exe == "nvcc":
            return mocker.Mock(returncode=0, stdout="Cuda compilation tools, release 13.0\n", stderr="")
        if exe == str(llama_root / "bin" / "llama-server"):
            return mocker.Mock(returncode=0, stdout="llama-server version test\n", stderr="")
        if exe == "id":
            return mocker.Mock(returncode=0, stdout="uid=1000(service)\n", stderr="")
        return mocker.Mock(returncode=0, stdout="", stderr="")

    def fake_find_unit(name):
        return unit_path if name == "llama-cpp" else None

    mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("subprocess.check_output", return_value=b"0\n")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.config.VLLM_BIN", tmp_path / "missing-vllm")
    mocker.patch("vserve.config.VLLM_ROOT", tmp_path / "missing-vllm-root")
    mocker.patch("vserve.config.LOGS_DIR", logs_dir)
    mocker.patch("vserve.config.active_yaml_path", return_value=tmp_path / "missing-active.yaml")
    mocker.patch("vserve.config.try_read_profile_yaml", return_value=None)
    mocker.patch("vserve.config.find_systemd_unit_path", side_effect=fake_find_unit)
    mocker.patch("vserve.config.cfg", return_value=fake_cfg)
    mocker.patch("vserve.backends._BACKENDS", [])
    mocker.patch("vserve.backends.running_backend", return_value=None)
    mocker.patch("vserve.cli._all_models", return_value=[])

    result = runner.invoke(app, ["doctor", "--json", "--strict"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["summary"]["fail"] == 0
    assert not any("vLLM not found" in check["message"] for check in data["checks"])


def test_init_command_exists():
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


def test_init_reports_nvml_fan_control_without_coolbits(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    vllm_root = tmp_path / "vllm"

    fake_gpu = mocker.Mock(name="GPU")
    fake_gpu.name = "NVIDIA RTX PRO 5000 Blackwell"
    fake_gpu.vram_total_gb = 48
    fake_gpu.driver = "595.58.03"
    fake_gpu.cuda = "13.1"

    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.gpu.get_fan_count", return_value=1)
    mocker.patch("vserve.config._discover_cuda_home", return_value=tmp_path / "cuda")
    mocker.patch("vserve.config._discover_vllm_root", return_value=vllm_root)
    mocker.patch("vserve.config._discover_service", return_value=("vllm", "vllm"))
    mocker.patch("vserve.config._discover_port", return_value=8888)
    mocker.patch("vserve.config._build_config", return_value=mocker.Mock())
    mocker.patch("vserve.config.save_config")
    mocker.patch("vserve.config.reset_config")
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    mocker.patch("typer.confirm", return_value=False)

    def _which(name: str) -> str | None:
        if name == "nvcc":
            return "/usr/bin/nvcc"
        return None

    mocker.patch("shutil.which", side_effect=_which)
    mocker.patch("subprocess.run", return_value=mocker.Mock(returncode=0, stdout="vLLM 0.1.0", stderr=""))

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Fan control 1 fan(s) exposed via NVML" in result.output
    assert "Coolbits" not in result.output
    assert "Install login banner?" not in result.output


def test_init_llamacpp_only_skips_vllm_prompt_and_banner_install(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"

    fake_gpu = mocker.Mock(name="GPU")
    fake_gpu.name = "NVIDIA RTX PRO 5000 Blackwell"
    fake_gpu.vram_total_gb = 48
    fake_gpu.driver = "595.58.03"
    fake_gpu.cuda = "13.1"

    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.gpu.get_fan_count", return_value=1)
    mocker.patch("vserve.config._discover_cuda_home", return_value=tmp_path / "cuda")
    mocker.patch("vserve.config._discover_vllm_root", return_value=None)
    mocker.patch("vserve.config._discover_service", return_value=("vllm", "vllm"))
    mocker.patch("vserve.config._discover_port", return_value=8888)
    mocker.patch("vserve.config._build_config", return_value=mocker.Mock())
    mocker.patch("vserve.config.save_config")
    mocker.patch("vserve.config.reset_config")
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    mocker.patch("typer.confirm", return_value=False)
    prompt = mocker.patch("typer.prompt")

    def _which(name: str) -> str | None:
        if name == "nvcc":
            return "/usr/bin/nvcc"
        if name == "llama-server":
            return "/opt/llama-cpp/bin/llama-server"
        return None

    mocker.patch("shutil.which", side_effect=_which)
    mocker.patch("subprocess.run", return_value=mocker.Mock(returncode=0, stdout="ok", stderr=""))

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    prompt.assert_not_called()
    assert "optional — needed for safetensors models" in result.output
    assert "required for vserve run/stop" not in result.output
    assert "Install login banner?" not in result.output


def test_fan_off_treats_permission_denied_pid_probe_as_live(mocker, tmp_path):
    import os

    import vserve.fan as fan_module

    fan_module._paths_resolved = False
    mocker.patch("vserve.fan.cfg", return_value=mocker.Mock(
        logs_dir=tmp_path / "logs",
        run_dir=tmp_path / "run",
    ))
    fan_module._resolve_paths()
    fan_module.PID_PATH.parent.mkdir(parents=True)
    fan_module.PID_PATH.write_text("4242")
    fan_module.STATE_PATH.write_text('{"fixed": 55}')

    def fake_kill(pid: int, signal_number: int) -> None:
        if signal_number == 0:
            raise PermissionError("not owner")
        raise AssertionError("should re-exec with sudo before sending SIGTERM")

    original_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if str(path) == "/proc/4242/cmdline":
            return b"python\0-m\0vserve.fan\0"
        return original_read_bytes(path)

    mocker.patch("os.kill", side_effect=fake_kill)
    mocker.patch("os.geteuid", return_value=1000)
    execvp = mocker.patch("os.execvp", side_effect=SystemExit(42))
    mocker.patch.object(Path, "read_bytes", fake_read_bytes)

    result = runner.invoke(app, ["fan", "off"])

    assert result.exit_code == 42
    execvp.assert_called_once()
    assert execvp.call_args.args[0] == "sudo"
    assert fan_module.PID_PATH.exists()
    assert fan_module.STATE_PATH.exists()
    assert os.kill.call_count == 1


def test_fan_off_does_not_remove_state_when_daemon_refuses_to_stop(mocker, tmp_path):
    import signal

    import vserve.fan as fan_module

    fan_module._paths_resolved = False
    mocker.patch("vserve.fan.cfg", return_value=mocker.Mock(
        logs_dir=tmp_path / "logs",
        run_dir=tmp_path / "run",
    ))
    fan_module._resolve_paths()
    fan_module.PID_PATH.parent.mkdir(parents=True)
    fan_module.PID_PATH.write_text("4242")
    fan_module.STATE_PATH.write_text('{"fixed": 55}')

    original_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if str(path) == "/proc/4242/cmdline":
            return b"python\0-m\0vserve.fan\0"
        return original_read_bytes(path)

    def fake_kill(pid: int, signal_number: int) -> None:
        assert pid == 4242
        assert signal_number in (0, signal.SIGTERM)

    mocker.patch("os.kill", side_effect=fake_kill)
    mocker.patch("os.geteuid", return_value=0)
    mocker.patch("time.sleep")
    mocker.patch.object(Path, "read_bytes", fake_read_bytes)

    result = runner.invoke(app, ["fan", "off"])

    assert result.exit_code == 1
    assert "did not stop" in result.output
    assert fan_module.PID_PATH.exists()
    assert fan_module.STATE_PATH.exists()


def test_doctor_summary_label_counts_warnings():
    from vserve.cli import _doctor_summary_label

    assert _doctor_summary_label(0, 0) == "All clear"
    assert _doctor_summary_label(1, 0) == "1 warning(s) found"
    assert _doctor_summary_label(2, 1) == "1 issue(s) and 2 warning(s) found"


# --- Old command names must not work ---

def test_old_start_command_removed():
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code != 0


def test_old_models_command_removed():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code != 0


def test_old_download_command_removed():
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code != 0


# --- Dashboard shows new command names ---

def test_dashboard_shows_new_commands(mocker):
    mocker.patch("vserve.cli._all_models", return_value=[])
    mocker.patch("vserve.cli.read_limits", return_value=None)
    mocker.patch("vserve.cli.CONFIG_FILE", new_callable=lambda: type("P", (), {"exists": lambda self: True})())
    result = runner.invoke(app, [])
    assert "run [model]" in result.output or "run" in result.output
    assert "list" in result.output
    assert "add" in result.output
    assert "rm" in result.output


# --- list command ---

def test_list_detail_view(mocker):
    """list <model> shows detail view."""
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.read_limits", return_value=None)

    result = runner.invoke(app, ["list", "Model-7B"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output
    assert "Not probed" in result.output or "not tuned" in result.output.lower() or "TestLM" in result.output


def test_list_empty(mocker):
    mocker.patch("vserve.cli.scan_models", return_value=[])
    mocker.patch("vserve.cli._all_models", return_value=[])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models" in result.output


def test_list_with_limits(mocker):
    """list shows tuned status from cached limits."""
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    mocker.patch("vserve.cli.scan_models", return_value=[fake])
    mocker.patch("vserve.cli.read_limits", return_value={
        "limits": {"4096": {"auto": 64, "fp8": 128}},
        "tool_call_parser": "hermes",
    })
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output


# --- rm edge cases ---

def test_rm_cleans_empty_parent(mocker, tmp_path):
    """rm removes empty parent provider dir."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "orphan-provider" / "Model-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake = ModelInfo(
        path=model_dir, provider="orphan-provider", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method=None,
        max_position_embeddings=32768, is_moe=False, model_size_gb=1.0,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.limits_path", return_value=tmp_path / "no.yaml")
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "no.yaml")

    result = runner.invoke(app, ["rm", "Model-7B", "--force"])
    assert result.exit_code == 0
    assert not model_dir.exists()
    assert not model_dir.parent.exists()  # empty provider dir removed


def test_rm_interactive_picker(mocker, tmp_path):
    """rm without args shows picker."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "test" / "Model-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake = ModelInfo(
        path=model_dir, provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method=None,
        max_position_embeddings=32768, is_moe=False, model_size_gb=1.0,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli._pick", return_value=0)
    mocker.patch("vserve.cli.limits_path", return_value=tmp_path / "no.yaml")
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "no.yaml")

    result = runner.invoke(app, ["rm", "--force"])
    assert result.exit_code == 0
    assert "Deleted" in result.output


# --- run command ---

def test_run_no_models(mocker):
    mocker.patch("vserve.cli._all_models", return_value=[])
    mocker.patch("vserve.cli.check_session")
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "No models" in result.output


def test_run_backend_override(mocker):
    """run --backend forces specific backend."""
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.check_session")
    mock_get = mocker.patch("vserve.backends.get_backend_by_name", side_effect=KeyError("nope"))

    runner.invoke(app, ["run", "Model-7B", "--backend", "nope"])
    mock_get.assert_called_once_with("nope")


def test_run_forced_backend_rejects_model_it_cannot_serve(mocker):
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    backend = mocker.Mock()
    backend.name = "llamacpp"
    backend.display_name = "llama.cpp"
    backend.can_serve.return_value = False
    backend.compatibility.return_value = mocker.Mock(supported=True)

    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.backends.get_backend_by_name", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")

    result = runner.invoke(app, ["run", "Model-7B", "--backend", "llamacpp"])

    assert result.exit_code == 1
    assert "llama.cpp cannot serve test/Model-7B" in result.output
    assert "GGUF" in result.output


def test_launch_backend_aborts_on_restart_stop_failure(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nmax-model-len: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"
    backend.is_running.return_value = True
    backend.stop.side_effect = RuntimeError("stop failed")

    lock = mocker.Mock()
    clear_session = mocker.patch("vserve.cli.clear_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([backend], False))
    mocker.patch("vserve.config.read_profile_yaml", return_value={"port": 8888, "max-model-len": 4096})
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("time.sleep")

    with pytest.raises(click.exceptions.Exit):
        _launch_backend(backend, cfg_path, "test-model")

    backend.start.assert_not_called()
    clear_session.assert_not_called()
    lock.release.assert_called_once()


def test_launch_backend_noninteractive_requires_replace(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nmax-model-len: 4096\n")
    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([backend], False))
    mocker.patch("vserve.config.read_profile_yaml", return_value={"port": 8888, "max-model-len": 4096})

    with pytest.raises(click.exceptions.Exit):
        _launch_backend(backend, cfg_path, "test-model", non_interactive=True)

    backend.stop.assert_not_called()
    backend.start.assert_not_called()
    lock.release.assert_called_once()


def test_launch_backend_replace_stops_without_prompt(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nmax-model-len: 4096\n")
    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"
    backend.health_url.return_value = "http://localhost:8888/health"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([backend], False), ([], False)])
    mocker.patch("vserve.config.read_profile_yaml", return_value={"port": 8888, "max-model-len": 4096})
    confirm = mocker.patch("typer.confirm")
    mocker.patch("time.sleep")

    class Resp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return None

    mocker.patch("urllib.request.urlopen", return_value=Resp())

    _launch_backend(backend, cfg_path, "test-model", non_interactive=True, replace=True)

    confirm.assert_not_called()
    backend.stop.assert_called_once()
    backend.start.assert_called_once_with(cfg_path, non_interactive=True)
    lock.release.assert_called_once()


def test_launch_backend_aborts_on_invalid_yaml_profile(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("- not\n- a mapping\n")

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"

    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)

    with pytest.raises(click.exceptions.Exit):
        _launch_backend(backend, cfg_path, "test-model")

    backend.start.assert_not_called()
    lock.release.assert_called_once()


def test_launch_backend_restart_keeps_session_until_new_start(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nmax-model-len: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"
    backend.is_running.side_effect = [True, False]
    backend.health_url.return_value = "http://localhost:8888/health"

    lock = mocker.Mock()
    clear_session = mocker.patch("vserve.cli.clear_session")
    write_session = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([backend], False), ([], False)])
    mocker.patch("vserve.config.read_profile_yaml", return_value={"port": 8888, "max-model-len": 4096})
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("time.sleep")
    resp = mocker.MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    mocker.patch("urllib.request.urlopen", return_value=resp)
    mocker.patch("vserve.config.cfg", return_value=mocker.Mock(vllm_root=tmp_path))

    _launch_backend(backend, cfg_path, "test-model")

    backend.stop.assert_called_once()
    clear_session.assert_not_called()
    write_session.assert_called_once_with("test-model")


def test_launch_backend_restart_times_out_if_service_never_stops(mocker, tmp_path):
    from vserve.cli import _launch_backend
    import typer

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nmax-model-len: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"
    backend.is_running.side_effect = [True] + [True] * 20

    lock = mocker.Mock()
    clear_session = mocker.patch("vserve.cli.clear_session")
    write_session = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch(
        "vserve.backends.probe_running_backends",
        side_effect=[([backend], False)] + [([backend], False)] * 15,
    )
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("time.sleep")

    with pytest.raises(typer.Exit) as exc:
        _launch_backend(backend, cfg_path, "test-model")

    assert exc.value.exit_code == 1
    backend.stop.assert_called_once()
    backend.start.assert_not_called()
    write_session.assert_not_called()
    clear_session.assert_not_called()


def test_launch_backend_stops_other_running_backend_before_start(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nctx_size: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "llama.cpp"
    backend.service_name = "llama-cpp"
    backend.name = "llamacpp"
    backend.health_url.return_value = "http://localhost:8888/health"

    active = mocker.Mock()
    active.display_name = "vLLM"
    active.service_name = "vllm"
    active.name = "vllm"

    lock = mocker.Mock()
    write_session = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([active], False), ([], False)])
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("time.sleep")
    resp = mocker.MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    mocker.patch("urllib.request.urlopen", return_value=resp)

    _launch_backend(backend, cfg_path, "profile")

    active.stop.assert_called_once()
    backend.start.assert_called_once_with(cfg_path, non_interactive=False)
    write_session.assert_called_once_with("profile")


def test_launch_backend_probe_uncertainty_stops_all_known_backends(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nctx_size: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "llama.cpp"
    backend.service_name = "llama-cpp"
    backend.name = "llamacpp"
    backend.health_url.return_value = "http://localhost:8888/health"

    active = mocker.Mock()
    active.display_name = "vLLM"
    active.service_name = "vllm"
    active.name = "vllm"

    uncertain = mocker.Mock()
    uncertain.display_name = "llama.cpp"
    uncertain.service_name = "llama-cpp"
    uncertain.name = "llamacpp"

    lock = mocker.Mock()
    write_session = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends._BACKENDS", [active, uncertain])
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([active], True), ([], False)])
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("time.sleep")
    resp = mocker.MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    mocker.patch("urllib.request.urlopen", return_value=resp)

    _launch_backend(backend, cfg_path, "profile")

    active.stop.assert_called_once()
    uncertain.stop.assert_called_once()
    backend.start.assert_called_once_with(cfg_path, non_interactive=False)
    write_session.assert_called_once_with("profile")


def test_launch_backend_timeout_returns_when_service_stays_running(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nctx_size: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "llama.cpp"
    backend.service_name = "llama-cpp"
    backend.name = "llamacpp"
    backend.is_running.side_effect = [True] * 200
    backend.health_url.return_value = "http://localhost:8888/health"

    lock = mocker.Mock()
    clear_session = mocker.patch("vserve.cli.clear_session")
    write_session = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
    mocker.patch("vserve.cli._is_interactive", return_value=True)
    mocker.patch("time.sleep")
    mocker.patch("urllib.request.urlopen", side_effect=RuntimeError("not ready"))

    _launch_backend(backend, cfg_path, "profile")

    write_session.assert_called_once_with("profile")
    clear_session.assert_not_called()


def test_launch_backend_timeout_noninteractive_requires_health(mocker, tmp_path):
    from vserve.cli import _launch_backend
    import typer

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nctx_size: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "llama.cpp"
    backend.service_name = "llama-cpp"
    backend.name = "llamacpp"
    backend.is_running.side_effect = [True] * 200
    backend.health_url.return_value = "http://localhost:8888/health"

    lock = mocker.Mock()
    clear_session = mocker.patch("vserve.cli.clear_session")
    write_session = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
    mocker.patch("vserve.cli._is_interactive", return_value=False)
    mocker.patch("time.sleep")
    mocker.patch("urllib.request.urlopen", side_effect=RuntimeError("not ready"))

    with pytest.raises(typer.Exit) as exc:
        _launch_backend(backend, cfg_path, "profile")

    assert exc.value.exit_code == 1
    write_session.assert_called_once_with("profile")
    clear_session.assert_not_called()


def test_launch_backend_yes_requires_health_even_from_tty(mocker, tmp_path):
    from vserve.cli import _launch_backend
    import typer

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nctx_size: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "llama.cpp"
    backend.service_name = "llama-cpp"
    backend.name = "llamacpp"
    backend.is_running.side_effect = [True] * 200
    backend.health_url.return_value = "http://localhost:8888/health"

    mocker.patch("vserve.cli._lock_or_exit", return_value=mocker.Mock())
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.backends.probe_running_backends", return_value=([], False))
    mocker.patch("vserve.cli._is_interactive", return_value=True)
    mocker.patch("time.sleep")
    mocker.patch("urllib.request.urlopen", side_effect=RuntimeError("not ready"))

    with pytest.raises(typer.Exit) as exc:
        _launch_backend(backend, cfg_path, "profile", non_interactive=True)

    assert exc.value.exit_code == 1


def test_stop_probe_failure_uses_fallback_stop(mocker):
    first = mocker.Mock()
    first.display_name = "vLLM"
    second = mocker.Mock()
    second.display_name = "llama.cpp"
    second.stop.side_effect = ValueError("systemctl stop llama-cpp failed")

    lock = mocker.Mock()
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([], True), ([], False)])
    mocker.patch("vserve.backends._BACKENDS", [first, second])
    clear_session = mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    assert "Backend state was uncertain" in result.output
    first.stop.assert_called_once()
    second.stop.assert_called_once()
    clear_session.assert_called_once()


def test_stop_stops_all_running_backends(mocker):
    first = mocker.Mock()
    first.display_name = "vLLM"
    second = mocker.Mock()
    second.display_name = "llama.cpp"

    lock = mocker.Mock()
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([first, second], False), ([], False)])
    clear_session = mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    assert "Stopped: vLLM, llama.cpp." in result.output
    first.stop.assert_called_once()
    second.stop.assert_called_once()
    clear_session.assert_called_once()


def test_stop_probe_failure_with_running_backend_stops_all_known_backends(mocker):
    first = mocker.Mock()
    first.display_name = "vLLM"
    second = mocker.Mock()
    second.display_name = "llama.cpp"

    lock = mocker.Mock()
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.backends._BACKENDS", [first, second])
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([first], True), ([], False)])
    clear_session = mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    assert "Stopped: vLLM, llama.cpp." in result.output
    first.stop.assert_called_once()
    second.stop.assert_called_once()
    clear_session.assert_called_once()


def _wait_for_health_inputs(mocker, *, urlopen_results, tails, service_running=None):
    """Build (sleep_fn, urlopen_fn, log_tail_fn, service_running_fn) for tests."""
    sleeps: list[float] = []

    def sleep_fn(s):
        sleeps.append(s)

    url_iter = iter(urlopen_results)

    def urlopen_fn(url, timeout=2):
        outcome = next(url_iter)
        if isinstance(outcome, Exception):
            raise outcome
        resp = mocker.MagicMock()
        resp.status = outcome
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    tail_iter = iter(tails)

    def log_tail_fn():
        try:
            return next(tail_iter)
        except StopIteration:
            return None

    running_iter = iter(service_running or [])

    def service_running_fn():
        try:
            return next(running_iter)
        except StopIteration:
            return True

    return sleep_fn, urlopen_fn, log_tail_fn, service_running_fn, sleeps


def test_wait_for_health_returns_ready_on_first_200(mocker):
    from vserve.cli import _wait_for_health

    sleep_fn, urlopen_fn, log_tail_fn, svc_fn, sleeps = _wait_for_health_inputs(
        mocker, urlopen_results=[200], tails=[]
    )

    outcome = _wait_for_health(
        health_url="http://h",
        timeout_s=300,
        poll_s=3,
        log_tail_fn=log_tail_fn,
        service_running_fn=svc_fn,
        sleep_fn=sleep_fn,
        urlopen_fn=urlopen_fn,
    )

    assert outcome == "ready"
    assert sleeps == [3]


def test_wait_for_health_deduplicates_log_block(capsys, mocker):
    from vserve.cli import _wait_for_health

    sleep_fn, urlopen_fn, log_tail_fn, svc_fn, _ = _wait_for_health_inputs(
        mocker,
        urlopen_results=[Exception("not ready"), Exception("not ready"), 200],
        tails=["AAA\nBBB\nCCC", "AAA\nBBB\nCCC", "AAA\nBBB\nCCC"],
    )

    outcome = _wait_for_health(
        health_url="http://h",
        timeout_s=300,
        poll_s=3,
        log_tail_fn=log_tail_fn,
        service_running_fn=svc_fn,
        sleep_fn=sleep_fn,
        urlopen_fn=urlopen_fn,
    )

    captured = capsys.readouterr().out
    assert outcome == "ready"
    occurrences = captured.count("Latest service logs")
    assert occurrences == 1, f"expected 1 header, got {occurrences}: {captured!r}"


def test_wait_for_health_prints_new_block_when_tail_changes(capsys, mocker):
    from vserve.cli import _wait_for_health

    # 4 polls: tail A, tail A (dedup), tail B, then 200. New tail prints again.
    sleep_fn, urlopen_fn, log_tail_fn, svc_fn, _ = _wait_for_health_inputs(
        mocker,
        urlopen_results=[Exception("x"), Exception("x"), Exception("x"), 200],
        tails=["AAA\nBBB", "AAA\nBBB", "CCC\nDDD"],
    )

    _wait_for_health(
        health_url="http://h",
        timeout_s=300,
        poll_s=3,
        log_tail_fn=log_tail_fn,
        service_running_fn=svc_fn,
        sleep_fn=sleep_fn,
        urlopen_fn=urlopen_fn,
    )

    captured = capsys.readouterr().out
    assert captured.count("Latest service logs") == 2


def test_wait_for_health_returns_stopped_when_service_dies(mocker):
    from vserve.cli import _wait_for_health

    sleep_fn, urlopen_fn, log_tail_fn, svc_fn, _ = _wait_for_health_inputs(
        mocker,
        urlopen_results=[Exception("x"), Exception("x"), Exception("x"), Exception("x")],
        tails=[None, None, None, None],
        # elapsed_s only checks service after >10s, so set False on 4th poll (elapsed=12s)
        service_running=[True, True, True, False],
    )

    outcome = _wait_for_health(
        health_url="http://h",
        timeout_s=300,
        poll_s=3,
        log_tail_fn=log_tail_fn,
        service_running_fn=svc_fn,
        sleep_fn=sleep_fn,
        urlopen_fn=urlopen_fn,
    )

    assert outcome == "stopped"


def test_wait_for_health_returns_timeout_after_budget(mocker):
    from vserve.cli import _wait_for_health

    fail = [Exception("nope")] * 5
    sleep_fn, urlopen_fn, log_tail_fn, svc_fn, sleeps = _wait_for_health_inputs(
        mocker,
        urlopen_results=fail,
        tails=[None] * 5,
    )

    outcome = _wait_for_health(
        health_url="http://h",
        timeout_s=12,  # ceil(12/3) = 4 iterations
        poll_s=3,
        log_tail_fn=log_tail_fn,
        service_running_fn=svc_fn,
        sleep_fn=sleep_fn,
        urlopen_fn=urlopen_fn,
    )

    assert outcome == "timeout"
    assert sleeps == [3, 3, 3, 3]


def test_stop_warns_once_when_owner_unknown(mocker):
    """The unknown-owner warning fires once, not on every TOCTOU re-check."""
    from vserve.lock import SessionUnknown

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch(
        "vserve.backends.probe_running_backends",
        side_effect=[([backend], False), ([], False)],
    )
    # First call (loud) raises SessionUnknown. Subsequent calls (quiet TOCTOU
    # re-check) also raise — without the quiet flag they'd print again.
    mocker.patch(
        "vserve.cli.check_session",
        side_effect=SessionUnknown("vLLM is running, but no vserve session marker exists."),
    )
    # read_session() returns None so the orphan-claim path fires.
    mocker.patch("vserve.cli.read_session", return_value=None)
    write = mocker.patch("vserve.cli.write_session")
    clear_session = mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    occurrences = result.output.count("no vserve session marker exists")
    assert occurrences == 1, f"expected 1 warning, got {occurrences}: {result.output!r}"
    backend.stop.assert_called_once()
    clear_session.assert_called_once()
    write.assert_called_with("orphan-claim")


def test_stop_claims_orphan_session_before_release(mocker):
    """After warning, stop() auto-claims the orphan via write_session."""
    from vserve.lock import SessionUnknown

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch(
        "vserve.backends.probe_running_backends",
        side_effect=[([backend], False), ([], False)],
    )
    mocker.patch("vserve.cli.check_session", side_effect=SessionUnknown())
    mocker.patch("vserve.cli.read_session", return_value=None)
    write = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    write.assert_called_once_with("orphan-claim")


def test_stop_does_not_claim_when_session_already_exists(mocker):
    """When read_session returns a valid marker, don't overwrite it."""
    from vserve.lock import SessionInfo

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch(
        "vserve.backends.probe_running_backends",
        side_effect=[([backend], False), ([], False)],
    )
    mocker.patch("vserve.cli.check_session")  # owner known, no exception
    mocker.patch(
        "vserve.cli.read_session",
        return_value=SessionInfo(user="mohan", since="2026-05-19 12:00:00", model="x"),
    )
    write = mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    write.assert_not_called()


def test_stop_warns_once_under_probe_failed_fallback(mocker):
    """Probe-failed fallback path also warns once and runs fallback stop."""
    from vserve.lock import SessionUnknown

    first = mocker.Mock()
    first.display_name = "vLLM"
    lock = mocker.Mock()
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch(
        "vserve.backends.probe_running_backends",
        side_effect=[([], True), ([], False)],
    )
    mocker.patch("vserve.backends._BACKENDS", [first])
    mocker.patch(
        "vserve.cli.check_session",
        side_effect=SessionUnknown("vLLM is running, but no vserve session marker exists."),
    )
    mocker.patch("vserve.cli.read_session", return_value=None)
    mocker.patch("vserve.cli.write_session")
    mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    occurrences = result.output.count("no vserve session marker exists")
    assert occurrences == 1, f"expected 1 warning, got {occurrences}: {result.output!r}"


def test_stop_probe_failure_exits_when_state_remains_uncertain(mocker):
    first = mocker.Mock()
    first.display_name = "vLLM"

    lock = mocker.Mock()
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.backends.probe_running_backends", side_effect=[([], True), ([], True)])
    mocker.patch("vserve.backends._BACKENDS", [first])
    clear_session = mocker.patch("vserve.cli.clear_session")

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 1
    assert "Could not verify backend state after fallback stop attempts" in result.output
    first.stop.assert_called_once()
    clear_session.assert_not_called()


def test_update_aborts_when_latest_version_unknown(mocker):
    mocker.patch("vserve.version.check_pypi", return_value=None)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "Could not determine the latest published version" in result.output


def test_update_aborts_when_no_updater_is_available(mocker):
    mocker.patch("vserve.version.check_pypi", return_value="9999.0.0")
    mocker.patch("vserve.version.write_cache")
    mocker.patch("shutil.which", return_value=None)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "Could not find uv or pip to perform upgrade" in result.output


def test_update_falls_back_to_pip_when_uv_inspection_fails(mocker):
    check_pypi = mocker.patch("vserve.version.check_pypi", return_value="9999.0.0")
    write_cache = mocker.patch("vserve.version.write_cache")
    mocker.patch("shutil.which", side_effect=["/usr/bin/uv", "/usr/bin/pip"])

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["/usr/bin/uv", "tool", "list"]:
            return mocker.Mock(returncode=1, stdout="", stderr="uv failed")
        if cmd[:3] == ["/usr/bin/pip", "install", "--upgrade"]:
            return mocker.Mock(returncode=0, stdout="", stderr="")
        raise AssertionError(cmd)

    mocker.patch("subprocess.run", side_effect=fake_run)
    refresh = mocker.patch("vserve.cli._refresh_banner")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "uv inspection failed; trying pip fallback" in result.output
    check_pypi.assert_called_once()
    write_cache.assert_called_once()
    refresh.assert_called_once()


def test_update_uses_discovered_pip_when_current_interpreter_has_no_pip(mocker):
    check_pypi = mocker.patch("vserve.version.check_pypi", return_value="9999.0.0")
    mocker.patch("vserve.version.write_cache")

    def fake_which(name):
        if name == "uv":
            return None
        if name == "pip":
            return "/usr/bin/pip"
        return None

    mocker.patch("shutil.which", side_effect=fake_which)

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["/usr/bin/pip", "install", "--upgrade"]:
            return mocker.Mock(returncode=0, stdout="", stderr="")
        if cmd[1:] == ["-c", "import vserve; print(vserve.__version__)"]:
            return mocker.Mock(returncode=0, stdout="", stderr="")
        raise AssertionError(cmd)

    run = mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("vserve.cli._refresh_banner")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    check_pypi.assert_called_once()
    assert any(call.args[0][:3] == ["/usr/bin/pip", "install", "--upgrade"] for call in run.call_args_list)


def test_llamacpp_slots_helper_accepts_nested_schema():
    from vserve.cli import _llamacpp_slots_from_limits_entry

    assert _llamacpp_slots_from_limits_entry({"auto": 4, "fp8": 8}) == 8


def test_vllm_limits_helper_accepts_flat_schema():
    from vserve.cli import _vllm_limits_entry

    assert _vllm_limits_entry(6) == {"auto": 6}


def test_llamacpp_interactive_slot_ceiling_uses_effective_dtype_column():
    """Regression: picking 128k with default-dtype runtime must not advertise
    the q4_0 ceiling. The cached gemma-4-26B-A4B matrix returns 42 across
    columns (q4_0) but only 11 at f16 — the run-time default in the absence
    of an explicit dtype. The picker must reflect the dtype that will
    actually be used."""
    from vserve.cli import (
        _llamacpp_interactive_runtime_defaults,
        _llamacpp_interactive_slot_ceiling,
    )

    lim = {
        "recommended_kv_dtype": "q8_0",
        "limits": {
            "131072": {"f16": 11, "q8_0": 22, "q4_1": 38, "q4_0": 42},
            "65536": {"f16": 21, "q8_0": 40, "q4_1": 69, "q4_0": 76},
        },
    }
    effective_kv, effective_limits = _llamacpp_interactive_runtime_defaults(lim)
    assert effective_kv == "q8_0"
    # Slot ceiling at 128k must come from the q8_0 column, not max across dtypes.
    assert _llamacpp_interactive_slot_ceiling(effective_limits, 131072, effective_kv) == 22
    assert _llamacpp_interactive_slot_ceiling(effective_limits, 65536, effective_kv) == 40


def test_llamacpp_interactive_runtime_defaults_does_not_promote_moe_offload_table():
    """Regression: when the model fits on GPU, the interactive picker must
    advertise the regular (no-`-ot`) slot ceilings — the `limits_with_ot`
    table is for users who explicitly opt into trading throughput for KV
    headroom. The previous implementation auto-promoted the `-ot` table for
    any MoE model, which silently relocated expert FFN weights to RAM and
    made inference 10–100× slower with no benefit when the model fit."""
    from vserve.cli import (
        _llamacpp_interactive_runtime_defaults,
        _llamacpp_interactive_slot_ceiling,
    )

    lim = {
        "recommended_kv_dtype": "q8_0",
        "full_offload": True,
        "limits": {"131072": {"f16": 11, "q8_0": 22}},
        "moe": {
            "is_moe": True,
            "expert_count": 128,
            "ot_pattern": ".ffn_.*_exps.=CPU",
            "limits_with_ot": {"131072": {"f16": 14, "q8_0": 26}},
        },
    }
    effective_kv, effective_limits = _llamacpp_interactive_runtime_defaults(lim)
    assert effective_kv == "q8_0"
    # Picker sees the no-`-ot` ceiling (22), not the inflated `-ot` one (26).
    assert _llamacpp_interactive_slot_ceiling(effective_limits, 131072, effective_kv) == 22


def test_llamacpp_needs_moe_offload_only_when_slots_exceed_no_ot_capacity():
    """Auto-apply `-ot` only when the chosen run won't fit on GPU otherwise."""
    from vserve.cli import _llamacpp_needs_moe_offload

    lim = {
        "full_offload": True,
        "limits": {"131072": {"f16": 11, "q8_0": 22}},
        "moe": {"is_moe": True, "ot_pattern": ".ffn_.*_exps.=CPU"},
    }
    # Within capacity → no auto-offload (preserve GPU throughput).
    assert _llamacpp_needs_moe_offload(lim, 131072, 2, "q8_0") is False
    assert _llamacpp_needs_moe_offload(lim, 131072, 22, "q8_0") is False
    # Exceeds q8_0 capacity → need offload to fit.
    assert _llamacpp_needs_moe_offload(lim, 131072, 23, "q8_0") is True
    # Default-dtype runtime: gate against f16 column.
    assert _llamacpp_needs_moe_offload(lim, 131072, 12, "f16") is True
    assert _llamacpp_needs_moe_offload(lim, 131072, 11, "f16") is False


def test_llamacpp_needs_moe_offload_partial_offload_always_true():
    """Model already partially on CPU (full_offload=False) → `-ot` is strictly
    better than letting llama.cpp pick which tail layers to spill, since the
    expert FFN blocks are the cleanest chunk to move."""
    from vserve.cli import _llamacpp_needs_moe_offload

    lim = {
        "full_offload": False,
        "limits": {"4096": {"f16": 4}},
        "moe": {"is_moe": True, "ot_pattern": ".ffn_.*_exps.=CPU"},
    }
    assert _llamacpp_needs_moe_offload(lim, 4096, 1, "f16") is True


def test_llamacpp_needs_moe_offload_non_moe_is_false():
    """Non-MoE models can't benefit from the expert-FFN pattern at all."""
    from vserve.cli import _llamacpp_needs_moe_offload

    lim = {"full_offload": True, "limits": {"4096": {"f16": 4}}, "moe": None}
    assert _llamacpp_needs_moe_offload(lim, 4096, 99, "f16") is False
    # Also degrades cleanly when the moe block has is_moe=False.
    lim2 = {"full_offload": True, "limits": {"4096": {"f16": 4}}, "moe": {"is_moe": False}}
    assert _llamacpp_needs_moe_offload(lim2, 4096, 99, "f16") is False


def test_llamacpp_interactive_slot_ceiling_falls_back_to_f16_then_max():
    """Partial / legacy cache rows: prefer the picked dtype, then f16,
    then any non-null value in the row, so older limits remain usable."""
    from vserve.cli import _llamacpp_interactive_slot_ceiling

    # dtype absent → falls back to f16
    assert _llamacpp_interactive_slot_ceiling({"4096": {"f16": 7}}, 4096, "q8_0") == 7
    # dtype + f16 absent → falls back to max across the remaining columns
    assert _llamacpp_interactive_slot_ceiling({"4096": {"q4_0": 12}}, 4096, "q8_0") == 12
    # legacy flat-int entry (pre-L1 schema) — accept as-is
    assert _llamacpp_interactive_slot_ceiling({"4096": 5}, 4096, "q8_0") == 5
    # missing context
    assert _llamacpp_interactive_slot_ceiling({"4096": {"f16": 7}}, 8192, "f16") is None


# --- Polish 1: engine-failure diagnosis ---


class TestDiagnoseEngineFailure:
    """Regression coverage for every error pattern hit in the 2026-05-19 debug
    session. Each pattern was previously left to the user to grep out of
    journalctl / vllm.log by hand; the diagnosis surfaces it inline."""

    def test_vllm_triton_attn_rejects_turboquant(self):
        from vserve.cli import _diagnose_engine_failure
        log = """
        ERROR ... gemma4.py:489 ... Attention __init__
        ERROR ValueError: Selected backend AttentionBackendEnum.TRITON_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported']
        """
        findings = _diagnose_engine_failure(log, "vllm")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "TRITON_ATTN" in cause or "attention backend" in cause.lower()
        assert "fp8" in suggestion

    def test_vllm_turboquant_workspace_assertion(self):
        from vserve.cli import _diagnose_engine_failure
        log = """
        File "/.../turboquant_attn.py", line 879, in _decode_attention
        File "/.../workspace.py", line 157, in _ensure_workspace_size
        AssertionError: Workspace is locked but allocation from 'turboquant_attn.py:879:_decode_attention' requires 5.61 MB
        """
        findings = _diagnose_engine_failure(log, "vllm")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "turboquant" in cause.lower() or "workspace" in cause.lower()
        # Maintainer-canonical fix per vllm#40807/41403 is cudagraph_mode: NONE
        # (preserves torch.compile fusions). fp8 fallback is also acceptable.
        assert "cudagraph_mode" in suggestion
        assert "vllm#40807" in suggestion or "40807" in suggestion
        assert "fp8" in suggestion

    def test_vllm_chunked_mm_input_disabled(self):
        from vserve.cli import _diagnose_engine_failure
        log = """
        ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (2048). Please increase max_num_batched_tokens.
        """
        findings = _diagnose_engine_failure(log, "vllm")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "multimodal" in cause.lower() or "mm" in cause.lower()
        assert "batched-tokens" in suggestion or "max-num-batched-tokens" in suggestion

    def test_vllm_tool_choice_without_parser(self):
        from vserve.cli import _diagnose_engine_failure
        log = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        findings = _diagnose_engine_failure(log, "vllm")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "tool" in cause.lower()
        assert "--tools" in suggestion or "parser" in suggestion

    def test_vllm_cuda_oom(self):
        from vserve.cli import _diagnose_engine_failure
        log = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.34 GiB"
        findings = _diagnose_engine_failure(log, "vllm")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "GPU memory" in cause or "memory" in cause.lower()
        assert "--slots" in suggestion or "--context" in suggestion or "fp8" in suggestion

    def test_llamacpp_cuda_oom_kv_cache(self):
        from vserve.cli import _diagnose_engine_failure
        log = """
        ggml_backend_cuda_buffer_type_alloc_buffer: allocating 107520.00 MiB on device 0: cudaMalloc failed: out of memory
        alloc_tensor_range: failed to allocate CUDA0 buffer of size 112742891520
        failed to allocate buffer for kv cache
        """
        findings = _diagnose_engine_failure(log, "llamacpp")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "GPU memory" in cause or "KV cache" in cause
        assert "--slots" in suggestion or "--context" in suggestion

    def test_llamacpp_common_fit_params_explicit_ngl(self):
        from vserve.cli import _diagnose_engine_failure
        log = "common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 30, abort"
        findings = _diagnose_engine_failure(log, "llamacpp")
        assert len(findings) == 1
        cause, suggestion = findings[0]
        assert "n-gpu-layers" in cause.lower() or "auto-fitter" in cause
        assert "--override-tensor" in suggestion or "override-tensor" in suggestion or "n-gpu-layers" in suggestion

    def test_llamacpp_segv_surfaced(self):
        from vserve.cli import _diagnose_engine_failure
        log = "llama-cpp.service: Main process exited, code=dumped, status=11/SEGV"
        findings = _diagnose_engine_failure(log, "llamacpp")
        assert len(findings) == 1
        cause, _suggestion = findings[0]
        assert "segfault" in cause.lower() or "SIGSEGV" in cause or "core-dump" in cause.lower()

    def test_no_match_returns_empty(self):
        from vserve.cli import _diagnose_engine_failure
        log = "INFO: Application startup complete.\nINFO: Listening on http://0.0.0.0:8888"
        assert _diagnose_engine_failure(log, "vllm") == []
        assert _diagnose_engine_failure(log, "llamacpp") == []

    def test_backend_dispatch_isolated(self):
        """vLLM patterns must not fire on llamacpp logs and vice versa."""
        from vserve.cli import _diagnose_engine_failure
        vllm_log = "AssertionError: Workspace is locked ... turboquant_attn ..."
        assert len(_diagnose_engine_failure(vllm_log, "llamacpp")) == 0
        llama_log = "cudaMalloc failed: out of memory"
        # vLLM uses different OOM error class — llama.cpp's message shouldn't trigger vLLM diagnosis.
        # (It would only match the vLLM CUDA OOM line if it included `CUDA out of memory`.)
        assert len(_diagnose_engine_failure(llama_log, "vllm")) == 0

    def test_empty_log_returns_empty(self):
        from vserve.cli import _diagnose_engine_failure
        assert _diagnose_engine_failure("", "vllm") == []
        assert _diagnose_engine_failure("", "llamacpp") == []

    def test_multiple_patterns_dedupe(self):
        """A log can contain a cascade (TurboQuant assertion then Shutdown).
        Each distinct cause is emitted once, not repeated per match line."""
        from vserve.cli import _diagnose_engine_failure
        log = """
        AssertionError: Workspace is locked but allocation from 'turboquant_attn.py:879:_decode_attention' requires 5.61 MB
        AssertionError: Workspace is locked but allocation from 'turboquant_attn.py:879:_decode_attention' requires 5.61 MB
        AssertionError: Workspace is locked but allocation from 'turboquant_attn.py:879:_decode_attention' requires 5.61 MB
        """
        findings = _diagnose_engine_failure(log, "vllm")
        assert len(findings) == 1  # not 3
