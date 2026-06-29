from pathlib import Path
from unittest.mock import Mock

from vserve.serve import start_vllm, stop_vllm, is_vllm_running


def _mock_cfg(mocker, service_name="vllm", active_yaml=None):
    mock = Mock()
    mock.service_name = service_name
    mock.active_yaml = active_yaml or Path("/tmp/active.yaml")
    mock.run_dir = mock.active_yaml.parent / "run"
    mock.vllm_root = mock.active_yaml.parent.parent
    mock.cuda_home = Path("/usr/local/cuda")
    mock.gpu_index = 0
    mocker.patch("vserve.serve.cfg", return_value=mock)
    mocker.patch("vserve.serve.find_systemd_unit_path", return_value=None)
    return mock


def test_is_vllm_running_active(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(True, "active", ""))
    assert is_vllm_running() is True


def test_is_vllm_running_inactive(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(True, "inactive", ""))
    assert is_vllm_running() is False


def test_is_vllm_running_activating_is_uncertain(mocker):
    import pytest

    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "activating", ""))
    with pytest.raises(RuntimeError, match="is transitional"):
        is_vllm_running()


def test_stop_vllm(mocker):
    _mock_cfg(mocker)
    mock_run = mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))
    stop_vllm()
    mock_run.assert_called_once_with("stop", non_interactive=False, service_name="vllm")


def test_stop_vllm_noninteractive_uses_noninteractive_systemctl(mocker):
    _mock_cfg(mocker)
    mock_run = mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))
    stop_vllm(non_interactive=True)
    mock_run.assert_called_once_with("stop", non_interactive=True, service_name="vllm")


def test_start_vllm_with_config(mocker, tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model: /opt/vllm/models/test\nport: 8888\n")

    active = tmp_path / "active.yaml"
    _mock_cfg(mocker, active_yaml=active)
    mock_systemctl = mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))

    start_vllm(config_path=config_file)
    assert active.is_symlink()
    mock_systemctl.assert_called_with("start", non_interactive=False)


def test_systemctl_start_noninteractive_uses_sudo_n(mocker):
    _mock_cfg(mocker)
    mock_run = mocker.patch("vserve.systemd_helpers.subprocess.run", return_value=Mock(returncode=0, stdout="", stderr=""))

    ok, _out, _err = __import__("vserve.serve", fromlist=["_systemctl"])._systemctl(
        "start",
        non_interactive=True,
    )

    assert ok is True
    assert mock_run.call_args.args[0][:3] == ["sudo", "-n", "systemctl"]


def test_service_uses_env_file_requires_exact_vserve_env_path(mocker, tmp_path):
    from vserve.serve import _service_uses_env_file

    active = tmp_path / "configs" / "active.yaml"
    active.parent.mkdir(parents=True)
    mock_c = _mock_cfg(mocker, active_yaml=active)
    env_path = mock_c.vllm_root / "configs" / ".env"
    unit = tmp_path / "vllm.service"
    unit.write_text("[Service]\nEnvironmentFile=/etc/default/other.env\n")
    mocker.patch("vserve.serve.find_systemd_unit_path", return_value=unit)

    assert _service_uses_env_file(env_path) is False

    unit.write_text(f"[Service]\nEnvironmentFile=-{env_path}\n")
    assert _service_uses_env_file(env_path) is True


def test_systemctl_rejects_unit_that_does_not_belong_to_vserve(mocker, tmp_path):
    _mock_cfg(mocker, service_name="ssh")
    unit = tmp_path / "ssh.service"
    unit.write_text("[Service]\nExecStart=/usr/sbin/sshd -D\n")
    # 0.6.3: assert_unit_safe (called via systemd_helpers) imports
    # find_systemd_unit_path from vserve.config directly, so we patch at
    # the systemd_helpers namespace.
    mocker.patch("vserve.systemd_helpers.find_systemd_unit_path", return_value=unit)
    run = mocker.patch("vserve.systemd_helpers.subprocess.run")

    ok, _out, err = __import__("vserve.serve", fromlist=["_systemctl"])._systemctl("start")

    assert ok is False
    # Case-insensitive — 0.6.3 systemd_helpers refactor lowercased the
    # backend label since `backend_name` is now passed as `"vllm"`.
    assert "does not look like a vserve vllm unit" in err.lower()
    run.assert_not_called()


def test_start_vllm_writes_active_manifest(mocker, tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model: /opt/vllm/models/test\nport: 8888\n")

    active = tmp_path / "active.yaml"
    mock_c = _mock_cfg(mocker, active_yaml=active)
    mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))

    start_vllm(config_path=config_file)

    from vserve.config import read_active_manifest

    manifest = read_active_manifest(mock_c.run_dir / "active-manifest.json")
    assert manifest is not None
    assert manifest["backend"] == "vllm"
    assert manifest["service_name"] == "vllm"
    assert manifest["config_path"] == str(config_file.resolve())
    assert manifest["status"] == "starting"


def test_start_vllm_rolls_back_active_link_on_systemctl_failure(mocker, tmp_path):
    previous_config = tmp_path / "previous.yaml"
    previous_config.write_text("model: previous\n")
    next_config = tmp_path / "next.yaml"
    next_config.write_text("model: next\n")
    active = tmp_path / "active.yaml"
    active.symlink_to(previous_config)

    mock_c = _mock_cfg(mocker, active_yaml=active)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Unit not found"))

    import pytest

    with pytest.raises(RuntimeError, match="systemctl start failed"):
        start_vllm(next_config)

    assert active.resolve() == previous_config.resolve()

    from vserve.config import read_active_manifest

    manifest = read_active_manifest(mock_c.run_dir / "active-manifest.json")
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert "Unit not found" in manifest["error"]


def test_systemctl_uses_service_name(mocker):
    _mock_cfg(mocker, service_name="my-vllm")
    mock_run = mocker.patch("vserve.systemd_helpers.subprocess.run", return_value=Mock(returncode=0, stdout="active", stderr=""))
    is_vllm_running()
    args = mock_run.call_args[0][0]
    assert "my-vllm" in args


def test_systemctl_uses_timeout(mocker):
    _mock_cfg(mocker)
    mock_run = mocker.patch("vserve.systemd_helpers.subprocess.run", return_value=Mock(returncode=0, stdout="active", stderr=""))
    is_vllm_running()
    assert mock_run.call_args.kwargs["timeout"] == 5


def test_is_vllm_running_probe_error_is_false(mocker):
    import pytest

    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "dbus error"))
    with pytest.raises(RuntimeError, match="systemctl is-active"):
        is_vllm_running()


def test_is_vllm_running_missing_unit_is_false(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Unit vllm.service could not be found."))
    assert is_vllm_running() is False


def test_systemctl_timeout_returns_error(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.systemd_helpers.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd="systemctl", timeout=5))
    ok, out, err = __import__("vserve.serve", fromlist=["_systemctl"])._systemctl("start", timeout=5)
    assert ok is False
    assert out == ""
    assert "timed out" in err


def test_start_vllm_failure(mocker, tmp_path):
    import pytest

    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: test\n")
    active = tmp_path / "active.yaml"
    _mock_cfg(mocker, active_yaml=active)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Unit not found"))
    with pytest.raises(RuntimeError, match="systemctl start failed"):
        start_vllm(config_file)


def test_stop_vllm_failure(mocker):
    import pytest

    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Failed to stop"))
    with pytest.raises(RuntimeError, match="systemctl stop failed"):
        stop_vllm()


# --- 0.7.0 item Q: NVFP4 / FlashInfer envs ---


def test_resolve_quant_envs_nvfp4(mocker, tmp_path):
    """NVFP4-quantized model on sm≥100 → emit FlashInfer envs into the env file."""
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=120))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert envs.get("VLLM_USE_FLASHINFER_MOE_FP4") == "1"
    assert envs.get("VLLM_FLASHINFER_MOE_BACKEND") == "throughput"


def test_resolve_quant_envs_modelopt_alias(mocker, tmp_path):
    """ModelOpt-NVFP4 checkpoints also need FlashInfer envs on sm≥100."""
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=120))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: modelopt\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert envs.get("VLLM_USE_FLASHINFER_MOE_FP4") == "1"


# --- 0.6.3 bug fix 3: hardware-gate FlashInfer FP4 envs on sm≥100 ---


def test_resolve_quant_envs_nvfp4_filtered_on_ada(mocker, tmp_path):
    """0.6.3 bug fix 3: FlashInfer FP4 envs require sm≥100. On Ada (sm89)
    the envs must be filtered out so vLLM falls back to the non-FlashInfer
    path instead of setting flags the kernel won't honor."""
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=89))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert "VLLM_USE_FLASHINFER_MOE_FP4" not in envs
    assert "VLLM_FLASHINFER_MOE_BACKEND" not in envs


def test_resolve_quant_envs_nvfp4_filtered_on_hopper(mocker, tmp_path):
    """Hopper (sm90) is also below the sm≥100 FlashInfer FP4 cutoff."""
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=90))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert "VLLM_USE_FLASHINFER_MOE_FP4" not in envs


def test_resolve_quant_envs_nvfp4_allowed_on_dc_blackwell(mocker, tmp_path):
    """Blackwell DC (B200, sm100) is exactly at the cutoff — must allow."""
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=100))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert envs.get("VLLM_USE_FLASHINFER_MOE_FP4") == "1"


def test_resolve_quant_envs_nvfp4_filtered_when_compute_cap_unknown(mocker, tmp_path):
    """Conservative: when compute_cap is None (older nvidia-smi or detection
    failure), filter the FP4 envs to avoid silent kernel-version mismatch."""
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=None))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert "VLLM_USE_FLASHINFER_MOE_FP4" not in envs


def test_resolve_quant_envs_nvfp4_filtered_when_gpu_info_raises(mocker, tmp_path):
    """When get_gpu_info raises (e.g. no nvidia-smi available), filter
    conservatively — better to lose a kernel optimization than crash."""
    mocker.patch("vserve.gpu.get_gpu_info", side_effect=RuntimeError("nvidia-smi missing"))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert "VLLM_USE_FLASHINFER_MOE_FP4" not in envs


def test_resolve_quant_envs_no_quant_returns_empty(tmp_path):
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model: /tmp/x\n")
    assert _resolve_quant_envs(cfg_path) == {}


def test_resolve_quant_envs_fp8_returns_empty(tmp_path):
    """FP8 doesn't need any per-quant env vars; the model just emits the
    --quantization flag."""
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: fp8\nmodel: /tmp/x\n")
    assert _resolve_quant_envs(cfg_path) == {}


def test_upsert_env_file_includes_extra(mocker, tmp_path):
    """The extra-envs hook merges quant envs into the base env-file write."""
    from vserve.serve import _upsert_env_file
    mock_cfg = Mock()
    mock_cfg.vllm_root = tmp_path
    mock_cfg.cuda_home = Path("/usr/local/cuda")
    mock_cfg.gpu_index = 0
    mocker.patch("vserve.serve.cfg", return_value=mock_cfg)
    env_path = _upsert_env_file({"VLLM_USE_FLASHINFER_MOE_FP4": "1"})
    text = env_path.read_text()
    assert "CUDA_HOME=" in text
    assert "VLLM_USE_FLASHINFER_MOE_FP4=1" in text


def test_recommend_quant_blackwell_dense():
    from vserve.models import recommend_quant_for_arch
    assert recommend_quant_for_arch(sm=120, is_moe=False) == "nvfp4"


def test_recommend_quant_blackwell_moe():
    from vserve.models import recommend_quant_for_arch
    assert recommend_quant_for_arch(sm=120, is_moe=True) == "mxfp4"


def test_recommend_quant_hopper_fp8():
    from vserve.models import recommend_quant_for_arch
    assert recommend_quant_for_arch(sm=90, is_moe=False) == "fp8"


def test_recommend_quant_filters_by_available():
    from vserve.models import recommend_quant_for_arch
    # NVFP4 not in available → fall back to FP8
    assert recommend_quant_for_arch(sm=120, is_moe=False, available_quants={"fp8"}) == "fp8"


def test_quant_flags_includes_new_entries():
    from vserve.models import QUANT_FLAGS
    for key in ("nvfp4", "modelopt", "mxfp4", "mxfp4_moe", "bitsandbytes", "gguf"):
        assert key in QUANT_FLAGS, f"Missing quant flag: {key}"


def test_quant_env_vars_includes_nvfp4_modelopt():
    from vserve.models import QUANT_ENV_VARS
    assert "VLLM_USE_FLASHINFER_MOE_FP4" in QUANT_ENV_VARS["nvfp4"]
    assert "VLLM_USE_FLASHINFER_MOE_FP4" in QUANT_ENV_VARS["modelopt"]


# --- 0.6.3: vLLM 0.22 deprecates the FlashInfer MoE env vars ---


def test_resolve_quant_envs_dropped_on_022(mocker, tmp_path):
    """vLLM 0.22 wraps VLLM_USE_FLASHINFER_MOE_* in deprecated_env()
    (removal targeted 0.23) and its moe-backend=auto default is
    hardware-aware — on a known >=0.22 runtime, emit nothing."""
    from packaging.version import Version
    mocker.patch("vserve.serve._runtime_vllm_version", return_value=Version("0.22.0"))
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=120))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    assert _resolve_quant_envs(cfg_path) == {}


def test_resolve_quant_envs_kept_below_022(mocker, tmp_path):
    """Pre-0.22 runtimes still need the env vars (no --moe-backend flag)."""
    from packaging.version import Version
    mocker.patch("vserve.serve._runtime_vllm_version", return_value=Version("0.21.0"))
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=120))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert envs.get("VLLM_USE_FLASHINFER_MOE_FP4") == "1"
    assert envs.get("VLLM_FLASHINFER_MOE_BACKEND") == "throughput"


def test_resolve_quant_envs_kept_on_unknown_version(mocker, tmp_path):
    """Unknown runtime version must behave pre-0.22: a broken probe may
    cause FutureWarnings on 0.22, never the 0.21 slow path."""
    mocker.patch("vserve.serve._runtime_vllm_version", return_value=None)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=Mock(compute_cap=120))
    from vserve.serve import _resolve_quant_envs
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("quantization: nvfp4\nmodel: /tmp/x\n")
    envs = _resolve_quant_envs(cfg_path)
    assert envs.get("VLLM_USE_FLASHINFER_MOE_FP4") == "1"
