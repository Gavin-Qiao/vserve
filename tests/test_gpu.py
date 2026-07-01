import pytest
from unittest.mock import MagicMock, call, patch

from vserve.gpu import (
    get_gpu_info,
    compute_gpu_memory_utilization,
    resolve_gpu_memory_utilization,
    get_fan_count,
    get_fan_speed,
    set_fan_speed,
    restore_fan_auto,
)


def test_parse_gpu_info(mocker):
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA RTX PRO 5000 Blackwell, 48935, 0, 1024, 590.48.01",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="13.1")
    info = get_gpu_info()
    assert info.name == "NVIDIA RTX PRO 5000 Blackwell"
    assert info.vram_total_mb == 48935
    assert abs(info.vram_total_gb - 47.8) < 0.1
    assert info.driver == "590.48.01"
    assert info.cuda == "13.1"


def test_parse_gpu_info_uses_selected_row(mocker):
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value=(
            "GPU 0, 24576, 0, 1024, 590.48.01\n"
            "GPU 1, 49152, 0, 2048, 590.48.01"
        ),
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="13.1")
    cfg = MagicMock(gpu_index=1)

    info = get_gpu_info(config=cfg)

    assert info.name == "GPU 1"
    assert info.index == 1
    assert info.vram_total_mb == 49152
    assert info.vram_used_mb == 2048


def test_compute_gpu_memory_utilization():
    util = compute_gpu_memory_utilization(vram_total_gb=47.8, overhead_gb=2.0)
    assert abs(util - 0.958) < 0.001


def test_compute_gpu_memory_utilization_small_gpu():
    util = compute_gpu_memory_utilization(vram_total_gb=8.0, overhead_gb=2.0)
    assert abs(util - 0.75) < 0.001


def test_compute_gpu_memory_utilization_zero_vram():
    with pytest.raises(ValueError, match="VRAM total"):
        compute_gpu_memory_utilization(0.0)
    with pytest.raises(ValueError, match="VRAM total"):
        compute_gpu_memory_utilization(-1.0)


def test_resolve_gpu_memory_utilization_rejects_auto_policy_below_floor():
    cfg = MagicMock(gpu_memory_utilization=None, gpu_overhead_gb=7.0)
    with pytest.raises(ValueError, match="gpu.overhead_gb"):
        resolve_gpu_memory_utilization(8.0, config=cfg)


def test_resolve_gpu_memory_utilization_uses_one_policy():
    cfg = MagicMock(gpu_memory_utilization=None, gpu_overhead_gb=None)
    assert abs(resolve_gpu_memory_utilization(48.0, config=cfg) - compute_gpu_memory_utilization(48.0)) < 0.001
    assert resolve_gpu_memory_utilization(48.0, requested=0.82, config=cfg) == 0.82

    cfg.gpu_memory_utilization = 0.91
    assert resolve_gpu_memory_utilization(48.0, config=cfg) == 0.91

    cfg.gpu_memory_utilization = None
    cfg.gpu_overhead_gb = 2.0
    assert abs(resolve_gpu_memory_utilization(48.0, config=cfg) - compute_gpu_memory_utilization(48.0, 2.0)) < 0.001


def test_resolve_gpu_memory_utilization_rejects_invalid_persisted_values():
    cfg = MagicMock(gpu_memory_utilization=1.2, gpu_overhead_gb=None)
    with pytest.raises(ValueError, match="gpu.memory_utilization"):
        resolve_gpu_memory_utilization(48.0, config=cfg)

    cfg.gpu_memory_utilization = None
    cfg.gpu_overhead_gb = 100.0
    with pytest.raises(ValueError, match="between 0.5 and 0.99"):
        resolve_gpu_memory_utilization(48.0, config=cfg)


def _mock_pynvml():
    """Create a mock pynvml module for testing."""
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0
    mock.nvmlDeviceSetDefaultFanSpeed_v2 = None
    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex.return_value = handle
    return mock, handle


def test_get_fan_speed():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetFanSpeed_v2.return_value = 80
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        speed = get_fan_speed()
    assert speed == 80
    mock_nvml.nvmlInit.assert_called_once()
    mock_nvml.nvmlShutdown.assert_called_once()


def test_get_fan_count():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetNumFans.return_value = 2
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        assert get_fan_count() == 2


def test_set_fan_speed_valid():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetNumFans.return_value = 2
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        set_fan_speed(80)
    assert mock_nvml.nvmlDeviceSetFanSpeed_v2.call_args_list == [
        call(handle, 0, 80),
        call(handle, 1, 80),
    ]


def test_set_fan_speed_rejects_out_of_range():
    with pytest.raises(ValueError, match="30-100"):
        set_fan_speed(10)
    with pytest.raises(ValueError, match="30-100"):
        set_fan_speed(101)


def test_restore_fan_auto():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetNumFans.return_value = 2
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        restore_fan_auto()
    assert mock_nvml.nvmlDeviceSetFanControlPolicy.call_args_list == [
        call(handle, 0, 0),
        call(handle, 1, 0),
    ]


def test_restore_fan_auto_prefers_default_api():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetNumFans.return_value = 2
    mock_nvml.nvmlDeviceSetDefaultFanSpeed_v2 = MagicMock()
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        restore_fan_auto()
    assert mock_nvml.nvmlDeviceSetDefaultFanSpeed_v2.call_args_list == [
        call(handle, 0),
        call(handle, 1),
    ]
    mock_nvml.nvmlDeviceSetFanControlPolicy.assert_not_called()


def test_restore_fan_auto_falls_back_when_default_api_fails():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetNumFans.return_value = 2
    mock_nvml.nvmlDeviceSetDefaultFanSpeed_v2 = MagicMock(side_effect=RuntimeError("unsupported"))
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        restore_fan_auto()
    assert mock_nvml.nvmlDeviceSetFanControlPolicy.call_args_list == [
        call(handle, 0, 0),
        call(handle, 1, 0),
    ]


# --- 0.6.1 item R: compute capability parsing ---


def test_compute_cap_parsed_from_nvidia_smi_blackwell(mocker):
    """Blackwell RTX (PRO 5000) reports compute_cap 12.0 → packed sm120."""
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA RTX PRO 5000 Blackwell, 48935, 0, 1024, 595.58.03, 12.0",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="13.0")
    info = get_gpu_info()
    assert info.compute_cap == 120


def test_compute_cap_parsed_for_hopper(mocker):
    """H100 reports compute_cap 9.0 → packed sm90."""
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA H100 80GB, 81920, 0, 0, 535.171.04, 9.0",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="12.8")
    info = get_gpu_info()
    assert info.compute_cap == 90


def test_compute_cap_parsed_for_ada(mocker):
    """4090 reports compute_cap 8.9 → packed sm89."""
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA GeForce RTX 4090, 24564, 0, 1024, 550.0.0, 8.9",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="12.8")
    info = get_gpu_info()
    assert info.compute_cap == 89


def test_compute_cap_falls_back_to_none_when_missing(mocker):
    """Older nvidia-smi without compute_cap column → field is None."""
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA RTX PRO 5000, 48935, 0, 1024, 595.58.03",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="13.0")
    info = get_gpu_info()
    assert info.compute_cap is None


def test_compute_cap_falls_back_to_none_when_garbled(mocker):
    """Unparseable compute_cap string → None (silent fallthrough)."""
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA RTX PRO 5000, 48935, 0, 1024, 595.58.03, garbage",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="13.0")
    info = get_gpu_info()
    assert info.compute_cap is None


def _headroom_model(*, is_moe=False, is_multimodal=False):
    from vserve.models import ModelInfo
    from pathlib import Path
    return ModelInfo(
        path=Path("/fake"),
        provider="p",
        model_name="m",
        architecture="Arch",
        model_type="t",
        quant_method=None,
        max_position_embeddings=32768,
        is_moe=is_moe,
        model_size_gb=20.0,
        is_multimodal=is_multimodal,
    )


def test_resolve_gpu_util_auto_no_model_matches_base_overhead():
    # model=None preserves the prior behaviour: base 3.5 GB overhead, no extra.
    assert resolve_gpu_memory_utilization(47.8, model=None) == pytest.approx((47.8 - 3.5) / 47.8)


def test_resolve_gpu_util_moe_reserves_extra_headroom():
    base = resolve_gpu_memory_utilization(47.8, model=None)
    moe = resolve_gpu_memory_utilization(47.8, model=_headroom_model(is_moe=True))
    assert moe < base
    assert moe == pytest.approx((47.8 - 5.0) / 47.8)  # 3.5 base + 1.5 MoE


def test_resolve_gpu_util_multimodal_reserves_most():
    moe = resolve_gpu_memory_utilization(47.8, model=_headroom_model(is_moe=True))
    mm = resolve_gpu_memory_utilization(47.8, model=_headroom_model(is_multimodal=True))
    mm_moe = resolve_gpu_memory_utilization(
        47.8, model=_headroom_model(is_moe=True, is_multimodal=True)
    )
    assert mm == pytest.approx((47.8 - 6.0) / 47.8)      # 3.5 + 2.5
    assert mm_moe == pytest.approx((47.8 - 7.5) / 47.8)  # 3.5 + 2.5 + 1.5
    assert mm_moe < mm and mm_moe < moe


def test_resolve_gpu_util_explicit_request_ignores_model_headroom():
    # An explicit --gpu-util wins as-is; model headroom is not applied.
    got = resolve_gpu_memory_utilization(
        47.8, requested=0.95, model=_headroom_model(is_moe=True, is_multimodal=True)
    )
    assert got == pytest.approx(0.95)


def test_resolve_gpu_util_configured_util_ignores_model_headroom():
    cfg = MagicMock(gpu_memory_utilization=0.88, gpu_overhead_gb=None)
    got = resolve_gpu_memory_utilization(47.8, config=cfg, model=_headroom_model(is_moe=True))
    assert got == pytest.approx(0.88)
