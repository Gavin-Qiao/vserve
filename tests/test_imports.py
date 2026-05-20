"""Smoke tests — verify every public import path works at runtime.

This file exists because a lazy-resolution bug in fan.py broke `vserve fan`
but passed all unit tests. Every import pattern used by CLI code must
be tested here.
"""


def test_config_compat_constants():
    """Module-level __getattr__ provides backward-compatible Path constants."""
    from vserve.config import (  # noqa: F401
        MODELS_DIR, CONFIGS_DIR, BENCHMARKS_DIR, LOGS_DIR,
        VLLM_BIN, VLLM_PYTHON, VLLM_ROOT,
    )
    from pathlib import Path
    assert isinstance(MODELS_DIR, Path)
    assert isinstance(VLLM_BIN, Path)


def test_config_functions():
    from vserve.config import (  # noqa: F401
        cfg, load_config, save_config, reset_config,
        limits_path, profile_path, active_yaml_path,
        read_limits, write_limits,
        read_profile_yaml, write_profile_yaml,
        VserveConfig, CONFIG_FILE,
        _discover_vllm_root, _discover_cuda_home, _discover_service,
        _discover_port, _build_config,
    )
    assert callable(cfg)


def test_fan_module_imports():
    """The exact import pattern cli.py uses for fan."""
    import vserve.fan as _fan
    from vserve.fan import read_state, compute_fan_speed  # noqa: F401
    from vserve.fan import (  # noqa: F401
        EMERGENCY_TEMP, MIN_FAN, HYSTERESIS, MAX_RAMP_PER_CYCLE,
        MAX_CONSECUTIVE_FAILURES, FAILURE_RESET_THRESHOLD,
        POLL_INTERVAL, SUBPROCESS_TIMEOUT,
    )
    # These must be accessible after _resolve_paths
    _fan._resolve_paths()
    from pathlib import Path
    assert isinstance(_fan.PID_PATH, Path)
    assert isinstance(_fan.STATE_PATH, Path)
    assert isinstance(_fan.LOG_PATH, Path)


def test_fan_resolve_paths_idempotent():
    """Calling _resolve_paths twice must not error."""
    import vserve.fan as _fan
    _fan._paths_resolved = False
    _fan._resolve_paths()
    first = _fan.PID_PATH
    _fan._resolve_paths()
    assert _fan.PID_PATH == first


def test_gpu_module_imports():
    from vserve.gpu import (  # noqa: F401
        get_gpu_info, compute_gpu_memory_utilization,
        get_fan_count, get_fan_speed, set_fan_speed, restore_fan_auto, GpuInfo,
    )
    assert callable(get_gpu_info)
    assert callable(set_fan_speed)


def test_serve_module_imports():
    from vserve.serve import start_vllm, stop_vllm, is_vllm_running  # noqa: F401
    assert callable(start_vllm)
    assert callable(is_vllm_running)


def test_models_module_imports():
    from vserve.models import scan_models, fuzzy_match, detect_model, ModelInfo  # noqa: F401
    assert callable(scan_models)
    assert callable(fuzzy_match)


def test_cli_app_exists():
    from vserve.cli import app
    assert app is not None
