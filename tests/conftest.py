import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """Create a fake model directory with config.json."""
    model_dir = tmp_path / "models" / "testprovider" / "TestModel-7B-FP8"
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["TestForCausalLM"],
        "model_type": "test",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "max_position_embeddings": 131072,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)

    return model_dir


@pytest.fixture
def fake_moe_model_dir(tmp_path: Path) -> Path:
    """Create a fake MoE model directory."""
    model_dir = tmp_path / "models" / "testprovider" / "TestMoE-35B-FP8"
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["TestMoeForCausalLM"],
        "model_type": "test_moe",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "max_position_embeddings": 262144,
            "num_experts": 256,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 64,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 2048)

    return model_dir


@pytest.fixture
def fake_limits() -> dict:
    """Sample limits.json data."""
    return {
        "model_path": "/opt/vllm/models/testprovider/TestModel-7B-FP8",
        "calculated_at": "2026-03-27T10:00:00",
        "vram_total_gb": 48.0,
        "gpu_memory_utilization": 0.9270833333333334,
        "available_kv_gb": 37.3,  # 48.0 * 0.927 - 7.2
        "model_size_gb": 7.2,
        "quant_method": "fp8",
        "architecture": "TestForCausalLM",
        "is_moe": False,
        "max_position_embeddings": 131072,
        "limits": {
            "4096": {"auto": 64, "fp8": 128},
            "8192": {"auto": 32, "fp8": 64},
            "16384": {"auto": 16, "fp8": 32},
            "32768": {"auto": 8, "fp8": 16},
            "65536": {"auto": None, "fp8": 8},
            "131072": {"auto": None, "fp8": None},
        },
    }


@pytest.fixture
def fake_gguf_model_dir(tmp_path: Path) -> Path:
    """Create a fake GGUF model directory."""
    model_dir = tmp_path / "models" / "bartowski" / "TestModel-8B-GGUF"
    model_dir.mkdir(parents=True)

    (model_dir / "TestModel-8B-Q4_K_M.gguf").write_bytes(b"\0" * 4096)

    tok_config = {"chat_template": "{% if tools %}tools{% endif %}{{ messages }}"}
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tok_config))

    return model_dir


@pytest.fixture
def fake_embedding_model_dir(tmp_path: Path) -> Path:
    """Create a fake GGUF embedding model directory."""
    model_dir = tmp_path / "models" / "nomic-ai" / "nomic-embed-text-v1.5-GGUF"
    model_dir.mkdir(parents=True)

    (model_dir / "nomic-embed-text-v1.5-Q8_0.gguf").write_bytes(b"\0" * 2048)

    return model_dir


@pytest.fixture
def models_root(tmp_path: Path, fake_model_dir: Path) -> Path:
    """Return the models root containing fake_model_dir."""
    return tmp_path / "models"


@pytest.fixture(autouse=True)
def _isolate_lock_dir(tmp_path, monkeypatch):
    """Point vserve.lock at a per-test directory.

    The suite previously exercised the REAL /run/lock/vserve — a pytest
    run on a workstation with a live backend could delete the operator's
    session marker (found in the 0.6.3 on-GPU sweep: the pre-commit hook's
    test run cleared the session and the next `run --replace` refused with
    "session owner unknown"). Tests that need specific paths re-patch on
    top of this.
    """
    lock_dir = tmp_path / "lock" / "vserve"
    # _ensure_lock_dir() mkdirs LOCK_DIR itself but not its parents.
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("vserve.lock.LOCK_DIR", lock_dir)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", lock_dir / "vserve-session.json")


@pytest.fixture(autouse=True)
def _pin_unknown_vllm_runtime_version(monkeypatch):
    """Pin the vLLM runtime-version probe seams to None for every test.

    The 0.22 emission gates probe the configured vLLM venv. On the
    workstation that returns a real version; on GitHub CI there is no
    venv at all. Pinning to None (= conservative pre-0.22 behavior)
    makes the suite behave identically everywhere. Tests that exercise
    ≥0.22 behavior re-patch the same targets explicitly.

    Only the consumer-module wrappers are pinned — NOT
    ``vserve.runtime.installed_vllm_version`` itself, which has its own
    direct unit tests. ``raising=False`` tolerates the wrappers landing
    in later commits of the 0.6.3 series.
    """
    monkeypatch.setattr(
        "vserve.backends.vllm._runtime_vllm_version",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        "vserve.serve._runtime_vllm_version",
        lambda: None,
        raising=False,
    )
