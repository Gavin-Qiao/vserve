"""CLI-level tests for the MTP toggle on `vserve run` (0.6.8).

`--mtp / --no-mtp / --mtp-tokens` resolve into `choices["spec"]` for the
backend. These tests verify the flag → choices plumbing plus the two guard
rails (runtime version gate, no-MTP-weights refusal); the SpecConfig
semantics themselves are covered by `tests/test_recipes_spec_decode.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from packaging.version import Version
from typer.testing import CliRunner

from _helpers import strip_ansi
from vserve.cli import app

runner = CliRunner()


@pytest.fixture
def fake_mtp_model_dir(tmp_path: Path) -> Path:
    """A vLLM-servable checkpoint with in-checkpoint MTP draft layers
    (Qwen3.5/3.6 shape: keys on text_config)."""
    model_dir = tmp_path / "models" / "testprovider" / "TestMtp-7B-FP8"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "max_position_embeddings": 131072,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        },
    }))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)
    return model_dir


def _setup_vllm_yes_path(mocker, model_dir: Path, tmp_path: Path, *, vllm_version: str | None = "0.24.0"):
    """Wire up a scripted `vserve run` happy path against a vLLM model.

    Mirrors `_setup_llamacpp_yes_path` in test_run_llamacpp_flags.py.
    ``vllm_version=None`` leaves the conftest autouse pin (unknown runtime)
    in place.
    """
    from vserve.config import LIMITS_SCHEMA_VERSION
    from vserve.models import detect_model

    model = detect_model(model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.can_serve.return_value = True
    backend.compatibility.return_value = mocker.Mock(supported=True)
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    backend.available_tool_parsers.return_value = set()
    backend.available_reasoning_parsers.return_value = set()
    backend.root_dir = tmp_path

    mocker.patch("vserve.cli._all_models", return_value=[model])
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.cli.read_limits", return_value={
        "schema_version": LIMITS_SCHEMA_VERSION,
        "backend": "vllm",
        "limits": {"8192": {"auto": 8, "fp8": 16}},
    })
    mocker.patch(
        "vserve.gpu.get_gpu_info",
        return_value=mocker.Mock(name="GPU", vram_total_gb=48.0),
    )
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "profile.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._launch_backend")
    if vllm_version is not None:
        mocker.patch(
            "vserve.cli._runtime_vllm_version",
            return_value=Version(vllm_version),
        )
    return backend


# Explicit context/kv/slots skip the tuned-defaults derivation so the test
# exercises only the MTP plumbing.
_EXPLICIT = ["--context", "8192", "--slots", "4", "--kv-cache-dtype", "auto"]


class TestRunHelpShowsMtpFlags:
    @pytest.mark.parametrize("flag", ["--mtp", "--no-mtp", "--mtp-tokens"])
    def test_help_lists_flag(self, flag):
        result = runner.invoke(app, ["run", "--help"])
        assert flag in strip_ansi(result.stdout), f"{flag} missing from --help output"


class TestMtpFlagThreadsIntoChoices:
    def test_mtp_on_threads_spec_config(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--mtp"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec is not None
        assert spec.method == "mtp"
        assert spec.draft_model_path is None
        assert spec.n_max == 3

    def test_mtp_tokens_implies_mtp_and_sets_depth(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--mtp-tokens", "6"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec is not None
        assert spec.n_max == 6

    def test_no_mtp_leaves_spec_unset(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--no-mtp"])
        assert result.exit_code == 0, result.stdout
        assert backend.build_config.call_args.args[1]["spec"] is None

    def test_default_is_off(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--yes"])
        assert result.exit_code == 0, result.stdout
        assert backend.build_config.call_args.args[1]["spec"] is None


class TestMtpGuards:
    @staticmethod
    def _flat(result) -> str:
        """Console output with Rich line-wrapping collapsed."""
        return " ".join(strip_ansi(result.stdout).split())

    def test_runtime_below_024_refused(self, mocker, fake_mtp_model_dir, tmp_path):
        _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path, vllm_version="0.23.0")
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--mtp"])
        assert result.exit_code == 1
        assert "requires vLLM >=" in self._flat(result)

    def test_unknown_runtime_version_refused(self, mocker, fake_mtp_model_dir, tmp_path):
        # vllm_version=None keeps the conftest autouse pin: version unknown.
        _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path, vllm_version=None)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--mtp"])
        assert result.exit_code == 1
        assert "could not be determined" in self._flat(result)

    def test_model_without_mtp_weights_refused(self, mocker, fake_model_dir, tmp_path):
        _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--mtp"])
        assert result.exit_code == 1
        assert "no MTP weights" in self._flat(result)

    def test_runtime_info_version_satisfies_gate_directly(self, mocker, fake_mtp_model_dir, tmp_path):
        """When the runtime probe returned a real version string, the gate
        reads it without touching the module seam."""
        from vserve.cli import _vllm_mtp_gate_reason
        info = mocker.Mock(vllm_version="0.24.0")
        assert _vllm_mtp_gate_reason(info) is None
        old = mocker.Mock(vllm_version="0.22.1")
        assert "requires vLLM >=" in _vllm_mtp_gate_reason(old)


class TestSpecFlag:
    """0.6.8: `--spec auto|off|ngram|mtp|draft` — the auto-pick recipe wired
    into `vserve run`. --mtp/--no-mtp stay as shorthand."""

    def test_spec_mtp_equals_mtp_flag(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--spec", "mtp"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec.method == "mtp" and spec.n_max == 3

    def test_spec_off_forces_none_even_on_mtp_model(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--spec", "off"])
        assert result.exit_code == 0, result.stdout
        assert backend.build_config.call_args.args[1]["spec"] is None

    def test_spec_ngram_on_vllm(self, mocker, fake_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--spec", "ngram"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec.method == "ngram" and spec.n_max == 5

    def test_spec_auto_picks_mtp_on_mtp_model(self, mocker, fake_mtp_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--spec", "auto"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec.method == "mtp"
        assert spec.draft_model_path is None

    def test_spec_auto_degrades_to_ngram_on_unknown_runtime(self, mocker, fake_mtp_model_dir, tmp_path):
        """auto is best-effort: an un-gateable runtime downgrades mtp→ngram
        instead of blocking the launch (explicit --mtp still refuses)."""
        backend = _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path, vllm_version=None)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--spec", "auto"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec.method == "ngram"

    def test_spec_auto_picks_ngram_for_plain_dense_model(self, mocker, fake_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--spec", "auto"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        # TestForCausalLM is unknown to the arch table → ngram fallback on vLLM.
        assert spec.method == "ngram"

    def test_spec_bogus_value_rejected(self, mocker, fake_model_dir, tmp_path):
        _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--spec", "eagle9"])
        assert result.exit_code == 1
        assert "Unknown --spec value" in " ".join(strip_ansi(result.stdout).split())

    def test_spec_conflicts_with_mtp_flag(self, mocker, fake_mtp_model_dir, tmp_path):
        _setup_vllm_yes_path(mocker, fake_mtp_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmtp", *_EXPLICIT, "--spec", "ngram", "--no-mtp"])
        assert result.exit_code == 1
        assert "conflicts" in strip_ansi(result.stdout)

    def test_mtp_tokens_rejected_for_non_mtp_spec(self, mocker, fake_model_dir, tmp_path):
        _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--spec", "ngram", "--mtp-tokens", "3"])
        assert result.exit_code == 1
        assert "--mtp-tokens only applies" in " ".join(strip_ansi(result.stdout).split())


class TestScriptedTriggerFlags:
    """0.6.8 fix: flags the wizard doesn't consume (--thinking, --moe-backend,
    --spec) force the scripted path instead of being silently dropped."""

    def test_thinking_alone_triggers_scripted_run(self, mocker, fake_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--no-thinking"])
        assert result.exit_code == 0, result.stdout
        assert backend.build_config.call_args.args[1]["thinking"] is False

    def test_moe_backend_alone_triggers_scripted_run(self, mocker, fake_model_dir, tmp_path):
        backend = _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(
            app, ["run", "testmodel", *_EXPLICIT, "--moe-backend", "flashinfer_trtllm"],
        )
        assert result.exit_code == 0, result.stdout
        assert backend.build_config.call_args.args[1]["moe_backend"] == "flashinfer_trtllm"


class TestDraftDiscoveryHelpers:
    def test_params_b_from_name(self):
        from vserve.cli import _params_b_from_name
        assert _params_b_from_name("Qwen3.5-0.8B") == 0.8
        assert _params_b_from_name("TestModel-7B-FP8") == 7.0
        # A3B names carry active + total; the max (total) is the size proxy.
        assert _params_b_from_name("Qwen3.6-35B-A3B-NVFP4") == 35.0
        assert _params_b_from_name("NoSizeHere") is None

    def test_special_token_ids_read_text_config_and_list_eos(self, tmp_path):
        import json
        from vserve.cli import _model_special_token_ids
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "text_config": {"bos_token_id": 11, "eos_token_id": [22, 23]},
        }))
        assert _model_special_token_ids(d) == (11, 22)

    def test_spec_draft_picks_compatible_local_model(self, mocker, tmp_path):
        """End-to-end --spec draft: a same-family ≤1.5B local model with
        matching BOS/EOS becomes the draft."""
        import json
        from vserve.models import detect_model

        def mk(name, arch, extra=None):
            d = tmp_path / "models" / "prov" / name
            d.mkdir(parents=True)
            cfg = {
                "architectures": [arch],
                "model_type": "llama",
                "bos_token_id": 1, "eos_token_id": 2,
                "max_position_embeddings": 8192,
                "num_key_value_heads": 4, "num_attention_heads": 16,
                "hidden_size": 1024, "num_hidden_layers": 8,
            }
            cfg.update(extra or {})
            (d / "config.json").write_text(json.dumps(cfg))
            (d / "model.safetensors").write_bytes(b"\0" * 64)
            return d

        target_dir = mk("BigLlama-8B", "LlamaForCausalLM")
        draft_dir = mk("TinyLlama-1B", "LlamaForCausalLM")
        target = detect_model(target_dir)
        draft = detect_model(draft_dir)

        backend = _setup_vllm_yes_path(mocker, target_dir, tmp_path)
        mocker.patch("vserve.cli._all_models", return_value=[target, draft])
        result = runner.invoke(app, ["run", "bigllama", *_EXPLICIT, "--spec", "draft"])
        assert result.exit_code == 0, result.stdout
        spec = backend.build_config.call_args.args[1]["spec"]
        assert spec.method == "draft"
        assert spec.draft_model_path == draft.path

    def test_spec_draft_refuses_when_no_candidate(self, mocker, fake_model_dir, tmp_path):
        _setup_vllm_yes_path(mocker, fake_model_dir, tmp_path)
        result = runner.invoke(app, ["run", "testmodel", *_EXPLICIT, "--spec", "draft"])
        assert result.exit_code == 1
        assert "No compatible draft model" in " ".join(strip_ansi(result.stdout).split())


def _run_vllm_wizard(mocker, model_dir, tmp_path, *, confirms: list[bool], vllm_version: str | None = "0.24.0"):
    """Drive _custom_config_vllm directly with the pick/prompt/confirm seams
    mocked. Returns the backend mock (build_config captures choices)."""
    from vserve.models import detect_model
    from vserve.cli import _custom_config_vllm

    model = detect_model(model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    backend.detect_tools.return_value = {}
    backend.runtime_info = None  # not callable → wizard skips the live probe

    mocker.patch("vserve.runtime.build_tuning_fingerprint", return_value={})
    mocker.patch("vserve.config.limits_cache_matches", return_value=True)
    mocker.patch("vserve.config.read_limits", return_value={
        "backend": "vllm",
        "limits": {"8192": {"auto": 8}},
    })
    mocker.patch(
        "vserve.gpu.get_gpu_info",
        return_value=mocker.Mock(name="GPU", vram_total_gb=48.0),
    )
    mocker.patch("vserve.cli._pick", side_effect=[0, 0, 0])  # ctx, kv dtype, batched tokens
    mocker.patch("typer.prompt", return_value="4")           # slots
    mocker.patch("typer.confirm", side_effect=confirms)
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "custom.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli.console.clear")
    if vllm_version is not None:
        mocker.patch("vserve.cli._runtime_vllm_version", return_value=Version(vllm_version))
    _custom_config_vllm(model, backend)
    return backend


@pytest.fixture
def fake_multimodal_model_dir(tmp_path: Path) -> Path:
    """A vLLM-servable multimodal checkpoint (vision tower, no MTP)."""
    model_dir = tmp_path / "models" / "testprovider" / "TestVision-9B-FP8"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["TestVLForConditionalGeneration"],
        "model_type": "test_vl",
        "quantization_config": {"quant_method": "fp8"},
        "vision_config": {"hidden_size": 1024},
        "text_config": {
            "max_position_embeddings": 131072,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        },
    }))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)
    return model_dir


class TestWizardSelections:
    """0.6.8: the interactive wizard offers MTP (when the checkpoint has draft
    layers) and text-only serving (when the checkpoint is multimodal)."""

    def test_wizard_text_only_yes_sets_language_model_only(self, mocker, fake_multimodal_model_dir, tmp_path):
        # confirms: [text-only? YES, start? YES]
        backend = _run_vllm_wizard(mocker, fake_multimodal_model_dir, tmp_path, confirms=[True, True])
        choices = backend.build_config.call_args.args[1]
        assert choices["language_model_only"] is True

    def test_wizard_text_only_no_keeps_full_multimodal(self, mocker, fake_multimodal_model_dir, tmp_path):
        backend = _run_vllm_wizard(mocker, fake_multimodal_model_dir, tmp_path, confirms=[False, True])
        choices = backend.build_config.call_args.args[1]
        assert choices["language_model_only"] is False

    def test_wizard_offers_mtp_and_yes_sets_spec(self, mocker, fake_mtp_model_dir, tmp_path):
        # confirms: [MTP? YES, start? YES] (not multimodal → no text-only question)
        backend = _run_vllm_wizard(mocker, fake_mtp_model_dir, tmp_path, confirms=[True, True])
        choices = backend.build_config.call_args.args[1]
        assert choices["spec"] is not None
        assert choices["spec"].method == "mtp"

    def test_wizard_mtp_not_offered_on_old_runtime(self, mocker, fake_mtp_model_dir, tmp_path):
        # Runtime gate fails → wizard never asks about MTP; confirms: [start? YES]
        backend = _run_vllm_wizard(
            mocker, fake_mtp_model_dir, tmp_path, confirms=[True], vllm_version="0.23.0",
        )
        choices = backend.build_config.call_args.args[1]
        assert choices["spec"] is None
