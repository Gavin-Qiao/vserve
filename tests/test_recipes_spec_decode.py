"""Tests for the speculative-decoding recipe picker (item M)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_config(model_dir: Path, config: dict) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


class TestPickSpecConfig:
    def test_blocklist_returns_none_for_a3b(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        out = pick_spec_config(
            architecture="Qwen3A3BForCausalLM", backend="llamacpp",
            model_path=tmp_path, available_drafts=[],
        )
        assert out is None

    def test_blocklist_can_be_forced(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        # Force=True allows past blocklist but falls back to ngram on vLLM
        # (no draft, no MTP variant).
        out = pick_spec_config(
            architecture="Qwen3A3BForCausalLM", backend="vllm",
            model_path=tmp_path, available_drafts=[], force=True,
        )
        assert out is not None
        assert out.method == "ngram"

    def test_mtp_picked_when_variant_present(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        # Place an MTP-suffixed sibling next to the model dir.
        model_dir = tmp_path / "Qwen3.6-27B-Instruct-GGUF"
        model_dir.mkdir()
        mtp_dir = tmp_path / "Qwen3.6-27B-MTP-GGUF"
        mtp_dir.mkdir()
        out = pick_spec_config(
            architecture="Qwen36ForCausalLM", backend="llamacpp",
            model_path=model_dir, available_drafts=[],
            bos_token_id=1, eos_token_id=2,
        )
        assert out is not None
        assert out.method == "mtp"
        assert out.draft_model_path == mtp_dir

    def test_falls_back_to_ngram_for_vllm_dense(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        out = pick_spec_config(
            architecture="LlamaForCausalLM", backend="vllm",
            model_path=tmp_path, available_drafts=[],
        )
        assert out is not None
        assert out.method == "ngram"

    def test_llamacpp_without_draft_returns_none(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        out = pick_spec_config(
            architecture="LlamaForCausalLM", backend="llamacpp",
            model_path=tmp_path, available_drafts=[],
        )
        assert out is None  # llama.cpp ngram exists but is disabled (net-negative, batched)

    def test_picks_draft_model_when_compatible(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, pick_spec_config
        draft = DraftCandidate(
            path=tmp_path / "tiny",
            architecture="LlamaForCausalLM",
            size_b=1.0,
            bos_token_id=1, eos_token_id=2,
        )
        out = pick_spec_config(
            architecture="LlamaForCausalLM", backend="llamacpp",
            model_path=tmp_path, available_drafts=[draft],
            bos_token_id=1, eos_token_id=2,
        )
        assert out is not None
        assert out.method == "draft"
        assert out.draft_model_path == draft.path

    def test_rejects_draft_with_mismatched_vocab(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, pick_spec_config
        draft = DraftCandidate(
            path=tmp_path / "tiny",
            architecture="LlamaForCausalLM",
            size_b=1.0,
            bos_token_id=5, eos_token_id=6,  # different vocab
        )
        out = pick_spec_config(
            architecture="LlamaForCausalLM", backend="llamacpp",
            model_path=tmp_path, available_drafts=[draft],
            bos_token_id=1, eos_token_id=2,
        )
        # Vocab mismatch → can't use draft; ngram disabled on llama.cpp → None.
        assert out is None

    def test_rejects_draft_over_1_5b(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, pick_spec_config
        big_draft = DraftCandidate(
            path=tmp_path / "big",
            architecture="LlamaForCausalLM",
            size_b=3.0,  # too big
            bos_token_id=1, eos_token_id=2,
        )
        out = pick_spec_config(
            architecture="LlamaForCausalLM", backend="vllm",
            model_path=tmp_path, available_drafts=[big_draft],
            bos_token_id=1, eos_token_id=2,
        )
        # Big drafts are skipped; falls back to ngram on vLLM.
        assert out is not None
        assert out.method == "ngram"


class TestVocabCompatible:
    def test_same_family_same_bos_eos(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="Qwen3ForCausalLM", size_b=1.0,
            bos_token_id=1, eos_token_id=2,
        )
        assert vocab_compatible("Qwen3ForCausalLM", 1, 2, draft) is True

    def test_different_family_rejected(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="LlamaForCausalLM", size_b=1.0,
            bos_token_id=1, eos_token_id=2,
        )
        assert vocab_compatible("Qwen3ForCausalLM", 1, 2, draft) is False

    def test_different_bos_rejected(self, tmp_path):
        from vserve.recipes.spec_decode import DraftCandidate, vocab_compatible
        draft = DraftCandidate(
            path=tmp_path, architecture="Qwen3ForCausalLM", size_b=1.0,
            bos_token_id=99, eos_token_id=2,
        )
        assert vocab_compatible("Qwen3ForCausalLM", 1, 2, draft) is False


class TestSpecBlocklistContents:
    """0.6.3 bug fix 4: SPEC_BLOCKLIST contained `GptOssMoeForCausalLM` (a
    typo — every other registry uses `GptOssForCausalLM`), so the entry was
    dead code and the actual `gpt-oss` model wasn't blocked from spec-decode."""

    def test_blocklist_uses_canonical_gptoss_arch_name(self):
        from vserve.recipes.spec_decode import SPEC_BLOCKLIST
        assert "GptOssForCausalLM" in SPEC_BLOCKLIST
        assert "GptOssMoeForCausalLM" not in SPEC_BLOCKLIST  # the typo


class TestNativeMtpLayers:
    """0.6.8: in-checkpoint MTP detection, mirroring vLLM 0.24's
    SpeculativeConfig config-key rewriting."""

    def test_qwen35_style_text_config_key(self, tmp_path):
        from vserve.recipes.spec_decode import native_mtp_layers
        model = _write_config(tmp_path / "m", {
            "model_type": "qwen3_5_moe",
            "text_config": {"mtp_num_hidden_layers": 1},
        })
        assert native_mtp_layers(model) == 1

    def test_deepseek_style_top_level_key(self, tmp_path):
        from vserve.recipes.spec_decode import native_mtp_layers
        model = _write_config(tmp_path / "m", {"num_nextn_predict_layers": 2})
        assert native_mtp_layers(model) == 2

    def test_minimax_style_num_mtp_modules(self, tmp_path):
        from vserve.recipes.spec_decode import native_mtp_layers
        model = _write_config(tmp_path / "m", {"num_mtp_modules": 3})
        assert native_mtp_layers(model) == 3

    def test_zero_layers_means_no_mtp_head(self, tmp_path):
        from vserve.recipes.spec_decode import native_mtp_layers
        model = _write_config(tmp_path / "m", {"text_config": {"mtp_num_hidden_layers": 0}})
        assert native_mtp_layers(model) is None

    def test_bool_value_rejected(self, tmp_path):
        from vserve.recipes.spec_decode import native_mtp_layers
        model = _write_config(tmp_path / "m", {"mtp_num_hidden_layers": True})
        assert native_mtp_layers(model) is None

    def test_no_config_json_returns_none(self, tmp_path):
        from vserve.recipes.spec_decode import native_mtp_layers
        assert native_mtp_layers(tmp_path) is None


class TestResolveMtpRequest:
    """0.6.8: explicit `vserve run --mtp` resolution — MTP or a precise error,
    never a silent substitute."""

    def test_vllm_native_defaults_to_three_tokens(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"text_config": {"mtp_num_hidden_layers": 1}})
        out = resolve_mtp_request(backend="vllm", model_path=model)
        assert out.method == "mtp"
        assert out.draft_model_path is None  # in-checkpoint → no draft model
        # Depth 1 is net-negative (-19% decode, research 2026-05-20 §5) —
        # the default must reuse the single MTP layer 3×.
        assert out.n_max == 3

    def test_vllm_native_deeper_stack_runs_at_own_depth(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"num_nextn_predict_layers": 2})
        out = resolve_mtp_request(backend="vllm", model_path=model)
        assert out.n_max == 2

    def test_vllm_explicit_tokens_win(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"text_config": {"mtp_num_hidden_layers": 1}})
        out = resolve_mtp_request(backend="vllm", model_path=model, num_tokens=6)
        assert out.n_max == 6

    def test_vllm_divisibility_enforced_before_boot(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"num_nextn_predict_layers": 2})
        with pytest.raises(ValueError, match="multiple of"):
            resolve_mtp_request(backend="vllm", model_path=model, num_tokens=3)

    def test_tokens_below_one_rejected(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"text_config": {"mtp_num_hidden_layers": 1}})
        with pytest.raises(ValueError, match=">= 1"):
            resolve_mtp_request(backend="vllm", model_path=model, num_tokens=0)

    def test_vllm_falls_back_to_sibling_variant(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "base", {"model_type": "other"})
        mtp_dir = tmp_path / "base-MTP"
        mtp_dir.mkdir()
        out = resolve_mtp_request(backend="vllm", model_path=model)
        assert out.method == "mtp"
        assert out.draft_model_path == mtp_dir

    def test_vllm_no_weights_no_sibling_raises(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"model_type": "other"})
        with pytest.raises(ValueError, match="no MTP weights"):
            resolve_mtp_request(backend="vllm", model_path=model)

    def test_llamacpp_requires_sibling_variant(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model_dir = tmp_path / "Qwen3.6-27B-GGUF"
        model_dir.mkdir()
        with pytest.raises(ValueError, match="GGUF sibling"):
            resolve_mtp_request(backend="llamacpp", model_path=model_dir)

    def test_llamacpp_uses_sibling_variant(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_mtp_request
        model_dir = tmp_path / "Qwen3.6-27B-GGUF"
        model_dir.mkdir()
        mtp_dir = tmp_path / "Qwen3.6-27B-MTP-GGUF"
        mtp_dir.mkdir()
        out = resolve_mtp_request(backend="llamacpp", model_path=model_dir, num_tokens=4)
        assert out.method == "mtp"
        assert out.draft_model_path == mtp_dir
        assert out.n_max == 4

    def test_llamacpp_ignores_native_safetensors_mtp(self, tmp_path):
        """GGUF conversions don't carry the safetensors MTP head — the
        in-checkpoint keys must NOT satisfy an llama.cpp MTP request."""
        from vserve.recipes.spec_decode import resolve_mtp_request
        model = _write_config(tmp_path / "m", {"text_config": {"mtp_num_hidden_layers": 1}})
        with pytest.raises(ValueError, match="GGUF sibling"):
            resolve_mtp_request(backend="llamacpp", model_path=model)


class TestResolveSpecRequest:
    """0.6.8: the --spec METHOD dispatcher."""

    def test_ngram_vllm_returns_config(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_spec_request
        out = resolve_spec_request(method="ngram", backend="vllm", model_path=tmp_path)
        assert out is not None and out.method == "ngram" and out.n_max == 5

    def test_ngram_llamacpp_refused_net_negative(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_spec_request
        with pytest.raises(ValueError, match="vLLM-only"):
            resolve_spec_request(method="ngram", backend="llamacpp", model_path=tmp_path)

    def test_auto_returns_none_when_nothing_positive(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_spec_request
        out = resolve_spec_request(
            method="auto", backend="llamacpp", model_path=tmp_path,
            architecture="LlamaForCausalLM",
        )
        assert out is None  # llama.cpp, no draft, no MTP sibling → nothing

    def test_unknown_method_raises(self, tmp_path):
        from vserve.recipes.spec_decode import resolve_spec_request
        with pytest.raises(ValueError, match="Unknown spec method"):
            resolve_spec_request(method="medusa", backend="vllm", model_path=tmp_path)


class TestPickSpecConfigNativeMtp:
    """0.6.8 gate fix: pick_spec_config's mtp step was gated on
    find_mtp_variant() alone, so vLLM safetensors checkpoints with
    in-checkpoint MTP layers (no sibling dir) never got the mtp
    recommendation their arch-table entry promised."""

    def test_vllm_native_checkpoint_gets_mtp_without_sibling(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        model = _write_config(
            tmp_path / "Qwen3.6-27B-FP8",
            {"text_config": {"mtp_num_hidden_layers": 1}},
        )
        out = pick_spec_config(
            architecture="Qwen3_5ForConditionalGeneration", backend="vllm",
            model_path=model, available_drafts=[],
        )
        assert out is not None
        assert out.method == "mtp"
        assert out.draft_model_path is None
        assert out.n_max == 3  # research-tuned depth, was 5 pre-0.6.8

    def test_a3b_moe_blocklisted_for_auto_but_forceable(self, tmp_path):
        """2026-07-10 bench: in-checkpoint MTP measured −52%/−53% on the
        Qwen3.6-35B-A3B (sm120, NVFP4) despite 60% acceptance — auto must
        not recommend it; force (and explicit --mtp) still can."""
        from vserve.recipes.spec_decode import pick_spec_config
        model = _write_config(
            tmp_path / "Qwen3.6-35B-A3B-NVFP4",
            {"text_config": {"mtp_num_hidden_layers": 1}},
        )
        auto = pick_spec_config(
            architecture="Qwen3_5MoeForConditionalGeneration", backend="vllm",
            model_path=model, available_drafts=[],
        )
        assert auto is None
        forced = pick_spec_config(
            architecture="Qwen3_5MoeForConditionalGeneration", backend="vllm",
            model_path=model, available_drafts=[], force=True,
        )
        assert forced is not None and forced.method == "mtp"

    def test_llamacpp_does_not_use_native_safetensors_path(self, tmp_path):
        from vserve.recipes.spec_decode import pick_spec_config
        model = _write_config(
            tmp_path / "Qwen3.6-27B",
            {"text_config": {"mtp_num_hidden_layers": 1}},
        )
        out = pick_spec_config(
            architecture="Qwen36ForCausalLM", backend="llamacpp",
            model_path=model, available_drafts=[],
        )
        # No sibling variant, no draft → llama.cpp gets nothing (ngram is
        # disabled there), never the unloadable in-checkpoint form.
        assert out is None


class TestSpecBuildConfig:
    """Wire-through tests for the build_config integration."""

    def test_vllm_emits_speculative_config_for_ngram(self, fake_model_dir):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model
        from vserve.recipes.spec_decode import SpecConfig
        b = VllmBackend()
        m = detect_model(fake_model_dir)
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
            "spec": SpecConfig(method="ngram", n_max=5, n_min=1),
        })
        assert cfg["speculative-config"]["method"] == "ngram"
        assert cfg["speculative-config"]["prompt_lookup_max"] == 5

    def test_vllm_refuses_mtp_with_gemma4_tools(self, tmp_path):
        import pytest
        import json
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import ModelInfo
        from vserve.recipes.spec_decode import SpecConfig
        model_dir = tmp_path / "models" / "u" / "gemma4"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForCausalLM"]}))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value={"gemma4"})  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = ModelInfo(
            path=model_dir, provider="u", model_name="gemma4",
            architecture="Gemma4ForCausalLM", model_type="gemma4", quant_method=None,
            max_position_embeddings=131072, is_moe=False, model_size_gb=1.0,
        )
        with pytest.raises(ValueError, match="41967"):
            b.build_config(m, {
                "context": 8192, "kv_dtype": "auto", "slots": 4,
                "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
                "tools": True, "tool_parser": "gemma4", "reasoning_parser": None,
                "spec": SpecConfig(method="mtp", draft_model_path=Path("/tmp/draft"), n_max=5),
            })

    def test_llamacpp_emits_spec_draft_dict(self, fake_gguf_model_dir, mocker):
        from vserve.backends.llamacpp import LlamaCppBackend
        from vserve.models import detect_model
        from vserve.recipes.spec_decode import SpecConfig
        b = LlamaCppBackend()
        mocker.patch.object(
            b, "_read_gguf_metadata",
            return_value={"arch": "llama", "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        )
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
            "spec": SpecConfig(method="draft", draft_model_path=Path("/draft.gguf"), n_max=3, n_min=1, p_min=0.6),
        })
        assert cfg["spec_draft"]["draft_model_path"] == "/draft.gguf"
        assert cfg["spec_draft"]["n_max"] == 3

    def test_llamacpp_start_emits_spec_draft_flags(self, fake_gguf_model_dir, tmp_path, mocker):
        from unittest.mock import Mock
        from vserve.backends.llamacpp import LlamaCppBackend
        from vserve.models import detect_model
        from vserve.recipes.spec_decode import SpecConfig
        b = LlamaCppBackend()
        mocker.patch.object(
            b, "_read_gguf_metadata",
            return_value={"arch": "llama", "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        )
        m = detect_model(fake_gguf_model_dir)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
            "spec": SpecConfig(method="draft", draft_model_path=Path("/draft.gguf"), n_max=3, n_min=1, p_min=0.6),
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
        assert "-md /draft.gguf" in script
        assert "--spec-draft-n-max 3" in script
        assert "--spec-draft-p-min 0.6" in script


class TestNativeMtpBuildConfig:
    """0.6.8: in-checkpoint MTP emission — the speculative-config block must
    NOT carry a `model` key (vLLM loads the draft layers from the target
    checkpoint when the key is absent)."""

    def _model(self, tmp_path):
        from vserve.models import detect_model
        model_dir = tmp_path / "models" / "qwen" / "Qwen36-Test-FP8"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
                "max_position_embeddings": 131072,
                "num_key_value_heads": 4,
                "num_attention_heads": 32,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
            },
        }))
        (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)
        return detect_model(model_dir)

    def test_vllm_emits_mtp_without_model_key(self, tmp_path):
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.recipes.spec_decode import SpecConfig
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = self._model(tmp_path)
        cfg = b.build_config(m, {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": False, "tool_parser": None, "reasoning_parser": None,
            "spec": SpecConfig(method="mtp", draft_model_path=None, n_max=3),
        })
        block = cfg["speculative-config"]
        assert block["method"] == "mtp"
        assert block["num_speculative_tokens"] == 3
        assert "model" not in block
        assert "prompt_lookup_min" not in block  # ngram-only keys

    def test_tune_stamps_supports_mtp(self, tmp_path):
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = self._model(tmp_path)
        gpu = Mock(vram_total_gb=48.0)
        result = b.tune(m, gpu, gpu_mem_util=0.9)
        assert result["supports_mtp"] is True
        assert result["mtp_num_layers"] == 1

    def test_tune_does_not_stamp_without_mtp_weights(self, fake_model_dir):
        from unittest.mock import Mock
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value=set())  # type: ignore[method-assign]
        m = detect_model(fake_model_dir)
        gpu = Mock(vram_total_gb=48.0)
        result = b.tune(m, gpu, gpu_mem_util=0.9)
        assert "supports_mtp" not in result
        assert "mtp_num_layers" not in result
