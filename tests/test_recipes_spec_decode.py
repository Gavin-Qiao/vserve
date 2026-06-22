"""Tests for the speculative-decoding recipe picker (item M)."""

from __future__ import annotations

from pathlib import Path


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
