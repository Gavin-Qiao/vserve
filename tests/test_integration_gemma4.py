"""Cross-feature integration tests for Gemma-4 hosting.

These tests exercise multiple 0.6.1 items together — they catch
regressions where one item's change silently disables another's emission
(for example, an arch table mismatch causing the tool parser to skip
emitting the vendored chat template that item F resolved).

The model fixture is synthetic but architecture-faithful: config.json
matches the real Gemma-4 shape (heterogeneous head_dim, vision_config,
sliding_window_pattern), and the picker / backend code paths run end
to end without spinning up an actual vLLM or llama.cpp server.
"""

from __future__ import annotations

import json
from unittest.mock import Mock


def _make_gemma4_vllm_model_dir(tmp_path, *, with_vision=False, with_audio=False):
    """Architecture-faithful Gemma-4 fixture for the vLLM path."""
    model_dir = tmp_path / "models" / "nvidia" / "Gemma-4-31B-IT-NVFP4"
    model_dir.mkdir(parents=True)
    cfg = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "quantization_config": {"quant_method": "nvfp4"},
        "text_config": {
            "head_dim": 256,
            "global_head_dim": 512,
            "sliding_window_pattern": 6,
            "num_attention_heads": 10,
            "num_key_value_heads": 2,
            "hidden_size": 2880,
            "num_hidden_layers": 60,
            "max_position_embeddings": 131072,
        },
    }
    if with_vision:
        cfg["vision_config"] = {"hidden_size": 1024, "image_seq_length": 280}
    if with_audio:
        cfg["audio_config"] = {"hidden_size": 768, "audio_seq_length": 750}
    (model_dir / "config.json").write_text(json.dumps(cfg))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
    return model_dir


class TestGemma4VllmLaunch:
    """End-to-end: every auto-emission for Gemma-4 + tools + reasoning
    fires together. If any of items D, F, B, C, AA, Q, R, J, A is
    silently dropped, this test catches it."""

    def _backend_and_model(self, tmp_path, *, with_vision=False, with_audio=False):
        from vserve.backends.vllm import VllmBackend
        from vserve.models import detect_model
        model_dir = _make_gemma4_vllm_model_dir(tmp_path, with_vision=with_vision, with_audio=with_audio)
        b = VllmBackend()
        b.available_tool_parsers = Mock(return_value={"gemma4", "hermes"})  # type: ignore[method-assign]
        b.available_reasoning_parsers = Mock(return_value={"gemma4", "qwen3"})  # type: ignore[method-assign]
        m = detect_model(model_dir)
        return b, m

    def _base_choices(self, tools=True):
        return {
            "context": 8192, "kv_dtype": "auto", "slots": 4,
            "batched_tokens": 4096, "gpu_mem_util": 0.9, "port": 8888,
            "tools": tools, "tool_parser": None, "reasoning_parser": None,
            "gpu_compute_cap": 120,
        }

    def test_full_launch_emits_every_auto_default(self, tmp_path):
        """Gemma-4-31B-NVFP4 + tools + audio multimodal — verify the
        emitted YAML has every item 0.6.1 added wired through correctly."""
        b, m = self._backend_and_model(tmp_path, with_vision=True, with_audio=True)
        choices = self._base_choices(tools=True)
        cfg = b.build_config(m, choices)

        # Item D: tool parser auto-discovered from arch.
        assert cfg["tool-call-parser"] == "gemma4"
        assert cfg["enable-auto-tool-choice"] is True
        # Item F: vendored chat template auto-resolved.
        assert "chat-template" in cfg
        assert cfg["chat-template"].endswith("tool_chat_template_gemma4.jinja")
        # Item B: reasoning parser auto-discovered.
        assert cfg["reasoning-parser"] == "gemma4"
        # Item Q: NVFP4 auto-pins KV to fp8 (override from "auto").
        assert cfg["kv-cache-dtype"] == "fp8"
        # Item J: audio model gets 8192 floor (vs 4096 for vision-only).
        assert cfg["max-num-batched-tokens"] >= 8192
        # Item A: Gemma-4 sampler defaults emitted.
        ogc = cfg.get("override-generation-config", {})
        assert ogc.get("temperature") == 1.0
        assert ogc.get("top_p") == 0.95
        assert ogc.get("top_k") == 64
        # Item R: heterogeneous head_dim → TRITON_ATTN forced (even on SM120).
        ac = cfg.get("attention-config", {})
        assert ac.get("backend") == "TRITON_ATTN"
        # Item Q (env-side, but we verify the quant flag is in the YAML).
        assert cfg["quantization"] == "nvfp4"

    def test_vision_only_uses_4096_floor_not_8192(self, tmp_path):
        b, m = self._backend_and_model(tmp_path, with_vision=True, with_audio=False)
        cfg = b.build_config(m, self._base_choices())
        # Vision (no audio) → 4096 floor.
        assert cfg["max-num-batched-tokens"] == 4096

    def test_thinking_kwarg_routes_to_enable_thinking_for_gemma4(self, tmp_path):
        b, m = self._backend_and_model(tmp_path)
        choices = self._base_choices()
        choices["thinking"] = True
        cfg = b.build_config(m, choices)
        assert cfg["chat-template-kwargs"] == {"enable_thinking": True}

    def test_filters_turboquant_cells_in_tune(self, tmp_path):
        """The whole tune path filters TurboQuant for Gemma-4 (TRITON_ATTN
        rejects all of them). This is the 0.6.0 fix that 0.6.1 builds on."""
        from vserve.backends.vllm import _filter_incompatible_kv_dtypes
        model_dir = _make_gemma4_vllm_model_dir(tmp_path)
        limits = {
            "limits": {"4096": {"auto": 4, "fp8": 8, "turboquant_3bit_nc": 21}},
            "kv_cache_dtypes": {"turboquant_3bit_nc": {"bytes_per_token": 200}},
        }
        out = _filter_incompatible_kv_dtypes(limits, model_dir)
        assert "turboquant_3bit_nc" not in out["limits"]["4096"]


class TestGemma4LlamaCppLaunch:
    """End-to-end on the llama.cpp path — items BB, A, C, K, J, I, H, L."""

    def _fixture(self, tmp_path, mocker, *, arch="gemma4", num_layers=48):
        from vserve.backends.llamacpp import LlamaCppBackend
        from vserve.models import detect_model
        model_dir = tmp_path / "models" / "unsloth" / "gemma-4-26B-A4B-it-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
        (model_dir / "mmproj-BF16.gguf").write_bytes(b"GGUF")
        b = LlamaCppBackend()
        mocker.patch.object(
            b, "_read_gguf_metadata",
            return_value={"arch": arch, "num_layers": num_layers, "num_kv_heads": 2, "head_dim": 256},
        )
        mocker.patch.object(
            b, "_read_gguf_chat_template",
            return_value="...{%- if enable_thinking %}<|channel|>thought\n{%- endif %}<channel|>...",
        )
        m = detect_model(model_dir)
        return b, m

    def test_full_build_config_emits_all_items(self, tmp_path, mocker):
        b, m = self._fixture(tmp_path, mocker)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 60, "parallel": 1,
            "port": 8888, "tools": True,
            "kv_cache_k": "q8_0", "kv_cache_v": "q8_0",
        })
        # Item A: sampler from arch=gemma4.
        assert cfg["temp"] == 1.0
        assert cfg["top_p"] == 0.95
        assert cfg["top_k"] == 64
        # Item K: reasoning-format detected from <|channel|> marker.
        assert cfg["reasoning_format"] == "harmony"
        # Item J: mmproj auto-discovered.
        assert "mmproj" in cfg
        assert cfg["mmproj"].endswith("mmproj-BF16.gguf")
        # Item I: K==V symmetric (passes the invariant).
        assert cfg["cache_type_k"] == cfg["cache_type_v"] == "q8_0"
        # Item C: jinja on for tools.
        assert cfg.get("jinja") is True

    def test_gemma4_31b_forces_v_to_f16_even_when_user_picked_q8_0(self, tmp_path, mocker):
        """31B = 60 layers → force V=f16 (llamacpp#22527)."""
        b, m = self._fixture(tmp_path, mocker, num_layers=60)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 60, "parallel": 1,
            "port": 8888, "tools": False,
            "kv_cache_k": "q8_0", "kv_cache_v": "q8_0",
        })
        assert cfg["cache_type_v"] == "f16"
        # Surface the note for the status display.
        assert any("22527" in n or "Gemma-4-31B" in n for n in cfg.get("notes", []))

    def test_launch_script_contains_every_flag(self, tmp_path, mocker):
        b, m = self._fixture(tmp_path, mocker)
        cfg = b.build_config(m, {
            "context": 4096, "n_gpu_layers": 60, "parallel": 1,
            "port": 8888, "tools": True,
            "kv_cache_k": "q8_0", "kv_cache_v": "q8_0",
            "thinking": False,
        })
        cfg_path = tmp_path / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        active = tmp_path / "active.sh"
        active.parent.mkdir(parents=True, exist_ok=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "_assert_unit_safe_for_privileged_action")
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))
        b.start(cfg_path)
        script = active.read_text()
        # BB
        assert "--fit off" in script
        # A
        assert "--temp 1.0" in script
        assert "--top-k 64" in script
        # K
        assert "--reasoning-format harmony" in script
        # J
        assert "--mmproj" in script
        # C — chat_template_kwargs JSON serialized
        assert "--chat-template-kwargs" in script
        assert '{"enable_thinking": false}' in script
