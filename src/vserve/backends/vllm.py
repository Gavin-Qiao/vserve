"""vLLM backend — wraps existing serve/probe/tools modules."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


# ---------------------------------------------------------------------------
# Backend × dtype × architecture compatibility
# ---------------------------------------------------------------------------
# vLLM forces a specific attention backend for some architectures and not all
# backends accept every KV-cache dtype. When the tuner emits a cell that the
# engine refuses (e.g. TurboQuant with TRITON_ATTN), engine init crashes on
# startup with `ValueError: Selected backend ... is not valid for this
# configuration. Reason: ['kv_cache_dtype not supported']`. Filter those cells
# out of the tune output so the picker never offers them.

# KV dtypes that the TRITON_ATTN backend does NOT support.
_TRITON_ATTN_INCOMPATIBLE_KV_DTYPES: frozenset[str] = frozenset({
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
})


# Architecture → tool-call parser name. The right-hand side must match a key
# in vLLM's `_TOOL_PARSERS_TO_REGISTER` table. The mapping is intentionally
# explicit (not derived from architecture string parsing) so that surprises
# show up at code-review time rather than at deploy time.
# Cross-checked against vLLM 0.21 tool_parsers/ directory.
_ARCH_TO_TOOL_PARSER: dict[str, str] = {
    # Gemma 3 / 4 family
    "Gemma3ForCausalLM":              "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM":              "gemma4",
    "Gemma4ForConditionalGeneration": "gemma4",
    # Llama 3.x / 4 — 3.1/3.2/3.3 use the JSON-style parser, 4 uses pythonic
    "LlamaForCausalLM":               "llama3_json",
    "Llama4ForCausalLM":              "llama4_pythonic",
    "Llama4MoeForCausalLM":           "llama4_pythonic",
    # Qwen3 family — Hermes-format on base, qwen3_coder for coder variants
    "Qwen3ForCausalLM":               "hermes",
    "Qwen35ForCausalLM":              "hermes",
    "Qwen36ForCausalLM":              "hermes",
    "Qwen3MoeForCausalLM":            "hermes",
    "Qwen3A3BForCausalLM":            "hermes",
    "Qwen36MoeForCausalLM":           "hermes",
    "Qwen3CoderForCausalLM":          "qwen3_coder",
    "Qwen3XmlForCausalLM":            "qwen3_xml",
    # DeepSeek V-series — per-version parser
    "DeepseekV3ForCausalLM":          "deepseek_v3",
    "DeepseekV31ForCausalLM":         "deepseek_v31",
    "DeepseekV32ForCausalLM":         "deepseek_v32",
    "DeepseekV4ForCausalLM":          "deepseek_v4",
    # Moonshot Kimi K2 (instruct + thinking)
    "KimiK2ForCausalLM":              "kimi_k2",
    "KimiK2ThinkingForCausalLM":      "kimi_k2",
    # GLM 4 family (Z.ai)
    "Glm4MoeForCausalLM":             "glm45",
    "Glm47MoeForCausalLM":            "glm47",
    # IBM Granite (text-only Granite-3 + Granite-4 with mixed schema)
    "GraniteForCausalLM":             "granite",
    "Granite4ForCausalLM":            "granite4",
    # Cohere Command R/R+ (Command-4 parser supersedes Command-3)
    "CohereForCausalLM":              "cohere_command4",
    # Baidu ERNIE 4.5
    "Ernie4ForCausalLM":              "ernie45",
    # AI21 Jamba
    "JambaForCausalLM":               "jamba",
    # Salesforce xLAM
    "XlamForCausalLM":                "xlam",
    # Liquid LFM 2 / 2.5
    "Lfm2ForCausalLM":                "lfm2",
    "Lfm25ForCausalLM":               "lfm25",
    # Mistral
    "MistralForCausalLM":             "mistral",
    "MistralThinkingForCausalLM":     "mistral",
    # GPT-OSS (OpenAI Harmony)
    "GptOssForCausalLM":              "openai",
    # InternLM (2.x uses same parser)
    "InternLMForCausalLM":            "internlm",
    "InternLM2ForCausalLM":           "internlm",
}


# Architecture → reasoning-parser name (vLLM 0.21+). Reasoning parsers split
# the thinking trace from the answer in OpenAI-format responses (so clients
# see `message.reasoning_content` distinct from `message.content`). Names
# must match the keys registered by `ReasoningParserManager` — verified
# against https://docs.vllm.ai/en/latest/features/reasoning_outputs.html.
_ARCH_TO_REASONING_PARSER: dict[str, str] = {
    "Gemma3ForCausalLM":              "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM":              "gemma4",
    "Gemma4ForConditionalGeneration": "gemma4",
    "DeepseekV3ForCausalLM":          "deepseek_r1",
    "DeepseekV31ForCausalLM":         "deepseek_r1",
    "DeepseekV32ForCausalLM":         "deepseek_r1",
    "DeepseekV4ForCausalLM":          "deepseek_r1",
    "Qwen3ForCausalLM":               "qwen3",
    "Qwen35ForCausalLM":              "qwen3",
    "Qwen36ForCausalLM":              "qwen3",
    "Qwen3MoeForCausalLM":            "qwen3",
    "Qwen3A3BForCausalLM":            "qwen3",
    "KimiK2ThinkingForCausalLM":      "deepseek_r1",  # uses <think> markers
    "MistralForCausalLM":             "mistral",
    "MistralThinkingForCausalLM":     "mistral",
    "GptOssForCausalLM":              "openai_gptoss",
}


def _read_model_config(model_path: Path) -> dict:
    """Read the model's `config.json`, returning `{}` on any error."""
    try:
        data = json.loads((model_path / "config.json").read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _architecture_forces_triton_attn(model_path: Path) -> bool:
    """True when the architecture has heterogeneous per-layer head dimensions,
    in which case vLLM forces the TRITON_ATTN backend to prevent mixed-backend
    numerical divergence (see `Gemma4` model_executor: "head_dim != global_head_dim").
    """
    cfg = _read_model_config(model_path)
    if not cfg:
        return False
    subconfigs: list[dict] = [cfg]
    text_config = cfg.get("text_config")
    if isinstance(text_config, dict):
        subconfigs.append(text_config)
    for sub in subconfigs:
        head_dim = sub.get("head_dim")
        global_head_dim = sub.get("global_head_dim")
        if (
            isinstance(head_dim, int) and not isinstance(head_dim, bool)
            and isinstance(global_head_dim, int) and not isinstance(global_head_dim, bool)
            and head_dim != global_head_dim
        ):
            return True
    return False


# Item R: architecture → forced attention backend. MLA architectures (DeepSeek
# V2/V3/V4, Kimi K2, LongCat-Flash) ship with their own attention layout — the
# default backend auto-pick can mis-route. Heterogeneous head_dim still forces
# TRITON_ATTN (handled by _architecture_forces_triton_attn above).
_ARCH_FORCES_BACKEND: dict[str, str] = {
    "DeepseekV2ForCausalLM":   "FLASHMLA",
    "DeepseekV3ForCausalLM":   "FLASHMLA",
    "DeepseekV31ForCausalLM":  "FLASHMLA",
    "DeepseekV32ForCausalLM":  "FLASHMLA",
    "DeepseekV4ForCausalLM":   "FLASHMLA",
    "KimiK2ForCausalLM":       "FLASHMLA",
    "KimiK2ThinkingForCausalLM": "FLASHMLA",
    "LongcatFlashForCausalLM": "FLASHMLA",
    # GPT-OSS on SM120 (Blackwell RTX) forces TRITON_ATTN because FlashInfer
    # doesn't support attention sinks on that compute capability (vllm#40153).
    # Conditional — handled in _forced_attention_backend below.
    "GptOssForCausalLM":       "TRITON_ATTN",
    # Gemma-4 already forces TRITON_ATTN via the heterogeneous-head_dim path.
    # Keep explicit here so the table is the single source of truth.
    "Gemma4ForCausalLM":              "TRITON_ATTN",
    "Gemma4ForConditionalGeneration": "TRITON_ATTN",
}

# Per-backend KV dtypes that aren't supported on that backend.
BACKEND_INCOMPATIBLE_KV_DTYPES: dict[str, frozenset[str]] = {
    "TRITON_ATTN": _TRITON_ATTN_INCOMPATIBLE_KV_DTYPES,
    # FLASHMLA / TOKENSPEED_MLA support fp8 KV and most non-quant dtypes;
    # TurboQuant 3-bit specifically fails on FLASHMLA per maintainer notes.
    "FLASHMLA":         frozenset({"turboquant_3bit_nc", "turboquant_k3v4_nc"}),
    "TOKENSPEED_MLA":   frozenset(),
}


def _forced_attention_backend(model_path: Path, gpu_compute_cap: int) -> str | None:
    """Return the vLLM attention backend string vserve should pin for this
    architecture and GPU. None means "let vLLM auto-pick".

    Compute-cap routing:
      sm>=100 (Blackwell) — MLA architectures prefer TOKENSPEED_MLA
      sm<100             — MLA architectures prefer FLASHMLA
      sm==120 + GptOss   — TRITON_ATTN required (vllm#40153)
    """
    cfg = _read_model_config(model_path)
    archs = cfg.get("architectures") if cfg else None
    if not isinstance(archs, list):
        return None
    for arch in archs:
        if not isinstance(arch, str):
            continue
        base = _ARCH_FORCES_BACKEND.get(arch)
        if base is None:
            continue
        # MLA archs upgrade to TOKENSPEED_MLA on Blackwell-class compute.
        if base == "FLASHMLA" and gpu_compute_cap >= 100:
            return "TOKENSPEED_MLA"
        # GPT-OSS only forces TRITON_ATTN on SM120 specifically — other
        # compute caps can use the default backend.
        if base == "TRITON_ATTN" and arch == "GptOssForCausalLM" and gpu_compute_cap != 120:
            return None
        return base
    return None


def _is_multimodal_model(model_path: Path) -> bool:
    """True when the model has a vision/audio encoder, as evidenced by the
    presence of `vision_config` / `audio_config` in `config.json`."""
    cfg = _read_model_config(model_path)
    if not cfg:
        return False
    return bool(cfg.get("vision_config") or cfg.get("audio_config"))


def _mm_batched_tokens_floor(model_path: Path) -> int | None:
    """Floor for `max-num-batched-tokens` so a single multimodal item fits in
    one batch. vLLM disables MM-input chunking for bidirectional-attention
    encoders, so when an image expands to e.g. 2496 tokens and the default
    batch is 2048, engine init crashes with `Chunked MM input disabled but
    max_tokens_per_mm_item ... is larger than max_num_batched_tokens`.

    Floors per modality:
    - Vision worst case (Gemma-4 pan-and-scan): ~280 tokens/crop × ~9 crops
      ≈ 2520; 4096 covers it.
    - Audio (Gemma-4 E-class): 750 tokens / 30 s segment; multi-segment
      requests push to 6-9k tokens. 8192 covers the common case.
    """
    cfg = _read_model_config(model_path)
    if not cfg:
        return None
    has_vision = bool(cfg.get("vision_config"))
    has_audio = bool(cfg.get("audio_config"))
    if not (has_vision or has_audio):
        return None
    return 8192 if has_audio else 4096


def _suggested_tool_parser(model_path: Path, available_parsers: set[str] | None) -> str | None:
    """Map the model's architecture to a vLLM tool parser that is installed
    in the configured runtime. Returns None when no mapping or no installed
    parser matches."""
    cfg = _read_model_config(model_path)
    archs = cfg.get("architectures") if cfg else None
    if not isinstance(archs, list):
        return None
    for arch in archs:
        if not isinstance(arch, str):
            continue
        candidate = _ARCH_TO_TOOL_PARSER.get(arch)
        if candidate is None:
            continue
        if available_parsers is None or candidate in available_parsers:
            return candidate
    return None


def _suggested_reasoning_parser(model_path: Path, available_parsers: set[str] | None) -> str | None:
    """Mirror of _suggested_tool_parser for reasoning parsers.

    Returns the recipe-mapped reasoning parser if it exists in the runtime
    registry (or unconditionally when ``available_parsers`` is None — used
    for tests that want to inspect the mapping without standing up vLLM).
    """
    cfg = _read_model_config(model_path)
    archs = cfg.get("architectures") if cfg else None
    if not isinstance(archs, list):
        return None
    for arch in archs:
        if not isinstance(arch, str):
            continue
        candidate = _ARCH_TO_REASONING_PARSER.get(arch)
        if candidate is None:
            continue
        if available_parsers is None or candidate in available_parsers:
            return candidate
    return None


def _locate_gemma4_chat_template(vllm_root: Path | None) -> Path | None:
    """Locate the vendored Gemma-4 tool chat template inside the configured
    vLLM install. The bundled ``gemma4`` parser expects this template's
    specific encoding (``<|"|>`` string delimiter, ``<|tool_call>`` outer
    tag, bare unquoted JSON keys) — the stock HF template does NOT emit it,
    so tool calls silently fail.

    Returns the resolved path or None when neither the install copy nor the
    vserve-packaged fallback is present.
    """
    candidates: list[Path] = []
    if vllm_root is not None:
        # Try common venv layouts inside the configured vLLM root.
        for venv_dir in ("venv", ".venv"):
            for py in ("python3.13", "python3.12", "python3.11", "python3.10"):
                candidates.append(
                    vllm_root / venv_dir / "lib" / py / "site-packages"
                    / "vllm" / "examples" / "tool_chat_template_gemma4.jinja"
                )
    # Packaged fallback shipped inside the vserve wheel.
    candidates.append(Path(__file__).resolve().parent.parent / "templates" / "tool_chat_template_gemma4.jinja")
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                return cand
        except OSError:
            continue
    return None


def _served_model_name_aliases(provider: str, model_name: str) -> list[str]:
    """Generate `served-model-name` aliases so OpenAI-compatible clients can
    address the model by a short name instead of the full filesystem path
    that vLLM defaults to.

    Order: `provider/model` first (canonical Hugging Face identifier), then
    a lowercase slug with common chat-template suffixes stripped.
    """
    aliases: list[str] = [f"{provider}/{model_name}"]
    slug = model_name.lower()
    for suffix in ("-it-gguf", "-it", "-instruct", "-chat", "-gguf"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    if slug and slug not in {alias.lower() for alias in aliases}:
        aliases.append(slug)
    return aliases


def _filter_incompatible_kv_dtypes(limits_data: dict, model_path: Path) -> dict:
    """Remove KV dtypes from the tune output that the forced attention backend
    can't actually run. Modifies `limits_data` in place (and returns it).

    For Gemma-4 (heterogeneous head dims → TRITON_ATTN), drops every
    `turboquant_*` cell from `limits` and removes the corresponding entries
    from `kv_cache_dtypes`. Without this filter, the tuner picks a TurboQuant
    cell that crashes at engine init.
    """
    if not _architecture_forces_triton_attn(model_path):
        return limits_data
    incompatible = _TRITON_ATTN_INCOMPATIBLE_KV_DTYPES

    limits = limits_data.get("limits")
    if isinstance(limits, dict):
        for ctx_key, row in list(limits.items()):
            if isinstance(row, dict):
                for dtype in list(row.keys()):
                    if dtype in incompatible:
                        del row[dtype]
                if not row:
                    del limits[ctx_key]

    profiles = limits_data.get("kv_cache_dtypes")
    if isinstance(profiles, dict):
        for dtype in list(profiles.keys()):
            if dtype in incompatible:
                del profiles[dtype]

    rec_profile = limits_data.get("recommended_profile")
    if isinstance(rec_profile, dict):
        rec_dtype = rec_profile.get("kv_cache_dtype")
        if isinstance(rec_dtype, str) and rec_dtype in incompatible:
            # The recommendation is now invalid; drop it so the CLI falls
            # back to picker-driven selection.
            limits_data["recommended_profile"] = None

    limits_data["forced_attn_backend"] = "TRITON_ATTN"
    return limits_data




class VllmBackend:
    name = "vllm"
    display_name = "vLLM"

    def __init__(self) -> None:
        self._runtime_parser_registry_cache: dict[str, set[str] | None] | None = None
        self._runtime_parser_registry_cache_loaded = False

    @property
    def service_name(self) -> str:
        from vserve.config import cfg

        return cfg().service_name

    @property
    def service_user(self) -> str:
        from vserve.config import cfg

        return cfg().service_user

    @property
    def root_dir(self) -> Path:
        from vserve.config import cfg
        return cfg().vllm_root

    def can_serve(self, model: ModelInfo) -> bool:
        """True if model has top-level safetensors or .bin weights."""
        from vserve.model_files import iter_top_level_files_with_suffix

        p = model.path
        has_safetensors = bool(iter_top_level_files_with_suffix(p, ".safetensors"))
        has_bin = bool(iter_top_level_files_with_suffix(p, ".bin"))
        return has_safetensors or has_bin

    def find_entrypoint(self) -> Path | None:
        from vserve.config import cfg
        vllm_bin = cfg().vllm_bin
        return vllm_bin if vllm_bin.exists() else None

    def runtime_info(self, *, prefer_cache: bool = False, with_pip_check: bool = True):
        from vserve.runtime import collect_vllm_runtime_info

        return collect_vllm_runtime_info(
            prefer_cache=prefer_cache,
            with_pip_check=with_pip_check,
        )

    def compatibility(self, *, prefer_cache: bool = False, with_pip_check: bool = True):
        from vserve.runtime import check_vllm_compatibility

        return check_vllm_compatibility(
            self.runtime_info(prefer_cache=prefer_cache, with_pip_check=with_pip_check)
        )

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        from vserve.probe import calculate_limits
        result = calculate_limits(
            model_info=model,
            vram_total_gb=gpu.vram_total_gb,
            gpu_mem_util=gpu_mem_util,
        )
        result["backend"] = self.name
        # Drop (ctx × kv_dtype) cells that the forced attention backend can't
        # actually run. Without this, the tuner emits TurboQuant cells for
        # Gemma-4 (which forces TRITON_ATTN) and engine init crashes.
        result = _filter_incompatible_kv_dtypes(result, model.path)
        # Surface a suggested tool parser when the architecture maps to one
        # of vLLM's bundled parsers — so `vserve run --tools` doesn't need
        # the user to know the parser name.
        suggested = _suggested_tool_parser(model.path, self.available_tool_parsers())
        if suggested is not None:
            result["tool_call_parser"] = suggested
            result["supports_tools"] = True
        # Same lookup for reasoning parsers — keep them split because tool
        # calls and reasoning channels are independent concerns.
        rp_suggested = _suggested_reasoning_parser(model.path, self.available_reasoning_parsers())
        if rp_suggested is not None:
            result["reasoning_parser"] = rp_suggested
            result["supports_reasoning"] = True
        return result

    def available_tool_parsers(self) -> set[str] | None:
        """Return tool parsers supported by the installed vLLM runtime."""
        registry = self._runtime_parser_registry()
        return registry.get("tool_parsers") if registry is not None else None

    def available_reasoning_parsers(self) -> set[str] | None:
        """Return reasoning parsers supported by the installed vLLM runtime."""
        registry = self._runtime_parser_registry()
        return registry.get("reasoning_parsers") if registry is not None else None

    def _runtime_parser_registry(self) -> dict[str, set[str] | None] | None:
        if self._runtime_parser_registry_cache_loaded:
            return self._runtime_parser_registry_cache

        from vserve.config import cfg

        script = r"""
import json

def registered(manager, attr_names):
    list_registered = getattr(manager, "list_registered", None)
    if callable(list_registered):
        try:
            values = list_registered()
            if isinstance(values, dict):
                return sorted(str(key) for key in values)
            if isinstance(values, (list, tuple, set)):
                return sorted(str(value) for value in values)
        except Exception:
            pass
    for attr_name in attr_names:
        parsers = getattr(manager, attr_name, None)
        if isinstance(parsers, dict):
            return sorted(str(key) for key in parsers)
    return None

def tool_parsers():
    managers = []
    for module_name in (
        "vllm.tool_parsers",
        "vllm.entrypoints.openai.tool_parsers",
    ):
        try:
            module = __import__(module_name, fromlist=["ToolParserManager"])
            managers.append(getattr(module, "ToolParserManager"))
        except Exception:
            pass
    for manager in managers:
        values = registered(manager, ("tool_parsers", "_tool_parsers"))
        if values is not None:
            return values
    return None

def reasoning_parsers():
    managers = []
    for module_name in (
        "vllm.reasoning",
        "vllm.entrypoints.openai.reasoning_parsers",
    ):
        try:
            module = __import__(module_name, fromlist=["ReasoningParserManager"])
            managers.append(getattr(module, "ReasoningParserManager"))
        except Exception:
            pass
    for manager in managers:
        values = registered(manager, ("reasoning_parsers", "_reasoning_parsers"))
        if values is not None:
            return values
    return None

print(json.dumps({
    "tool_parsers": tool_parsers(),
    "reasoning_parsers": reasoning_parsers(),
}))
"""
        try:
            result = subprocess.run(
                [str(cfg().vllm_python), "-c", script],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            self._runtime_parser_registry_cache = None
            self._runtime_parser_registry_cache_loaded = True
            return None
        if result.returncode != 0:
            self._runtime_parser_registry_cache = None
            self._runtime_parser_registry_cache_loaded = True
            return None
        try:
            data = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError:
            self._runtime_parser_registry_cache = None
            self._runtime_parser_registry_cache_loaded = True
            return None
        out: dict[str, set[str] | None] = {}
        for key in ("tool_parsers", "reasoning_parsers"):
            value = data.get(key)
            out[key] = set(value) if isinstance(value, list) else None
        self._runtime_parser_registry_cache = out
        self._runtime_parser_registry_cache_loaded = True
        return out

    @staticmethod
    def _format_available(values: set[str] | None) -> str:
        return ", ".join(sorted(values)) if values else "(none reported)"

    def _validate_parsers(self, choices: dict) -> None:
        tool_parser = choices.get("tool_parser")
        if tool_parser:
            available = self.available_tool_parsers()
            if available is None:
                raise RuntimeError(
                    "Could not inspect installed vLLM tool parsers. "
                    "Activate the configured vLLM runtime or pass a parser supported by the installed runtime."
                )
            if tool_parser not in available:
                raise ValueError(
                    f"Unknown vLLM tool parser '{tool_parser}'. "
                    f"Available: {self._format_available(available)}"
                )
        reasoning_parser = choices.get("reasoning_parser")
        if reasoning_parser:
            available = self.available_reasoning_parsers()
            if available is None:
                raise RuntimeError(
                    "Could not inspect installed vLLM reasoning parsers. "
                    "Activate the configured vLLM runtime or pass a parser supported by the installed runtime."
                )
            if reasoning_parser not in available:
                raise ValueError(
                    f"Unknown vLLM reasoning parser '{reasoning_parser}'. "
                    f"Available: {self._format_available(available)}"
                )

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build vLLM serving config dict.

        Expected choices keys:
            context, kv_dtype, slots, batched_tokens, gpu_mem_util,
            port, tools, tool_parser, reasoning_parser

        Auto-applied unless explicitly overridden by `choices`:
            served-model-name aliases (provider/model + short slug) — so
              OpenAI clients don't need the full filesystem path.
            max-num-batched-tokens floor for multimodal models (>= MM-item
              size) — without this, vLLM crashes at startup when a single
              image expands past the default 2048 batch.
            tool-call-parser via architecture lookup when --tools requested.
        """
        self._validate_parsers(choices)
        cfg: dict = {
            "model": str(model.path),
            "served-model-name": _served_model_name_aliases(model.provider, model.model_name),
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "dtype": "bfloat16",
            "gpu-memory-utilization": choices["gpu_mem_util"],
            "max-model-len": choices["context"],
            "max-num-seqs": choices["slots"],
            "kv-cache-dtype": choices["kv_dtype"],
            "enable-prefix-caching": True,
        }
        if choices.get("trust_remote_code"):
            cfg["trust-remote-code"] = True
        bt = choices.get("batched_tokens")
        min_bt = self._minimum_batched_tokens(model)
        mm_floor = _mm_batched_tokens_floor(model.path)
        # Combine: scheduler floor (hybrid-attn) AND multimodal floor.
        # Use max of both so a multimodal hybrid-attn model gets the larger.
        floors = [v for v in (min_bt, mm_floor) if isinstance(v, int)]
        effective_floor = max(floors) if floors else None
        if effective_floor is not None:
            if bt is None:
                bt = effective_floor
            elif isinstance(bt, int) and not isinstance(bt, bool):
                bt = max(bt, effective_floor)
        if bt is not None:
            cfg["max-num-batched-tokens"] = bt
        if choices.get("performance_mode") is not None:
            cfg["performance-mode"] = choices["performance_mode"]
        if choices.get("optimization_level") is not None:
            cfg["optimization-level"] = choices["optimization_level"]
        if choices.get("block_size") is not None:
            cfg["block-size"] = choices["block_size"]
        if choices.get("kv_cache_memory_bytes") is not None:
            cfg["kv-cache-memory-bytes"] = choices["kv_cache_memory_bytes"]
        if choices.get("enable_prefix_caching") is not None:
            cfg["enable-prefix-caching"] = bool(choices["enable_prefix_caching"])

        qf = self.quant_flag(model.quant_method)
        if qf:
            cfg["quantization"] = qf.split()[-1]
        # R: force the right attention backend for MLA architectures and
        # for GPT-OSS on SM120. ``choices["gpu_compute_cap"]`` overrides the
        # nvidia-smi probe (used by tests and CI).
        forced_backend = choices.get("attention_backend")
        if forced_backend is None:
            cc = choices.get("gpu_compute_cap")
            if cc is None:
                try:
                    from vserve.gpu import get_gpu_info
                    cc = get_gpu_info().compute_cap
                except Exception:
                    cc = None
            if isinstance(cc, int):
                forced_backend = _forced_attention_backend(model.path, cc)
        if isinstance(forced_backend, str) and forced_backend:
            cfg.setdefault("attention-config", {})
            ac = cfg["attention-config"]
            if isinstance(ac, dict):
                ac["backend"] = forced_backend
        # Q: NVFP4 + auto KV is broken (treats fp8-KV as fp8-checkpoint per
        # vllm#39133). Force fp8 KV when the model is NVFP4/ModelOpt and the
        # caller hasn't pinned a KV dtype. The user can still override
        # explicitly by passing kv_dtype="auto" with their own override.
        if (
            model.quant_method in {"nvfp4", "modelopt"}
            and (cfg.get("kv-cache-dtype") in {"auto", None, ""})
        ):
            cfg["kv-cache-dtype"] = "fp8"

        # Tool-call parser: prefer explicitly-passed parser, fall back to a
        # parser inferred from the model architecture, only emit when tools
        # are actually requested.
        tool_parser = choices.get("tool_parser")
        if choices.get("tools") and not tool_parser:
            tool_parser = _suggested_tool_parser(model.path, self.available_tool_parsers())
        if choices.get("tools") and tool_parser:
            cfg["enable-auto-tool-choice"] = True
            cfg["tool-call-parser"] = tool_parser
            # The Gemma-4 parser expects the vendored chat template — the
            # stock HF template's tag layout silently breaks tool-call
            # serialization (R4 §7). Auto-resolve unless the user has set
            # chat-template explicitly via choices.
            if tool_parser == "gemma4" and "chat-template" not in cfg and "chat_template" not in choices:
                template_path = _locate_gemma4_chat_template(self.root_dir)
                if template_path is not None:
                    cfg["chat-template"] = str(template_path)
            if "chat_template" in choices and choices["chat_template"]:
                cfg["chat-template"] = str(choices["chat_template"])
        # Reasoning parser: prefer explicit, else fall back to architecture
        # lookup. Symmetric with the tool-parser path above.
        rp = choices.get("reasoning_parser")
        if not rp:
            rp = _suggested_reasoning_parser(model.path, self.available_reasoning_parsers())
        if rp:
            cfg["reasoning-parser"] = rp

        # chat-template-kwargs (item C). Architectures differ on the kwarg
        # name used to flip thinking on/off:
        #   - DeepSeek V3.1+ hybrid uses "thinking"
        #   - Gemma 3/4, Qwen 3.x family use "enable_thinking"
        # ``choices["chat_template_kwargs"]`` is a free-form passthrough;
        # ``choices["thinking"]`` is a convenience that maps to the right
        # kwarg for the architecture.
        ctk: dict = {}
        explicit_ctk = choices.get("chat_template_kwargs")
        if isinstance(explicit_ctk, dict):
            ctk.update(explicit_ctk)
        thinking = choices.get("thinking")
        if thinking is not None and thinking != "auto":
            archs_raw = _read_model_config(model.path).get("architectures") or [""]
            first_arch = archs_raw[0] if isinstance(archs_raw, list) and archs_raw else ""
            key = "thinking" if isinstance(first_arch, str) and first_arch.startswith("Deepseek") else "enable_thinking"
            ctk[key] = bool(thinking) if isinstance(thinking, bool) else thinking in (True, "on", "true", 1)
        if ctk:
            cfg["chat-template-kwargs"] = ctk

        # Architecture-derived sampler defaults — unless the caller opted out
        # by passing recipe_sampling=False. vLLM accepts the parameters via
        # `override-generation-config` in the YAML (a nested dict). Caller-
        # supplied overrides win via setdefault.
        if choices.get("recipe_sampling", True) and not choices.get("override_generation_config"):
            from vserve.recipes.sampling import get_sampling_defaults
            arch = (_read_model_config(model.path).get("architectures") or [None])[0]
            defaults = get_sampling_defaults(arch if isinstance(arch, str) else None)
            if defaults is not None:
                gen_cfg: dict = {"temperature": defaults.temp}
                if defaults.top_p is not None:
                    gen_cfg["top_p"] = defaults.top_p
                if defaults.top_k is not None:
                    gen_cfg["top_k"] = defaults.top_k
                if defaults.min_p is not None:
                    gen_cfg["min_p"] = defaults.min_p
                if defaults.presence_penalty is not None:
                    gen_cfg["presence_penalty"] = defaults.presence_penalty
                # repeat_penalty is llama.cpp-only; vLLM uses repetition_penalty
                # at request time, not via override-generation-config.
                cfg["override-generation-config"] = gen_cfg
        elif choices.get("override_generation_config"):
            cfg["override-generation-config"] = choices["override_generation_config"]

        # Pre-emit cudagraph_mode: NONE for kernels known to lock the CUDA-graph
        # workspace at decode time. Workspace is sized during graph capture, so
        # any decode-time allocation that's larger asserts. Skipping capture
        # (cudagraph_mode: NONE) keeps torch.compile fusions; --enforce-eager
        # also disables fusions, which is unnecessary.
        kv_dtype = choices.get("kv_dtype")
        if isinstance(kv_dtype, str) and kv_dtype.startswith("turboquant"):
            cc = cfg.setdefault("compilation-config", {})
            if isinstance(cc, dict):
                cc.setdefault("cudagraph_mode", "NONE")
        # Same workaround for spec-decode + quantized KV (vllm#41559 — DFlash
        # spec-decode breaks with any KV quantization).
        if choices.get("spec") and isinstance(kv_dtype, str) and kv_dtype not in {"auto", ""}:
            cc = cfg.setdefault("compilation-config", {})
            if isinstance(cc, dict):
                cc.setdefault("cudagraph_mode", "NONE")

        # M: speculative-config block. Accept either a SpecConfig instance
        # (auto-picked recipe) or a pre-built dict (caller knows what they
        # want). vllm#41967 — refuse MTP + Gemma-4 + tools.
        spec = choices.get("spec")
        if spec:
            spec_cls: type | None
            try:
                from vserve.recipes.spec_decode import SpecConfig as _SpecConfig
                spec_cls = _SpecConfig
            except Exception:
                spec_cls = None
            if spec_cls is not None and isinstance(spec, spec_cls):
                if spec.method == "mtp" and cfg.get("tool-call-parser") == "gemma4":
                    raise ValueError(
                        "Refusing to enable MTP spec-decode with Gemma-4 + tools — "
                        "first call's args are corrupted (vllm#41967). Use "
                        "ngram or disable tools."
                    )
                block: dict = {"method": spec.method, "num_speculative_tokens": spec.n_max}
                if spec.draft_model_path:
                    block["model"] = str(spec.draft_model_path)
                if spec.method == "ngram":
                    block["prompt_lookup_min"] = spec.n_min
                    block["prompt_lookup_max"] = spec.n_max
                cfg["speculative-config"] = block
            elif isinstance(spec, dict):
                cfg["speculative-config"] = spec

        return cfg

    @staticmethod
    def _minimum_batched_tokens(model: ModelInfo) -> int | None:
        """Return vLLM scheduler floor for hybrid linear-attention/Mamba models."""
        try:
            data = json.loads((model.path / "config.json").read_text())
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        configs = [data]
        text_config = data.get("text_config")
        if isinstance(text_config, dict):
            configs.append(text_config)
        for config in configs:
            layer_types = config.get("layer_types")
            if isinstance(layer_types, list) and any("linear_attention" in str(item).lower() for item in layer_types):
                return 4096
            full_attention_interval = config.get("full_attention_interval")
            if (
                isinstance(full_attention_interval, int)
                and not isinstance(full_attention_interval, bool)
                and full_attention_interval > 0
            ):
                return 4096
        return None

    def quant_flag(self, method: str | None) -> str:
        from vserve.models import quant_flag as _qf
        return _qf(method)

    def start(self, config_path: Path, *, non_interactive: bool = False) -> None:
        from vserve.serve import start_vllm
        start_vllm(config_path, non_interactive=non_interactive)

    def stop(self, *, non_interactive: bool = False) -> None:
        from vserve.serve import stop_vllm
        stop_vllm(non_interactive=non_interactive)

    def is_running(self) -> bool:
        from vserve.serve import is_vllm_running
        return is_vllm_running()

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def active_manifest_path(self) -> Path:
        from vserve.config import active_manifest_path

        return active_manifest_path()

    def detect_tools(self, model_path: Path) -> dict:
        from vserve.tools import detect_tool_parser, detect_reasoning_parser
        return {
            "tool_call_parser": detect_tool_parser(model_path),
            "reasoning_parser": detect_reasoning_parser(model_path),
        }

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        from vserve.config import find_systemd_unit_path

        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            return find_systemd_unit_path(self.service_name) is not None

        return [
            (f"{self.display_name} binary", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]
