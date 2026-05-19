"""vLLM backend — wraps existing serve/probe/tools modules."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo




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
        """
        self._validate_parsers(choices)
        cfg: dict = {
            "model": str(model.path),
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
        if min_bt is not None:
            if bt is None:
                bt = min_bt
            elif isinstance(bt, int) and not isinstance(bt, bool):
                bt = max(bt, min_bt)
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

        if choices.get("tools") and choices.get("tool_parser"):
            cfg["enable-auto-tool-choice"] = True
            cfg["tool-call-parser"] = choices["tool_parser"]
        rp = choices.get("reasoning_parser")
        if rp:
            cfg["reasoning-parser"] = rp

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
