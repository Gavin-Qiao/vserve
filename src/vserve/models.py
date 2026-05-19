"""Model scanning, detection, and fuzzy matching."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from vserve.model_files import iter_top_level_files_with_suffix

_log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    path: Path
    provider: str
    model_name: str
    architecture: str
    model_type: str
    quant_method: str | None
    max_position_embeddings: int
    is_moe: bool
    model_size_gb: float
    num_kv_heads: int | None = None
    num_layers: int | None = None
    head_dim: int | None = None
    is_gguf: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model_name}"

    @property
    def is_embedding(self) -> bool:
        name = self.model_name.lower()
        arch = self.architecture.lower()
        return (
            "embed" in name or "embed" in arch
            or "e5" in name.split("-")
            or "bge" in name.split("-")
            or "rerank" in name
        )

    @property
    def is_unsloth_ud(self) -> bool:
        """True when this is an Unsloth Dynamic 2.0 (UD-) quantized GGUF.

        Unsloth's UD- quants (Q4_K_XL, Q5_K_XL, IQ4_XS variants in the
        unsloth/* repos) are imatrix-calibrated against calibration_v5,
        documented at ~99.9% KL divergence vs FP16. Detection is filename-
        based — the UD- prefix marks the calibrated build.
        """
        if not self.is_gguf or self.provider.lower() != "unsloth":
            return False
        # Match any GGUF file in the model dir whose name contains a
        # `UD-` prefix on the quant tag (e.g. `model-UD-Q4_K_XL.gguf`).
        try:
            for path in self.path.iterdir():
                if path.suffix.lower() == ".gguf" and "-UD-" in path.name:
                    return True
        except OSError:
            return False
        return False


QUANT_FLAGS = {
    "gptq": "--quantization gptq_marlin",
    "awq": "--quantization awq_marlin",
    "fp8": "--quantization fp8",
    "compressed-tensors": "",
    "none": "",
}


def _validate_index_shards(model_dir: Path) -> None:
    for index_path in sorted(model_dir.glob("*.index*.json")):
        if not ("safetensors.index" in index_path.name or "bin.index" in index_path.name):
            continue
        try:
            data = json.loads(index_path.read_text())
        except Exception as exc:
            raise ValueError(f"invalid weight index {index_path.name}: {exc}") from None
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight index {index_path.name}: missing weight_map")
        missing = sorted({
            str(shard)
            for shard in weight_map.values()
            if not (index_path.parent / str(shard)).exists()
        })
        if missing:
            raise ValueError(f"{model_dir} missing shard(s) referenced by {index_path.name}: {', '.join(missing)}")


def quant_flag(method: str | None) -> str:
    if method is None:
        return ""
    return QUANT_FLAGS.get(method, "")


def detect_model(model_dir: Path) -> ModelInfo:
    config_path = model_dir / "config.json"
    gguf_files = iter_top_level_files_with_suffix(model_dir, ".gguf")

    has_safetensors = bool(iter_top_level_files_with_suffix(model_dir, ".safetensors"))
    has_bin = bool(iter_top_level_files_with_suffix(model_dir, ".bin"))
    if gguf_files and not has_safetensors and not has_bin:
        # GGUF model — may or may not have config.json alongside
        size_bytes = sum(f.stat().st_size for f in gguf_files)
        size_gb = round(size_bytes / (1024**3), 1)
        return ModelInfo(
            path=model_dir,
            provider=model_dir.parent.name,
            model_name=model_dir.name,
            architecture="gguf",
            model_type="gguf",
            quant_method=None,
            max_position_embeddings=0,  # read from GGUF metadata at tune time
            is_moe=False,
            model_size_gb=size_gb,
            is_gguf=True,
        )

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)
    _validate_index_shards(model_dir)

    text_config = config.get("text_config", {})

    arch = (config.get("architectures") or ["unknown"])[0]
    model_type = config.get("model_type", "unknown")
    qmethod = (config.get("quantization_config") or {}).get("quant_method")
    max_pos = text_config.get(
        "max_position_embeddings",
        config.get("max_position_embeddings", 32768),
    )
    num_experts = text_config.get(
        "num_experts",
        config.get("num_experts",
                    text_config.get("num_local_experts",
                                    config.get("num_local_experts"))),
    )
    is_moe = num_experts is not None and num_experts > 0

    # Architecture fields for KV cache calculation
    num_kv_heads = text_config.get(
        "num_key_value_heads",
        config.get("num_key_value_heads"),
    )
    num_layers = text_config.get(
        "num_hidden_layers",
        config.get("num_hidden_layers"),
    )
    hidden_size = text_config.get("hidden_size", config.get("hidden_size"))
    num_attn_heads = text_config.get(
        "num_attention_heads",
        config.get("num_attention_heads"),
    )
    if num_kv_heads is None and num_attn_heads is not None:
        num_kv_heads = num_attn_heads
    head_dim_val = text_config.get("head_dim", config.get("head_dim"))
    if head_dim_val is None and hidden_size and num_attn_heads:
        head_dim_val = hidden_size // num_attn_heads

    size_bytes = sum(f.stat().st_size for f in iter_top_level_files_with_suffix(model_dir, ".safetensors"))
    size_bytes += sum(f.stat().st_size for f in iter_top_level_files_with_suffix(model_dir, ".bin"))
    size_gb = round(size_bytes / (1024**3), 1)

    return ModelInfo(
        path=model_dir,
        provider=model_dir.parent.name,
        model_name=model_dir.name,
        architecture=arch,
        model_type=model_type,
        quant_method=qmethod,
        max_position_embeddings=max_pos,
        is_moe=is_moe,
        model_size_gb=size_gb,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        head_dim=head_dim_val,
    )


def scan_models(models_root: Path) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    if not models_root.exists():
        return models
    try:
        provider_dirs = sorted(models_root.iterdir())
    except OSError as exc:
        _log.warning("Skipping models root %s: %s", models_root, exc)
        return models
    for provider_dir in provider_dirs:
        if provider_dir.is_symlink() or not provider_dir.is_dir():
            continue
        try:
            model_dirs = sorted(provider_dir.iterdir())
        except OSError as exc:
            _log.warning("Skipping provider directory %s: %s", provider_dir, exc)
            continue
        for model_dir in model_dirs:
            if model_dir.is_symlink() or not model_dir.is_dir():
                continue
            if (model_dir / ".vserve-ignore").exists():
                continue
            try:
                has_config = (model_dir / "config.json").exists()
                has_gguf = bool(iter_top_level_files_with_suffix(model_dir, ".gguf"))
                if not has_config and not has_gguf:
                    continue
                models.append(detect_model(model_dir))
            except Exception as exc:
                _log.warning("Skipping model directory %s: %s", model_dir, exc)
                continue
    return models


def fuzzy_match(query: str, models: list[ModelInfo]) -> list[ModelInfo]:
    query = query.strip()
    for m in models:
        if query == str(m.path):
            return [m]

    for m in models:
        if query == m.full_name:
            return [m]

    query_lower = query.lower()
    contiguous = [
        m
        for m in models
        if query_lower in m.model_name.lower()
        or query_lower in m.full_name.lower()
    ]
    if contiguous:
        return contiguous

    tokens = [t for t in re.split(r"[^a-z0-9]+", query_lower) if t]
    if not tokens:
        return []

    ranked: list[tuple[int, str, ModelInfo]] = []
    for m in models:
        features = [
            m.provider,
            m.model_name,
            m.full_name,
            m.architecture,
            m.model_type,
            m.quant_method or "",
            "gguf" if m.is_gguf else "safetensors",
            "embedding" if m.is_embedding else "",
        ]
        haystack = " ".join(features).lower()
        if not all(token in haystack for token in tokens):
            continue
        score = 0
        name_lower = m.model_name.lower()
        full_lower = m.full_name.lower()
        for token in tokens:
            if token == (m.quant_method or "").lower():
                score += 8
            if token in name_lower:
                score += 4
            if token in full_lower:
                score += 2
            if token == m.provider.lower():
                score += 1
        ranked.append((-score, m.full_name.lower(), m))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [m for _score, _name, m in ranked]
