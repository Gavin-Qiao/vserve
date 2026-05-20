"""llama.cpp backend — serves GGUF models via llama-server."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable

from vserve.model_files import iter_top_level_files_with_suffix

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


# llama.cpp KV-cache element sizes (bytes per element), computed from the
# packed block layout in ggml-quants.h. Block size is 32 elements; the per-
# block overhead (scale, min, packed weights) yields the fractional totals.
#
# - f16/bf16 = 2 bytes (default)
# - q8_0     = 32 q8 weights + 1×fp16 scale  → 34 / 32 = 1.0625
# - q5_1     = 32×5b packed (20) + 2×fp16     → 24 / 32 = 0.75
# - q5_0     = 32×5b packed (20) + 1×fp16     → 22 / 32 = 0.6875
# - q4_1     = 32×4b packed (16) + 2×fp16     → 20 / 32 = 0.625
# - q4_0     = 32×4b packed (16) + 1×fp16     → 18 / 32 = 0.5625
# - iq4_nl   ≈ q4_0
_KV_DTYPE_BYTES_PER_ELEMENT: dict[str, float] = {
    "f16": 2.0,
    "bf16": 2.0,
    "f32": 4.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}

# K/V dtype candidates surfaced in tune output. We restrict the matrix to
# symmetric pairs because llama.cpp's fused Flash-Attention kernel falls back
# to a much slower path when K and V types don't match (llamacpp#22411).
# iq4_nl is gated on the GGML_CUDA_FA_ALL_QUANTS build flag — without it,
# the iq4_nl pair silently falls back to non-fused FA.
_DEFAULT_KV_DTYPES: tuple[str, ...] = ("f16", "bf16", "q8_0", "q5_1", "q4_1", "q4_0")
_FA_ALL_QUANTS_GATED_DTYPES: frozenset[str] = frozenset({"iq4_nl"})

_LLAMACPP_KV_DTYPE_QUALITY: dict[str, str] = {
    "f16":    "native",
    "bf16":   "native (newer GPUs)",
    "q8_0":   "lossy; widely documented as near-zero quality cost",
    "q5_1":   "lossy; middle-ground between q8_0 and q4_1",
    "q4_1":   "lossy; aggressive — validate with eval before adopting",
    "q4_0":   "lossy; most aggressive — validate with eval before adopting",
    "iq4_nl": "lossy; most accurate 4-bit — requires GGML_CUDA_FA_ALL_QUANTS=ON",
}


class LlamaCppBackend:
    name = "llamacpp"
    display_name = "llama.cpp"

    @property
    def service_name(self) -> str:
        from vserve.config import cfg

        value = getattr(cfg(), "llamacpp_service_name", None)
        return value if isinstance(value, str) else "llama-cpp"

    @property
    def service_user(self) -> str:
        from vserve.config import cfg

        value = getattr(cfg(), "llamacpp_service_user", None)
        return value if isinstance(value, str) else "llama-cpp"

    @property
    def root_dir(self) -> Path:
        from vserve.config import cfg
        root = getattr(cfg(), "llamacpp_root", None)
        return root if isinstance(root, Path) else Path("/opt/llama-cpp")

    def can_serve(self, model: ModelInfo) -> bool:
        """True if model directory contains GGUF files."""
        return getattr(model, "is_gguf", False) or bool(iter_top_level_files_with_suffix(model.path, ".gguf"))

    def find_entrypoint(self) -> Path | None:
        candidate = self.root_dir / "bin" / "llama-server"
        if candidate.exists():
            return candidate
        found = shutil.which("llama-server")
        return Path(found) if found else None

    def _assert_unit_safe_for_privileged_action(self) -> None:
        from vserve.config import (
            find_systemd_unit_path,
            unit_content_matches_backend,
            validate_systemd_service_name,
        )

        validate_systemd_service_name(self.service_name)
        unit = find_systemd_unit_path(self.service_name)
        if unit is None:
            return
        try:
            content = unit.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"Cannot verify {self.service_name}.service before privileged systemctl action: {exc}"
            ) from None
        if not unit_content_matches_backend(
            content,
            backend_name=self.name,
            root=self.root_dir,
            expected_paths=[self.root_dir / "configs" / "active.sh"],
        ):
            raise RuntimeError(f"{self.service_name}.service does not look like a vserve llama.cpp unit")

    def runtime_info(self) -> dict:
        entrypoint = self.find_entrypoint()
        errors: list[str] = []
        version: str | None = None
        if entrypoint is None:
            errors.append("llama-server entrypoint not found")
        else:
            try:
                result = subprocess.run(
                    [str(entrypoint), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                output = result.stdout.strip() or result.stderr.strip()
                if result.returncode == 0:
                    version = output or None
                else:
                    errors.append(output or f"{entrypoint} --version failed")
            except Exception as exc:
                errors.append(f"{entrypoint} --version failed: {exc}")
        return {
            "backend": self.name,
            "executable": str(entrypoint or ""),
            "llama_server_version": version,
            "errors": errors,
        }

    def compatibility(self) -> dict:
        info = self.runtime_info()
        errors = list(info.get("errors") or [])
        return {
            "backend": self.name,
            "supported": not errors,
            "messages": ["llama.cpp runtime is available."] if not errors else [],
            "warnings": [],
            "errors": errors,
        }

    def _candidate_kv_dtypes(self) -> tuple[str, ...]:
        """Return the KV-cache dtype candidates surfaced in tune output,
        filtered by build flags. iq4_nl is gated on GGML_CUDA_FA_ALL_QUANTS
        being set at build time (otherwise the fused FA kernel can't use it).
        """
        candidates: list[str] = list(_DEFAULT_KV_DTYPES)
        try:
            from vserve.llamacpp_probe import probe_llama_cpp_build
            ep = self.find_entrypoint()
            info = probe_llama_cpp_build(ep) if ep is not None else None
            allow_fa_all_quants = bool(info and info.fa_all_quants)
        except Exception:
            allow_fa_all_quants = False
        if allow_fa_all_quants:
            # Append after q4_0 in quality order so existing matrices don't
            # change column ordering for builds without the flag.
            candidates.append("iq4_nl")
        return tuple(candidates)

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        """Calculate n-gpu-layers, context sizes, and a context × KV-dtype matrix.

        The matrix mirrors the vLLM tune output: rows = context steps, columns =
        symmetric (K, V) dtype pairs from `_DEFAULT_KV_DTYPES`. q8_0 typically
        halves KV memory at near-zero quality loss; q4_0/q4_1 cut further with
        more risk. Fused Flash-Attention requires K and V dtypes to match, so
        only symmetric pairs are surfaced.
        """
        selected = self._select_gguf_model_files(model.path)
        if selected is None:
            raise ValueError(f"No GGUF files in {model.path}")
        primary_file, model_files = selected
        model_size_bytes = sum(f.stat().st_size for f in model_files)
        model_size_gb = model_size_bytes / (1024**3)

        metadata = self._read_gguf_metadata(primary_file)
        num_layers = metadata.get("num_layers", 32)
        max_context = metadata.get("max_context", 4096)

        usable_gb = gpu.vram_total_gb * gpu_mem_util
        layer_size_gb = model_size_gb / num_layers if num_layers > 0 else model_size_gb

        # n-gpu-layers (with 10% scratch buffer for embeddings)
        available_for_layers = usable_gb * 0.9
        n_gpu_layers = min(num_layers, int(available_for_layers / layer_size_gb)) if layer_size_gb > 0 else num_layers
        full_offload = n_gpu_layers >= num_layers

        # Compute/runtime VRAM reserve held back from the KV-cache budget.
        # llama-server uses VRAM beyond model + KV for: prompt-prefill compute
        # buffers (scale ~linearly with -np × ubatch), CUDA workspace, pinned
        # host-side transfer buffers, the embedding/output projection, and a
        # small per-graph scratch. Without this reserve, the tuner produced
        # configs that fit in pure KV math but at runtime llama.cpp silently
        # shed tensors back to host RAM (observed: 22 slots × 128k q8_0 →
        # 40 GB RAM peak as experts traversed PCIe). 10% of total VRAM
        # (~5 GB on a 48 GB card) is a conservative floor that prevents the
        # silent CPU-spill failure mode on the high-parallel configs we
        # actually emit.
        compute_reserve_gb = gpu.vram_total_gb * 0.10
        gpu_model_gb = n_gpu_layers * layer_size_gb
        remaining_gb = usable_gb - gpu_model_gb - compute_reserve_gb
        remaining_bytes = int(max(0, remaining_gb) * (1024**3))

        from vserve.probe import _context_steps
        steps = _context_steps(max_context) if max_context >= 4096 else [4096]

        # I: filter iq4_nl out of the candidate matrix when the binary was
        # not built with GGML_CUDA_FA_ALL_QUANTS=ON. Without that flag, the
        # fused FA kernel silently falls back to a dequant-on-the-fly path.
        candidate_dtypes = self._candidate_kv_dtypes()

        # 2D matrix: { "<ctx>": { "f16": slots, "q8_0": slots, ... } }
        limits: dict[str, dict[str, int | None]] = {}
        for ctx in steps:
            row: dict[str, int | None] = {}
            for dtype in candidate_dtypes:
                slots = self._max_parallel_slots_for_context(
                    metadata, remaining_bytes, ctx, k_dtype=dtype, v_dtype=dtype
                )
                row[dtype] = slots if slots >= 1 else None
            limits[str(ctx)] = row

        kv_dtype_profiles = self._kv_dtype_profiles(metadata)
        recommended_kv = self._recommended_kv_dtype(limits)

        # MoE awareness — when the model has expert_count > 1, surface a
        # second slot table computed with `-ot ".ffn_.*_exps.=CPU"` so the
        # user can see how much additional KV space CPU-offloading the
        # expert FFNs would free.
        moe_block = self._moe_block(
            metadata,
            model_size_gb=model_size_gb,
            num_layers=num_layers,
            usable_gb=usable_gb,
            steps=steps,
        )

        # Embedding model detection
        is_embedding = model.is_embedding
        pooling = metadata.get("pooling")
        if is_embedding and not pooling:
            pooling = self._guess_pooling(model.model_name)

        tool_info = self.detect_tools(model.path) if not is_embedding else {}

        from datetime import datetime, timezone
        result: dict = {
            "backend": "llamacpp",
            "model_path": str(model.path),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "vram_total_gb": gpu.vram_total_gb,
            "gpu_memory_utilization": gpu_mem_util,
            "model_size_gb": round(model_size_gb, 1),
            "n_gpu_layers": n_gpu_layers,
            "num_layers": num_layers,
            "full_offload": full_offload,
            "max_context": max_context,
            "supports_tools": tool_info.get("supports_tools", False),
            "supports_reasoning": tool_info.get("supports_reasoning", False),
            "kv_cache_dtypes": kv_dtype_profiles,
            "recommended_kv_dtype": recommended_kv,
            "limits": limits,
        }
        if moe_block is not None:
            result["moe"] = moe_block
        if not metadata:
            result["metadata_estimated"] = True
        if is_embedding:
            result["is_embedding"] = True
            result["pooling"] = pooling or "mean"
        return result

    def _moe_block(
        self,
        metadata: dict,
        *,
        model_size_gb: float,
        num_layers: int,
        usable_gb: float,
        steps: list[int],
    ) -> dict | None:
        """Compute MoE expert-CPU-offload (`-ot`) capacity numbers.

        Returns None for non-MoE models. For MoE models, returns:
        - the recommended -ot pattern
        - estimated GPU-resident model size with `-ot` applied
        - estimated VRAM freed
        - a second (context × kv_dtype) slot table assuming `-ot` is on

        Math: per-layer GPU element count without -ot is approximated as
        attention + shared FFN + (expert_count × expert FFN). With -ot, the
        expert FFN block moves to CPU, leaving attention + shared FFN.
        The ratio gives an estimate of the GPU-resident fraction.
        """
        expert_count = self._positive_int(metadata.get("expert_count")) or 0
        if expert_count <= 1:
            return None

        embedding = self._positive_int(metadata.get("embedding_length")) or 0
        expert_ffn = self._positive_int(metadata.get("expert_feed_forward_length")) or 0
        shared_ffn = self._positive_int(metadata.get("feed_forward_length")) or 0
        if embedding <= 0 or expert_ffn <= 0:
            # Can't estimate without these — surface a degenerate MoE block
            return {
                "is_moe": True,
                "expert_count": expert_count,
                "expert_used_count": self._positive_int(metadata.get("expert_used_count")) or 0,
                "ot_pattern": ".ffn_.*_exps.=CPU",
                "estimated_gpu_freed_gb": None,
                "limits_with_ot": None,
                "note": "MoE detected but expert FFN dimensions missing — could not estimate -ot savings",
            }

        # Element counts per layer (SwiGLU uses 3 projections: gate, up, down).
        attn_elements = 4 * embedding * embedding  # q,k,v,o
        shared_ffn_elements = 3 * embedding * shared_ffn if shared_ffn else 0
        expert_ffn_elements = expert_count * 3 * embedding * expert_ffn
        non_expert_elements = attn_elements + shared_ffn_elements
        total_layer_elements = non_expert_elements + expert_ffn_elements
        if total_layer_elements <= 0:
            return None

        expert_fraction = expert_ffn_elements / total_layer_elements
        non_expert_fraction = 1.0 - expert_fraction

        # Approximate GPU-resident model size with `-ot` applied. We keep the
        # non-layer overhead (embedding table, output projection) on GPU; that
        # overhead is approximated as 5% of the file in the absence of a
        # better signal.
        non_layer_fraction = 0.05
        gpu_resident_with_ot_gb = (
            model_size_gb * non_layer_fraction
            + model_size_gb * (1 - non_layer_fraction) * non_expert_fraction
        )
        freed_gb = model_size_gb - gpu_resident_with_ot_gb

        # Recompute KV cache room with the freed VRAM.
        with_ot_usable = usable_gb - gpu_resident_with_ot_gb
        with_ot_remaining_gb = max(0.0, with_ot_usable - usable_gb * 0.1)  # keep 10% scratch
        with_ot_remaining_bytes = int(with_ot_remaining_gb * (1024**3))

        limits_with_ot: dict[str, dict[str, int | None]] = {}
        for ctx in steps:
            row: dict[str, int | None] = {}
            for dtype in _DEFAULT_KV_DTYPES:
                slots = self._max_parallel_slots_for_context(
                    metadata, with_ot_remaining_bytes, ctx, k_dtype=dtype, v_dtype=dtype
                )
                row[dtype] = slots if slots >= 1 else None
            limits_with_ot[str(ctx)] = row

        return {
            "is_moe": True,
            "expert_count": expert_count,
            "expert_used_count": self._positive_int(metadata.get("expert_used_count")) or 0,
            "expert_ffn_fraction": round(expert_fraction, 3),
            "ot_pattern": ".ffn_.*_exps.=CPU",
            "estimated_gpu_resident_gb": round(gpu_resident_with_ot_gb, 1),
            "estimated_gpu_freed_gb": round(freed_gb, 1),
            "limits_with_ot": limits_with_ot,
        }

    @staticmethod
    def _kv_dtype_profiles(metadata: dict) -> dict[str, dict[str, object]]:
        """Per-dtype byte cost relative to f16, mirroring the vLLM tune block."""
        out: dict[str, dict[str, object]] = {}
        f16_bpe = _KV_DTYPE_BYTES_PER_ELEMENT["f16"]
        for dtype in _DEFAULT_KV_DTYPES:
            bpe = _KV_DTYPE_BYTES_PER_ELEMENT.get(dtype, 2.0)
            out[dtype] = {
                "bytes_per_element": bpe,
                "compression_ratio_vs_f16": round(f16_bpe / bpe, 2) if bpe else None,
                "quality": _LLAMACPP_KV_DTYPE_QUALITY[dtype],
            }
        return out

    @staticmethod
    def _recommended_kv_dtype(limits: dict[str, dict[str, int | None]]) -> str | None:
        """Default to f16 for safety; prefer q8_0 if it strictly beats f16 at
        the largest context that fits any dtype, since q8_0 KV is documented
        as having near-zero quality cost and unlocks ~2× more slots."""
        if not limits:
            return None
        # Pick the largest context where at least one dtype fits.
        for ctx_str in sorted(limits, key=int, reverse=True):
            row = limits[ctx_str]
            f16 = row.get("f16")
            q8 = row.get("q8_0")
            if q8 is not None and (f16 is None or q8 > f16):
                return "q8_0"
            if f16 is not None:
                return "f16"
        return None

    @staticmethod
    def _select_gguf_model_files(model_path: Path) -> tuple[Path, list[Path]] | None:
        """Select one coherent GGUF variant from a model directory."""
        import re

        gguf_files = iter_top_level_files_with_suffix(model_path, ".gguf")
        if not gguf_files:
            return None

        shard_pattern = re.compile(r"(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", flags=re.IGNORECASE)
        groups: dict[str, dict[int, Path]] = {}
        totals: dict[str, int] = {}
        singles: list[Path] = []
        for path in gguf_files:
            match = shard_pattern.match(path.name)
            if match:
                base = match.group("base")
                groups.setdefault(base, {})[int(match.group("idx"))] = path
                totals[base] = int(match.group("total"))
            else:
                singles.append(path)

        incomplete: list[str] = []
        candidates: list[list[Path]] = []
        for base, paths_by_index in groups.items():
            total = totals[base]
            expected = set(range(1, total + 1))
            found = set(paths_by_index)
            if found != expected:
                missing = ", ".join(f"{idx:05d}" for idx in sorted(expected - found))
                incomplete.append(f"{base} missing shard(s): {missing}")
                continue
            candidates.append([paths_by_index[idx] for idx in sorted(paths_by_index)])
        candidates.extend([[path] for path in singles])
        if not candidates and incomplete:
            raise ValueError(f"Incomplete split GGUF shard set in {model_path}: {'; '.join(incomplete)}")
        best = max(candidates, key=lambda files: (sum(path.stat().st_size for path in files), files[0].name))
        return best[0], best

    @staticmethod
    def _positive_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value > 0 else None

    @classmethod
    def _total_kv_heads(cls, value: object, num_layers: int) -> int:
        default = 8 * max(1, num_layers)
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return max(1, value) * max(1, num_layers)
        if isinstance(value, list):
            heads = [item for item in (cls._positive_int(item) for item in value) if item is not None]
            if not heads:
                return default
            if len(heads) >= num_layers:
                return sum(heads[:num_layers])
            return sum(heads) + heads[-1] * max(0, num_layers - len(heads))
        return default

    @classmethod
    def _max_parallel_slots_for_context(
        cls,
        metadata: dict,
        remaining_bytes: int,
        context: int,
        *,
        k_dtype: str = "f16",
        v_dtype: str = "f16",
    ) -> int:
        if context <= 0 or remaining_bytes <= 0:
            return 0
        first_slot_bytes = cls._llamacpp_kv_cache_bytes(
            metadata, context=context, parallel=1, k_dtype=k_dtype, v_dtype=v_dtype
        )
        if first_slot_bytes <= 0 or first_slot_bytes > remaining_bytes:
            return 0

        low = 1
        high = 1
        while high < 4096:
            candidate = min(high * 2, 4096)
            if cls._llamacpp_kv_cache_bytes(
                metadata, context=context, parallel=candidate, k_dtype=k_dtype, v_dtype=v_dtype
            ) > remaining_bytes:
                high = candidate - 1
                break
            high = candidate
            if high == 4096:
                break

        while low < high:
            mid = (low + high + 1) // 2
            if cls._llamacpp_kv_cache_bytes(
                metadata, context=context, parallel=mid, k_dtype=k_dtype, v_dtype=v_dtype
            ) <= remaining_bytes:
                low = mid
            else:
                high = mid - 1
        return low

    @classmethod
    def _llamacpp_kv_cache_bytes(
        cls,
        metadata: dict,
        *,
        context: int,
        parallel: int,
        k_dtype: str = "f16",
        v_dtype: str = "f16",
    ) -> int:
        num_layers = cls._positive_int(metadata.get("num_layers")) or 32
        key_length = cls._positive_int(metadata.get("key_length")) or cls._positive_int(metadata.get("head_dim")) or 128
        value_length = cls._positive_int(metadata.get("value_length")) or cls._positive_int(metadata.get("head_dim")) or key_length
        key_length_swa = cls._positive_int(metadata.get("key_length_swa")) or key_length
        value_length_swa = cls._positive_int(metadata.get("value_length_swa")) or value_length
        num_kv_heads = metadata.get("num_kv_heads", 8)
        k_bpe = _KV_DTYPE_BYTES_PER_ELEMENT.get(k_dtype, 2.0)
        v_bpe = _KV_DTYPE_BYTES_PER_ELEMENT.get(v_dtype, 2.0)

        total_f = 0.0
        for layer_index in range(num_layers):
            if cls._is_recurrent_layer(metadata, layer_index):
                continue
            heads = cls._kv_heads_for_layer(num_kv_heads, layer_index)
            if cls._is_swa_layer(metadata, layer_index):
                cells = cls._swa_cache_cells(metadata, context)
                total_f += heads * key_length_swa * k_bpe * cells * parallel
                total_f += heads * value_length_swa * v_bpe * cells * parallel
            else:
                total_f += heads * key_length * k_bpe * context * parallel
                total_f += heads * value_length * v_bpe * context * parallel
        # Recurrent state size is fp32-sized in llama.cpp and does not honour
        # --cache-type-k/v — keep it as the existing helper computes.
        recurrent = cls._llamacpp_recurrent_state_bytes(metadata, parallel=parallel)
        return int(total_f) + recurrent

    @classmethod
    def _kv_heads_for_layer(cls, value: object, layer_index: int) -> int:
        if isinstance(value, bool):
            return 8
        if isinstance(value, int):
            return max(1, value)
        if isinstance(value, list):
            heads = [item for item in (cls._positive_int(item) for item in value) if item is not None]
            if heads:
                return heads[layer_index] if layer_index < len(heads) else heads[-1]
        return 8

    @classmethod
    def _is_recurrent_layer(cls, metadata: dict, layer_index: int) -> bool:
        full_attention_interval = cls._positive_int(metadata.get("full_attention_interval"))
        if full_attention_interval is None:
            return False
        return (layer_index + 1) % full_attention_interval != 0

    @classmethod
    def _is_swa_layer(cls, metadata: dict, layer_index: int) -> bool:
        if cls._positive_int(metadata.get("sliding_window")) is None:
            return False
        pattern = metadata.get("sliding_window_pattern")
        if isinstance(pattern, list):
            values = []
            for item in pattern:
                if isinstance(item, bool):
                    values.append(int(item))
                elif isinstance(item, int):
                    values.append(int(item))
            if values:
                return bool(values[layer_index % len(values)])
        period = cls._positive_int(pattern)
        if period is not None:
            return layer_index % period < period - 1
        return False

    @classmethod
    def _swa_cache_cells(cls, metadata: dict, context: int) -> int:
        n_swa = cls._positive_int(metadata.get("sliding_window"))
        if n_swa is None:
            return context
        n_ubatch = 512
        size = min(context, n_swa + n_ubatch)
        return ((size + 255) // 256) * 256

    @classmethod
    def _llamacpp_recurrent_state_bytes(cls, metadata: dict, *, parallel: int) -> int:
        num_layers = cls._positive_int(metadata.get("num_layers")) or 32
        recurrent_layers = sum(1 for layer_index in range(num_layers) if cls._is_recurrent_layer(metadata, layer_index))
        if recurrent_layers <= 0:
            return 0
        n_embd_r, n_embd_s = cls._recurrent_state_dimensions(metadata)
        if n_embd_r <= 0 and n_embd_s <= 0:
            return 0
        rs_size = max(1, parallel)
        return recurrent_layers * (n_embd_r + n_embd_s) * rs_size * 4

    @classmethod
    def _recurrent_state_dimensions(cls, metadata: dict) -> tuple[int, int]:
        embedding_length = cls._positive_int(metadata.get("embedding_length")) or 0
        wkv_head_size = cls._positive_int(metadata.get("wkv_head_size")) or 0
        if wkv_head_size and embedding_length:
            token_shift_count = cls._positive_int(metadata.get("token_shift_count")) or 2
            return token_shift_count * embedding_length, embedding_length * wkv_head_size

        shortconv_l_cache = cls._positive_int(metadata.get("shortconv_l_cache")) or 0
        if shortconv_l_cache and embedding_length:
            return embedding_length * max(0, shortconv_l_cache - 1), 0

        kda_head_dim = cls._positive_int(metadata.get("kda_head_dim")) or 0
        if kda_head_dim:
            num_attn_heads = cls._positive_int(metadata.get("num_attn_heads")) or 1
            ssm_conv_kernel = cls._positive_int(metadata.get("ssm_conv_kernel")) or 4
            d_inner = num_attn_heads * kda_head_dim
            return 3 * max(0, ssm_conv_kernel - 1) * d_inner, kda_head_dim * kda_head_dim * num_attn_heads

        ssm_conv_kernel = cls._positive_int(metadata.get("ssm_conv_kernel")) or 0
        ssm_inner_size = cls._positive_int(metadata.get("ssm_inner_size")) or 0
        ssm_state_size = cls._positive_int(metadata.get("ssm_state_size")) or 0
        ssm_group_count = cls._positive_int(metadata.get("ssm_group_count")) or 0
        if not (ssm_conv_kernel and ssm_inner_size and ssm_state_size):
            return 0, 0
        n_embd_r = max(0, ssm_conv_kernel - 1) * (ssm_inner_size + 2 * ssm_group_count * ssm_state_size)
        n_embd_s = ssm_state_size * ssm_inner_size
        return n_embd_r, n_embd_s

    def _read_gguf_metadata(self, gguf_path: Path) -> dict:
        """Read model metadata from GGUF file header."""
        metadata = self._read_gguf_header_metadata(gguf_path)
        if metadata:
            return metadata
        try:
            from gguf import GGUFReader  # type: ignore[import-not-found, import-untyped]
            reader = GGUFReader(str(gguf_path))

            arch = "llama"
            arch_field = reader.fields.get("general.architecture")
            if arch_field is not None:
                raw = arch_field.parts[-1]
                if isinstance(raw, bytes):
                    arch = raw.decode("utf-8")
                elif hasattr(raw, 'tobytes'):
                    arch = raw.tobytes().decode("utf-8")
                else:
                    arch = str(raw)

            def _get_int(key: str, default: int) -> int:
                field = reader.fields.get(key)
                if field is not None:
                    val = field.parts[-1]
                    # gguf returns numpy arrays; extract scalar
                    return int(val[0]) if hasattr(val, '__len__') and not isinstance(val, (str, bytes)) else int(val)
                return default

            num_layers = _get_int(f"{arch}.block_count", 32)
            max_context = _get_int(f"{arch}.context_length", 4096)
            num_kv_heads = _get_int(f"{arch}.attention.head_count_kv", 8)
            num_attn_heads = _get_int(f"{arch}.attention.head_count", 32)
            embedding_length = _get_int(f"{arch}.embedding_length", 4096)
            head_dim = embedding_length // num_attn_heads if num_attn_heads else 128
            key_length = _get_int(f"{arch}.attention.key_length", head_dim)
            value_length = _get_int(f"{arch}.attention.value_length", head_dim)

            # Detect pooling type (embedding models)
            pooling_field = reader.fields.get("tokenizer.ggml.pooling_type")
            pooling = None
            if pooling_field is not None:
                val = pooling_field.parts[-1]
                pval = int(val[0]) if hasattr(val, '__len__') and not isinstance(val, (str, bytes)) else int(val)
                # llama.cpp: 0=none, 1=mean, 2=cls, 3=last, 4=rank
                pooling = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}.get(pval)

            return {
                "arch": arch,
                "num_layers": num_layers,
                "max_context": max_context,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "key_length": key_length,
                "value_length": value_length,
                "pooling": pooling,
            }
        except ImportError:
            metadata = self._read_gguf_header_metadata(gguf_path)
            if metadata:
                return metadata
            import sys
            print(
                "[vserve] warning: gguf package not installed; estimating llama.cpp tuning metadata",
                file=sys.stderr,
            )
            return {}
        except Exception:
            metadata = self._read_gguf_header_metadata(gguf_path)
            if metadata:
                return metadata
            import sys
            print(f"[vserve] warning: failed to read GGUF metadata from {gguf_path}", file=sys.stderr)
            return {}

    @classmethod
    def _read_gguf_header_metadata(cls, gguf_path: Path) -> dict:
        """Read the small GGUF metadata table without importing the optional gguf package."""
        try:
            values: dict[str, Any] = {}
            with gguf_path.open("rb") as fh:
                if cls._read_exact(fh, 4) != b"GGUF":
                    return {}
                version = cls._read_u32(fh)
                if version not in {1, 2, 3}:
                    return {}
                _tensor_count = cls._read_u64(fh)
                metadata_count = cls._read_u64(fh)
                arch: str | None = None
                for _ in range(metadata_count):
                    key = cls._read_gguf_string(fh)
                    value_type = cls._read_u32(fh)
                    if key == "tokenizer.ggml.tokens" and arch and cls._has_core_gguf_metadata(values, arch):
                        break
                    keep = cls._is_wanted_gguf_metadata_key(key)
                    value = cls._read_gguf_value(fh, value_type, keep=keep)
                    if keep:
                        values[key] = value
                    if key == "general.architecture" and isinstance(value, str):
                        arch = value
            return cls._normalize_gguf_metadata(values)
        except (OSError, UnicodeDecodeError, ValueError, struct.error):
            return {}

    @staticmethod
    def _is_wanted_gguf_metadata_key(key: str) -> bool:
        return (
            key == "general.architecture"
            or key == "tokenizer.ggml.pooling_type"
            or key.endswith(".block_count")
            or key.endswith(".context_length")
            or key.endswith(".embedding_length")
            or key.endswith(".feed_forward_length")
            or key.endswith(".expert_count")
            or key.endswith(".expert_used_count")
            or key.endswith(".expert_feed_forward_length")
            or key.endswith(".expert_shared_count")
            or key.endswith(".attention.head_count")
            or key.endswith(".attention.head_count_kv")
            or key.endswith(".attention.key_length")
            or key.endswith(".attention.value_length")
            or key.endswith(".attention.key_length_swa")
            or key.endswith(".attention.value_length_swa")
            or key.endswith(".attention.sliding_window")
            or key.endswith(".attention.sliding_window_pattern")
            or key.endswith(".attention.shared_kv_layers")
            or key.endswith(".full_attention_interval")
            or key.endswith(".ssm.conv_kernel")
            or key.endswith(".ssm.inner_size")
            or key.endswith(".ssm.state_size")
            or key.endswith(".ssm.time_step_rank")
            or key.endswith(".ssm.group_count")
            or key.endswith(".kda.head_dim")
            or key.endswith(".wkv.head_size")
            or key.endswith(".token_shift_count")
            or key.endswith(".shortconv.l_cache")
        )

    @staticmethod
    def _has_core_gguf_metadata(values: dict[str, Any], arch: str) -> bool:
        return (
            f"{arch}.block_count" in values
            and f"{arch}.context_length" in values
            and f"{arch}.attention.head_count" in values
            and f"{arch}.attention.head_count_kv" in values
            and (
                f"{arch}.embedding_length" in values
                or (
                    f"{arch}.attention.key_length" in values
                    and f"{arch}.attention.value_length" in values
                )
            )
        )

    @classmethod
    def _normalize_gguf_metadata(cls, values: dict[str, Any]) -> dict:
        arch = values.get("general.architecture")
        if not isinstance(arch, str) or not arch:
            return {}
        num_layers = cls._gguf_int(values.get(f"{arch}.block_count"), 32)
        max_context = cls._gguf_int(values.get(f"{arch}.context_length"), 4096)
        num_attn_heads = cls._gguf_int(values.get(f"{arch}.attention.head_count"), 32)
        embedding_length = cls._gguf_int(values.get(f"{arch}.embedding_length"), 4096)
        head_dim = embedding_length // num_attn_heads if num_attn_heads else 128
        key_length = cls._gguf_int(values.get(f"{arch}.attention.key_length"), head_dim)
        value_length = cls._gguf_int(values.get(f"{arch}.attention.value_length"), head_dim)
        key_length_swa = cls._gguf_int(values.get(f"{arch}.attention.key_length_swa"), key_length)
        value_length_swa = cls._gguf_int(values.get(f"{arch}.attention.value_length_swa"), value_length)

        kv_heads_value = values.get(f"{arch}.attention.head_count_kv")
        if isinstance(kv_heads_value, list):
            num_kv_heads: int | list[int] = [
                int(item) for item in kv_heads_value
                if isinstance(item, int) and not isinstance(item, bool)
            ] or [8]
        else:
            num_kv_heads = cls._gguf_int(kv_heads_value, 8)

        pooling_raw = cls._gguf_int(values.get("tokenizer.ggml.pooling_type"), -1)
        pooling = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}.get(pooling_raw)
        sliding_window_pattern = values.get(f"{arch}.attention.sliding_window_pattern")
        if isinstance(sliding_window_pattern, list):
            pattern_values = []
            for item in sliding_window_pattern:
                if isinstance(item, bool):
                    pattern_values.append(int(item))
                elif isinstance(item, int):
                    pattern_values.append(int(item))
            sliding_window_pattern = pattern_values
        elif isinstance(sliding_window_pattern, int) and not isinstance(sliding_window_pattern, bool):
            sliding_window_pattern = int(sliding_window_pattern)
        else:
            sliding_window_pattern = None

        return {
            "arch": arch,
            "num_layers": num_layers,
            "max_context": max_context,
            "embedding_length": embedding_length,
            "feed_forward_length": cls._gguf_int(values.get(f"{arch}.feed_forward_length"), 0),
            "expert_count": cls._gguf_int(values.get(f"{arch}.expert_count"), 0),
            "expert_used_count": cls._gguf_int(values.get(f"{arch}.expert_used_count"), 0),
            "expert_feed_forward_length": cls._gguf_int(
                values.get(f"{arch}.expert_feed_forward_length"), 0
            ),
            "expert_shared_count": cls._gguf_int(values.get(f"{arch}.expert_shared_count"), 0),
            "num_attn_heads": num_attn_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": key_length,
            "key_length": key_length,
            "value_length": value_length,
            "key_length_swa": key_length_swa,
            "value_length_swa": value_length_swa,
            "sliding_window": cls._gguf_int(values.get(f"{arch}.attention.sliding_window"), 0),
            "sliding_window_pattern": sliding_window_pattern,
            "shared_kv_layers": cls._gguf_int(values.get(f"{arch}.attention.shared_kv_layers"), 0),
            "full_attention_interval": cls._gguf_int(values.get(f"{arch}.full_attention_interval"), 0),
            "ssm_conv_kernel": cls._gguf_int(values.get(f"{arch}.ssm.conv_kernel"), 0),
            "ssm_inner_size": cls._gguf_int(values.get(f"{arch}.ssm.inner_size"), 0),
            "ssm_state_size": cls._gguf_int(values.get(f"{arch}.ssm.state_size"), 0),
            "ssm_time_step_rank": cls._gguf_int(values.get(f"{arch}.ssm.time_step_rank"), 0),
            "ssm_group_count": cls._gguf_int(values.get(f"{arch}.ssm.group_count"), 0),
            "kda_head_dim": cls._gguf_int(values.get(f"{arch}.kda.head_dim"), 0),
            "wkv_head_size": cls._gguf_int(values.get(f"{arch}.wkv.head_size"), 0),
            "token_shift_count": cls._gguf_int(values.get(f"{arch}.token_shift_count"), 0),
            "shortconv_l_cache": cls._gguf_int(values.get(f"{arch}.shortconv.l_cache"), 0),
            "pooling": pooling,
        }

    @staticmethod
    def _gguf_int(value: object, default: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return int(value)
        return default

    @staticmethod
    def _read_exact(fh: BinaryIO, size: int) -> bytes:
        data = fh.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF metadata")
        return data

    @classmethod
    def _read_u32(cls, fh: BinaryIO) -> int:
        return struct.unpack("<I", cls._read_exact(fh, 4))[0]

    @classmethod
    def _read_u64(cls, fh: BinaryIO) -> int:
        return struct.unpack("<Q", cls._read_exact(fh, 8))[0]

    @classmethod
    def _read_gguf_string(cls, fh: BinaryIO) -> str:
        size = cls._read_u64(fh)
        return cls._read_exact(fh, size).decode("utf-8")

    @classmethod
    def _read_gguf_value(cls, fh: BinaryIO, value_type: int, *, keep: bool) -> Any:
        formats = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            10: "<Q",
            11: "<q",
            12: "<d",
        }
        if value_type in formats:
            fmt = formats[value_type]
            data = cls._read_exact(fh, struct.calcsize(fmt))
            return struct.unpack(fmt, data)[0] if keep else None
        if value_type == 7:
            data = cls._read_exact(fh, 1)
            return bool(struct.unpack("<?", data)[0]) if keep else None
        if value_type == 8:
            size = cls._read_u64(fh)
            if keep:
                return cls._read_exact(fh, size).decode("utf-8")
            cls._skip_bytes(fh, size)
            return None
        if value_type == 9:
            element_type = cls._read_u32(fh)
            length = cls._read_u64(fh)
            if keep:
                return [
                    cls._read_gguf_value(fh, element_type, keep=True)
                    for _ in range(length)
                ]
            fixed_size = cls._gguf_fixed_value_size(element_type)
            if fixed_size is not None:
                cls._skip_bytes(fh, fixed_size * length)
            else:
                for _ in range(length):
                    cls._read_gguf_value(fh, element_type, keep=False)
            return None
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")

    @staticmethod
    def _gguf_fixed_value_size(value_type: int) -> int | None:
        sizes = {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 1,
            10: 8,
            11: 8,
            12: 8,
        }
        return sizes.get(value_type)

    @staticmethod
    def _skip_bytes(fh: BinaryIO, size: int) -> None:
        if size <= 0:
            return
        try:
            fh.seek(size, 1)
        except OSError:
            remaining = size
            while remaining:
                chunk = fh.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("truncated GGUF metadata")
                remaining -= len(chunk)

    @staticmethod
    def _guess_pooling(model_name: str) -> str:
        """Guess pooling type from model name when GGUF metadata is absent."""
        name = model_name.lower()
        if "bge" in name:
            return "cls"
        if "rerank" in name:
            return "rank"
        # nomic, e5, jina, mxbai, snowflake, qwen-embed all use mean
        return "mean"

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build llama-server JSON config.

        Accepted choices keys:
            context, n_gpu_layers, parallel, port,
            tools, embedding, pooling,
            kv_cache_k, kv_cache_v   — symmetric pair recommended (fused FA)
            batch_size, ubatch_size  — llama-server -b / -ub
            override_tensors         — list of patterns for -ot, e.g.
                                       [".ffn_.*_exps.=CPU"] for MoE expert
                                       CPU offloading.

        Note on context semantics: `choices["context"]` is the per-slot
        sequence length the user actually wants (matching the column header
        in the tune output). llama-server's `-c / --ctx-size` is the *total*
        KV-cache size across all slots — per-slot context = ctx_size /
        parallel. We multiply here so each slot ends up with the requested
        window and the analytic KV-byte math stays consistent (the same math
        treats `context` as per-slot and multiplies by `parallel` for total
        cost).
        """
        from vserve.config import cfg as vserve_cfg

        selected = self._select_gguf_model_files(model.path)
        if selected is None:
            raise ValueError(f"No GGUF files in {model.path}")
        model_file = str(selected[0])

        parallel = int(choices.get("parallel", 1) or 1)
        per_slot_context = int(choices["context"])
        cfg: dict = {
            "model": model_file,
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            # ctx_size is the llama-server -c value: total context across slots.
            "ctx_size": per_slot_context * parallel,
            # Informational record of the per-slot window that vserve picked,
            # so a human reading the JSON can see what they asked for.
            "ctx_per_slot": per_slot_context,
            "n_gpu_layers": choices["n_gpu_layers"],
            "parallel": parallel,
            "flash_attn": True,
            "gpu_index": int(getattr(vserve_cfg(), "gpu_index", 0) or 0),
        }
        k_dtype = choices.get("kv_cache_k")
        v_dtype = choices.get("kv_cache_v")
        # I: Fused Flash-Attention requires K and V to share a dtype
        # (llamacpp#22411). Without symmetric pairs the engine silently falls
        # back to a much slower dequant-on-the-fly path. Refuse to emit
        # asymmetric pairs when -fa is on (the default).
        if (
            k_dtype is not None and v_dtype is not None
            and str(k_dtype) != str(v_dtype)
            and choices.get("flash_attn", True)
        ):
            raise ValueError(
                f"llama.cpp fused Flash-Attention requires K==V; got "
                f"K={k_dtype} V={v_dtype}. Pass symmetric --kv-cache-k/v, or "
                f"disable -fa via flash_attn=False (slower non-fused path)."
            )
        if k_dtype:
            cfg["cache_type_k"] = str(k_dtype)
        if v_dtype:
            cfg["cache_type_v"] = str(v_dtype)
        if choices.get("batch_size") is not None:
            cfg["batch_size"] = int(choices["batch_size"])
        if choices.get("ubatch_size") is not None:
            cfg["ubatch_size"] = int(choices["ubatch_size"])
        override = choices.get("override_tensors")
        if isinstance(override, list) and override:
            cfg["override_tensors"] = [str(item) for item in override]
        # J: mmproj-BF16.gguf for vision/audio models (Gemma-4 etc.).
        # Caller may opt out by passing mmproj=False or mmproj="" explicitly.
        mmproj_override = choices.get("mmproj")
        if mmproj_override is None:
            if model.mmproj_path is not None:
                cfg["mmproj"] = str(model.mmproj_path)
        elif mmproj_override:
            cfg["mmproj"] = str(mmproj_override)
        # H: prefer --n-cpu-moe N over the all-experts-CPU regex. ncmoe is
        # cleaner, version-stable, and well-defined ("first N MoE layers'
        # experts on CPU" counting from highest layer down). Keep the regex
        # path open for surgical layer-range cases.
        n_cpu_moe = choices.get("n_cpu_moe")
        if n_cpu_moe is None and isinstance(override, list) and override == [".ffn_.*_exps.=CPU"]:
            n_cpu_moe = 99  # 99 = "all expert layers" — llama.cpp clamps to actual count
            cfg.pop("override_tensors", None)
        if isinstance(n_cpu_moe, int) and not isinstance(n_cpu_moe, bool):
            cfg["n_cpu_moe"] = max(0, n_cpu_moe)
        if choices.get("embedding"):
            cfg["embedding"] = True
            if choices.get("pooling"):
                cfg["pooling"] = choices["pooling"]
        elif choices.get("tools"):
            cfg["jinja"] = True

        # Architecture-derived sampler defaults — unless the caller opted out.
        # Read the architecture from GGUF metadata (already used for KV math)
        # and map to recipe defaults. Caller-supplied sampler keys win.
        sampler_keys = (
            "temp", "top_p", "top_k", "min_p",
            "presence_penalty", "repeat_penalty",
        )
        for key in sampler_keys:
            if key in choices and choices[key] is not None:
                cfg[key] = choices[key]
        try:
            metadata = self._read_gguf_metadata(Path(model_file))
        except Exception:
            metadata = {}
        gguf_arch = metadata.get("arch") if metadata else None
        if choices.get("recipe_sampling", True):
            from vserve.recipes.sampling import get_sampling_defaults_from_gguf_arch
            defaults = get_sampling_defaults_from_gguf_arch(gguf_arch)
            if defaults is not None:
                for key in sampler_keys:
                    if key not in cfg:
                        val = getattr(defaults, key, None)
                        if val is not None:
                            cfg[key] = val

        # chat-template-kwargs (item C). Same architecture-aware kwarg
        # selection as the vLLM path. Stored as a JSON string that start()
        # passes to ``--chat-template-kwargs``.
        ctk: dict = {}
        explicit_ctk = choices.get("chat_template_kwargs")
        if isinstance(explicit_ctk, dict):
            ctk.update(explicit_ctk)
        thinking = choices.get("thinking")
        if thinking is not None and thinking != "auto":
            arch_lower = gguf_arch.lower() if isinstance(gguf_arch, str) else ""
            key = "thinking" if arch_lower.startswith("deepseek") else "enable_thinking"
            ctk[key] = bool(thinking) if isinstance(thinking, bool) else thinking in (True, "on", "true", 1)
        if ctk:
            cfg["chat_template_kwargs"] = ctk

        # Kimi-K2 thinking models hide `<think>` tokens unless --special is
        # passed (R2 §13). Detect by arch lower.
        if isinstance(gguf_arch, str) and gguf_arch.lower().startswith("kimi"):
            cfg.setdefault("special", True)

        # I: Gemma-4-31B + -fa + q8_0 V → illegal memory access after the
        # second SWA checkpoint (llamacpp#22527). Force V to f16 for that
        # specific architecture, regardless of what the picker chose.
        is_gemma4 = isinstance(gguf_arch, str) and gguf_arch.lower().startswith("gemma4")
        num_layers = (metadata or {}).get("num_layers") if isinstance(metadata, dict) else None
        if (
            is_gemma4 and isinstance(num_layers, int) and num_layers >= 60
            and cfg.get("flash_attn", True)
            and cfg.get("cache_type_v") == "q8_0"
        ):
            cfg["cache_type_v"] = "f16"
            # Surface a note so the tuner output / status display can warn.
            cfg.setdefault("notes", []).append(
                "Gemma-4-31B+FA+q8_0 V is unstable (llamacpp#22527); forcing V=f16."
            )

        # K: reasoning-format auto-emission. llama.cpp's `--reasoning-format`
        # picks the framing (harmony for <|channel|>, deepseek for <think>,
        # none for no separation). Without it, both leak into message.content
        # and clients can't split.
        if "reasoning_format" in choices and choices["reasoning_format"] is not None:
            cfg["reasoning_format"] = str(choices["reasoning_format"])
        else:
            template = self._read_gguf_chat_template(model.path) or ""
            if "<|channel|>" in template or "<|channel>" in template:
                cfg["reasoning_format"] = "harmony"
            elif "<think>" in template or "&lt;think&gt;" in template:
                cfg["reasoning_format"] = "deepseek"
        budget = choices.get("reasoning_budget")
        if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
            cfg["reasoning_budget"] = budget

        # S: prompt-caching primitives. --cache-reuse N (min chunk for KV
        # shifting within a slot), --slot-save-path (persistent snapshots),
        # --cram <MB> (host-memory prompt cache for shared prefixes), and
        # --swa-full (required for Gemma-4 + cache-reuse per llamacpp#21468).
        cache_reuse = choices.get("cache_reuse")
        slot_save_path = choices.get("slot_save_path")
        cram_mb = choices.get("cram_mb")
        swa_full = choices.get("swa_full")
        if isinstance(cache_reuse, int) and not isinstance(cache_reuse, bool) and cache_reuse > 0:
            cfg["cache_reuse"] = cache_reuse
        if isinstance(slot_save_path, str) and slot_save_path:
            cfg["slot_save_path"] = slot_save_path
        if isinstance(cram_mb, int) and not isinstance(cram_mb, bool) and cram_mb > 0:
            cfg["cram_mb"] = cram_mb
        if swa_full is True:
            cfg["swa_full"] = True
        # Gating: Gemma-4 + cache-reuse needs --swa-full (shared-KV layers
        # break the reuse heuristic without it — llamacpp#21468).
        if (
            is_gemma4 and cfg.get("cache_reuse")
            and not cfg.get("swa_full")
        ):
            raise ValueError(
                "Gemma-4 + --cache-reuse requires --swa-full to avoid "
                "stale-KV corruption (llamacpp#21468). Pass swa_full=True."
            )

        # M: speculative-decoding flags. Accept either a recipes.SpecConfig
        # (auto-picked) or a plain dict ({draft_model_path, n_max, n_min,
        # p_min}). Stored under ``spec_draft`` so start() can route to
        # --spec-draft-* flags.
        spec = choices.get("spec")
        if spec:
            spec_cls: type | None
            try:
                from vserve.recipes.spec_decode import SpecConfig as _SpecConfig
                spec_cls = _SpecConfig
            except Exception:
                spec_cls = None
            spec_dict: dict | None = None
            if spec_cls is not None and isinstance(spec, spec_cls):
                if spec.method == "ngram":
                    # llama.cpp has no equivalent zero-cost ngram path; skip.
                    spec_dict = None
                elif spec.draft_model_path is not None:
                    spec_dict = {
                        "draft_model_path": str(spec.draft_model_path),
                        "n_max": spec.n_max,
                        "n_min": spec.n_min,
                        "p_min": spec.p_min,
                    }
            elif isinstance(spec, dict):
                spec_dict = spec
            if spec_dict:
                cfg["spec_draft"] = spec_dict
        return cfg

    def quant_flag(self, method: str | None) -> str:
        """llama.cpp quant is baked into GGUF — no CLI flag needed."""
        return ""

    def start(self, config_path: Path, *, non_interactive: bool = False) -> None:
        """Write active launch script from JSON config and start systemd service."""
        import json

        # Read config and build CLI flags
        try:
            cfg = json.loads(config_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Invalid llama.cpp config {config_path}: {exc}") from None
        if not isinstance(cfg, dict):
            raise RuntimeError(f"Invalid llama.cpp config {config_path}: expected JSON object")
        entrypoint = self.find_entrypoint() or "llama-server"
        args = [str(entrypoint)]
        flag_map = {
            "model": "-m",
            "host": "--host",
            "port": "--port",
            "ctx_size": "-c",
            "n_gpu_layers": "-ngl",
            "parallel": "-np",
            "batch_size": "-b",
            "ubatch_size": "-ub",
            "cache_type_k": "-ctk",
            "cache_type_v": "-ctv",
            "n_cpu_moe": "-ncmoe",
            "mmproj": "--mmproj",
            "reasoning_format": "--reasoning-format",
            "reasoning_budget": "--reasoning-budget",
            "cache_reuse": "--cache-reuse",
            "slot_save_path": "--slot-save-path",
            "cram_mb": "--cram",
            "temp": "--temp",
            "top_p": "--top-p",
            "top_k": "--top-k",
            "min_p": "--min-p",
            "presence_penalty": "--presence-penalty",
            "repeat_penalty": "--repeat-penalty",
        }
        for key, flag in flag_map.items():
            if key in cfg:
                args.extend([flag, str(cfg[key])])
        # Boolean flags
        if cfg.get("flash_attn"):
            args.extend(["-fa", "on"])
        if cfg.get("embedding"):
            args.append("--embedding")
        if cfg.get("pooling"):
            args.extend(["--pooling", str(cfg["pooling"])])
        if cfg.get("jinja"):
            args.append("--jinja")
        if cfg.get("special"):
            # Kimi-K2 etc. — expose hidden tokens like `<think>` to the parser.
            args.append("--special")
        if cfg.get("swa_full"):
            # S: required for Gemma-4 + --cache-reuse (llamacpp#21468).
            args.append("--swa-full")
        # M: speculative decoding (draft model). llama.cpp uses -md plus
        # --spec-draft-n-{min,max} / --spec-draft-p-min.
        spec_draft = cfg.get("spec_draft")
        if isinstance(spec_draft, dict) and spec_draft.get("draft_model_path"):
            args.extend(["-md", str(spec_draft["draft_model_path"])])
            if "n_max" in spec_draft:
                args.extend(["--spec-draft-n-max", str(spec_draft["n_max"])])
            if "n_min" in spec_draft:
                args.extend(["--spec-draft-n-min", str(spec_draft["n_min"])])
            if "p_min" in spec_draft:
                args.extend(["--spec-draft-p-min", str(spec_draft["p_min"])])
        # chat_template_kwargs (item C). Pass as a single JSON string.
        ctk = cfg.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk:
            args.extend(["--chat-template-kwargs", json.dumps(ctk)])
        # Repeatable: --override-tensor / -ot for MoE expert CPU offloading
        # and other selective offload patterns.
        override_tensors = cfg.get("override_tensors", []) or []
        for pattern in override_tensors:
            args.extend(["-ot", str(pattern)])
        # llama.cpp 67ace02+ emits a perf warning when --override-tensor or
        # --n-cpu-moe is combined with mmap-enabled loading. Auto-disable
        # mmap so the MoE-offload path runs on the fast loader by default.
        # Users who need mmap on can hand-edit the generated config.
        if (override_tensors or cfg.get("n_cpu_moe")) and not cfg.get("mmap", False):
            args.append("--no-mmap")
        # vserve computes n_gpu_layers ourselves; tell llama.cpp's auto-fitter
        # to stand down. Without this, every start logs
        #   common_fit_params: failed to fit params to free device memory:
        #   n_gpu_layers already set by user to N, abort
        # which looks like an error in journals but is cosmetic (llamacpp#21801).
        args.extend(["--fit", "off"])

        # Write per-model launch script + JSON alongside the config,
        # then symlink active.sh/active.json to them.
        # Symlink pattern (matching vLLM's active.yaml) avoids permission
        # issues — any user in the llm group can unlink + recreate symlinks
        # in the group-writable configs dir.
        import shlex
        from vserve.config import cfg as vserve_cfg
        active = self._active_config_path()
        active.parent.mkdir(parents=True, exist_ok=True)
        json_link = active.with_suffix(".json")
        previous_active_target = active.readlink() if active.is_symlink() else None
        previous_json_target = json_link.readlink() if json_link.is_symlink() else None

        # Per-model script in configs/models/
        model_script = config_path.with_suffix(".sh")
        gpu_index_raw: object = cfg.get("gpu_index")
        if gpu_index_raw is None:
            gpu_index_raw = getattr(vserve_cfg(), "gpu_index", 0)
        try:
            gpu_index = int(str(gpu_index_raw))
        except (TypeError, ValueError):
            gpu_index = 0
        script = "#!/bin/bash\n"
        script += f"export CUDA_VISIBLE_DEVICES={gpu_index}\n"
        script += "exec " + shlex.join(args) + "\n"
        model_script.write_text(script)
        model_script.chmod(0o755)

        # Per-model JSON
        model_json = config_path

        self._assert_unit_safe_for_privileged_action()

        # Symlink active.sh → per-model script
        active.unlink(missing_ok=True)
        active.symlink_to(model_script.resolve())

        # Symlink active.json → per-model JSON
        json_link.unlink(missing_ok=True)
        json_link.symlink_to(model_json.resolve())

        command = ["sudo", "systemctl", "start", self.service_name]
        if non_interactive:
            command.insert(1, "-n")
        result = subprocess.run(
            command,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            self._restore_active_links(
                active,
                json_link,
                previous_active_target=previous_active_target,
                previous_json_target=previous_json_target,
            )
            self._write_failed_manifest(config_path, result.stderr.strip() or result.stdout.strip())
            raise RuntimeError(f"systemctl start {self.service_name} failed: {result.stderr}")

    def stop(self, *, non_interactive: bool = False) -> None:
        self._assert_unit_safe_for_privileged_action()
        command = ["sudo", "systemctl", "stop", self.service_name]
        if non_interactive:
            command.insert(1, "-n")
        result = subprocess.run(
            command,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl stop {self.service_name} failed: {result.stderr}")

    def is_running(self) -> bool:
        result = subprocess.run(
            ["systemctl", "is-active", self.service_name],
            capture_output=True, text=True, timeout=10,
        )
        status = result.stdout.strip().lower()
        if result.returncode == 0 and status == "active":
            return True
        if status in {"inactive", "failed"}:
            return False
        if status in {"activating", "deactivating", "reloading"}:
            raise RuntimeError(f"systemctl is-active {self.service_name} is transitional: {status}")
        if "could not be found" in result.stderr.lower():
            return False
        if result.stderr.strip():
            raise RuntimeError(f"systemctl is-active {self.service_name} failed: {result.stderr.strip()}")
        return False

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def active_manifest_path(self) -> Path:
        return self.root_dir / "run" / "active-manifest.json"

    def detect_tools(self, model_path: Path) -> dict:
        """Check if model supports tool calling and reasoning via chat template or GGUF metadata."""
        from vserve.tools import supports_tools, _read_chat_template
        import re

        # Try tokenizer_config.json first
        template = _read_chat_template(model_path)
        if template is None:
            # Fall back to GGUF embedded template
            template = self._read_gguf_chat_template(model_path)

        has_tools = bool(template and re.search(r"\btools\b", template))
        has_reasoning = bool(template and ("<think>" in template or "<|channel>" in template))

        return {
            "supports_tools": has_tools or supports_tools(model_path),
            "supports_reasoning": has_reasoning,
        }

    def _read_gguf_chat_template(self, model_path: Path) -> str | None:
        """Read chat template from GGUF file metadata."""
        gguf_files = iter_top_level_files_with_suffix(model_path, ".gguf")
        if not gguf_files:
            return None
        try:
            from gguf import GGUFReader  # type: ignore[import-not-found, import-untyped]
            reader = GGUFReader(str(gguf_files[0]))
            field = reader.fields.get("tokenizer.chat_template")
            if field is None:
                return None
            raw = field.parts[-1]
            if hasattr(raw, "tobytes"):
                return raw.tobytes().decode("utf-8", errors="replace")
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
        except Exception:
            return None

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        from vserve.config import find_systemd_unit_path

        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            return find_systemd_unit_path(self.service_name) is not None

        return [
            (f"{self.display_name} binary (llama-server)", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]

    def _active_config_path(self) -> Path:
        return self.root_dir / "configs" / "active.sh"

    @staticmethod
    def _restore_active_links(
        active: Path,
        json_link: Path,
        *,
        previous_active_target: Path | None,
        previous_json_target: Path | None,
    ) -> None:
        for link, target in ((active, previous_active_target), (json_link, previous_json_target)):
            try:
                link.unlink(missing_ok=True)
                if target is not None:
                    link.symlink_to(target)
            except OSError:
                pass

    def _write_failed_manifest(self, config_path: Path, error: str) -> None:
        from datetime import datetime, timezone
        from vserve.config import write_active_manifest

        try:
            write_active_manifest({
                "backend": self.name,
                "service_name": self.service_name,
                "config_path": str(config_path.resolve()),
                "status": "failed",
                "error": error,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, self.active_manifest_path())
        except Exception:
            pass
