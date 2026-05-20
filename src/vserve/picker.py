"""Picker-data helpers: limits-table parsing and scripted-config defaults.

Extracted from `cli.py` in 0.6.3 per audit
`docs/audits/2026-05-20-cli-sprawl.md`. cli.py's `run` and `tune`
commands sit on top of a layer of "given a tuned limits table, what
should the scripted (--yes) defaults be" logic that had nothing to do
with CLI plumbing.

Public surface:
- :func:`llamacpp_slots_from_limits_entry`,
  :func:`llamacpp_needs_moe_offload`,
  :func:`llamacpp_interactive_runtime_defaults`,
  :func:`llamacpp_interactive_slot_ceiling`
- :func:`vllm_limits_entry`, :func:`vllm_limit_dtype_order`,
  :func:`vllm_kv_label`
- :data:`VLLM_AUTOMATIC_KV_DTYPES`
- :func:`choose_vllm_scripted_defaults`,
  :func:`choose_llamacpp_scripted_defaults`

The interactive ``_custom_config_*`` / ``_scripted_config`` orchestrators
stay in cli.py — they need ``console`` and the interactive picker, which
would create circular imports if moved here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vserve.models import ModelInfo


# ── llama.cpp limits-table helpers ────────────────────────────────────────


def llamacpp_slots_from_limits_entry(entry: object) -> int | None:
    """Best slot count out of a tuned limits-row entry.

    Entry is either an int (legacy schema, single value), a bool (treat
    as missing — caller did `entry: bool | int` wrong), or a dict of
    kv-dtype → slots. Returns the max non-None value or None.
    """
    if isinstance(entry, dict):
        values = [v for v in entry.values() if v is not None]
        return max(values) if values else None
    if entry is None or isinstance(entry, bool) or not isinstance(entry, int):
        return None
    return entry


def llamacpp_needs_moe_offload(
    limits_data: dict, chosen_context: int, chosen_slots: int, effective_kv: str
) -> bool:
    """True only when the chosen ``(context, slots, kv)`` can't fit on GPU
    without ``-ot``.

    Auto-applying ``-ot`` when the model already fits on GPU pushes hot
    expert weights to system RAM — every lookup then traverses PCIe and
    tokens/s collapses. Use this gate before auto-enabling the pattern.
    """
    moe = limits_data.get("moe")
    if not isinstance(moe, dict) or not moe.get("is_moe") or not moe.get("ot_pattern"):
        return False
    if not limits_data.get("full_offload", True):
        # Model is already partially on CPU; `-ot` gives a cleaner partition.
        return True
    base = limits_data.get("limits")
    if not isinstance(base, dict):
        return False
    entry = base.get(str(chosen_context))
    if not isinstance(entry, dict):
        return False
    cap = entry.get(effective_kv)
    if not isinstance(cap, int):
        cap = entry.get("f16")
    if not isinstance(cap, int):
        return False
    return chosen_slots > cap


def llamacpp_interactive_runtime_defaults(lim: dict) -> tuple[str, dict]:
    """Return ``(effective_kv_dtype, effective_limits_table)``.

    Mirrors the scripted-CLI defaults so the interactive picker advertises
    slot ceilings that match the run the user will actually get. KV dtype
    defaults to ``recommended_kv_dtype`` (q8_0 if it strictly beats f16
    in the matrix, else f16). Without this the picker showed the max
    across *every* KV-dtype column — including q4_0 — even though the
    run defaults to f16, producing wildly inflated ceilings.

    Does NOT auto-promote the MoE ``limits_with_ot`` table. ``-ot``
    sacrifices throughput for KV-cache headroom and is only worth
    applying when the model can't fit on GPU; that decision happens
    later via :func:`llamacpp_needs_moe_offload`.
    """
    recommended_kv = lim.get("recommended_kv_dtype") or "f16"
    base = lim.get("limits")
    effective_limits = base if isinstance(base, dict) else {}
    return recommended_kv, effective_limits


def llamacpp_interactive_slot_ceiling(
    effective_limits: dict, chosen_ctx: int, effective_kv: str
) -> int | None:
    """Slot ceiling for the ``(context, dtype)`` the interactive flow will
    actually run with.

    Falls back to f16 (the in-library default), then to any value in the
    row, so legacy / partial caches still produce a usable number.
    """
    entry = effective_limits.get(str(chosen_ctx))
    if not isinstance(entry, dict):
        return llamacpp_slots_from_limits_entry(entry)
    slots = entry.get(effective_kv)
    if not isinstance(slots, int) or isinstance(slots, bool):
        slots = entry.get("f16")
    if not isinstance(slots, int) or isinstance(slots, bool):
        slots = llamacpp_slots_from_limits_entry(entry)
    return slots if isinstance(slots, int) and slots >= 1 else None


# ── vLLM limits-table helpers ─────────────────────────────────────────────


VLLM_AUTOMATIC_KV_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc")


def vllm_limits_entry(entry: object) -> dict[str, int | None]:
    """Normalise a vLLM limits-row entry into ``{kv_dtype: slots}``."""
    if isinstance(entry, dict):
        cleaned: dict[str, int | None] = {}
        for key, value in entry.items():
            if not isinstance(key, str):
                continue
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                cleaned[key] = value
        return cleaned
    if entry is None or isinstance(entry, bool) or not isinstance(entry, int):
        return {}
    return {"auto": entry}


def vllm_limit_dtype_order(limits_data: dict, limits: dict) -> list[str]:
    """Order of kv-dtype columns to surface in the picker matrix.

    Falls back to a stable preferred-ordering ("auto" first, then "fp8",
    then everything else) when the tuner didn't pin a column order.
    """
    dtypes_obj = limits_data.get("kv_cache_dtypes")
    if isinstance(dtypes_obj, dict):
        ordered = [key for key in dtypes_obj if isinstance(key, str)]
        if ordered:
            return ordered
    seen: list[str] = []
    for _ctx, entry in sorted(limits.items(), key=lambda item: int(str(item[0]))):
        for dtype in vllm_limits_entry(entry):
            if dtype not in seen:
                seen.append(dtype)
    preferred = ["auto", "fp8"]
    return [dtype for dtype in preferred if dtype in seen] + [
        dtype for dtype in seen if dtype not in preferred
    ]


def vllm_kv_label(dtype: str) -> str:
    """Human-readable label for a kv-dtype column header."""
    labels = {
        "auto": "Auto KV",
        "fp8": "FP8 KV",
        "fp8_e4m3": "FP8 e4m3",
        "fp8_e5m2": "FP8 e5m2",
        "turboquant_k8v4": "TQ k8v4",
        "turboquant_4bit_nc": "TQ 4bit",
        "turboquant_k3v4_nc": "TQ k3v4",
        "turboquant_3bit_nc": "TQ 3bit",
        # llama.cpp K/V dtypes — labels are intentionally symmetric since
        # fused Flash-Attention requires K and V to match.
        "f16": "F16 KV",
        "bf16": "BF16 KV",
        "f32": "F32 KV",
        "q8_0": "Q8 KV",
        "q5_1": "Q5_1 KV",
        "q5_0": "Q5_0 KV",
        "q4_1": "Q4_1 KV",
        "q4_0": "Q4_0 KV",
        "iq4_nl": "IQ4_NL KV",
    }
    return labels.get(dtype, dtype)


# ── scripted-config (--yes) default chooser ───────────────────────────────


def choose_vllm_scripted_defaults(
    m: ModelInfo,
    limits_data: dict,
    *,
    context: int | None,
    slots: int | None,
    kv_cache_dtype: str | None,
) -> tuple[int, str, int, dict]:
    """Pick ``(context, kv_cache_dtype, slots, recommendation)`` for a
    scripted vLLM ``--yes`` run.

    When the user didn't pin any value, prefer the tuner's
    "balanced" recommendation if it's still in the limits matrix.
    Otherwise pick the largest working context and the first available
    automatic-KV column.
    """
    limits = limits_data.get("limits", {})
    limits = limits if isinstance(limits, dict) else {}
    recommendation: dict = {}
    if context is None and slots is None and kv_cache_dtype is None:
        recommendations = limits_data.get("recommendations")
        if isinstance(recommendations, dict):
            balanced = recommendations.get("balanced")
            if isinstance(balanced, dict):
                rec_context = balanced.get("context")
                rec_kv = balanced.get("kv_cache_dtype")
                rec_slots = balanced.get("max_num_seqs")
                if (
                    isinstance(rec_context, int)
                    and isinstance(rec_kv, str)
                    and isinstance(rec_slots, int)
                    and vllm_limits_entry(limits.get(str(rec_context), {})).get(rec_kv) is not None
                ):
                    return rec_context, rec_kv, rec_slots, dict(balanced)

    working_contexts: list[int] = []
    for ctx_str, entry in limits.items():
        try:
            ctx = int(str(ctx_str))
        except ValueError:
            continue
        choices = vllm_limits_entry(entry)
        if kv_cache_dtype is not None:
            if choices.get(kv_cache_dtype) is not None:
                working_contexts.append(ctx)
        elif any(choices.get(dtype) is not None for dtype in VLLM_AUTOMATIC_KV_DTYPES):
            working_contexts.append(ctx)

    if not working_contexts:
        raise ValueError(f"No tuned vLLM capacity is available for {m.full_name}; run vserve tune {m.full_name}")
    if context is not None and context not in working_contexts:
        raise ValueError(f"No tuned vLLM capacity for context {context}; run vserve tune {m.full_name} --recalc")
    chosen_context = context or max(working_contexts)
    ctx_entry = vllm_limits_entry(limits.get(str(chosen_context), {}))
    if kv_cache_dtype is not None:
        if ctx_entry.get(kv_cache_dtype) is None:
            raise ValueError(f"No tuned vLLM capacity for context {chosen_context} with KV dtype {kv_cache_dtype}")
        chosen_kv = kv_cache_dtype
    else:
        candidate_kv = next((dtype for dtype in VLLM_AUTOMATIC_KV_DTYPES if ctx_entry.get(dtype) is not None), None)
        if candidate_kv is None:
            raise ValueError(f"No tuned vLLM capacity for context {chosen_context}")
        chosen_kv = candidate_kv

    if slots is not None:
        chosen_slots = slots
    else:
        tuned_slots = ctx_entry.get(chosen_kv)
        if tuned_slots is None:
            raise ValueError(f"No tuned vLLM slot count for context {chosen_context} with KV dtype {chosen_kv}")
        chosen_slots = int(tuned_slots)
    return chosen_context, chosen_kv, chosen_slots, recommendation


def choose_llamacpp_scripted_defaults(
    m: ModelInfo,
    limits_data: dict,
    *,
    context: int | None,
    slots: int | None,
    gpu_layers: int | None,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
) -> tuple[int, int, int, str, str]:
    """Pick ``(context, slots, n_gpu_layers, kv_cache_k, kv_cache_v)`` for a
    scripted llama.cpp ``--yes`` run.

    KV dtype defaults to the tuner's recommendation (f16 unless q8_0
    strictly fits more slots — see
    :py:meth:`LlamaCppBackend._recommended_kv_dtype`). The fused
    Flash-Attention path requires K and V to match, so a single dtype is
    picked unless the user explicitly passes asymmetric K and V values.
    """
    limits = limits_data.get("limits", {})
    limits = limits if isinstance(limits, dict) else {}
    working_contexts: list[int] = []
    for ctx_str, entry in limits.items():
        try:
            ctx = int(str(ctx_str))
        except ValueError:
            continue
        if llamacpp_slots_from_limits_entry(entry) is not None:
            working_contexts.append(ctx)
    if not working_contexts:
        raise ValueError(f"No tuned llama.cpp capacity is available for {m.full_name}; run vserve tune {m.full_name}")
    if context is not None and context not in working_contexts:
        raise ValueError(f"No tuned llama.cpp capacity for context {context}; run vserve tune {m.full_name} --recalc")
    chosen_context = context or max(working_contexts)

    recommended_kv = limits_data.get("recommended_kv_dtype") or "f16"
    chosen_k = kv_cache_k or recommended_kv
    chosen_v = kv_cache_v or kv_cache_k or recommended_kv

    if slots is not None:
        chosen_slots = slots
    else:
        entry = limits.get(str(chosen_context))
        if isinstance(entry, dict):
            tuned_slots: int | None = entry.get(chosen_k)
            if tuned_slots is None:
                # Fall back to f16 row then to max across dtypes.
                tuned_slots = entry.get("f16")
            if tuned_slots is None:
                tuned_slots = llamacpp_slots_from_limits_entry(entry)
        else:
            tuned_slots = llamacpp_slots_from_limits_entry(entry)
        if tuned_slots is None:
            raise ValueError(f"No tuned llama.cpp slot count for context {chosen_context}")
        chosen_slots = int(tuned_slots)

    tuned_layers = limits_data.get("n_gpu_layers")
    if gpu_layers is not None:
        chosen_layers = gpu_layers
    else:
        if not isinstance(tuned_layers, int):
            raise ValueError(f"No tuned llama.cpp GPU layer count for {m.full_name}; run vserve tune {m.full_name}")
        chosen_layers = tuned_layers
    return chosen_context, chosen_slots, chosen_layers, chosen_k, chosen_v
