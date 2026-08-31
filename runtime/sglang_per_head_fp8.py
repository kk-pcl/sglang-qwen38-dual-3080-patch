"""Static per-KV-head FP8 scale support for SGLang attention backends.

The FlashInfer adapter uses the algebraic equivalence

    (Q @ (K_fp8 * s_k).T) @ (V_fp8 * s_v)
      == ((Q * s_k) @ K_fp8.T) @ V_fp8 * s_v

when the K/V scales are constant for a KV head. Scale tensors are persistent
attributes on RadixAttention so their addresses remain CUDA-graph stable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import torch


PATCH_API_VERSION = 1
_ATTR_PREFIX = "_sglang_static_per_head_fp8"


def _as_positive_values(value: Real | Sequence[Real], name: str) -> list[float]:
    if isinstance(value, Real):
        values = [float(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
    else:
        raise TypeError(f"{name} must be a number or a sequence of numbers")
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError(f"{name} must contain only finite positive values: {values}")
    return values


def _local_values(
    values: list[float],
    *,
    local_heads: int,
    tp_rank: int,
    tp_size: int,
    name: str,
) -> list[float]:
    """Accept one scalar, rank-local values, or global contiguous head values."""
    if len(values) == 1:
        return values * local_heads
    if len(values) == local_heads:
        return values
    global_heads = local_heads * tp_size
    if len(values) == global_heads:
        start = tp_rank * local_heads
        return values[start : start + local_heads]
    raise ValueError(
        f"{name} has {len(values)} values; expected 1, {local_heads} rank-local, "
        f"or {global_heads} global values for TP={tp_size}"
    )


def install_static_per_head_scales(
    layer: Any,
    k_scale: Real | Sequence[Real],
    v_scale: Real | Sequence[Real],
    *,
    tp_rank: int,
    tp_size: int,
) -> None:
    """Install persistent rank-local scale tensors on RadixAttention."""
    local_k_heads = int(layer.tp_k_head_num)
    local_v_heads = int(layer.tp_v_head_num)
    local_q_heads = int(layer.tp_q_head_num)
    if local_k_heads <= 0 or local_v_heads <= 0 or local_q_heads <= 0:
        raise ValueError("attention head counts must be positive")
    if local_k_heads != local_v_heads:
        raise ValueError(
            "static per-head FP8 currently requires matching K/V head counts: "
            f"K={local_k_heads}, V={local_v_heads}"
        )
    if local_q_heads % local_k_heads:
        raise ValueError(
            f"Q heads ({local_q_heads}) must be divisible by KV heads ({local_k_heads})"
        )

    k_values = _local_values(
        _as_positive_values(k_scale, "k_scale"),
        local_heads=local_k_heads,
        tp_rank=tp_rank,
        tp_size=tp_size,
        name="k_scale",
    )
    v_values = _local_values(
        _as_positive_values(v_scale, "v_scale"),
        local_heads=local_v_heads,
        tp_rank=tp_rank,
        tp_size=tp_size,
        name="v_scale",
    )

    reference = getattr(layer, "k_scale", None)
    device = reference.device if isinstance(reference, torch.Tensor) else torch.device("cuda")
    local_k = torch.tensor(k_values, dtype=torch.float32, device=device)
    local_v = torch.tensor(v_values, dtype=torch.float32, device=device)
    q_per_kv = local_q_heads // local_k_heads

    k_write = local_k.view(1, local_k_heads, 1)
    v_write = local_v.view(1, local_v_heads, 1)
    k_for_q = local_k.repeat_interleave(q_per_kv).view(1, local_q_heads, 1)
    v_for_o = local_v.repeat_interleave(q_per_kv).view(1, local_q_heads, 1)

    setattr(layer, f"{_ATTR_PREFIX}_k", local_k)
    setattr(layer, f"{_ATTR_PREFIX}_v", local_v)
    setattr(layer, f"{_ATTR_PREFIX}_k_write", k_write)
    setattr(layer, f"{_ATTR_PREFIX}_v_write", v_write)
    setattr(layer, f"{_ATTR_PREFIX}_k_for_q", k_for_q)
    setattr(layer, f"{_ATTR_PREFIX}_v_for_o", v_for_o)
    setattr(layer, f"{_ATTR_PREFIX}_enabled", True)
    layer.k_scale_float = 1.0
    layer.v_scale_float = 1.0


def has_static_per_head_scales(layer: Any) -> bool:
    return bool(getattr(layer, f"{_ATTR_PREFIX}_enabled", False))


def get_kv_write_scales(layer: Any, default_k: Any, default_v: Any) -> tuple[Any, Any]:
    if not has_static_per_head_scales(layer):
        return default_k, default_v
    return (
        getattr(layer, f"{_ATTR_PREFIX}_k_write"),
        getattr(layer, f"{_ATTR_PREFIX}_v_write"),
    )


def scale_q_for_paged_attention(q: torch.Tensor, layer: Any) -> torch.Tensor:
    if not has_static_per_head_scales(layer):
        return q
    expected = (int(layer.tp_q_head_num), int(layer.head_dim))
    if q.ndim != 3 or tuple(q.shape[-2:]) != expected:
        raise ValueError(
            f"paged Q has shape {tuple(q.shape)}, expected [tokens, {expected[0]}, {expected[1]}]"
        )
    return q.mul_(getattr(layer, f"{_ATTR_PREFIX}_k_for_q"))


def scale_output_from_paged_attention(output: torch.Tensor, layer: Any) -> torch.Tensor:
    if not has_static_per_head_scales(layer):
        return output
    expected = (int(layer.tp_q_head_num), int(layer.v_head_dim))
    if output.ndim != 3 or tuple(output.shape[-2:]) != expected:
        raise ValueError(
            "paged attention output has shape "
            f"{tuple(output.shape)}, expected [tokens, {expected[0]}, {expected[1]}]"
        )
    return output.mul_(getattr(layer, f"{_ATTR_PREFIX}_v_for_o"))


def describe_static_per_head_scales(layer: Any) -> dict[str, Any] | None:
    if not has_static_per_head_scales(layer):
        return None
    return {
        "k": getattr(layer, f"{_ATTR_PREFIX}_k").detach().cpu().tolist(),
        "v": getattr(layer, f"{_ATTR_PREFIX}_v").detach().cpu().tolist(),
        "q_heads": int(layer.tp_q_head_num),
        "kv_heads": int(layer.tp_k_head_num),
    }
