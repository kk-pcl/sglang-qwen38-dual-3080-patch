"""Per-rank, per-layer scalar FP8 KV loader for the DFlash draft model."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


FORMAT = "sglang-dflash2-separate-kv-v1"


def _set_scale(attn: Any, name: str, value: float) -> None:
    import torch

    current = getattr(attn, name, None)
    if isinstance(current, torch.Tensor):
        with torch.no_grad():
            current.fill_(value)
    else:
        setattr(attn, name, value)
    setattr(attn, f"{name}_float", value)


def _positive_scalar(value: Any, label: str) -> float:
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"{label} contains per-head values; this minimal package requires one scalar"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return result


def install_dflash_scalar_loader(logger: Any) -> None:
    from sglang.srt.models.dflash import DFlash2DraftModel

    def load_kv_cache_scales(self: Any, _target_scale_path: str) -> None:
        configured = os.environ.get("DFLASH_KV_SCALE_PATH", "").strip()
        if not configured:
            raise RuntimeError(
                "DFlash scalar FP8 KV requires DFLASH_KV_SCALE_PATH"
            )
        scale_path = Path(configured)
        payload = json.loads(scale_path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(
                f"Unsupported DFlash scalar scale format: {payload.get('format')!r}"
            )

        from sglang.srt.runtime_context import get_parallel

        rank = int(get_parallel().tp_rank)
        ranks = payload.get("ranks")
        entries = ranks.get(str(rank)) if isinstance(ranks, dict) else None
        if not isinstance(entries, dict):
            raise ValueError(f"No DFlash scalar scales for TP rank {rank}")

        loaded: list[int] = []
        for layer_id, layer in enumerate(self.layers):
            pair = entries.get(str(layer_id))
            if not isinstance(pair, dict) or "k" not in pair or "v" not in pair:
                raise ValueError(
                    f"No scalar DFlash K/V pair for rank={rank}, layer={layer_id}"
                )
            k_scale = _positive_scalar(pair["k"], f"rank {rank} layer {layer_id} K")
            v_scale = _positive_scalar(pair["v"], f"rank {rank} layer {layer_id} V")
            radix_attn = layer.self_attn.attn
            _set_scale(radix_attn, "k_scale", k_scale)
            _set_scale(radix_attn, "v_scale", v_scale)
            loaded.append(layer_id)

        logger.info(
            "DFlash scalar FP8 KV scales loaded: rank=%d, layers=%s, source=%s",
            rank,
            loaded,
            scale_path,
        )

    load_kv_cache_scales._qwen38_dflash_scalar_fp8 = True
    DFlash2DraftModel.load_kv_cache_scales = load_kv_cache_scales
