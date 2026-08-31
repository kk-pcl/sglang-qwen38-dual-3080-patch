"""Static per-KV-head FP8 scale loader for the Qwen3.8 target model only."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable


PER_HEAD_FORMAT = "sglang-static-per-head-kv-v1"


def _is_array(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _parallel() -> tuple[int, int]:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return int(parallel.tp_rank), int(parallel.tp_size)


def _find_layers(model: Any) -> Any:
    current = model
    visited: set[int] = set()
    for _ in range(5):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        layers = getattr(current, "layers", None)
        if layers is not None:
            return layers
        current = getattr(current, "model", None)
    raise RuntimeError(f"Cannot locate Qwen3.8 decoder layers in {type(model).__name__}")


def _wrap_loader(original: Callable[..., Any], logger: Any) -> Callable[..., Any]:
    def load(self: Any, scale_path_text: str) -> None:
        scale_path = Path(scale_path_text)
        payload = json.loads(scale_path.read_text(encoding="utf-8"))
        if payload.get("format") != PER_HEAD_FORMAT:
            return original(self, scale_path_text)

        entries = payload.get("layers")
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"No per-head target scales in {scale_path}")

        rank, size = _parallel()
        expected_tp = payload.get("calibration_tp_size")
        if expected_tp is not None and int(expected_tp) != size:
            raise ValueError(
                f"Target scale file was calibrated for TP={expected_tp}, current TP={size}"
            )

        from sglang.srt.layers.attention.sglang_per_head_fp8 import (
            install_static_per_head_scales,
        )

        layers = _find_layers(self)
        loaded: list[int] = []
        for layer_text, pair in entries.items():
            layer_id = int(layer_text)
            if layer_id >= len(layers):
                continue
            attn = getattr(layers[layer_id], "attn", None)
            if attn is None:
                continue
            if not isinstance(pair, dict) or "k" not in pair or "v" not in pair:
                raise ValueError(f"Malformed target scale pair for layer {layer_id}")
            if not (_is_array(pair["k"]) or _is_array(pair["v"])):
                raise ValueError(
                    f"Target layer {layer_id} is not per-head; use a scalar loader instead"
                )
            install_static_per_head_scales(
                attn,
                pair["k"],
                pair["v"],
                tp_rank=rank,
                tp_size=size,
            )
            loaded.append(layer_id)

        if len(layers) >= 64 and len(loaded) != len(entries):
            raise RuntimeError(
                f"Target did not receive every per-head scale: loaded={loaded}, "
                f"expected={sorted(map(int, entries))}"
            )
        logger.info(
            "Qwen3.8 target per-head FP8 KV scales loaded: rank=%d/%d, "
            "layers=%s, source=%s",
            rank,
            size,
            loaded,
            scale_path,
        )

    load._qwen38_target_per_head_fp8 = True
    return load


def install_target_per_head_loader(logger: Any) -> None:
    from sglang.srt.models import qwen3_5

    for model_cls in (
        qwen3_5.Qwen3_5ForCausalLM,
        qwen3_5.Qwen3_5ForConditionalGeneration,
    ):
        original = model_cls.load_kv_cache_scales
        if not getattr(original, "_qwen38_target_per_head_fp8", False):
            model_cls.load_kv_cache_scales = _wrap_loader(original, logger)
