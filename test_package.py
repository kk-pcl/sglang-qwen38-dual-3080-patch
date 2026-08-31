#!/usr/bin/env python3
"""Offline validation for the minimal Qwen3.8 patch package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def load_helper():
    path = ROOT / "runtime/sglang_per_head_fp8.py"
    spec = importlib.util.spec_from_file_location("sglang_per_head_fp8_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_files() -> None:
    target = json.loads(
        (ROOT / "scales/qwen38-fp8-kv-scales-per-head.json").read_text()
    )
    assert target["format"] == "sglang-static-per-head-kv-v1"
    assert int(target["calibration_tp_size"]) == 2
    assert len(target["layers"]) == 16
    assert all(
        isinstance(pair["k"], list) and isinstance(pair["v"], list)
        for pair in target["layers"].values()
    )

    draft = json.loads(
        (ROOT / "scales/dflash2-fp8-kv-scales-scalar.json").read_text()
    )
    assert draft["format"] == "sglang-dflash2-separate-kv-v1"
    assert set(draft["ranks"]) == {"0", "1"}
    for layers in draft["ranks"].values():
        assert len(layers) == 5
        assert all(
            isinstance(pair["k"], (int, float))
            and isinstance(pair["v"], (int, float))
            for pair in layers.values()
        )


def test_per_head_math() -> None:
    helper = load_helper()

    class Layer:
        tp_k_head_num = 2
        tp_v_head_num = 2
        tp_q_head_num = 8
        head_dim = 4
        v_head_dim = 4
        k_scale = torch.tensor(1.0)
        v_scale = torch.tensor(1.0)

    layer = Layer()
    helper.install_static_per_head_scales(
        layer,
        [2.0, 3.0, 5.0, 7.0],
        [11.0, 13.0, 17.0, 19.0],
        tp_rank=1,
        tp_size=2,
    )
    k_write, v_write = helper.get_kv_write_scales(layer, 1.0, 1.0)
    assert tuple(k_write.shape) == (1, 2, 1)
    assert tuple(v_write.shape) == (1, 2, 1)
    assert k_write.flatten().tolist() == [5.0, 7.0]
    assert v_write.flatten().tolist() == [17.0, 19.0]

    q = torch.ones(1, 8, 4)
    helper.scale_q_for_paged_attention(q, layer)
    assert q[0, :, 0].tolist() == [5.0] * 4 + [7.0] * 4

    output = torch.ones(1, 8, 4)
    helper.scale_output_from_paged_attention(output, layer)
    assert output[0, :, 0].tolist() == [17.0] * 4 + [19.0] * 4


def main() -> None:
    test_scale_files()
    test_per_head_math()
    print("minimal patch package tests: PASS")


if __name__ == "__main__":
    main()
