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
    assert len(draft["single_gpu"]) == 5
    for layer_id, pair in draft["single_gpu"].items():
        rank0 = draft["ranks"]["0"][layer_id]
        rank1 = draft["ranks"]["1"][layer_id]
        assert pair["k"] == max(rank0["k"], rank1["k"])
        assert pair["v"] == max(rank0["v"], rank1["v"])

    loader_path = ROOT / "runtime/dflash_scalar_kv_loader.py"
    loader_spec = importlib.util.spec_from_file_location("dflash_scale_test", loader_path)
    assert loader_spec is not None and loader_spec.loader is not None
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)
    assert loader._entries_for_topology(draft, rank=0, tp_size=1) == draft["single_gpu"]
    assert loader._entries_for_topology(draft, rank=1, tp_size=2) == draft["ranks"]["1"]


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

    class SingleGpuLayer:
        tp_k_head_num = 4
        tp_v_head_num = 4
        tp_q_head_num = 16
        head_dim = 4
        v_head_dim = 4
        k_scale = torch.tensor(1.0)
        v_scale = torch.tensor(1.0)

    single = SingleGpuLayer()
    helper.install_static_per_head_scales(
        single,
        [2.0, 3.0, 5.0, 7.0],
        [11.0, 13.0, 17.0, 19.0],
        tp_rank=0,
        tp_size=1,
    )
    single_k, single_v = helper.get_kv_write_scales(single, 1.0, 1.0)
    assert single_k.flatten().tolist() == [2.0, 3.0, 5.0, 7.0]
    assert single_v.flatten().tolist() == [11.0, 13.0, 17.0, 19.0]


def main() -> None:
    test_scale_files()
    test_per_head_math()
    print("minimal patch package tests: PASS")


if __name__ == "__main__":
    main()
