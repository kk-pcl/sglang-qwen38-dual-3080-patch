#!/usr/bin/env python3
"""Verify that runtime/sitecustomize.py installed only the intended loaders."""

from sglang.srt.models import qwen3_5
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag


target_loader = qwen3_5.Qwen3_5ForConditionalGeneration.load_kv_cache_scales
draft_loader = DFlash2DraftModel.load_kv_cache_scales

assert getattr(target_loader, "_qwen38_target_per_head_fp8", False)
assert getattr(draft_loader, "_qwen38_dflash_scalar_fp8", False)
assert not getattr(draft_loader, "_sglang_static_per_head_fp8", False)
assert getattr(
    triton_symm_mem_ag.MultimemAllGatherer.__init__,
    "_qwen38_ampere_safe",
    False,
)

print("runtime loader install: PASS (target=per-head, draft=scalar, ampere-safe)")
