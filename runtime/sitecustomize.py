"""Runtime loaders for the minimal Qwen3.8 target-per-head/DFlash-scalar patch."""

from __future__ import annotations

import logging
import sys
from typing import Any


LOGGER = logging.getLogger("sglang.qwen38_minimal_patch")


def _install_ampere_multimem_fallback() -> None:
    """Use NCCL instead of Hopper-only multimem gather on pre-SM90 GPUs.

    The upstream gatherer creates its symmetric-memory state lazily.  On the
    WSL2/Ampere combination the rendezvous can terminate the scheduler with
    SIGFPE before its Python exception fallback is reached.
    """
    import torch
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    gatherer = triton_symm_mem_ag.MultimemAllGatherer
    original = gatherer.__init__
    if getattr(original, "_qwen38_ampere_safe", False):
        return

    def safe_init(
        self: Any,
        max_tokens: int,
        *,
        enabled: bool = True,
        skip_entry_sync: bool = False,
    ) -> None:
        try:
            major, _minor = torch.cuda.get_device_capability()
        except Exception:
            major = 0

        if enabled and major < 9:
            enabled = False
            LOGGER.info(
                "Disabled MultimemAllGatherer on SM%d; using NCCL all-gather",
                major,
            )

        original(
            self,
            max_tokens,
            enabled=enabled,
            skip_entry_sync=skip_entry_sync,
        )

    safe_init._qwen38_ampere_safe = True
    gatherer.__init__ = safe_init


def _install_target_loader_entrypoint() -> None:
    from sglang.srt.models import qwen3_5

    def unsupported_scalar_loader(self: Any, scale_path: str) -> None:
        del self
        raise ValueError(
            "This minimal package expects the bundled Qwen3.8 target per-head "
            f"scale file, but received an unsupported file: {scale_path}"
        )

    for model_cls in (
        qwen3_5.Qwen3_5ForCausalLM,
        qwen3_5.Qwen3_5ForConditionalGeneration,
    ):
        if not hasattr(model_cls, "load_kv_cache_scales"):
            model_cls.load_kv_cache_scales = unsupported_scalar_loader


def _install() -> None:
    _install_ampere_multimem_fallback()
    _install_target_loader_entrypoint()

    from dflash_scalar_kv_loader import install_dflash_scalar_loader
    from qwen38_target_per_head_loader import install_target_per_head_loader

    install_dflash_scalar_loader(LOGGER)
    install_target_per_head_loader(LOGGER)
    LOGGER.info(
        "Installed minimal Qwen3.8 runtime: target=per-head FP8, draft=scalar FP8"
    )


process_command = " ".join(
    str(value) for value in getattr(sys, "orig_argv", sys.argv)
)
skip_helper_process = any(
    marker in process_command
    for marker in (
        "multiprocessing.resource_tracker",
        "torch._inductor.compile_worker",
        "/torch/_inductor/compile_worker/",
    )
)

if not skip_helper_process:
    try:
        _install()
    except Exception:
        LOGGER.exception("Failed to install the minimal Qwen3.8 runtime patch")
        raise
