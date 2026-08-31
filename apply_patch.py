#!/usr/bin/env python3
"""Install the minimal Qwen3.8 SGLang optimization patch.

Scope:
1. TP-shard DFlash's large fc.weight with RowParallelLinear.
2. Add static per-KV-head FP8 scales to the target FlashInfer path.

The DFlash KV cache deliberately remains on SGLang's scalar/per-tensor path.
The patcher is semantic, idempotent, and refuses unknown upstream drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import time
from pathlib import Path
from typing import Any


PATCH_ID = "qwen38-minimal-tpfc-target-perhead-v1"
FLASH_IMPORT_MARKER = (
    "from sglang.srt.layers.attention.sglang_per_head_fp8 import ("
)
TPFC_MARKER = (
    "# Qwen3.8 minimal patch: shard the large DFlash context projection over TP."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"upstream drift at {label}: expected one anchor, found {count}"
        )
    return text.replace(old, new, 1)


def replace_exact_count(
    text: str, old: str, new: str, expected: int, label: str
) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"upstream drift at {label}: expected {expected} anchors, found {count}"
        )
    return text.replace(old, new)


def patch_flashinfer(text: str) -> str:
    if FLASH_IMPORT_MARKER in text:
        return text

    text = replace_once(
        text,
        "from sglang.srt.layers.attention.base_attn_backend import AttentionBackend\n",
        "from sglang.srt.layers.attention.base_attn_backend import AttentionBackend\n"
        "from sglang.srt.layers.attention.sglang_per_head_fp8 import (\n"
        "    get_kv_write_scales,\n"
        "    scale_output_from_paged_attention,\n"
        "    scale_q_for_paged_attention,\n"
        ")\n",
        "FlashInfer imports",
    )
    text = replace_once(
        text,
        "    def _kv_write_scales(self, layer: RadixAttention):\n"
        "        if self.kv_cache_quant_method.needs_global_scale():\n"
        "            return None, None\n"
        "        return layer.k_scale, layer.v_scale\n",
        "    def _kv_write_scales(self, layer: RadixAttention):\n"
        "        if self.kv_cache_quant_method.needs_global_scale():\n"
        "            return None, None\n"
        "        return get_kv_write_scales(layer, layer.k_scale, layer.v_scale)\n",
        "KV write scales",
    )
    text = replace_once(
        text,
        "            o = prefill_wrapper_paged.forward(\n"
        "                q.view(-1, layer.tp_q_head_num, layer.head_dim),\n",
        "            o = prefill_wrapper_paged.forward(\n"
        "                scale_q_for_paged_attention(\n"
        "                    q.view(-1, layer.tp_q_head_num, layer.head_dim), layer\n"
        "                ),\n",
        "paged prefill Q",
    )
    text = replace_once(
        text,
        "                k_scale=layer.k_scale_float,\n"
        "                v_scale=layer.v_scale_float,\n"
        "            )\n"
        "        else:\n",
        "                k_scale=layer.k_scale_float,\n"
        "                v_scale=layer.v_scale_float,\n"
        "            )\n"
        "            o = scale_output_from_paged_attention(o, layer)\n"
        "        else:\n",
        "paged prefill output",
    )
    text = replace_once(
        text,
        "                o2, s2 = prefill_wrapper_paged.forward_return_lse(\n"
        "                    q.view(-1, layer.tp_q_head_num, layer.head_dim),\n",
        "                o2, s2 = prefill_wrapper_paged.forward_return_lse(\n"
        "                    scale_q_for_paged_attention(\n"
        "                        q.view(-1, layer.tp_q_head_num, layer.head_dim), layer\n"
        "                    ),\n",
        "mixed prefill paged Q",
    )
    text = replace_once(
        text,
        "                    k_scale=layer.k_scale_float,\n"
        "                    v_scale=layer.v_scale_float,\n"
        "                )\n\n"
        "                o, _ = _safe_merge_state(o1, s1, o2, s2)\n",
        "                    k_scale=layer.k_scale_float,\n"
        "                    v_scale=layer.v_scale_float,\n"
        "                )\n"
        "                o2 = scale_output_from_paged_attention(o2, layer)\n\n"
        "                o, _ = _safe_merge_state(o1, s1, o2, s2)\n",
        "mixed prefill paged output",
    )
    text = replace_once(
        text,
        "        o = decode_wrapper.forward(\n"
        "            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),\n",
        "        o = decode_wrapper.forward(\n"
        "            scale_q_for_paged_attention(\n"
        "                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),\n"
        "                layer,\n"
        "            ),\n",
        "decode Q",
    )
    text = replace_once(
        text,
        "            k_scale=layer.k_scale_float,\n"
        "            v_scale=layer.v_scale_float,\n"
        "        )\n\n"
        "        return o.view(-1, layer.tp_q_head_num * layer.head_dim)\n",
        "            k_scale=layer.k_scale_float,\n"
        "            v_scale=layer.v_scale_float,\n"
        "        )\n"
        "        o = scale_output_from_paged_attention(o, layer)\n\n"
        "        return o.view(-1, layer.tp_q_head_num * layer.head_dim)\n",
        "decode output",
    )
    return text


def dflash_fc_is_compatible(text: str) -> bool:
    return (
        "self.fc = RowParallelLinear(" in text
        and "self.num_context_features * hidden_size," in text
        and "input_is_parallel=False," in text
        and "expected = int(self.fc.input_size)" in text
    )


def patch_dflash_tp_fc_current_main(text: str) -> str:
    """Patch the newer upstream layout that also supports Nemotron drafts."""
    text = replace_once(
        text,
        "        else:\n"
        "            self.fc = nn.Linear(\n"
        "                self.num_context_features * hidden_size, hidden_size, bias=False\n"
        "            )\n",
        "        else:\n"
        "            fc_prefix = f\"{prefix}.fc\" if prefix else \"fc\"\n"
        f"            {TPFC_MARKER}\n"
        "            self.fc = RowParallelLinear(\n"
        "                self.num_context_features * hidden_size,\n"
        "                hidden_size,\n"
        "                bias=False,\n"
        "                input_is_parallel=False,\n"
        "                quant_config=quant_config,\n"
        "                prefix=fc_prefix,\n"
        "            )\n",
        "DFlash current-main fc construction",
    )
    text = replace_exact_count(
        text,
        "        expected = int(\n"
        "            self.fc.input_size if self.is_nemotron_35_draft else self.fc.in_features\n"
        "        )\n",
        "        expected = int(self.fc.input_size)\n",
        2,
        "DFlash current-main fc input size",
    )
    text = replace_once(
        text,
        "        projected = self.fc(target_hidden)\n"
        "        if self.is_nemotron_35_draft:\n"
        "            projected = projected[0]\n"
        "        return self.hidden_norm(projected)\n",
        "        projected, _ = self.fc(target_hidden)\n"
        "        return self.hidden_norm(projected)\n",
        "DFlash current-main fc forward",
    )
    text = replace_once(
        text,
        "                if resolved_name.endswith(\"fc.weight\"):\n"
        "                    if self.is_nemotron_35_draft:\n"
        "                        expected_shape = (\n"
        "                            int(self.config.hidden_size),\n"
        "                            int(self.num_context_features * self.config.hidden_size),\n"
        "                        )\n"
        "                        loaded_shape = _logical_linear_weight_shape(\n"
        "                            param,\n"
        "                            loaded_weight,\n"
        "                            output_features=expected_shape[0],\n"
        "                        )\n"
        "                        shape_matches = loaded_shape == expected_shape or (\n"
        "                            getattr(param, \"pack_factor\", None) is None\n"
        "                            and tuple(loaded_weight.shape) == tuple(param.shape)\n"
        "                        )\n"
        "                    else:\n"
        "                        expected_shape = tuple(param.shape)\n"
        "                        loaded_shape = tuple(loaded_weight.shape)\n"
        "                        shape_matches = loaded_shape == expected_shape\n",
        "                if resolved_name.endswith(\"fc.weight\"):\n"
        "                    expected_shape = (\n"
        "                        int(self.fc.output_size),\n"
        "                        int(self.fc.input_size),\n"
        "                    )\n"
        "                    loaded_shape = _logical_linear_weight_shape(\n"
        "                        param,\n"
        "                        loaded_weight,\n"
        "                        output_features=expected_shape[0],\n"
        "                    )\n"
        "                    shape_matches = loaded_shape == expected_shape or (\n"
        "                        getattr(param, \"pack_factor\", None) is None\n"
        "                        and tuple(loaded_weight.shape) == tuple(param.shape)\n"
        "                    )\n",
        "DFlash current-main fc global shape validation",
    )
    text = replace_once(
        text,
        "        compute_dtype = self.fc.weight.dtype\n",
        "        compute_dtype = self.hidden_norm.weight.dtype\n",
        "Laguna current-main fc compute dtype",
    )
    text = replace_once(
        text,
        "        projected = self.fc(fused)\n"
        "        if self.is_nemotron_35_draft:\n"
        "            projected = projected[0]\n"
        "        return self.hidden_norm(projected)\n",
        "        projected, _ = self.fc(fused)\n"
        "        return self.hidden_norm(projected)\n",
        "Laguna current-main fc forward",
    )
    return text


def patch_dflash_tp_fc(text: str) -> str:
    if TPFC_MARKER in text or dflash_fc_is_compatible(text):
        return text

    current_main_anchor = (
        "        else:\n"
        "            self.fc = nn.Linear(\n"
        "                self.num_context_features * hidden_size, hidden_size, bias=False\n"
        "            )\n"
    )
    if current_main_anchor in text:
        return patch_dflash_tp_fc_current_main(text)

    text = replace_once(
        text,
        "        self.fc = nn.Linear(\n"
        "            self.num_context_features * hidden_size, hidden_size, bias=False\n"
        "        )\n",
        f"        {TPFC_MARKER}\n"
        "        self.fc = RowParallelLinear(\n"
        "            self.num_context_features * hidden_size,\n"
        "            hidden_size,\n"
        "            bias=False,\n"
        "            input_is_parallel=False,\n"
        "            quant_config=quant_config,\n"
        "            prefix=\"fc\",\n"
        "        )\n",
        "DFlash fc construction",
    )
    text = replace_exact_count(
        text,
        "        expected = int(self.fc.in_features)\n",
        "        expected = int(self.fc.input_size)\n",
        2,
        "DFlash fc input size",
    )
    text = replace_once(
        text,
        "        return self.hidden_norm(self.fc(target_hidden))\n",
        "        projected, _ = self.fc(target_hidden)\n"
        "        return self.hidden_norm(projected)\n",
        "DFlash fc forward",
    )
    text = replace_once(
        text,
        "                if resolved_name.endswith(\"fc.weight\") and tuple(\n"
        "                    loaded_weight.shape\n"
        "                ) != tuple(param.shape):\n"
        "                    raise ValueError(\n"
        "                        \"DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's \"\n"
        "                        \"number of context features (K) does not match this config. \"\n"
        "                        f\"Expected fc.weight.shape={tuple(param.shape)} \"\n",
        "                if resolved_name.endswith(\"fc.weight\") and tuple(\n"
        "                    loaded_weight.shape\n"
        "                ) != (int(self.fc.output_size), int(self.fc.input_size)):\n"
        "                    raise ValueError(\n"
        "                        \"DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's \"\n"
        "                        \"number of context features (K) does not match this config. \"\n"
        "                        \"Expected global fc.weight.shape=\"\n"
        "                        f\"{(int(self.fc.output_size), int(self.fc.input_size))} \"\n",
        "DFlash fc global shape validation",
    )
    text = replace_once(
        text,
        "        compute_dtype = self.fc.weight.dtype\n",
        "        compute_dtype = self.hidden_norm.weight.dtype\n",
        "Laguna fc compute dtype",
    )
    text = replace_once(
        text,
        "        return self.hidden_norm(self.fc(fused))\n",
        "        projected, _ = self.fc(fused)\n"
        "        return self.hidden_norm(projected)\n",
        "Laguna fc forward",
    )
    return text


def default_sglang_root() -> Path:
    import sglang

    return Path(sglang.__file__).resolve().parent


def paths_for(root: Path) -> dict[str, Path]:
    return {
        "flashinfer": root / "srt/layers/attention/flashinfer_backend.py",
        "dflash": root / "srt/models/dflash.py",
        "helper": root / "srt/layers/attention/sglang_per_head_fp8.py",
    }


def status(root: Path) -> dict[str, Any]:
    paths = paths_for(root)
    flash_text = (
        paths["flashinfer"].read_text(encoding="utf-8")
        if paths["flashinfer"].is_file()
        else ""
    )
    dflash_text = (
        paths["dflash"].read_text(encoding="utf-8")
        if paths["dflash"].is_file()
        else ""
    )
    return {
        "sglang_root": str(root),
        "flashinfer_per_head": FLASH_IMPORT_MARKER in flash_text,
        "dflash_tp_fc": dflash_fc_is_compatible(dflash_text),
        "helper_exists": paths["helper"].is_file(),
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def apply(args: argparse.Namespace) -> None:
    package_dir = Path(__file__).resolve().parent
    root = (args.sglang_root or default_sglang_root()).resolve()
    paths = paths_for(root)
    helper_source = package_dir / "runtime/sglang_per_head_fp8.py"
    required = [paths["flashinfer"], paths["dflash"], helper_source]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))

    original_flash = paths["flashinfer"].read_text(encoding="utf-8")
    original_dflash = paths["dflash"].read_text(encoding="utf-8")
    patched_flash = patch_flashinfer(original_flash)
    patched_dflash = patch_dflash_tp_fc(original_dflash)

    before = status(root)
    helper_same = paths["helper"].is_file() and sha256(paths["helper"]) == sha256(
        helper_source
    )
    if (
        patched_flash == original_flash
        and patched_dflash == original_dflash
        and helper_same
    ):
        print(json.dumps({"patch": PATCH_ID, "status": "already-applied", **before}, indent=2))
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = args.backup_root.expanduser().resolve() / stamp
    backup.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "patch": PATCH_ID,
        "created": stamp,
        "sglang_root": str(root),
        "files": {},
    }

    for name, path in paths.items():
        metadata: dict[str, Any] = {"existed": path.exists()}
        if path.exists():
            destination = backup / f"{name}-{path.name}"
            shutil.copy2(path, destination)
            metadata.update(
                backup=str(destination),
                sha256_before=sha256(path),
            )
        manifest["files"][str(path)] = metadata

    atomic_write(paths["flashinfer"], patched_flash)
    atomic_write(paths["dflash"], patched_dflash)
    shutil.copy2(helper_source, paths["helper"])

    for path in paths.values():
        py_compile.compile(str(path), doraise=True)
    for path_text, metadata in manifest["files"].items():
        metadata["sha256_after"] = sha256(Path(path_text))
    (backup / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    after = status(root)
    if not all(
        after[key] for key in ("flashinfer_per_head", "dflash_tp_fc", "helper_exists")
    ):
        raise RuntimeError(f"post-install verification failed: {after}")
    print(
        json.dumps(
            {"patch": PATCH_ID, "status": "applied", "backup": str(backup), **after},
            indent=2,
        )
    )


def verify(args: argparse.Namespace) -> None:
    root = (args.sglang_root or default_sglang_root()).resolve()
    result = status(root)
    ok = all(
        result[key] for key in ("flashinfer_per_head", "dflash_tp_fc", "helper_exists")
    )
    result.update(patch=PATCH_ID, status="ok" if ok else "incomplete")
    print(json.dumps(result, indent=2))
    if not ok:
        raise SystemExit(1)


def rollback(args: argparse.Namespace) -> None:
    backup = args.rollback.resolve()
    manifest_path = backup / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"backup manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("patch") != PATCH_ID:
        raise ValueError(f"backup is not for {PATCH_ID}")
    for path_text, metadata in manifest["files"].items():
        path = Path(path_text)
        if metadata["existed"]:
            shutil.copy2(Path(metadata["backup"]), path)
        elif path.exists():
            path.unlink()
    print(json.dumps({"patch": PATCH_ID, "status": "rolled-back", "backup": str(backup)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("~/.local/share/qwen38-minimal-patch/backups"),
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rollback", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollback:
        rollback(args)
    elif args.verify:
        verify(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
