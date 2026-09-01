#!/usr/bin/env bash

# Upstream-only DFlash2 launcher.
# This script deliberately does not load this repository's runtime/ directory,
# does not run apply_patch.py, and does not pass any custom KV-scale options.

set -Eeuo pipefail

CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/etc/profile.d/conda.sh}"
SGLANG_CONDA_ENV="${SGLANG_CONDA_ENV:-sglang}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-27B-AWQ}"
# Use the upstream-compatible BF16 DFlash2 checkpoint by default. The local
# W4A16 draft is intentionally not selected here because its low-bit loader is
# one of the adaptations covered by this repository's patch.
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-incoai/Qwen3.8-27B-DFlash2}"
TP_SIZE="${TP_SIZE:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"

[[ -f "$CONDA_INIT" ]] || {
  echo "Conda init script not found: $CONDA_INIT" >&2
  exit 1
}

case "$TP_SIZE" in
  1) default_cuda_visible_devices=0 ;;
  2) default_cuda_visible_devices=0,1 ;;
  *)
    echo "TP_SIZE must be 1 or 2; got: $TP_SIZE" >&2
    exit 1
    ;;
esac

source "$CONDA_INIT"
conda activate "$SGLANG_CONDA_ENV"

# Ensure no PYTHONPATH entry from the patched launcher can leak into the
# official-source run. No sitecustomize or project runtime is injected here.
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_cuda_visible_devices}"

args=(
  --model-path "$MODEL_PATH"
  --served-model-name qwen3.8-27b
  --tp-size "$TP_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT_MODEL_PATH"
  --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --host 0.0.0.0
  --port "$SGLANG_PORT"
)

if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  args+=(--api-key "$SGLANG_API_KEY")
fi

echo "Starting upstream-only Qwen3.8-27B + DFlash2"
echo "  conda env: $SGLANG_CONDA_ENV"
echo "  target:    $MODEL_PATH"
echo "  draft:     $DRAFT_MODEL_PATH"
echo "  TP:        $TP_SIZE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  context:   $CONTEXT_LENGTH"
echo "  draft tok: $SPECULATIVE_NUM_DRAFT_TOKENS"
echo "  custom repository patches: disabled"

exec sglang serve "${args[@]}"
