#!/usr/bin/env bash

# Upstream-only DFlash2 launcher.
#
# This entry point intentionally does not inject this repository's runtime
# directory, does not run apply_patch.py, and does not pass custom KV scales.
# It is a clean upstream baseline for comparing the official DFlash2 path.

set -Eeuo pipefail

CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/etc/profile.d/conda.sh}"
SGLANG_CONDA_ENV="${SGLANG_CONDA_ENV:-sglang}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-27B-AWQ}"
# The BF16 upstream checkpoint is used by default.  A local or another
# Hugging Face DFlash2 checkpoint can be supplied through this variable.
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-incoai/Qwen3.8-27B-DFlash2}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-20}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-2}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SGLANG_API_KEY="${SGLANG_API_KEY:-}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"

if [[ -z "$SGLANG_API_KEY" ]]; then
  echo "SGLANG_API_KEY is required; no API key is stored in this repository." >&2
  exit 1
fi

case "$TP_SIZE" in
  1) default_cuda_visible_devices=0 ;;
  2) default_cuda_visible_devices=0,1 ;;
  *)
    echo "TP_SIZE must be 1 or 2; got: $TP_SIZE" >&2
    exit 1
    ;;
esac

[[ -f "$CONDA_INIT" ]] || {
  echo "Conda init script not found: $CONDA_INIT" >&2
  exit 1
}

[[ -f "$MODEL_PATH/config.json" ]] || {
  echo "Target model config not found: $MODEL_PATH/config.json" >&2
  exit 1
}

# DRAFT_MODEL_PATH may be either a local directory or a Hugging Face model ID.
# Validate local paths without rejecting a remote ID that SGLang can download.
if [[ "$DRAFT_MODEL_PATH" == /* || "$DRAFT_MODEL_PATH" == ./* || "$DRAFT_MODEL_PATH" == ../* || "$DRAFT_MODEL_PATH" == ~/* ]]; then
  [[ -f "$DRAFT_MODEL_PATH/config.json" ]] || {
    echo "Draft model config not found: $DRAFT_MODEL_PATH/config.json" >&2
    exit 1
  }
fi

source "$CONDA_INIT"
conda activate "$SGLANG_CONDA_ENV"

# Do not allow a patched launcher/runtime from the parent shell to leak into
# this official-source baseline.  In particular, no sitecustomize is loaded.
unset PYTHONPATH
unset TARGET_KV_SCALE_PATH
unset DFLASH_KV_SCALE_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_cuda_visible_devices}"

echo "Starting upstream-only Qwen3.8-27B + DFlash2"
echo "  conda env: $SGLANG_CONDA_ENV"
echo "  target:    $MODEL_PATH"
echo "  draft:     $DRAFT_MODEL_PATH"
echo "  TP:        $TP_SIZE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  context:   $CONTEXT_LENGTH"
echo "  static mem: $MEM_FRACTION_STATIC"
echo "  draft tok: $SPECULATIVE_NUM_DRAFT_TOKENS"
echo "  custom repository patches: disabled"
echo "  custom KV scales:           disabled"

if [[ "$TP_SIZE" == 1 ]]; then
  cat >&2 <<'EOF'
NOTE: TP=1 is provided for topology compatibility only.  The Qwen3.8-27B
target plus DFlash2 does not fit on a 20 GiB RTX 3080; use TP=2 or a larger
GPU/smaller model pair.
EOF
fi

exec sglang serve \
  --model-path "$MODEL_PATH" \
  --load-format safetensors \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --json-model-override-args '{"language_model_only": false}' \
  --served-model-name qwen3.8-27b \
  --tp-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE" \
  --schedule-policy lpm \
  --disable-custom-all-reduce \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --api-key "$SGLANG_API_KEY" \
  --stream-response-default-include-usage \
  --enable-cache-report
