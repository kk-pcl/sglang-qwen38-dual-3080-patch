#!/usr/bin/env bash

set -Eeuo pipefail

PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/etc/profile.d/conda.sh}"
SGLANG_CONDA_ENV="${SGLANG_CONDA_ENV:-sglang}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-27B-AWQ}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$HOME/models/Qwen3.8-27B-DFlash2-W4A16}"
TARGET_KV_SCALE_PATH="${TARGET_KV_SCALE_PATH:-$PATCH_DIR/scales/qwen38-fp8-kv-scales-per-head.json}"
DFLASH_KV_SCALE_PATH="${DFLASH_KV_SCALE_PATH:-$PATCH_DIR/scales/dflash2-fp8-kv-scales-scalar.json}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SGLANG_API_KEY="${SGLANG_API_KEY:-}"

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

for required in \
  "$CONDA_INIT" \
  "$MODEL_PATH/config.json" \
  "$DRAFT_MODEL_PATH/config.json" \
  "$DRAFT_MODEL_PATH/model.safetensors" \
  "$TARGET_KV_SCALE_PATH" \
  "$DFLASH_KV_SCALE_PATH" \
  "$PATCH_DIR/runtime/sitecustomize.py"; do
  [[ -e "$required" ]] || { echo "Required file not found: $required" >&2; exit 1; }
done

source "$CONDA_INIT"
conda activate "$SGLANG_CONDA_ENV"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_cuda_visible_devices}"
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export TORCHINDUCTOR_COMPILE_THREADS=1
export DFLASH_KV_SCALE_PATH
export PYTHONPATH="$PATCH_DIR/runtime${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting minimal Qwen3.8 deployment"
echo "  conda env: $SGLANG_CONDA_ENV"
echo "  tensor parallel: $TP_SIZE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  target KV: static per-head FP8 E4M3"
echo "  draft KV:  per-layer scalar FP8 E4M3"
echo "  DFlash fc: TP-sharded RowParallelLinear"

if [[ "$TP_SIZE" == 1 ]]; then
  cat >&2 <<'EOF'
NOTE: TP=1 uses the bundled conservative single-GPU Draft scale profile.
      Qwen3.8-27B-AWQ + DFlash2 does not fit on a 20 GiB RTX 3080. Use a
      larger single GPU or a smaller target/draft pair; this mode is provided
      for topology compatibility, not to bypass VRAM requirements.
EOF
fi

exec sglang serve \
  --model-path "$MODEL_PATH" \
  --load-format safetensors \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --json-model-override-args '{"language_model_only": false}' \
  --kv-cache-dtype fp8_e4m3 \
  --quantization-param-path "$TARGET_KV_SCALE_PATH" \
  --served-model-name qwen3.8-27b \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --chunked-prefill-size 1024 \
  --max-running-requests 2 \
  --max-mamba-cache-size 20 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --cuda-graph-max-bs-decode 2 \
  --schedule-policy lpm \
  --disable-custom-all-reduce \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
  --speculative-draft-load-format safetensors \
  --speculative-draft-model-quantization compressed-tensors \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --speculative-num-draft-tokens 8 \
  --speculative-draft-window-size 2048 \
  --speculative-draft-attention-backend flashinfer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --api-key "$SGLANG_API_KEY" \
  --stream-response-default-include-usage \
  --enable-cache-report
