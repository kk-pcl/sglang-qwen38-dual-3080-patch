#!/usr/bin/env bash

# Convenience wrapper. The common launcher remains dual-GPU by default.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TP_SIZE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$SCRIPT_DIR/start_example.sh" "$@"
