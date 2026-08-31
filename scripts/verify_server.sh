#!/usr/bin/env bash

set -Eeuo pipefail

SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:8000}"
SGLANG_API_KEY="${SGLANG_API_KEY:-}"
SGLANG_MODEL="${SGLANG_MODEL:-qwen3.8-27b}"

[[ -n "$SGLANG_API_KEY" ]] || { echo "SGLANG_API_KEY is required." >&2; exit 1; }

echo "Model info:"
curl --fail --silent --show-error "$SGLANG_BASE_URL/model_info"
echo

payload="$(printf '{"model":"%s","messages":[{"role":"user","content":"请用一句话说明前缀缓存的作用。"}],"max_tokens":64,"stream":false}' "$SGLANG_MODEL")"

for run in 1 2; do
  echo "Chat request $run:"
  curl --fail --silent --show-error \
    "$SGLANG_BASE_URL/v1/chat/completions" \
    -H "Authorization: Bearer $SGLANG_API_KEY" \
    -H 'Content-Type: application/json' \
    --data "$payload"
  echo
done

echo "The second response should report cached_tokens when the prefix is retained."
