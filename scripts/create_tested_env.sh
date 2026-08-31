#!/usr/bin/env bash

set -Eeuo pipefail

BASE_ENV="${BASE_ENV:-sglang}"
TARGET_ENV="${TARGET_ENV:-sglang-qwen38}"
SGLANG_COMMIT="${SGLANG_COMMIT:-4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3}"
SGLANG_SOURCE_DIR="${SGLANG_SOURCE_DIR:-$HOME/src/sglang-qwen38-base}"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/etc/profile.d/conda.sh}"

[[ -f "$CONDA_INIT" ]] || { echo "Conda init not found: $CONDA_INIT" >&2; exit 1; }
[[ "$BASE_ENV" != "$TARGET_ENV" ]] || { echo "BASE_ENV and TARGET_ENV must differ." >&2; exit 1; }
[[ "$TARGET_ENV" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe TARGET_ENV: $TARGET_ENV" >&2; exit 1; }

source "$CONDA_INIT"

if conda env list | awk '{print $1}' | grep -Fxq "$TARGET_ENV"; then
  echo "Target environment already exists: $TARGET_ENV" >&2
  echo "Remove or rename it explicitly, then rerun this script." >&2
  exit 1
fi

conda create --yes --name "$TARGET_ENV" --clone "$BASE_ENV"
conda activate "$TARGET_ENV"

python -m pip uninstall --yes sglang || true

site_packages="$(python - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith('site-packages')]
if len(paths) != 1:
    raise SystemExit(f'Expected one site-packages path, got: {paths}')
print(paths[0])
PY
)"
sglang_package="$site_packages/sglang"

case "$sglang_package" in
  "$CONDA_PREFIX"/*/site-packages/sglang) ;;
  *) echo "Refusing to clean unexpected path: $sglang_package" >&2; exit 1 ;;
esac

if [[ -d "$sglang_package" ]]; then
  echo "Removing inherited untracked SGLang files from cloned environment: $sglang_package"
  rm -rf -- "$sglang_package"
fi

mkdir -p -- "$(dirname -- "$SGLANG_SOURCE_DIR")"
if [[ ! -d "$SGLANG_SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git "$SGLANG_SOURCE_DIR"
fi

git -C "$SGLANG_SOURCE_DIR" fetch origin "$SGLANG_COMMIT"
git -C "$SGLANG_SOURCE_DIR" checkout --detach "$SGLANG_COMMIT"

SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-deps "$SGLANG_SOURCE_DIR/python"

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in (
    'sglang',
    'torch',
    'transformers',
    'flashinfer-python',
    'compressed-tensors',
    'triton',
    'nvidia-nccl-cu13',
):
    try:
        print(f'{package}=={version(package)}')
    except PackageNotFoundError:
        print(f'{package}=NOT_INSTALLED')
PY

echo "Clean official SGLang environment is ready: $TARGET_ENV"
echo "Next: conda activate $TARGET_ENV && python apply_patch.py"
