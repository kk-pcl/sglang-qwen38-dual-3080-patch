# Environment and compatibility

## Real runtime matrix

The package was validated on 2026-08-31 with:

```text
WSL2 distribution: Ubuntu-vLLM-E
GPU:               2 x NVIDIA GeForce RTX 3080 20 GiB
Compute capability: 8.6
NVIDIA driver:     610.62
Python:            3.11.15
SGLang:            0.5.19.dev228+g4cb5aebfe
PyTorch:           2.13.0+cu130
Transformers:      5.12.1
FlashInfer:        0.6.17
compressed-tensors: 0.18.1a20260816
Triton:            3.7.1
NCCL:              2.29.7
```

The exact SGLang source commit is:

```text
4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3
```

The patch fixtures have also been checked against upstream commit
`881cbfe54c98356cfa1eaa134aa4d0be702fc90f`, but the full model startup test
was performed only on the locked commit above.

## Why clone an existing Conda environment?

SGLang GPU environments include tightly coupled PyTorch, CUDA, FlashInfer,
Triton and compressed-tensors builds. Cloning a known-good environment keeps
that dependency set intact. The setup script then removes SGLang only from the
new environment and installs a clean official checkout at the locked commit.

This makes the result substantially more reproducible than resolving the
latest packages in a brand-new environment.

## Upstream drift policy

`apply_patch.py` checks semantic source anchors. If the expected source shape
has changed, it exits before writing files. Do not weaken these checks just to
make a newer SGLang release accept the patch. Port and test the affected code
against the new upstream implementation instead.

## Model-specific data

The two JSON files under `scales/` are calibration artifacts, not generic FP8
defaults:

- target scales: 16 full-attention layers in **global contiguous KV-head**
  order. The calibration was collected under TP=2, but the same four global
  heads can be consumed directly under TP=1 or partitioned under TP=2;
- DFlash2 scales: exact scalar K/V values for TP=2 rank 0/1 plus a TP=1
  conservative profile formed by `max(K_rank0, K_rank1)` and
  `max(V_rank0, V_rank1)` for each Draft layer.

Using different weights, a different architecture or TP>2 requires new
calibration and validation.

## WSL2 / Ampere safety guard

The runtime `sitecustomize.py` detects GPUs below SM90 and disables the
Hopper-oriented `MultimemAllGatherer`, falling back to SGLang's normal TP NCCL
all-gather. Without this guard, the tested clean environment failed with a
floating-point exception while preparing the target verify CUDA Graph.

The guard is scoped to the launcher through `PYTHONPATH`; it is not copied into
the Conda environment globally.
