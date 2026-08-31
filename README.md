# 双 RTX 3080 部署 Qwen3.8-27B + DFlash2

这是一个面向 SGLang 的小型、可回滚补丁包，保留本机部署中收益最明确的两项优化：

1. 将 DFlash2 巨大的 `fc.weight` 从每张卡完整复制的 `nn.Linear` 改成 TP 分片的 `RowParallelLinear`。
2. 为主模型实现静态 **Per-KV-Head FP8 E4M3** 缩放；DFlash2 草稿模型继续使用更稳妥的逐层标量 FP8 缩放。

针对双 RTX 3080 / WSL2，运行时还包含一个必要的兼容保护：在 SM90 以下禁用 Hopper 专用的 symmetric-memory logits gather，回退到普通 TP NCCL all-gather，避免目标 verify CUDA Graph 阶段触发 `SIGFPE`。

> 这不是通用 SGLang 分支，也不包含模型权重。补丁只面向下方锁定的 SGLang 基线、Qwen3.8-27B 主模型和 DFlash2 草稿模型。

## 实测结果

在 WSL2、两张 20 GiB RTX 3080、TP=2 上完成了真实启动和 API 验证：

- 主模型 16 个全注意力层在两个 TP rank 上加载 Per-Head KV 缩放；
- DFlash2 五层草稿模型在两个 TP rank 上加载逐层标量缩放；
- DFlash2 权重约占 0.94 GiB/卡，`fc.weight` 采用 TP 分片；
- `MEM_FRACTION_STATIC=0.90` 时，设备 KV 池为 218,278 tokens；
- Target Prefill/Verify CUDA Graph 捕获成功；
- `/v1/chat/completions` 返回 HTTP 200，重复请求报告 `cached_tokens=64`；
- 测试请求中可观察到 DFlash `accept len` / `accept rate`；
- 未出现 CUDA OOM 或 illegal-memory-access。

在该保守显存比例下，DFlash Draft CUDA Graph 可能因初始化后仅剩约 0.20 GiB 而保持 eager。这影响 Draft 延迟，不影响功能正确性。

## 锁定环境

| 组件 | 实测版本 |
|---|---|
| SGLang | `0.5.19.dev228+g4cb5aebfe` |
| 上游 commit | `4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3` |
| Python | `3.11.15` |
| PyTorch | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| FlashInfer | `0.6.17` |
| compressed-tensors | `0.18.1a20260816` |
| Triton | `3.7.1` |
| NCCL | `2.29.7` |
| GPU/驱动 | `2 x RTX 3080 20 GiB / 610.62 / SM86` |

安装器使用语义锚点而非固定行号；未来版本源码漂移时会在写文件前停止，不会盲目套补丁。详细兼容说明见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 目录结构

```text
.
├── apply_patch.py                  # 安装、验证、回滚
├── start_example.sh                # 完整启动命令
├── scripts/
│   ├── create_tested_env.sh        # 从已有 SGLang 环境复制并重建锁定基线
│   └── verify_server.sh            # API 和缓存命中自检
├── runtime/
│   ├── sitecustomize.py            # 只注入本包运行时补丁
│   ├── qwen38_target_per_head_loader.py
│   ├── dflash_scalar_kv_loader.py
│   └── sglang_per_head_fp8.py
├── scales/
│   ├── qwen38-fp8-kv-scales-per-head.json
│   └── dflash2-fp8-kv-scales-scalar.json
└── test_*.py
```

补丁会修改当前环境中的：

```text
sglang/srt/models/dflash.py
sglang/srt/layers/attention/flashinfer_backend.py
```

并新增：

```text
sglang/srt/layers/attention/sglang_per_head_fp8.py
```

`dflash_worker_v2.py` 不会被修改。

## 1. 准备模型

默认目录为：

```text
~/models/Qwen3.8-27B-AWQ/
~/models/Qwen3.8-27B-DFlash2-W4A16/
```

本机实测对应的公开模型仓库是：

- 主模型：[cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4)
- Draft：[syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)

使用 Hugging Face 官方 CLI 下载：

```bash
python -m pip install -U huggingface_hub

hf download cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --local-dir "$HOME/models/Qwen3.8-27B-AWQ"

hf download syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --local-dir "$HOME/models/Qwen3.8-27B-DFlash2-W4A16"
```

至少应存在：

```text
~/models/Qwen3.8-27B-AWQ/config.json
~/models/Qwen3.8-27B-DFlash2-W4A16/config.json
~/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors
```

也可以不改脚本，通过 `MODEL_PATH` 和 `DRAFT_MODEL_PATH` 指向其他目录。缩放文件是针对当前模型结构、TP=2 和这组校准结果生成的，不应直接套到不同架构或不同 TP 数量。

## 2. 创建干净且可复现的环境

最稳妥的做法是从一个依赖齐全、能正常导入 SGLang 的 Conda 环境复制，再只重装锁定的官方源码。假设现有环境名为 `sglang`：

```bash
chmod +x scripts/create_tested_env.sh
BASE_ENV=sglang TARGET_ENV=sglang-qwen38 ./scripts/create_tested_env.sh
```

脚本会：

1. 克隆 Conda 环境，保留 CUDA/PyTorch/FlashInfer 等已验证依赖；
2. 仅清理新环境里的 SGLang 包目录；
3. 拉取官方 `sgl-project/sglang`；
4. checkout 到 `4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3`；
5. 以 `SGLANG_BUILD_RUST_EXTS=none` 安装官方 Python 包；
6. 打印关键版本。

它不会修改原来的 `sglang` 环境。若已具备同一官方基线，可以跳过这一步。

## 3. 安装补丁

```bash
conda activate sglang-qwen38
python apply_patch.py
python apply_patch.py --verify
```

首次安装会把所有将修改或覆盖的文件备份到：

```text
~/.local/share/qwen38-minimal-patch/backups/<timestamp>/
```

重复执行是安全的，会返回 `already-applied`。

## 4. 离线验证

```bash
python test_package.py
PYTHONPATH="$PWD/runtime" python test_runtime_install.py
```

测试覆盖缩放文件 schema、TP 头切分、KV write scale 形状、Q/output 缩放和运行时注入范围，不需要加载模型。

## 5. 启动服务

脚本不会内置 API key。请在启动时显式提供：

```bash
chmod +x start_example.sh

SGLANG_CONDA_ENV=sglang-qwen38 \
SGLANG_API_KEY='请替换为你自己的本地密钥' \
./start_example.sh
```

完整命令在 [start_example.sh](start_example.sh) 中。关键默认值：

```text
主模型 KV: Per-Head FP8 E4M3
Draft KV: 每 TP rank、每层一个 K/V 标量 FP8 E4M3
DFlash fc: RowParallelLinear，TP=2
context length: 模型配置默认值
mem fraction static: 0.90
chunked prefill: 1024
max running requests: 2
port: 8000
```

覆盖常用参数示例：

```bash
MODEL_PATH="$HOME/models/Qwen3.8-27B-AWQ" \
DRAFT_MODEL_PATH="$HOME/models/Qwen3.8-27B-DFlash2-W4A16" \
MEM_FRACTION_STATIC=0.90 \
SGLANG_PORT=8000 \
SGLANG_API_KEY='replace-me' \
./start_example.sh
```

## 6. API 验证

服务显示 `Uvicorn running on http://0.0.0.0:8000` 后：

```bash
SGLANG_API_KEY='与启动时相同的密钥' ./scripts/verify_server.sh
```

脚本会检查 `/model_info`，再发送两次相同的 OpenAI Chat Completions 请求，第二次用于观察 `prompt_tokens_details.cached_tokens`。

手动调用：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "用一句话介绍 SGLang。"}],
    "max_tokens": 64,
    "stream": false
  }'
```

## 回滚

使用安装器打印的备份目录：

```bash
python apply_patch.py \
  --rollback ~/.local/share/qwen38-minimal-patch/backups/<timestamp>
```

## 不包含的本机实验补丁

- DFlash Per-Head KV；
- compressed-tensors KV-only context projection；
- HiCache safe-copy/host-reserve；
- 延迟 DFlash CUDA Graph 捕获；
- EAGLE/MTP 兼容和 early weight sharing。

这些功能可以单独叠加，但不属于本仓库的最小可复现范围。

## 风险和适用范围

- 这是针对特定模型结构与版本的研究/部署补丁，不是 SGLang 官方功能。
- 主模型 Per-Head scale 为静态校准值，不是运行时动态校准。
- 仅验证 TP=2；不要直接用于 TP=1/4。
- RTX 3080 不原生执行 FP8 Tensor Core，FP8 在这里主要用于 KV 容量和带宽优化。
- 升级 SGLang 后应先在副本环境执行 `apply_patch.py --verify` 和离线测试。

## License

Apache-2.0。SGLang 本身及模型权重遵循各自的上游许可证。
