# GitHub 发布说明

## 推荐仓库信息

```text
Repository: sglang-qwen38-dual-3080-patch
Description: Minimal SGLang patch for Qwen3.8-27B + DFlash2 on dual RTX 3080: TP-sharded Draft fc and static per-head FP8 KV.
Visibility: Public
License: Apache-2.0
```

推荐 Topics：

```text
sglang
qwen
qwen3
dflash
speculative-decoding
fp8
kv-cache
rtx-3080
wsl2
tensor-parallel
```

## 使用 GitHub CLI 发布

先安装并登录 GitHub CLI，然后在本目录执行：

```bash
git init -b main
git config user.name "你的 GitHub 显示名"
git config user.email "你的 GitHub noreply 邮箱"
git add .
git commit -m "Initial release: TP-sharded DFlash fc and per-head FP8 KV"

gh auth login
gh repo create sglang-qwen38-dual-3080-patch \
  --public \
  --source . \
  --remote origin \
  --push \
  --description "Minimal SGLang patch for Qwen3.8-27B + DFlash2 on dual RTX 3080: TP-sharded Draft fc and static per-head FP8 KV."

gh repo edit --add-topic sglang \
  --add-topic qwen \
  --add-topic qwen3 \
  --add-topic dflash \
  --add-topic speculative-decoding \
  --add-topic fp8 \
  --add-topic kv-cache \
  --add-topic rtx-3080 \
  --add-topic wsl2 \
  --add-topic tensor-parallel
```

如果仓库名已存在，将命令中的名称换成其他名称。

## 使用 GitHub 网页发布

1. 在 GitHub 新建一个空的 Public repository；
2. 不要让网页额外生成 README、License 或 `.gitignore`，本包已经包含；
3. 上传压缩包解压后的全部文件；
4. 确认 `runtime/`、`scales/`、`scripts/` 和 `docs/` 均已上传；
5. 创建 commit；
6. 在仓库 About 区域添加 Description 和 Topics。

## 发布前检查

```bash
git diff --check
python test_package.py
PYTHONPATH="$PWD/runtime" python test_runtime_install.py
python apply_patch.py --verify
```

不要提交：

- API key、Token 或 `.env`；
- 模型权重；
- Conda 环境；
- SGLang 日志、CUDA dump；
- 带用户名的本机绝对路径；
- `~/.local/share/qwen38-minimal-patch/backups/` 下的备份。

## 建议首个 Release

Tag 可使用 `v0.1.0`，Release title：

```text
v0.1.0 — dual RTX 3080 tested baseline
```

Release notes 可直接写：

```text
- Pin SGLang to 4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3.
- TP-shard DFlash2 fc.weight with RowParallelLinear.
- Add static per-KV-head FP8 E4M3 scales for the Qwen3.8 target model.
- Keep DFlash2 Draft KV on per-rank/per-layer scalar FP8 E4M3.
- Add an Ampere/WSL2 safety fallback for symmetric-memory logits gather.
- Validate real startup, OpenAI API responses and prefix-cache reporting on 2 x RTX 3080 20 GiB.
```
