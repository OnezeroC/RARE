# RARE

RARE 的全称是 **Retrieval-Aware Routing with Sparse Expert Rectification**。

这个仓库包含最终版 `RARE` 实现，以及一个用于在 `LLMRouterBench`
上运行它的最小 pipeline。

## 环境

需要 GPU。

```bash
conda activate rare-gpu
pip install -r requirements.txt
```

本地准备一份 `LLMRouterBench`：

```bash
git clone https://github.com/ynulihao/LLMRouterBench.git third_party/LLMRouterBench
```

如果你的本地路径不同，设置：

```bash
export RARE_LLMROUTERBENCH_ROOT=/path/to/LLMRouterBench
```

## 运行

先构建官方 split cache：

```bash
python pipeline.py prepare split-cache \
  --seeds 42 999 2024 2025 3407
```

运行 `performance`：

```bash
python pipeline.py run \
  --setting performance \
  --seeds 42 999 2024 2025 3407 \
  --skip-existing
```

运行 `performance-cost`：

```bash
python pipeline.py run \
  --setting performance-cost \
  --seeds 42 999 2024 2025 3407 \
  --skip-existing
```

查看当前 CLI：

```bash
python pipeline.py --help
python pipeline.py run --help
```
