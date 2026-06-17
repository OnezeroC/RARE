# RARE

RARE 的全称是 **Retrieval-Aware Routing with Sparse Expert Rectification**。

当前仓库已经收敛为只做一件事：

- 保留最终版 `RARE` 方法实现
- 保留它与 `LLMRouterBench` 的评测接入入口
- 保留最小可运行的 `prepare + run` 流程

之前本地复现过的其他 baseline 运行逻辑，已经从主 pipeline 中移除。

## 目录结构

```text
RARE/
├── pipeline.py
├── README.md
├── README_zh.md
├── requirements.txt
├── src/
│   ├── cli/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── methods/
│   │   └── rare/
│   ├── models/
│   └── pipelines/
├── artifacts/
├── models/
└── .hf_home/
```

## 环境要求

正式运行需要 GPU。

```bash
conda activate rare-gpu
pip install -r requirements.txt
```

核心依赖：

- `numpy`
- `torch`
- `pyyaml`
- `pandas`
- `sentence-transformers`

## 默认本地资源

默认从当前仓库解析以下资源：

```bash
models/gte_Qwen2-7B-instruct/
.hf_home/
artifacts/
```

`LLMRouterBench` 需要你自行在本地提供。

推荐目录形式：

```bash
git clone https://github.com/ynulihao/LLMRouterBench.git third_party/LLMRouterBench
```

或者通过环境变量指定已有目录：

```bash
export RARE_LLMROUTERBENCH_ROOT=/path/to/LLMRouterBench
```

也支持环境变量覆盖：

```bash
export RARE_EMBEDDING_MODEL=/path/to/gte_Qwen2-7B-instruct
export RARE_HS_CACHE_ROOT=/path/to/hs_cache
export HF_HOME=/path/to/.hf_home
```

## Pipeline

根入口是：

```bash
python pipeline.py
```

当前 pipeline 只保留两个能力：

- `prepare split-cache`
- `run`

其中 `run` 永远执行最新版最终 `RARE` 方法。

## 标准流程

下面所有命令默认先执行：

```bash
conda activate rare-gpu
```

### 1. 构建官方 split cache

```bash
git clone https://github.com/ynulihao/LLMRouterBench.git third_party/LLMRouterBench

python pipeline.py prepare split-cache \
  --seeds 42 999 2024 2025 3407
```

### 2. 运行 `performance`

```bash
python pipeline.py run \
  --setting performance \
  --seeds 42 999 2024 2025 3407 \
  --skip-existing
```

### 3. 运行 `performance-cost`

```bash
python pipeline.py run \
  --setting performance-cost \
  --seeds 42 999 2024 2025 3407 \
  --skip-existing
```

## 常见运行方式

单个 seed 运行并输出到自定义结果文件：

```bash
python pipeline.py run \
  --setting performance \
  --seeds 42 \
  --result-json artifacts/results/custom_rare_seed42.json
```

在 `performance-cost` 下使用自定义 RARE 参数：

```bash
python pipeline.py run \
  --setting performance-cost \
  --seeds 42 999 2024 2025 3407 \
  --local-k 24 \
  --local-alpha 1.0 \
  --local-tau 0.03 \
  --local-uncertainty-threshold 2.0 \
  --gate-policy quantile_masked \
  --gate-threshold-quantile 0.95 \
  --backbone-loss-preset baseline \
  --retrieval-feature-preset perf_only \
  --local-inference-mode hard_switch \
  --blend-beta 0.75 \
  --blend-gamma 0.5
```

## `prepare split-cache` 参数

- `--seeds`
- `--llmrouterbench-root`
- `--config-path`
- `--train-ratio`

示例：

```bash
python pipeline.py prepare split-cache \
  --seeds 42 999 \
  --train-ratio 0.7
```

## `run` 参数

核心参数：

- `--setting {performance,performance-cost}`
- `--seeds`
- `--skip-existing`
- `--result-json`

RARE 调参参数：

- `--local-k`
- `--local-alpha`
- `--local-tau`
- `--local-uncertainty-threshold`
- `--gate-policy`
- `--gate-threshold-quantile`
- `--gate-target-rate`
- `--backbone-loss-preset`
- `--retrieval-feature-preset`
- `--local-inference-mode`
- `--blend-beta`
- `--blend-gamma`

以当前 CLI 为准：

```bash
python pipeline.py run --help
```
