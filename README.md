# RARE

RARE stands for **Retrieval-Aware Routing with Sparse Expert Rectification**.

This repository contains the latest research snapshot of the RARE router used in our `LLMRouterBench` reproduction and follow-up experiments.

## What Is Included

- `run_rare_performance.py`
  - RARE on `LLMRouterBench performance`
- `run_rare_performance_cost.py`
  - RARE on `LLMRouterBench performance-cost`
- `run_rare_cost_suite_aligned.py`
  - paper-style cost-suite evaluation
- `run_baselines_performance_cost.py`
  - local baselines already aligned to the same official prompt split
- `run_more_baselines_performance_cost.py`
  - bridge runner for additional baselines such as `EmbedLLM` and `GraphRouter`
- `build_llmrouterbench_performance_split_cache.py`
  - build official 5-seed prompt-split caches for the `performance` setting
- `export_avengers_official_data.py`
  - export official prompt splits into Avengers-compatible files

## Environment

This code assumes Python 3 with CUDA available for embedding and kNN-heavy runs.

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

Core packages:

- `numpy`
- `torch`
- `pyyaml`
- `pandas`
- `sentence-transformers`

## Required External Resources

This repository does not vendor the benchmark workspace or the embedding model.

Set these environment variables before running:

```bash
export RARE_POOL_EXP_ROOT=/path/to/PoolExp
export RARE_LLMROUTERBENCH_ROOT=/path/to/PoolExp/LLMRouterBench
export RARE_EMBEDDING_MODEL=/path/to/gte_Qwen2-7B-instruct
```

If `RARE_LLMROUTERBENCH_ROOT` is unset, the code falls back to `${RARE_POOL_EXP_ROOT}/LLMRouterBench`.

If `RARE_EMBEDDING_MODEL` is unset, the code tries these defaults:

- `${RARE_POOL_EXP_ROOT}/models/gte_Qwen2-7B-instruct`
- `../models/gte_Qwen2-7B-instruct`

## Reproducing Official 5-Seed Runs

### 1. Build `performance` split caches

```bash
python build_llmrouterbench_performance_split_cache.py --split-seeds 42 999 2024 2025 3407
```

### 2. Run RARE on `performance`

```bash
for seed in 42 999 2024 2025 3407; do
  python run_rare_performance.py --split-seed "$seed"
done
```

### 3. Run RARE on `performance-cost`

```bash
for seed in 42 999 2024 2025 3407; do
  python run_rare_performance_cost.py --split-seed "$seed"
done
```

### 4. Run aligned local baselines on `performance-cost`

```bash
python run_baselines_performance_cost.py --methods avengers knn routerbench_mlp --seeds 42 999 2024 2025 3407
```

## Metric Convention

For `LLMRouterBench performance-cost`, the primary metric in the current code is:

- `dataset_avg`

Auxiliary metric:

- `sample_avg`

This was changed to align local comparisons with `LLMRouterBench` dataset-level reporting.

## Current Best Local 5-Seed Summary

Under the unified official prompt split and `dataset_avg` reporting:

- `RARE`: `69.78%`
- `RouterBench-MLP`: `70.21%`
- `Avengers`: `68.54%`
- `Standard-kNN`: `66.46%`

These numbers are from our local runs in this workspace and are not copied from paper tables.

## Notes

- Large caches, benchmark dumps, and local experiment outputs are intentionally excluded from version control.
- Some extra baseline runners depend on baseline code that lives in the external `LLMRouterBench` workspace.
- `run_more_baselines_performance_cost.py` is a research bridge script and may need further cleanup for all third-party baselines.
