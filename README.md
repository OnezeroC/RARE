# RARE

RARE stands for **Retrieval-Aware Routing with Sparse Expert Rectification**.

This repository contains a compact research workspace for continuing reproduction and improvement of the RARE routing method on LLM routing benchmarks. It is extracted from a larger internal workspace and keeps only the core code needed to run the current RARE pipeline.

## Method Overview

RARE is a two-stage router:

1. **Retrieval-aware backbone**
   - Encode the query.
   - Retrieve nearest training examples.
   - Build a neighbor performance profile over candidate models.
   - Concatenate that profile with the query embedding and train a global router.

2. **Sparse local rectification**
   - Build a local residual correction candidate from retrieval neighbors.
   - Use a learned gate to decide when the local correction is actually helpful.
   - Trigger the second stage only on a small subset of queries.

In short, RARE combines a retrieval-aware global router with a sparse local correction module.

## Repository Layout

```text
RARE/
├── run_rare_performance.py
├── run_rare_performance_cost.py
├── common/
│   └── shared_embedding_cache.py
└── src/
    ├── glider_router.py
    ├── glider_v2_router.py
    └── rare_shared.py
```

## Main Entry Points

- `run_rare_performance.py`
  - Runs RARE on the `LLMRouterBench performance` setting.
- `run_rare_performance_cost.py`
  - Runs RARE on the `LLMRouterBench performance-cost` setting.

## Core Modules

- `src/glider_router.py`
  - Backbone training, local residual rectification, and embedding-cache-related logic reused by RARE.
- `src/glider_v2_router.py`
  - GPU kNN scoring utilities.
- `src/rare_shared.py`
  - Shared utilities for official splits, evaluation metrics, risk gating, and performance-cost data handling.
- `common/shared_embedding_cache.py`
  - SQLite-based shared embedding cache.

## Requirements

This code is currently a research snapshot rather than a fully packaged release. The current pipeline assumes:

- Python 3
- `numpy`
- `torch`
- A CUDA-enabled environment for training and GPU kNN retrieval
- Access to the embedding model used in the experiments:
  - `gte_Qwen2-7B-instruct`

## Data and Cache Setup

Large benchmark caches, local experiment results, reports, and internal handoff documents are intentionally **not** included in this public repository.

To run the scripts, you will need to prepare the following resources in your own environment:

- Official `LLMRouterBench performance` split cache
- Official `LLMRouterBench performance-cost` train/test split files
- Embedding caches for the chosen embedding model
- A writable SQLite embedding cache

The current codebase also contains local-path assumptions inherited from the original workspace, especially for the embedding model location. You may need to adjust these paths before running experiments in a fresh environment.

## Current Scope

This repository is intended for:

- reproducing the current RARE pipeline,
- iterating on routing improvements,
- and serving as a compact code release for the method.

It does not include:

- intermediate exploratory scripts from earlier development,
- large cached artifacts,
- benchmark result dumps,
- or internal project handoff notes.

## Notes

- The code is organized around the current research workflow, not around a general-purpose package API.
- If you plan to reuse it outside the original environment, start by checking path assumptions and data-loading utilities in `src/rare_shared.py`.
