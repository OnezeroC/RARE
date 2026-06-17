# RARE

RARE stands for **Retrieval-Aware Routing with Sparse Expert Rectification**.

This repository contains the final `RARE` implementation and a minimal pipeline
for running it on `LLMRouterBench`.

## Setup

GPU is required.

```bash
conda activate rare-gpu
pip install -r requirements.txt
```

Prepare a local `LLMRouterBench` checkout:

```bash
git clone https://github.com/ynulihao/LLMRouterBench.git third_party/LLMRouterBench
```

If your local path is different, set:

```bash
export RARE_LLMROUTERBENCH_ROOT=/path/to/LLMRouterBench
```

## Run

Build official split caches:

```bash
python pipeline.py prepare split-cache \
  --seeds 42 999 2024 2025 3407
```

Run `RARE` on `performance`:

```bash
python pipeline.py run \
  --setting performance \
  --seeds 42 999 2024 2025 3407 \
  --skip-existing
```

Run `RARE` on `performance-cost`:

```bash
python pipeline.py run \
  --setting performance-cost \
  --seeds 42 999 2024 2025 3407 \
  --skip-existing
```

For the exact live CLI:

```bash
python pipeline.py --help
python pipeline.py run --help
```
