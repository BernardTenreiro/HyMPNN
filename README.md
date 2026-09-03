# HyMPNN

HyMPNN provides EGNN and hybrid EGNN (HyEGNN) models for QM9, with fused CUDA
kernels and bounded CUDA-graph capture to hide hybrid message-passing overhead
on NVIDIA Hopper GPUs.

## Repository layout

- `src/hympnn/` contains all reusable Python code and CUDA kernels.
- `scripts/` contains training, plotting, experiment, and Slurm entrypoints.
- `data/raw/qm9/` contains source data; `data/processed/qm9/` contains cached splits.
- `logs/<experiment>/` contains metrics and console output for each run.
- `docs/` contains historical benchmark results and project notes.

The old top-level `EGNN/`, `src/EGNN/`, and `src/HyEGNN/` trees have been
removed. Legacy demos and duplicate model implementations that were not part of
the maintained QM9 training path were removed with them.

## Environment

The project requires Python 3.10+, PyTorch with CUDA support, NumPy, NetworkX,
the CUDA Toolkit, and Ninja. The custom kernels are compiled on first use via
PyTorch's JIT extension loader. The current compiler flags target compute
capability 9.0 (H100/H200).

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHON=/path/to/python
```

For an editable package install:

```bash
"$PYTHON" -m pip install -e .
```

The checkout entrypoint works without installing the package.

## Training

Standard five-layer EGNN:

```bash
"$PYTHON" -u scripts/train_qm9.py \
  --exp-name egnn_baseline \
  --property homo \
  --batch-size 128 \
  --n-layers 5 \
  --nf 128
```

Optimized 5+5 HyEGNN:

```bash
"$PYTHON" -u scripts/train_qm9.py \
  --exp-name hyegnn_optimized \
  --property homo \
  --batch-size 128 \
  --hybrid \
  --n-standard-layers 5 \
  --n-pairwise-layers 5 \
  --nf 128 \
  --nf-sparse 64 \
  --pairwise-layer-type sym_asym
```

The optimized CUDA execution profile is the trainer's only production path;
it requires an NVIDIA CUDA GPU and has no performance opt-in flags. Every run
automatically uses TF32, fused dense kernels, fused Adam, graph-safe GEMMs, and
full-step CUDA graphs. A `sym_asym` HybridEGNN also selects the fused pairwise
kernel automatically. Graph capture is limited to common batch shapes;
uncommon shapes safely fall back to eager execution within the same optimized
profile.

Run the full 12-experiment matrix with:

```bash
PYTHON="$PYTHON" bash scripts/run_experiments.sh
```

The sweep skips runs whose `metrics.json` contains `"completed": true`, so it
can safely resume after an interrupted allocation. Set `START_EXPERIMENT=N` to
skip directly to a numbered run.

Every experiment writes its console output to `logs/<name>/train.log` and its
structured results to `logs/<name>/metrics.json`.

## Code quality

```bash
uvx ruff format --check .
uvx ruff check .
bash -n scripts/*.sh scripts/slurm/*.sh
```
