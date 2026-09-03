# HyMPNN

HyMPNN contains EGNN and hybrid EGNN (HyEGNN) models for QM9, including CUDA
optimizations that reduce the hybrid model's launch overhead on NVIDIA Hopper GPUs.

## Repository layout

- `EGNN/main_qm9_pairwise.py` — QM9 training and evaluation entrypoint.
- `EGNN/qm9/models_pairwise.py` — EGNN, sparse, pairwise, and hybrid model definitions.
- `EGNN/qm9/cuda_graphs.py` — bounded, shape-bucketed CUDA graph runner.
- `src/EGNN/` — fused dense EGNN CUDA extension.
- `src/HyEGNN/` — fused symmetric/asymmetric pairwise CUDA extensions.
- `EGNN/run_experiments.sh` — reproducible baseline and hybrid experiment sweep.
- `EGNN/scripts/` — plotting and Slurm launch helpers.

## Environment

The training code requires Python, PyTorch with CUDA support, NumPy, NetworkX, and the
dependencies used by the original EGNN QM9 implementation. The custom kernels are JIT
compiled through `torch.utils.cpp_extension`, so CUDA Toolkit and Ninja must also be
available. The current compiler flags target compute capability 9.0 (H100/H200).

Set the CUDA toolkit and Python executable for your environment:

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHON=/path/to/python
```

The QM9 preparation code downloads and caches the dataset on first use.

## Training

A standard five-layer EGNN baseline:

```bash
"$PYTHON" -u EGNN/main_qm9_pairwise.py \
  --exp-name egnn_baseline \
  --property homo \
  --batch-size 128 \
  --n-layers 5 \
  --nf 128
```

An optimized 5+5 HyEGNN run:

```bash
"$PYTHON" -u EGNN/main_qm9_pairwise.py \
  --exp-name hyegnn_optimized \
  --property homo \
  --batch-size 128 \
  --hybrid \
  --n-standard-layers 5 \
  --n-pairwise-layers 5 \
  --nf 128 \
  --nf-sparse 64 \
  --pairwise-layer-type sym_asym \
  --tf32 \
  --fused-dense \
  --fused-pairwise \
  --fused-adam \
  --cuda-graphs
```

`--cuda-graphs` requires all of the fused options shown above. Graph capture is bounded
to common batch shapes; uncommon shapes safely use the eager path. The bucket capacities
can be tuned with `EGNN_GRAPH_DENSE_CAP`, `EGNN_GRAPH_DENSE_QUANTUM`,
`EGNN_GRAPH_SPARSE_QUANTUM`, `EGNN_GRAPH_SPARSE_SMALL_CAP`,
`EGNN_GRAPH_SPARSE_LARGE_CAP`, and `EGNN_GRAPH_MAX_BUCKETS`.

For the full experiment matrix, run:

```bash
PYTHON="$PYTHON" bash EGNN/run_experiments.sh
```

Results are written beneath `EGNN/qm9/logs/<exp_name>/`. The historical
`losess.json` filename and `losess` result key are retained for compatibility with
existing experiment consumers.

## Code quality

Python formatting and static checks are configured in `pyproject.toml`:

```bash
uvx ruff format --check .
uvx ruff check .
```

Shell scripts can be syntax-checked with `bash -n`.
