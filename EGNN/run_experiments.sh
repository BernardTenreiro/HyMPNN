#!/usr/bin/env bash
#
# QM9 pairwise-EGNN experiment sweep.
#
# Every run appends its stdout/stderr to EGNN/log/<exp_name>.log, so a run can
# be followed live with `tail -f EGNN/log/has_128_64.log`.
#
# The base conda env has a broken torch install, so point PYTHON at an env that
# actually has one.  Override on the command line if you use a different env:
#   PYTHON=/path/to/python bash EGNN/run_experiments.sh
#
# The CUDA kernels JIT-build on first use and need nvcc + ninja on PATH:
#   export CUDA_HOME=/usr/local/cuda-13.1
#   export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH
#
# Each run is ~1.5 h (EGNN) to ~2.3 h (HyEGNN); the whole sweep is ~24 h, so
# submit it with sbatch on h200x4-long -- a 2 h debug allocation will SIGKILL it.
#
# Set QM9_SUBSET=<n> to truncate all splits to n molecules for a quick smoke
# test (delete EGNN/data/qm9/*.npz first -- the processed splits are cached).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/lustre/nvwulf/software/miniconda3/envs/pytorch/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
LOG_DIR="$REPO_ROOT/EGNN/log"
mkdir -p "$LOG_DIR"

# Flags shared by every run in the sweep.
# --tf32        : tensor-core GEMMs (~1e-3 rel numerics; 1.27x on EGNN)
# --fused_dense : src/EGNN CUDA kernels for E_GCL_mask (shared by both models)
# --fused_adam  : one Adam kernel instead of ~110 launches
# sym_asym Hybrid runs add --fused_pairwise and --cuda_graphs.  The latter
# captures forward + backward + fused Adam so the extra HyEGNN dispatch is not
# exposed on the critical path.  Other pairwise architectures remain eager.
COMMON=(--num_workers 8 --lr 5e-4 --property homo --epochs 1000 --batch_size 128
        --tf32 --fused_dense --fused_adam)

run() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    {
        echo "###############################################################"
        echo "# exp_name : $name"
        echo "# started  : $(date -Is)"
        echo "# python   : $PYTHON"
        echo "# args     : ${COMMON[*]} --exp_name $name $*"
        echo "###############################################################"
    } >> "$log"

    "$PYTHON" -u EGNN/main_qm9_pairwise.py "${COMMON[@]}" --exp_name "$name" "$@" >> "$log" 2>&1
    local status=$?
    echo "# finished : $(date -Is)  exit=$status" >> "$log"
    echo "[$(date -Is)] $name -> exit $status  (log: $log)"
    return $status
}

# --- Hybrid models: 5 standard EGNN layers + 5 pairwise layers ---------------
HYBRID=(--nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid)

run has_128_64 --nf 128 "${HYBRID[@]}" --pairwise_layer_type sym_asym --fused_pairwise --cuda_graphs
run has_64_64  --nf 64  "${HYBRID[@]}" --pairwise_layer_type sym_asym --fused_pairwise --cuda_graphs

run he_128_64  --nf 128 "${HYBRID[@]}" --pairwise_layer_type egcl
run he_64_64   --nf 64  "${HYBRID[@]}" --pairwise_layer_type egcl

run hs_128_64  --nf 128 "${HYBRID[@]}" --pairwise_layer_type symmetric
run hs_64_64   --nf 64  "${HYBRID[@]}" --pairwise_layer_type symmetric

run hj_128_64  --nf 128 "${HYBRID[@]}" --pairwise_layer_type joint
run hj_64_64   --nf 64  "${HYBRID[@]}" --pairwise_layer_type joint

# --- Standard EGNN baselines ------------------------------------------------
run s5_128 --nf 128 --n_layers 5
run s5_64  --nf 64  --n_layers 5
run s7_128 --nf 128 --n_layers 7
run s7_64  --nf 64  --n_layers 7
