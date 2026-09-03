#!/usr/bin/env bash
#
# QM9 pairwise-EGNN experiment sweep.
#
# Every run appends its stdout/stderr to EGNN/log/<exp_name>.log, so a run can
# be followed live with `tail -f EGNN/log/has_128_64.log`.
#
# Select a Python environment with PyTorch and the project dependencies:
#   PYTHON=/path/to/python bash EGNN/run_experiments.sh
#
# The CUDA kernels JIT-build on first use and need nvcc + ninja on PATH:
#   export CUDA_HOME=/path/to/cuda
#   export PATH="$CUDA_HOME/bin:$PATH"
#
# Each run is ~1.5 h (EGNN) to ~2.3 h (HyEGNN); the whole sweep is ~24 h, so
# submit it with sbatch on h200x4-long -- a 2 h debug allocation will SIGKILL it.
#
# Set QM9_SUBSET=<n> to truncate all splits to n molecules for a quick smoke
# test (delete EGNN/data/qm9/*.npz first -- the processed splits are cached).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/EGNN/log"
mkdir -p "$LOG_DIR"

# Flags shared by every run in the sweep.
# --tf32        : tensor-core GEMMs (~1e-3 rel numerics; 1.27x on EGNN)
# --fused-dense : src/EGNN CUDA kernels for E_GCL_mask (shared by both models)
# --fused-adam  : one Adam kernel instead of ~110 launches
# sym_asym Hybrid runs add --fused-pairwise and --cuda-graphs.  The latter
# captures forward + backward + fused Adam so the extra HyEGNN dispatch is not
# exposed on the critical path.  Other pairwise architectures remain eager.
COMMON=(--num-workers 8 --lr 5e-4 --property homo --epochs 1000 --batch-size 128
        --tf32 --fused-dense --fused-adam)

run() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    {
        echo "###############################################################"
        echo "# exp_name : $name"
        echo "# started  : $(date -Is)"
        echo "# python   : $PYTHON"
        echo "# args     : ${COMMON[*]} --exp-name $name $*"
        echo "###############################################################"
    } >> "$log"

    "$PYTHON" -u EGNN/main_qm9_pairwise.py "${COMMON[@]}" --exp-name "$name" "$@" >> "$log" 2>&1
    local status=$?
    echo "# finished : $(date -Is)  exit=$status" >> "$log"
    echo "[$(date -Is)] $name -> exit $status  (log: $log)"
    return $status
}

# --- Hybrid models: 5 standard EGNN layers + 5 pairwise layers ---------------
HYBRID=(--nf-sparse 64 --n-layers 10 --n-standard-layers 5 --n-pairwise-layers 5 --hybrid)

run has_128_64 --nf 128 "${HYBRID[@]}" --pairwise-layer-type sym_asym --fused-pairwise --cuda-graphs
run has_64_64  --nf 64  "${HYBRID[@]}" --pairwise-layer-type sym_asym --fused-pairwise --cuda-graphs

run he_128_64  --nf 128 "${HYBRID[@]}" --pairwise-layer-type egcl
run he_64_64   --nf 64  "${HYBRID[@]}" --pairwise-layer-type egcl

run hs_128_64  --nf 128 "${HYBRID[@]}" --pairwise-layer-type symmetric
run hs_64_64   --nf 64  "${HYBRID[@]}" --pairwise-layer-type symmetric

run hj_128_64  --nf 128 "${HYBRID[@]}" --pairwise-layer-type joint
run hj_64_64   --nf 64  "${HYBRID[@]}" --pairwise-layer-type joint

# --- Standard EGNN baselines ------------------------------------------------
run s5_128 --nf 128 --n-layers 5
run s5_64  --nf 64  --n-layers 5
run s7_128 --nf 128 --n-layers 7
run s7_64  --nf 64  --n-layers 7
