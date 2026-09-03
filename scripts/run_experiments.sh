#!/usr/bin/env bash
#
# QM9 EGNN and HyEGNN experiment sweep.
#
# Every run appends stdout/stderr to logs/<exp_name>/train.log, so a run can be
# followed live with `tail -f logs/has_128_64/train.log`.
#
# Select a Python environment with PyTorch and the project dependencies:
#   PYTHON=/path/to/python bash scripts/run_experiments.sh
#
# The CUDA kernels JIT-build on first use and need nvcc + ninja on PATH:
#   export CUDA_HOME=/path/to/cuda
#   export PATH="$CUDA_HOME/bin:$PATH"
#
# Each run is ~1.5 h (EGNN) to ~2.3 h (HyEGNN); the whole sweep is ~24 h, so
# submit it with sbatch on h200x4-long -- a 2 h debug allocation will SIGKILL it.
#
# Set QM9_SUBSET=<n> to truncate all splits for a smoke test after deleting the
# cached files under data/processed/qm9/.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
START_EXPERIMENT="${START_EXPERIMENT:-1}"
if [[ ! "$START_EXPERIMENT" =~ ^[0-9]+$ ]] || (( START_EXPERIMENT < 1 || START_EXPERIMENT > 12 )); then
    echo "START_EXPERIMENT must be an integer from 1 through 12" >&2
    exit 2
fi
EXPERIMENT_INDEX=0

# Flags shared by every run in the sweep.
# --tf32        : tensor-core GEMMs (~1e-3 rel numerics; 1.27x on EGNN)
# --fused-dense : shared dense CUDA kernels used by both model families
# --fused-adam  : one Adam kernel instead of ~110 launches
# sym_asym Hybrid runs add --fused-pairwise and --cuda-graphs.  The latter
# captures forward + backward + fused Adam so the extra HyEGNN dispatch is not
# exposed on the critical path.  Other pairwise architectures remain eager.
COMMON=(--num-workers 8 --lr 5e-4 --property homo --epochs 1000 --batch-size 128
        --tf32 --fused-dense --fused-adam)

run() {
    local name="$1"; shift
    EXPERIMENT_INDEX=$((EXPERIMENT_INDEX + 1))
    local run_directory="$LOG_DIR/$name"
    local log="$run_directory/train.log"
    local metrics="$run_directory/metrics.json"

    if (( EXPERIMENT_INDEX < START_EXPERIMENT )); then
        echo "[$(date -Is)] [$EXPERIMENT_INDEX/12] $name -> skipped by START_EXPERIMENT"
        return 0
    fi
    if [[ -f "$metrics" ]] && grep -q '"completed": true' "$metrics"; then
        echo "[$(date -Is)] [$EXPERIMENT_INDEX/12] $name -> already complete"
        return 0
    fi

    mkdir -p "$run_directory"
    {
        echo "###############################################################"
        echo "# experiment: $EXPERIMENT_INDEX/12"
        echo "# exp_name : $name"
        echo "# started  : $(date -Is)"
        echo "# commit   : $(git rev-parse HEAD)"
        echo "# python   : $PYTHON"
        echo "# args     : ${COMMON[*]} --exp-name $name $*"
        echo "###############################################################"
    } >> "$log"

    "$PYTHON" -u scripts/train_qm9.py "${COMMON[@]}" --exp-name "$name" "$@" >> "$log" 2>&1
    local status=$?
    echo "# finished : $(date -Is)  exit=$status" >> "$log"
    echo "[$(date -Is)] [$EXPERIMENT_INDEX/12] $name -> exit $status  (log: $log)"
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
