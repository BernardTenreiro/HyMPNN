#!/usr/bin/env bash
# Controlled ablations for the HS-128 versus S7-128 accuracy gap.

set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
LOG_ROOT="$REPOSITORY_ROOT/logs/ablations"
START_ABLATION="${START_ABLATION:-1}"
END_ABLATION="${END_ABLATION:-3}"
ABLATION_INDEX=0

mkdir -p "$LOG_ROOT"

run() {
    local name="$1"
    shift
    ABLATION_INDEX=$((ABLATION_INDEX + 1))
    local run_directory="$LOG_ROOT/$name"
    local log="$run_directory/train.log"
    local metrics="$run_directory/metrics.json"

    if (( ABLATION_INDEX < START_ABLATION || ABLATION_INDEX > END_ABLATION )); then
        return 0
    fi
    if [[ -f "$metrics" ]] && grep -q '"completed": true' "$metrics"; then
        echo "[$(date -Is)] [$ABLATION_INDEX/3] $name -> already complete"
        return 0
    fi

    mkdir -p "$run_directory"
    {
        echo "###############################################################"
        echo "# ablation : $ABLATION_INDEX/3"
        echo "# exp_name : $name"
        echo "# started  : $(date -Is)"
        echo "# commit   : $(git rev-parse HEAD)"
        echo "# python   : $PYTHON"
        echo "# args     : --output-dir $LOG_ROOT --exp-name $name $*"
        echo "###############################################################"
    } >> "$log"

    "$PYTHON" -u scripts/train_qm9.py \
        --output-dir "$LOG_ROOT" --exp-name "$name" "$@" >> "$log" 2>&1
    local status=$?
    echo "# finished : $(date -Is)  exit=$status" >> "$log"
    echo "[$(date -Is)] [$ABLATION_INDEX/3] $name -> exit $status  (log: $log)"
    return $status
}

COMMON_HS=(
    --nf 128
    --n-standard-layers 5
    --hybrid
    --pairwise-layer-type symmetric
)

# Projection/decoder control: isolates the 128 -> 64 bottleneck without any HS updates.
run hs_projection_128_64 \
    "${COMMON_HS[@]}" --nf-sparse 64 --n-pairwise-layers 0 --n-pairwise-steps 0

# Width control: removes the 128 -> 64 bottleneck from the existing five-step HS model.
run hs_wide_128_128 \
    "${COMMON_HS[@]}" --nf-sparse 128 --n-pairwise-layers 5 --n-pairwise-steps 5

# Coverage control: reuse the same five learned modules for 31 matching steps.
# The current greedy coloring uses at most 31 colors for QM9's <=29 real atoms,
# so 31 consecutive schedule entries include at least one complete color sweep.
run hs_full_sweep_128_64 \
    "${COMMON_HS[@]}" --nf-sparse 64 --n-pairwise-layers 5 --n-pairwise-steps 31
