#!/usr/bin/env bash
# Claim unfinished experiments without duplicating work across GPU workers.

set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

EXPERIMENT_NAMES=(
    unused
    has_128_64 has_64_64 he_128_64 he_64_64
    hs_128_64 hs_64_64 hj_128_64 hj_64_64
    s5_128 s5_64 s7_128 s7_64
)
LOCK_DIRECTORY="$REPOSITORY_ROOT/logs/.dispatch-locks"
WORKER_LOG="$REPOSITORY_ROOT/logs/dispatch-$(hostname)-gpu${CUDA_VISIBLE_DEVICES:-unknown}.log"
mkdir -p "$LOCK_DIRECTORY"

if (( $# == 0 )); then
    set -- 8 9 10 11 12
fi

for experiment_index in "$@"; do
    if [[ ! "$experiment_index" =~ ^[0-9]+$ ]] || (( experiment_index < 1 || experiment_index > 12 )); then
        echo "Invalid experiment index: $experiment_index" >&2
        exit 2
    fi

    experiment_name="${EXPERIMENT_NAMES[$experiment_index]}"
    lock_file="$LOCK_DIRECTORY/$experiment_name.lock"
    (
        flock -n 9 || exit 75
        printf '[%s] claiming experiment %d/12: %s\n' \
            "$(date -Is)" "$experiment_index" "$experiment_name" >> "$WORKER_LOG"
        START_EXPERIMENT="$experiment_index" END_EXPERIMENT="$experiment_index" \
            bash scripts/run_experiments.sh >> "$WORKER_LOG" 2>&1

        metrics_file="$REPOSITORY_ROOT/logs/$experiment_name/metrics.json"
        if [[ -f "$metrics_file" ]] && grep -q '"completed": true' "$metrics_file"; then
            printf '[%s] completed experiment %d/12: %s\n' \
                "$(date -Is)" "$experiment_index" "$experiment_name" >> "$WORKER_LOG"
        else
            printf '[%s] experiment %d/12 did not complete: %s\n' \
                "$(date -Is)" "$experiment_index" "$experiment_name" >> "$WORKER_LOG"
            exit 1
        fi
    ) 9> "$lock_file"

    status=$?
    if (( status == 75 )); then
        continue
    fi
done
