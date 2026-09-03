#!/usr/bin/env bash
#SBATCH --job-name=egnn
#SBATCH --output=egnn-%j.out
#SBATCH --error=egnn-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$REPO_ROOT"

exec "$PYTHON" -u EGNN/main_qm9_pairwise.py \
    --exp-name "${EXP_NAME:-egnn_5_layer}" \
    --property "${PROPERTY:-homo}" \
    --epochs "${EPOCHS:-1000}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --num-workers "${NUM_WORKERS:-8}" \
    --n-layers 5 \
    --nf 128 \
    --tf32 \
    --fused-dense \
    --fused-adam \
    "$@"
