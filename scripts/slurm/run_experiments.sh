#!/usr/bin/env bash
#SBATCH --job-name=hympnn-sweep
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err
#SBATCH --partition=h200x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHON="${PYTHON:-/lustre/nvwulf/software/miniconda3/envs/pytorch/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPOSITORY_ROOT"
exec bash scripts/run_experiments.sh
