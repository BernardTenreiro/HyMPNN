#!/bin/bash
#SBATCH --job-name=egnn_std_a100
#SBATCH --output=qm9/logs/std_a1007L_ALPHA_%j.out
#SBATCH --error=qm9/logs/std_a100_%j.err
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

#change to your working directory
cd /gpfs/scratch/aschneble/MPNN/EGNN || exit 1
mkdir -p qm9/logs

module purge
module load anaconda

#change to your conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dimenet

echo "========================================"
echo "Standard EGNN_ALPHA on A100"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "========================================"
which python
python --version
nvidia-smi

python -u mqp.py \
  --exp_name full_egnn_baseline_a100 \
  --epochs 1000 \
  --n_layers 7 \
  --nf 128 \
  --property alpha \
  --batch_size 96 \
  --test_interval 25

echo "Finished: $(date)"
