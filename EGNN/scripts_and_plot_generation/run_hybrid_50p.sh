#!/bin/bash
#SBATCH --job-name=egnn_hybrid_5p5_nf64_a100
#SBATCH --output=qm9/logs/hybrid_half_gap%j.out
#SBATCH --error=qm9/logs/hybrid_5p5_nf64_bs96_%j.err
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
echo "HybridEGNN 5 standard + 5 pairwise, reduced width"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "========================================"

which python
python --version
nvidia-smi


python -u main_qm9_pairwise.py \
  --exp_name hybrid_5p5_nf64_bs96 \
  --hybrid \
  --n_standard_layers 5 \
  --n_pairwise_layers 5 \
  --nf 128 \
  --nf_sparse 96 \
  --epochs 1000 \
  --property gap \
  --batch_size 96 \
  --test_interval 25