#!/bin/bash
#SBATCH --job-name=mimic_rankings
#SBATCH --output=logs/mimic_rankings_%j.out
#SBATCH --error=logs/mimic_rankings_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
# Parts 1-4: cache top-10 rankings for the 12 full-MIMIC final-epoch checkpoints,
# qualitative examples (seed 3407), conditioned-on-rank-1 deltas, oracle ceilings.
# Inference only. SHA-256 of every checkpoint verified before loading.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u mimic/scripts/cache_rankings_and_examples_mimic.py
echo "END=$(date)"
