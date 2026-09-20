#!/bin/bash
#SBATCH --job-name=a2_preflight
#SBATCH --output=logs/a2_preflight_%j.out
#SBATCH --error=logs/a2_preflight_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
python3 -u mimic/scripts/preflight_scale_ablation_attempt2.py
echo "END=$(date)"
