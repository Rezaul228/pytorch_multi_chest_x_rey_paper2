#!/bin/bash
#SBATCH --job-name=a2_step4_eval
#SBATCH --output=logs/a2_step4_eval_%j.out
#SBATCH --error=logs/a2_step4_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u mimic/scripts/evaluate_scale_ablation_attempt2.py
echo "END=$(date)"
