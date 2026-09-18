#!/bin/bash
#SBATCH --job-name=A2_impression_spec
#SBATCH --output=logs/A2_impression_specificity_%j.out
#SBATCH --error=logs/A2_impression_specificity_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=volta
#SBATCH --qos=normal
# STEP A2: Impression-specificity addendum, MIMIC / ReXGradient / openi_sa test splits. CPU only.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
python3 -u impression_specificity_addendum.py
echo "END=$(date)"
