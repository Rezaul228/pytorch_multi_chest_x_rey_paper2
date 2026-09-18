#!/bin/bash
#SBATCH --job-name=section_coverage
#SBATCH --output=logs/section_coverage_%j.out
#SBATCH --error=logs/section_coverage_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=volta
#SBATCH --qos=normal
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
python3 -u openi/scripts/verify_section_boundary_coverage.py
echo "END=$(date)"
