#!/bin/bash
#SBATCH --job-name=openi_B0_resume_checks
#SBATCH --output=logs/openi_B0_resume_checks_%j.out
#SBATCH --error=logs/openi_B0_resume_checks_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=volta
#SBATCH --qos=normal
# STEP B0: openi_sa resume checks (report only, CPU). Non-zero exit = a check failed.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
python3 -u openi/scripts/resume_checks_openi_sa.py
echo "END=$(date)"
