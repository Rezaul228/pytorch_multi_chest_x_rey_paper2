#!/bin/bash
#SBATCH --job-name=hyp_tests_6seed
#SBATCH --output=logs/hypothesis_tests_6seed_%j.out
#SBATCH --error=logs/hypothesis_tests_6seed_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=volta
#SBATCH --qos=normal

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

python3 extend_hypothesis_tests_6seed_paper2.py

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
