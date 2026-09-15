#!/bin/bash
#SBATCH --job-name=verify_openi_baseline
#SBATCH --output=logs/verify_openi_baseline_%j.out
#SBATCH --error=logs/verify_openi_baseline_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=volta
#SBATCH --nodelist=pascal-node11.l3s.intra
#SBATCH --qos=normal

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

python3 verify_openi_baseline_checkpoints_paper2.py

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
