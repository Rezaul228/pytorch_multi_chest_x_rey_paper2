#!/bin/bash
#SBATCH --job-name=zero_shot_indiana
#SBATCH --output=logs/zero_shot_indiana_%j.out
#SBATCH --error=logs/zero_shot_indiana_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs

echo "Zero-shot: MIMIC-trained model -> Indiana train set"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Memory allocated: 64GB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python3 -u test_zero_shot_indiana.py 2>&1 | tee logs/zero_shot_indiana_results_${SLURM_JOB_ID}.txt

echo ""
echo "Completed at: $(date)"
echo "Results: logs/zero_shot_indiana_results_${SLURM_JOB_ID}.txt"
