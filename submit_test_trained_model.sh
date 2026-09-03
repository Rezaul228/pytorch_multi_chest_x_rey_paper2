#!/bin/bash
#SBATCH --job-name=test_trained_model
#SBATCH --output=logs/test_trained_model_%j.out
#SBATCH --error=logs/test_trained_model_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_ray1
mkdir -p logs

echo "Test trained model (cross-modal retrieval)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Memory allocated: 64GB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python3 -u test_trained_model.py 2>&1 | tee logs/test_trained_model_results_${SLURM_JOB_ID}.txt

echo ""
echo "Completed at: $(date)"
echo "Results: logs/test_trained_model_results_${SLURM_JOB_ID}.txt"
