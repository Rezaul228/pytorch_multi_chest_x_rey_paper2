#!/bin/bash
#SBATCH --job-name=shuffled_pairing_diag
#SBATCH --output=logs/shuffled_pairing_diag_%j.out
#SBATCH --error=logs/shuffled_pairing_diag_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_ray1
mkdir -p logs

echo "Shuffled-pairing diagnostic (baseline vs wrong co-attention pairing)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Memory allocated: 64GB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python3 -u diagnostic_shuffled_pairing.py 2>&1 | tee logs/shuffled_pairing_diag_results_${SLURM_JOB_ID}.txt

echo ""
echo "Completed at: $(date)"
echo "Results: logs/shuffled_pairing_diag_results_${SLURM_JOB_ID}.txt"
