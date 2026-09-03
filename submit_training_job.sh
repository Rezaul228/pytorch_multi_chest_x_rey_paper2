#!/bin/bash
#SBATCH --job-name=chest_xray_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=volta

# Load conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the training script with only experiment_name and device
# All other parameters (batch_size, learning_rate, epochs, etc.) come from config.py
python train_retrieval_v1.py \
    --experiment_name train_mimic_128_1e_04_30_full_pytorch \
    --device cuda 