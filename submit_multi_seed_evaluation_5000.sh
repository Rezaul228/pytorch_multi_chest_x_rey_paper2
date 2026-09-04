#!/bin/bash
#SBATCH --job-name=multi_seed_eval_5k
#SBATCH --output=logs/multi_seed_evaluation_5000_%j.out
#SBATCH --error=logs/multi_seed_evaluation_5000_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=volta

# Load conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

# Set working directory
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2

# Create logs directory
mkdir -p logs

echo "Starting multi-seed model evaluation with 10000 test samples..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Memory allocated: 64GB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Conda environment: multi_pytorch"

# Run multi-seed evaluation with 5000 test samples
python3 test_evaluate_all_seeds.py \
    --model_paths \
        "/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_3407/export/model_weights.pth" \
        "/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_2021/export/model_weights.pth" \
        "/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_123/export/model_weights.pth" \
        "/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42/export/model_weights.pth" \
        "/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model_weights.pth" \
    --seeds "v2,2021,123,42,17" \
    --config_name "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1" \
    --output_base_dir "multi_seed_evaluation_10000_with_all_seeds_results" \
    --num_samples 10000

echo "Multi-seed evaluation with 10000 samples completed!"
echo "Time: $(date)"
echo "Results saved to: multi_seed_evaluation_10000_results/" 