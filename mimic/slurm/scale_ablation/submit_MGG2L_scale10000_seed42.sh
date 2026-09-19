#!/bin/bash
#SBATCH --job-name=sc10000_MGG2L_s42
#SBATCH --output=logs/scale10000_MGG2L_seed42_%j.out
#SBATCH --error=logs/scale10000_MGG2L_seed42_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
# MIMIC data-scale ablation (pre-registered 2026-09-19): 10000 training studies,
# MGG2L arm, seed 42. Small-data recipe proven on Open-I: batch 128, lr 1e-4,
# wd 1e-4 (config), grad clip 1.0, loss weights 0.65/0.20/0.15 (+0.10 granularity
# for MG-G2L), best-val checkpoint on the fixed 3,000-study val subset, ties to the
# later epoch, NO early stopping. 150 epochs. The id lists are fixed and shared by
# both arms and all seeds, so seed variation is init + data order only.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID = $SLURM_JOB_ID   restart count = ${SLURM_RESTART_COUNT:-0}"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "GIT_HEAD = $(git rev-parse --short HEAD)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u train_retrieval_v2_paper2.py \
    --experiment_name mimic_shards_hybrid_full_ori_vo10805_to128_lr1e-4_b128_ep150_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_scale10000_seed_42 \
    --dataset_mode mimic_shards_hybrid_full_ori \
    --train_ids mimic/data/scale_ablation/train_ids_10000.csv \
    --val_ids mimic/data/scale_ablation/val_ids_3000.csv \
    --train_samples 200000 --val_samples 200000 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --grad_clip 1.0 \
    --max_epochs 150 \
    --save_best \
    --resume \
    --device cuda \
    --seed 42

echo "End: $(date)"
