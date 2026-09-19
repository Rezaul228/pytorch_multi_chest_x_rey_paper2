#!/bin/bash
#SBATCH --job-name=a2_10000_MGG2L_s123
#SBATCH --output=logs/a2_scale10000_MGG2L_seed123_%j.out
#SBATCH --error=logs/a2_scale10000_MGG2L_seed123_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
# MIMIC data-scale ablation ATTEMPT 2 -- MATCHED OPTIMIZATION BUDGET.
# Attempt 1 (jobs 115242-115253) is void: fixed epoch counts gave only 8,000 /
# 11,850 gradient steps against the full-MIMIC reference of 30,450, and neither
# arm converged. See mimic/results/scale_ablation/ATTEMPT1_VOID.md.
#
# Identical to attempt 1 in every respect EXCEPT the epoch count: same id files,
# same seeds, same lr 1e-4 / batch 128 / wd 1e-4 (config) / clip 1.0 / loss
# weights 0.65/0.20/0.15 (+0.10 granularity for MG-G2L), --save_best, no early
# stopping, checkpoint_best.pth + a single overwritten checkpoint_resume.pth only.
#
#   size 10000: 79 steps/epoch x 385 epochs = 30415 gradient steps
#   full-MIMIC reference budget                     = 30,450 gradient steps
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "=============================================================="
echo "BUDGET: size=10000  steps_per_epoch=79  epochs=385  total_planned_steps=30415"
echo "        full-MIMIC reference = 30450 steps  (deviation $(python3 -c "print(f'{100*(30415-30450)/30450:+.2f}%')"))"
echo "SLURM_JOB_ID = $SLURM_JOB_ID   restart count = ${SLURM_RESTART_COUNT:-0}"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "GIT_HEAD = $(git rev-parse --short HEAD)"
echo "Start: $(date)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u train_retrieval_v2_paper2.py \
    --experiment_name mimic_shards_hybrid_full_ori_vo10805_to128_lr1e-4_b128_ep385_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_scale10000_seed_123 \
    --dataset_mode mimic_shards_hybrid_full_ori \
    --train_ids mimic/data/scale_ablation/train_ids_10000.csv \
    --val_ids mimic/data/scale_ablation/val_ids_3000.csv \
    --train_samples 200000 --val_samples 200000 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --grad_clip 1.0 \
    --max_epochs 385 \
    --save_best \
    --resume \
    --device cuda \
    --seed 123

echo "End: $(date)"
