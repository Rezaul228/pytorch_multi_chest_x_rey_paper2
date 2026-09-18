#!/bin/bash
#SBATCH --job-name=openisa_v1_s1337_rerun
#SBATCH --output=logs/openisa_v1_s1337_rerun_%j.out
#SBATCH --error=logs/openisa_v1_s1337_rerun_%j.err
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=pascal,volta,ampere
#SBATCH --qos=normal
#SBATCH --requeue
# openi_sa Paper 1 RERUN (2026-09-18): from scratch under the symmetric best-val
# persistence fix (commit 6460db5). Same recipe as the original run (b128, lr 1e-4,
# wd 1e-4 from config, clip 1.0, fixed 100 epochs, no early stopping, --save_best).
# 12 h limit so no resume occurs (original hit 6 h at epoch ~69 on a 1080 Ti).
# NEW experiment folder (suffix _rerun) -- the original checkpoints are untouched.
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

python3 -u train_retrieval_v2.py \
    --experiment_name openi_sa_vo10805_to128_lr1e-4_b128_ep100_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_1337_rerun \
    --dataset_mode openi_sa \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --grad_clip 1.0 \
    --max_epochs 100 \
    --save_best \
    --resume \
    --device cuda \
    --seed 1337

echo "End: $(date)"
