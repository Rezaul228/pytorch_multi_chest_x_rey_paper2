#!/bin/bash
#SBATCH --job-name=smoke_MGG2L_openi_sa
#SBATCH --output=logs/smoke_MGG2L_openi_sa_%j.out
#SBATCH --error=logs/smoke_MGG2L_openi_sa_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=volta
#SBATCH --qos=normal

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs

echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"

python3 train_retrieval_v2_paper2.py \
    --experiment_name openi_sa_vo10805_to128_lr1e-4_b128_ep100_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_17_SMOKETEST \
    --dataset_mode openi_sa \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --grad_clip 1.0 \
    --max_epochs 2 \
    --save_best \
    --resume \
    --device cuda \
    --seed 17

echo "Smoke test done: $(date)"
