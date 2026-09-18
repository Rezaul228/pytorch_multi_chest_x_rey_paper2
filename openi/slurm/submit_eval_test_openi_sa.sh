#!/bin/bash
#SBATCH --job-name=openi_test_eval
#SBATCH --output=logs/openi_test_eval_%j.out
#SBATCH --error=logs/openi_test_eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
# Open-I (openi_sa) test evaluation: pre-flight gate -> GPU embedding pass ->
# R@K (both hit rules) -> CheXbert graded relevance -> per-query caches, then
# CPU statistics (seed-level primary, Wilcoxon secondary + MDE, missing-section).
# Inference only. Writes NEW files under openi/results/. MIMIC is never touched.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===== GPU: pre-flight + evaluation ====="
python3 -u openi/scripts/evaluate_test_openi_sa.py --seeds 17 42 123 1337 2021 3407
echo "===== CPU: statistics ====="
python3 -u openi/scripts/statistical_tests_openi_sa.py
echo "END=$(date)"
