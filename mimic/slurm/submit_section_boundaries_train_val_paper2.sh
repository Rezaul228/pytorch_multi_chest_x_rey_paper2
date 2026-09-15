#!/bin/bash
#SBATCH --job-name=section_boundaries_train_val
#SBATCH --output=logs/section_boundaries_train_val_%j.out
#SBATCH --error=logs/section_boundaries_train_val_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=volta

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs

echo "Section-boundary computation: MIMIC-CXR train + val (text-only, CPU-only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Memory requested: 8GB, CPUs: 1 (single task, no multiprocessing)"
echo ""

echo "=== TRAIN split (155,800 study_ids) ==="
python3 -u mimic/scripts/build_section_boundaries_incremental_paper2.py \
    --split train \
    --output mimic/data/section_boundaries_train_paper2.csv

echo ""
echo "=== VAL split (31,900 study_ids) ==="
python3 -u mimic/scripts/build_section_boundaries_incremental_paper2.py \
    --split val \
    --output mimic/data/section_boundaries_val_paper2.csv

echo ""
echo "Completed at: $(date)"
