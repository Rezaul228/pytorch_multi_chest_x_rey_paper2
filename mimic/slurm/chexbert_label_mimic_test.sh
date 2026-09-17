#!/bin/bash
#SBATCH --job-name=chexbert_mimic
#SBATCH --output=logs/chexbert_mimic_%j.out
#SBATCH --error=logs/chexbert_mimic_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=volta
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

REX=/home/abedin/Developments/chest_x_ray_data_processing/rexgradient
PROJ=/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
IN=$PROJ/mimic/data/chexbert_input_test.csv
OUT=$PROJ/mimic/data/chexbert_out
CKPT=$REX/CheXbert/chexbert.pth

cd "$PROJ"
mkdir -p logs

echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$SLURM_NODELIST  START=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ -f "$OUT/labeled_reports.csv" ]; then echo "refusing to overwrite $OUT/labeled_reports.csv"; exit 1; fi
mkdir -p "$OUT"
cd "$REX/CheXbert/src"
python3 -u label.py -d "$IN" -o "$OUT" -c "$CKPT"

echo "===== FINAL SUMMARY ====="
echo "input_rows=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$IN')))") output_rows=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$OUT/labeled_reports.csv')))")"
echo "output=$OUT/labeled_reports.csv"
echo "===== END SUMMARY ====="
echo "END=$(date)"
