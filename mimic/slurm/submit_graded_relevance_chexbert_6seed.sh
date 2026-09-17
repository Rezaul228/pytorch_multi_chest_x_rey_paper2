#!/bin/bash
#SBATCH --job-name=mimic_graded_chexbert
#SBATCH --output=logs/mimic_graded_chexbert_%j.out
#SBATCH --error=logs/mimic_graded_chexbert_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=volta
#SBATCH --qos=normal
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs

LABELS=mimic/data/test_labels_chexbert_binary.csv
SEEDS=17,42,123,1337,2021,3407
test -f "$LABELS" || { echo "missing $LABELS"; exit 1; }
for f in mimic/results/paper1_baseline_graded_relevance_chexbert_6seed.csv mimic/results/mg_g2l_graded_relevance_chexbert_6seed.csv; do
  test ! -f "$f" || { echo "refusing to overwrite $f"; exit 1; }
done

echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"

echo "===== PART A: aggregate graded relevance, both directions, 6 seeds, CheXbert labels ====="
python3 -u mimic/scripts/graded_relevance_multiseed_paper2.py \
    --binary-labels "$LABELS" \
    --seeds "$SEEDS" \
    --paper1-output mimic/results/paper1_baseline_graded_relevance_chexbert_6seed.csv \
    --mgg2l-output mimic/results/mg_g2l_graded_relevance_chexbert_6seed.csv

echo "===== PART B: per-query I->T cache (nDCG@10, P@5), 6 seeds, CheXbert labels ====="
python3 -u mimic/scripts/cache_and_group_i2t_metrics_paper2.py \
    --binary-labels "$LABELS" \
    --seeds "$SEEDS" \
    --output-template "per_query_i2t_metrics_chexbert_seed_{seed}.csv"

echo "END=$(date)"
