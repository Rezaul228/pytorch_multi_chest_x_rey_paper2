#!/bin/bash
#SBATCH --job-name=scale_flagoff_smoke
#SBATCH --output=logs/scale_flagoff_smoke_%j.out
#SBATCH --error=logs/scale_flagoff_smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --partition=volta,ampere
#SBATCH --qos=normal
# PART 2 verification + PART 4 smoke test for the MIMIC data-scale ablation.
#   A) flag-off equivalence: 2 epochs on the SAME 200 train / 100 val studies,
#      once WITHOUT --train_ids/--val_ids (original code path) and once WITH them
#      (new path). Same data, same order, same seed -> weights must be bit-identical
#      and the log format must match apart from the one ID FILTER line.
#   B) smoke: 3 epochs, 2,553-study subset, both arms, seed 17.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
mkdir -p logs
echo "SLURM_JOB_ID=$SLURM_JOB_ID NODE=$SLURM_NODELIST START=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SA=mimic/data/scale_ablation
COMMON="--dataset_mode mimic_shards_hybrid_full_ori --batch_size 128 --learning_rate 1e-4 --grad_clip 1.0 --device cuda --seed 17"

echo "########## A) FLAG-OFF EQUIVALENCE (2 epochs, 200 train / 100 val) ##########"
for ARM in v1 MGG2L; do
  [ "$ARM" = v1 ] && SCRIPT=train_retrieval_v2.py || SCRIPT=train_retrieval_v2_paper2.py
  for MODE in flagoff flagon; do
    EXP=equivcheck_${ARM}_${MODE}
    rm -rf "saved_models/$EXP"
    EXTRA=""
    [ "$MODE" = flagon ] && EXTRA="--train_ids $SA/equiv_train_ids_200.csv --val_ids $SA/equiv_val_ids_100.csv"
    echo "----- $ARM / $MODE -----"
    python3 -u $SCRIPT --experiment_name "$EXP" $COMMON --max_epochs 2 \
        --train_samples 200 --val_samples 100 $EXTRA > "logs/equiv_${ARM}_${MODE}.log" 2>&1
    echo "done: logs/equiv_${ARM}_${MODE}.log"
  done
done

echo "===== A1) bit-identical weights? ====="
python3 - <<'PY'
import torch
for arm in ("v1", "MGG2L"):
    a = torch.load(f"saved_models/equivcheck_{arm}_flagoff/export/model_weights.pth", map_location="cpu", weights_only=True)
    b = torch.load(f"saved_models/equivcheck_{arm}_flagon/export/model_weights.pth", map_location="cpu", weights_only=True)
    assert a.keys() == b.keys()
    same = all(torch.equal(a[k], b[k]) for k in a)
    mx = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
    print(f"{arm}: {len(a)} tensors  all torch.equal={same}  max|diff|={mx:.3e}")
PY

echo "===== A2) log-format diff (normalised: drop timings/paths/progress bars) ====="
for ARM in v1 MGG2L; do
  for MODE in flagoff flagon; do
    grep -vE "it/s|s/it|^ *$|Memory |elapsed|Start:|End:|saved to:|Checkpoint saved|[0-9]{2}:[0-9]{2}" "logs/equiv_${ARM}_${MODE}.log" \
      | sed -E 's/[0-9]+\.[0-9]+/N/g; s/[0-9]+/N/g' > "logs/equiv_${ARM}_${MODE}.norm"
  done
  echo "--- $ARM: lines present in flagon but not flagoff (expect only the ID FILTER line) ---"
  diff "logs/equiv_${ARM}_flagoff.norm" "logs/equiv_${ARM}_flagon.norm" || true
done

echo "########## B) SMOKE TEST (3 epochs, 2,553 studies, both arms, seed 17) ##########"
for ARM in v1 MGG2L; do
  [ "$ARM" = v1 ] && SCRIPT=train_retrieval_v2.py || SCRIPT=train_retrieval_v2_paper2.py
  EXP=smoke_scale2553_${ARM}
  rm -rf "saved_models/$EXP"
  echo "----- smoke $ARM -----"
  python3 -u $SCRIPT --experiment_name "$EXP" $COMMON --max_epochs 3 --save_best \
      --train_ids $SA/train_ids_2553.csv --val_ids $SA/val_ids_3000.csv \
      --train_samples 200000 --val_samples 200000 2>&1 | tee "logs/smoke_scale2553_${ARM}.log" | \
      grep -E "ID FILTER|Data loaded|section guard|Granularity Loss|Epoch [0-9]+/|recall@1:|Warning|fallback|Total Loss" || true
done

echo "===== B1) smoke checks ====="
grep -h "ID FILTER\|Data loaded" logs/smoke_scale2553_*.log
echo "-- MG-G2L granularity loss per epoch (must be NON-ZERO) --"
grep "Granularity Loss" logs/smoke_scale2553_MGG2L.log
echo "-- section guard lines --"
grep "section guard" logs/smoke_scale2553_MGG2L.log || echo "MISSING section guard"
echo "-- any warning / fallback --"
grep -i "warning\|fallback" logs/smoke_scale2553_*.log || echo "none"
echo "END=$(date)"
