# Agent Knowledge Transfer — `pytorch_multi_chest_x_rey_paper2`

**Purpose:** Hand off context from prior work so a new chat agent can continue safely without rediscovering the project.

**Last updated:** 2026-09-04  
**Owner:** Rezaul Abedin (`abedin` on pascal / GitHub `Rezaul228`)

---

## 1. What this project is

Multimodal **chest X-ray ↔ radiology report retrieval** (MIMIC-CXR / Indiana).

Core idea: a **dual-branch** model (Synergy + Difference) with co-attention, trained with contrastive + synergy + orthogonal losses.

This folder is a **fresh paper workspace**, not the long-running experiment dump.

| Path | Role |
|------|------|
| `/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2` | **This project** — clean paper codebase |
| `/home/abedin/Developments/mimic_ori_full_all_data` | **Proven source** — same code family; full logs, many checkpoints, analyses that already worked |
| `/home/abedin/Developments/pytorch_multi_chest_x_ray1` | Older sibling; efficiency table (Params/FLOPs/GPU mem/latency) was measured there |
| `/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/` | **Shared shard data** (do not duplicate into this repo) |

**GitHub:** https://github.com/Rezaul228/pytorch_multi_chest_x_rey_paper2 (public, branch `master`)

---

## 2. Critical agent rule (read this first)

> **In `mimic_ori_full_all_data`, the same scripts already worked.**  
> If something fails here, **do not rewrite architecture or invent new pipelines**.  
> Almost always the fix is local: wrong path, stale import after rename, missing checkpoint name, or config sample limits.

**Debug order:**

1. Compare failing file to the same logic in `mimic_ori_full_all_data` (or pre-rename name).
2. Check hardcoded `model_path` / `cd` / project root → must be `.../pytorch_multi_chest_x_rey_paper2`.
3. Check imports: `analysis_all_visualization_v1` (not `all_visualization_v1`).
4. Confirm checkpoint exists under `saved_models/.../export/model_weights.pth`.
5. Confirm `config.py` / `paths.py` dataset mode and shard paths.
6. Only then change logic — and keep changes minimal.

---

## 3. How the system works (pipeline)

```
Shard data (paths.py / config.py)
        ↓
enhanced_data_loader.py / data_loader_v1.py  → DataLoaders + tokenizer
        ↓
base_models_refactored_v1.py  → MultimodalFusion (dual branch)
        ↓
train_retrieval_v2.py  ← config.py hyperparams, analysis_all_visualization_v1.TrainingVisualizer
        ↓
saved_models/<experiment>/export/{model_weights.pth, model.pth}
        ↓
test_*.py / analysis_*.py  → eval, ablations, figures, noise tests
```

**Central knobs**

- `config.py` — `DATASET_MODE`, epochs, batch size, LR, train/val sample caps, loss weights.
- `paths.py` — `CENTRALIZED_DATA_BASE_PATH` and shard helpers (`get_train_shards_dir`, etc.).
- Submit: `submit_training_simple_v2.sh <experiment_name> [cuda]` → SLURM on `volta`, reads the rest from `config.py`.

**Default dataset mode (as of knowledge update):** `mimic_shards_hybrid_full_ori`  
Vocab 10805, seq 128, batch 256, LR 5e-5.  
**Caution:** config may still have **test caps** (`train_samples=1000`, `val_samples=200`, `epochs=5`). Full runs need those set to `None` / real epoch count.

**Env:** conda `multi_pytorch` on pascal (`source /opt/conda/etc/profile.d/conda.sh && conda activate multi_pytorch`).

---

## 4. File naming conventions (this repo)

After cleanup, scripts are grouped by prefix:

| Prefix | Meaning | Examples |
|--------|---------|----------|
| *(none / core)* | Train, config, data, model | `train_retrieval_v2.py`, `config.py`, `paths.py`, `base_models_refactored_v1.py`, loaders |
| `analysis_*` | Paper analysis / viz / ablation / noise / efficiency | `analysis_measure_efficiency.py`, `analysis_difference_branch.py` |
| `test_*` | Evaluation / checkpoint checks | `test_full_evaluation.py`, `test_evaluate_all_seeds.py`, `test_zero_shot_indiana.py` |
| `submit_*.sh` | SLURM wrappers | `submit_training_simple_v2.sh` |

**Important rename (imports):**

- `all_visualization_v1.py` → `analysis_all_visualization_v1.py`
- Training/eval import: `from analysis_all_visualization_v1 import TrainingVisualizer` / `visualize_retrieval_examples`

Old names like `evaluate_all_seeds.py` / `run_full_evaluation.py` were renamed to `test_evaluate_all_seeds.py` / `test_full_evaluation.py`.

---

## 5. What was done in the setup chats (session memory)

1. Confirmed efficiency metrics (Params/FLOPs/GPU/latency vs ResNet/ClinicalBERT) came from **`pytorch_multi_chest_x_ray1`**, not this folder.
2. Wiped `pytorch_multi_chest_x_rey_paper2` and copied a **clean** subset from `mimic_ori_full_all_data` (no logs, no bulk results, no old checkpoints).
3. Renamed analysis scripts to `analysis_*`, eval scripts to `test_*`; fixed imports and project paths.
4. Removed redundant scripts (`setup_new_laptop.sh`, duplicate multi-seed submit stub).
5. Re-init git history onto existing remote; pushed commit replacing old tree (`502924d` on top of initial `dd57e10`).
6. Submitted 5-epoch smoke train `paper2_5ep_seed42` (job **112920**) — **COMPLETED** OK but weak metrics (tiny data subset).
7. Copied a **known-good** full checkpoint into this project for testing (see §6).

---

## 6. Checkpoint for testing "do scripts still work?"

**Use this proven model** (copied from old project):

```text
/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/
  mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42/
    export/model_weights.pth   ← prefer this for eval
    export/model.pth
```

Also present:

- `saved_models/paper2_5ep_seed42/` — smoke run only; **not** for paper metrics.

**Many scripts still hardcode older experiment folder names** (e.g. `...branch_v2` or `...branch_v3`). When testing:

1. Point `model_path` / `MODEL_PATH` at the **seed_42** path above (or pass CLI args if the script supports them).
2. Do **not** assume every hardcoded path already matches the copied model.

Example weight path string:

```text
/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42/export/model_weights.pth
```

---

## 7. Typical commands

```bash
cd /home/abedin/Developments/pytorch_multi_chest_x_rey_paper2
source /opt/conda/etc/profile.d/conda.sh && conda activate multi_pytorch

# Train (params from config.py)
./submit_training_simple_v2.sh <experiment_name> cuda

# Zero-shot / eval examples (update MODEL_PATH inside script first if needed)
python test_zero_shot_indiana.py
python test_full_evaluation.py
python test_model_weights.py

# Analysis (same idea — fix hardcoded model_path if FileNotFound)
python analysis_measure_efficiency.py --ours-only
```

SLURM: partition `volta`, QoS `normal`. `--priority=TOP` is allowed; negative `--nice` is **not** for this account. Logs: `logs/training_<jobid>.out`.

---

## 8. Data paths (usually leave alone)

- Central data: `/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/`
- Modes include: `mimic_shards_hybrid_full_ori`, `indiana_shards`, `mimic_shards`, etc.
- Local `data/`, `logs/`, `outputs/`, `saved_models/` are gitignored (except we keep real checkpoints on disk for work).

---

## 9. Known pitfalls

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: all_visualization_v1` | Import `analysis_all_visualization_v1` |
| `FileNotFoundError` on model | Update hardcoded path to seed_42 export (or correct experiment) |
| Paths still say `mimic_ori_full_all_data` | Should already be paper2; grep and fix leftovers |
| Terrible metrics after short train | Config sample caps / few epochs — not a model bug |
| `gh` broken in shell | User has `alias gh='history\|grep'` — use `/home/abedin/.local/bin/gh` or `unalias gh` |
| Efficiency numbers missing here | Measured in `pytorch_multi_chest_x_ray1/logs/measure_efficiency_results_90535.txt` |

---

## 10. What "success" looks like when validating the port

For a script that worked in the old project:

1. It **loads** the seed_42 weights without error.
2. It **reads** shards via `paths`/`config` without path errors.
3. It **runs** to completion (eval metrics or a figure under `outputs/`).
4. If it fails, the traceback points to path/import/config — fix that, don't redesign the model.

Prefer **minimal diffs**. When unsure, open the twin file under `mimic_ori_full_all_data` and compare.

---

## 11. Related reference results (old project)

Noise / robustness result dumps live under `mimic_ori_full_all_data` (e.g. `enhanced_noise_sensitivity_results_*.txt`). They were **not** copied into paper2 (fresh start). Re-run analysis scripts here when needed.

Multi-seed / paper tables previously used several seed folders under the old `saved_models/`; only **seed_42** was copied into paper2 so far.

---

## 12. Quick map — core vs tooling

**Must-keep core:** `config.py`, `paths.py`, `base_models_refactored_v1.py`, `data_loader_v1.py`, `enhanced_data_loader.py`, `train_retrieval_v2.py`, `train_test_cross_modal_evaluation_v1.py`, `analysis_all_visualization_v1.py` (viz used by training).

**Eval:** `test_*.py`, `submit_zero_shot_indiana.sh`, `submit_multi_seed_evaluation_5000.sh`.

**Paper tooling:** other `analysis_*.py`.

---

*If this file conflicts with live code, trust the code + `mimic_ori_full_all_data` behavior, then update this document.*
