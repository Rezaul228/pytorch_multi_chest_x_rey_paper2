# MG-G2L (Paper 2) — Progress Handover

## Goal
Extend Paper 1's dual-branch (Synergy/Difference) MIMIC-CXR retrieval model with a
section-aware "MG-G2L" mechanism (Findings vs. Impression global feedback +
granularity loss), then rigorously compare MG-G2L vs. the original Paper 1
architecture across multiple seeds on: retrieval metrics (R@K/MRR), CheXpert-based
graded relevance (nDCG/mAP/Precision), and sub-group analyses (laterality, section
availability, Findings/Impression agreement).

## What's completed

**Architecture (`base_models_refactored_v1_paper2.py`, copy of the original — original untouched):**
- `HierarchicalCoAttention.compute_section_pools` / `compute_section_global_attention`
  / `section_aware_feedback`: masked mean-pool + attention + gated feedback for
  Findings/Impression sections separately, using new `section_find_gate_weights` /
  `section_imp_gate_weights` params.
- `forward()` on `HierarchicalCoAttention`, `BranchEncoder`, `MultimodalFusion` all
  gained optional args (`findings_token_count`, `has_find`, `has_imp`, `token_ids`),
  default `None` → **byte-for-byte identical** to original when omitted (verified via
  `torch.equal` at every level, including a full 50-epoch training run).
- `compute_granularity_loss`: InfoNCE loss aligning image embeddings to
  Findings/Impression-pooled text, added as `+0.10 * L_gran` on top of the original
  `0.65*synergy + 0.20*main + 0.15*ortho` (unchanged weights).

**Data pipeline (`data_loader_v1_paper2.py`, copy of original):**
- `IndianaDataset` now loads `section_boundaries_{split}_paper2.csv` once and attaches
  `findings_token_count`/`has_find`/`has_imp` per sample; unknown study_id → safe
  fallback (`0, False, False`), never crashes. Verified 0 fallback across all
  197,355 train/val/test samples.

**Training (`train_retrieval_v2_paper2.py`, copy of original):**
- Checkpoint save every 3rd epoch + final epoch, single fixed file
  (`checkpoint_resume.pth`, overwritten in place), auto-resume from saved epoch.
  Verified via a real save→resume round-trip (289/289 params `torch.equal`).
- `evaluate_recall_k_batched()` and `evaluate_cross_modal_retrieval_streaming()`
  (paper2 copies) fixed to pass section args through when present, else fall back to
  the original call — this was a real bug initially (validation during seed_42's
  training silently used the old, non-section-aware path).

**Tokenization / section-boundary tooling:**
- Reverse-engineered the exact original tokenizer (regex clean + stopword removal via
  locally-cached nltk + plain `.split()`, NOT `nltk.word_tokenize`) — verified 300/300
  exact match against real shard captions.
- `section_boundaries_{train,val,test}_paper2.csv`, `hard_negative_tags_test_paper2.csv`,
  `divergence_scores_test_paper2.csv` built and cached. **Per-patient files
  (CheXpert/NegBio labels, section boundaries, hard-negative tags) are gitignored**
  — PhysioNet DUA, credentialed-access data, kept local-only.

**Training runs completed (MG-G2L, all 50 epochs, all clean):**
- seed_42 (job 113710), seed_17 (113722), seed_123 (113724/rerun of 113723 due to a
  memory-collision on shared node), seed_3407 (113725, first-ever run on
  ampere/A6000 — verified healthy). Plus a seed_42 **reproducibility rerun**
  (113735): bit-identical final weights to the original (289/289 `torch.equal`, max
  diff 0.0) — confirms the pipeline is fully deterministic.
- **In progress**: seed_1337 (job 113798) and seed_2021 (job 113799), both on
  `pascal-node10`, started ~2026-09-14. ETA ~10-12h from submission.

**Evaluation results (4 matched seeds: 17, 42, 123, 3407):**
- R@K/MRR mean±std: MG-G2L beats Paper 1 on every metric. Biggest gains: I2T R@1
  (0.9874±0.0104 → 0.9995±0.0005), I2T MRR (0.9923±0.0064 → 0.9996±0.0003). T2I
  already near-ceiling for both models (little room to move).
- Graded relevance (CheXpert nDCG@5/10, mAP@10, Precision@5; restricted n=8,968):
  MG-G2L improves I2T consistently; **T2I is mixed** — regresses for 3 of 4 seeds
  (17, 123, 3407), only seed_42 improves T2I. Net T2I mean delta ≈ 0, and MG-G2L's
  T2I variance is *larger* than Paper 1's (opposite of the I2T variance-reduction
  pattern).
- Statistical tests (Wilcoxon/Mann-Whitney on cached per-query values,
  `statistical_tests_i2t_paper2.py`): overall MG-G2L>Paper1 improvement is
  significant (p<0.05) in **all 4 seeds × 2 metrics**. The narrower "single-section /
  Findings-Impression-divergence helps more" hypothesis is **seed-dependent, not
  universal**: holds cleanly for seeds 17 & 123, mixed for 42, **contradicted by
  3407** (its `both_sections`/`agreement` group shows the *larger* gain, opposite of
  the hypothesis).
- Data-quality catch: the MIMIC Paper-1-baseline `seed_2021` checkpoint on disk is a
  **byte-duplicate of `seed_123`** (confirmed via SHA-256) — not a real 4th seed.
  Real seed_1337/seed_2021 Paper-1-baseline retrains are running in a **different,
  unrelated project** (`mimic_ori_full_all_data`, jobs 113796/113797) — not managed
  from here.

## Key implementation decisions
- Every new arg defaults to `None`/unused → old call sites need zero changes;
  backward compatibility verified with `torch.equal`, not just "looks right."
- `global_text_report` (the "whole report" fallback) is kept **exactly** equal to the
  old unmasked `torch.mean(text_tokens, dim=1)` — deliberately not masked, per
  explicit correction mid-project.
- To force full-dataset training/eval without touching shared `config.py`:
  pass `--train_samples 200000 --val_samples 40000` (sentinels exceeding the real
  counts) rather than `None`, which would fall back to `config.py`'s (stale,
  quick-test) defaults.
- Graded relevance uses the **rigorous** convention: nDCG's IDCG and mAP's
  relevant-count normalizer both use the true best-case ranking across the *whole*
  corpus, not just the retrieved top-10.
- **This SLURM cluster does not enforce `--mem` as a real scheduling constraint** —
  always check `scontrol show node` (RealMemory/FreeMem/AllocTRES) before submitting;
  learned this after two jobs got stacked memory-unsafely on one node and had to be
  cancelled/resubmitted.

## Current problems / unfinished work
1. **Open-I baseline investigation — interrupted mid-task.** User asked to checksum-verify
   5 new Open-I (`aug_indiana_extended`) Paper-1-baseline checkpoints (seeds
   17/42/123/2021/3407) and evaluate them on the Open-I test set (5,313 samples).
   Found so far: these checkpoints are **not in this project** — they live in sibling
   repo `/home/abedin/Developments/pytorch_multi_chest_x_ray/saved_models/`. This
   project's `config.py` has **no `aug_indiana_extended` entry** in
   `DATASET_CONFIGS` (returned `None`). Need to either add that config here
   (vocab_size=10870, embed_dim=256, num_heads=8, num_layers=2 — per user's task
   description) or run the check from the sibling repo. **Checksums not yet computed.**
2. seed_1337/seed_2021 MG-G2L jobs (113798/113799) still training, not yet evaluated.
3. Real (non-duplicate) seed_2021 Paper-1-baseline MIMIC checkpoint pending from the
   other project — needed to properly extend the 4-seed MIMIC tables to 5-6 seeds.
4. Root cause of seed_3407's (and partially seed_42's) reversal of the
   "missing-section helps more" pattern not investigated — would need
   branch-specialization inspection or more seeds for statistical power.

## Exact next steps
1. Resume Open-I check: get/define the `aug_indiana_extended` config, compute
   SHA-256 for the 5 checkpoints in `pytorch_multi_chest_x_ray/saved_models/`, flag
   duplicates, strict-load distinct ones into the **original** `MultimodalFusion`,
   run `evaluate_cross_modal_retrieval_streaming()` on the Open-I test set (5,313
   samples, no section args), sanity-check seed_42 against job 10227's validation
   reference (~0.983 recall@1 — note: validation not test, so only a loose sanity
   check expected).
2. Once jobs 113798/113799 finish: evaluate via the established streaming +
   graded-relevance pipelines, add seed_1337/seed_2021 to the MG-G2L multi-seed
   tables.
3. Once the other project's Paper-1-baseline seed_1337/seed_2021 MIMIC retrains
   finish: pull those checkpoints in and extend the matched multi-seed MIMIC
   comparison tables accordingly.
4. Consider deeper investigation into why seed_3407 breaks the divergence-group
   hypothesis before stating it as a paper claim.
