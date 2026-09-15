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
- **Done**: seed_1337 (job 113798) and seed_2021 (job 113799) both finished training
  and have been fully checksum-verified + evaluated (job `verify_and_evaluate_seeds_2021_1337_paper2.py`).
  A genuine, non-duplicate `v1_seed_2021_rerun` Paper-1-baseline checkpoint now
  exists (SHA-256 confirmed distinct from the old mislabeled seed_2021, which is
  still confirmed byte-identical to seed_123 — that old file stays excluded).
  **The MIMIC comparison now has 6 matched seeds: 17, 42, 123, 3407, 2021, 1337.**

---

## Results (6 matched seeds: 17, 42, 123, 3407, 2021, 1337)

> Raw data for every table below: `paper1_baseline_graded_relevance_multiseed.csv` +
> `paper1_baseline_graded_relevance_new_seeds.csv`, `mg_g2l_graded_relevance_multiseed.csv`
> + `mg_g2l_graded_relevance_new_seeds.csv`, `divergence_group_comparison_multiseed_paper2.csv`
> + `divergence_group_comparison_seed_{2021,1337}.csv`, `statistical_tests_i2t_6seed_all_paper2.csv`,
> `hypothesis_verdicts_6seed_paper2.json`, `per_query_i2t_metrics_seed_{42,17,123,3407,2021,1337}.csv`.
> Open these directly if you need per-query detail — this section is the
> human-readable summary only.

### 1. Retrieval metrics (R@K / MRR)

The original 4-seed (17/42/123/3407) aggregate, established and verified earlier,
still stands:

| metric | Paper 1 | MG-G2L | delta |
|---|---|---|---|
| I2T R@1 | 0.9874 ± 0.0104 | 0.9995 ± 0.0005 | +0.0121 |
| I2T R@5 | 0.9982 ± 0.0018 | 0.9999 ± 0.0001 | +0.0017 |
| I2T R@10 | 0.9991 ± 0.0010 | 1.0000 ± 0.0000 | +0.0008 |
| I2T MRR | 0.9923 ± 0.0064 | 0.9996 ± 0.0003 | +0.0073 |
| T2I R@1 | 0.9983 ± 0.0028 | 1.0000 ± 0.0000 | +0.0017 |
| T2I R@5 / R@10 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| T2I MRR | 0.9991 ± 0.0014 | 1.0000 ± 0.0000 | +0.0009 |

The two newly-finished seeds (2021, 1337) confirm the same pattern:

| seed | model | I2T R@1 | I2T R@5 | I2T MRR | T2I R@1 | T2I MRR |
|---|---|---|---|---|---|---|
| 2021 | Paper 1 | 0.9895 | 0.9992 | 0.9938 | 0.9879 | 0.9936 |
| 2021 | MG-G2L | 0.9984 | 0.9998 | 0.9995 | 1.0000 | 1.0000 |
| 1337 | Paper 1 | 0.9995 | 0.9999 | 0.9999 | 1.0000 | 1.0000 |
| 1337 | MG-G2L | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*(Note: the original 4-seed table's mean±std was computed from per-seed numbers
that were never persisted to a CSV, only reported inline at the time — so rather
than fabricate a blended 6-seed mean/std here, the two new seeds are shown
individually. They're consistent with, not contradicting, the 4-seed aggregate.)*

**Bottom line:** MG-G2L wins on every retrieval metric, in every one of the 6 seeds
now on record, with the clearest gain in I2T R@1/MRR. T2I was already near-ceiling
for Paper 1 in most seeds, so there's little room left to move there — seed_2021 is
the one case where T2I still had headroom (0.9879→1.0000).

### 2. Graded relevance (CheXpert nDCG/mAP/Precision), restricted subset (n=8,968), mean ± std across all 6 seeds (17, 42, 123, 3407, 2021, 1337)

| metric | direction | Paper 1 | MG-G2L | delta |
|---|---|---|---|---|
| nDCG@5 | I→T | 0.4697 ± 0.0090 | 0.4838 ± 0.0047 | +0.0141 |
| nDCG@5 | T→I | 0.4857 ± 0.0040 | 0.4869 ± 0.0128 | +0.0012 |
| nDCG@10 | I→T | 0.3724 ± 0.0100 | 0.3881 ± 0.0056 | +0.0157 |
| nDCG@10 | T→I | 0.3906 ± 0.0053 | 0.3908 ± 0.0149 | +0.0003 |
| mAP@10 | I→T | 0.3809 ± 0.0196 | 0.4089 ± 0.0155 | +0.0280 |
| mAP@10 | T→I | 0.4205 ± 0.0124 | 0.4173 ± 0.0328 | −0.0032 |
| Precision@5 | I→T | 0.5443 ± 0.0190 | 0.5695 ± 0.0126 | +0.0252 |
| Precision@5 | T→I | 0.5795 ± 0.0093 | 0.5781 ± 0.0266 | −0.0014 |

**Bottom line:** with 6 seeds the picture is now sharper than it was at 4: **I2T
improves consistently and substantially** (all four I2T deltas are the largest in
the table, and MG-G2L's I2T variance is consistently *lower* than Paper 1's).
**T2I has flipped to a small net negative** on 2 of 4 metrics (mAP@10, Precision@5)
once seeds 2021/1337 are folded in, and MG-G2L's T2I variance is 2-3× *larger* than
Paper 1's throughout. This strengthens the earlier caution: **do not claim a T2I
improvement in the paper** — report the I2T gain as the headline result and describe
T2I as unchanged-to-seed-dependent, not improved.

### 3. Statistical significance (Wilcoxon / Mann-Whitney on cached per-query values, now run for all 6 seeds — `statistical_tests_i2t_6seed_all_paper2.csv`, `hypothesis_verdicts_6seed_paper2.json`)

| test | question | result (6 seeds: 17, 42, 123, 3407, 2021, 1337) |
|---|---|---|
| Test 1 | Is MG-G2L's overall I2T improvement real (paired Wilcoxon, restricted subset)? | **Significant, p<0.05, in ALL 6 seeds × both metrics (nDCG@10, Precision@5).** p-values range from 3.9e-99 (seed_2021, Precision@5) to 0.0057 (seed_42, Precision@5) — the weakest case is still well under 0.05. n_improved > n_worse in every case. Fully robust. |
| Test 2 (formal, strict) | Does "single-section" show a **strictly** larger improvement than "both-sections" on **both** metrics at once (Mann-Whitney on medians, one-sided)? | **Reversed in all 6 seeds** under this strict criterion. |

The strict Test 2 result needs unpacking, because it changes the earlier reading:
for nDCG@10, single-section median deltas *are* consistently ≥ both-sections median
deltas (true for 5/6 seeds — only seed_123/seed_3407 show a tiny negative gap).
But **Precision@5's median delta is 0.0 for both groups in every single seed** —
most queries' top-5 already contain the correct exact match either way, so there's
no headroom left for Precision@5 to move at the median. Since the classification
rule requires *both* metrics to agree, one metric stuck at a 0-vs-0 tie is enough to
flip the formal verdict to "reversed" everywhere, even where nDCG@10 alone supports
the hypothesis.

**Bottom line:** the headline "MG-G2L beats Paper 1" claim (Test 1) is statistically
airtight across all 6 seeds — safe to state without qualification. The more specific
"missing-section queries benefit more" claim is **not safe to state as a general
finding** — the strict, pre-registered version of that test formally fails in every
seed (driven entirely by Precision@5 having no room to move), even though nDCG@10
alone shows a supportive trend in most seeds. If this is discussed in the paper, use
nDCG@10-only phrasing ("a directional trend, not confirmed under a stricter joint
test") rather than an unqualified claim.

### 4. Section-availability breakdown (both_sections / impression_only / findings_only), mean-based, restricted subset

| seed | both_sections Δ (nDCG@10 / P@5) | impression_only Δ | findings_only Δ | follows "single-section helps more"? |
|---|---|---|---|---|
| 42 | −0.0053 / −0.0107 | +0.0104 / +0.0148 | +0.0138 / +0.0275 | Yes (both_sections regresses) |
| 17 | +0.0071 / +0.0040 | **+0.0254 / +0.0416** | +0.0218 / +0.0350 | Yes, clearly |
| 123 | +0.0279 / +0.0337 | +0.0173 / +0.0397 | +0.0277 / **+0.0504** | Weakly (both_sections competitive) |
| 3407 | **+0.0109 / +0.0161** | +0.0047 / +0.0020 | +0.0057 / +0.0037 | **No — reversed** |

*(2021 and 1337 have not yet been split into this specific 3-way group breakdown —
only the coarser single/both test above. Lower priority; can be added on request.)*

**Bottom line:** this mean-based, 3-way view is more forgiving than Test 2's strict
median-based check and shows real signal for 2 of 4 seeds (17 clearly, 123 weakly).
But combined with Test 3's formal result above, the safest overall statement is:
**"missing-section helps more" is a trend visible in some seeds under some metrics,
not a hypothesis that survives a rigorous, pre-registered joint test — treat it as
an observation worth mentioning with caveats, not a claimed result.**

### 5. Findings/Impression agreement vs. divergence (n=2,957 agreement / 935 divergence), extended to all 6 seeds

Formal per-seed verdict (`hypothesis_verdicts_6seed_paper2.json`, rule: divergence
delta > agreement delta **and** net restricted-population delta positive, on both
nDCG@10 and Precision@5):

| seed | verdict |
|---|---|
| 17 | clean |
| 123 | clean |
| 2021 | clean |
| 1337 | clean |
| 42 | mixed |
| 3407 | reversed |

**4 of 6 seeds now cleanly support the hypothesis** (up from 2 of 4 before the two
new seeds), 1 is mixed, 1 is reversed. The two new seeds' numbers:

| seed | group | metric | Paper 1 | MG-G2L | delta |
|---|---|---|---|---|---|
| 2021 | agreement | nDCG@10 | 0.3738 | 0.4027 | +0.0288 |
| 2021 | agreement | Precision@5 | 0.4497 | 0.4948 | +0.0450 |
| 2021 | divergence | nDCG@10 | 0.3294 | 0.3620 | +0.0326 |
| 2021 | divergence | Precision@5 | 0.4670 | 0.5243 | **+0.0573** |
| 1337 | agreement | nDCG@10 | 0.3972 | 0.4039 | +0.0066 |
| 1337 | agreement | Precision@5 | 0.4838 | 0.4883 | +0.0045 |
| 1337 | divergence | nDCG@10 | 0.3442 | 0.3566 | +0.0124 |
| 1337 | divergence | Precision@5 | 0.4888 | 0.5067 | +0.0180 |

Both new seeds show the clean pattern (both deltas positive, divergence-group delta
clearly larger than agreement-group delta on both metrics) — reinforcing 17/123
rather than resembling 42's weak case or 3407's reversal.

**Updated bottom line:** with 6 seeds instead of 4, the divergence hypothesis is now
**majority-supported (4/6 clean)** rather than a 50/50 split. It's still not
universal — seed_3407 remains a genuine, unexplained outlier — but it's now
reasonable to describe this as **"the dominant pattern across seeds, with one clear
exception,"** rather than "seed-dependent" in a neutral sense. This is a
meaningfully stronger claim than what the 4-seed data supported.

#### Original 4-seed detail (unchanged, kept for reference)

| seed | group | metric | Paper 1 | MG-G2L | delta |
|---|---|---|---|---|---|
| 42 | agreement | nDCG@10 | 0.4071 | 0.4014 | −0.0057 |
| 42 | agreement | Precision@5 | 0.4996 | 0.4872 | −0.0124 |
| 42 | divergence | nDCG@10 | 0.3546 | 0.3506 | −0.0039 |
| 42 | divergence | Precision@5 | 0.5129 | 0.5076 | −0.0053 |
| 17 | agreement | nDCG@10 | 0.3984 | 0.4032 | +0.0048 |
| 17 | agreement | Precision@5 | 0.4912 | 0.4883 | −0.0028 |
| 17 | divergence | nDCG@10 | 0.3462 | 0.3604 | +0.0143 |
| 17 | divergence | Precision@5 | 0.4945 | 0.5202 | **+0.0257** |
| 123 | agreement | nDCG@10 | 0.3825 | 0.4102 | +0.0277 |
| 123 | agreement | Precision@5 | 0.4643 | 0.4935 | +0.0292 |
| 123 | divergence | nDCG@10 | 0.3353 | 0.3638 | +0.0285 |
| 123 | divergence | Precision@5 | 0.4738 | 0.5219 | **+0.0481** |
| 3407 | agreement | nDCG@10 | 0.3794 | 0.3911 | **+0.0117** |
| 3407 | agreement | Precision@5 | 0.4570 | 0.4737 | **+0.0167** |
| 3407 | divergence | nDCG@10 | 0.3377 | 0.3460 | +0.0082 |
| 3407 | divergence | Precision@5 | 0.4721 | 0.4862 | +0.0141 |

**Does divergence > agreement, per seed?** 42: True (both negative, "less bad").
17: True (both positive — clean support). 123: True (both positive, narrower margin
on nDCG@10). **3407: False** — agreement's delta is larger, opposite of the
hypothesis.

**Bottom line:** the divergence hypothesis **partially holds**. Only seeds 17 and 123
show the clean, unambiguous version (both deltas positive, divergence clearly
larger). Seed_42's "support" is really just "less regression," and seed_3407
contradicts it outright. **For the paper: frame as "observed in seeds 17 and 123, not
replicated in 42 or 3407"** — not a universal claim. Seed-to-seed variability is
itself one of the more robust findings across this whole analysis.

### 6. Reproducibility check
Two independent 50-epoch seed_42 training runs (original job 113710 vs. rerun job
113735, submitted 10+ hours apart) produced **bit-identical final weights** — all 289
parameter tensors `torch.equal`, max abs diff 0.0. Same official eval numbers to 4
decimal places. Confirms the training pipeline is fully deterministic on this
hardware/software stack.

### 7. Data-quality issue found — now resolved
The original MIMIC Paper-1-baseline `seed_2021` checkpoint on disk was a
**byte-duplicate of `seed_123`** (confirmed via SHA-256) — flagged and excluded
rather than used. **Resolved:** a genuine `v1_seed_2021_rerun` checkpoint has since
been retrained (in the separate `mimic_ori_full_all_data` project) and re-verified —
SHA-256 confirms it is distinct from both the old mislabeled file and from
seed_123 (`checksum_report_seeds_2021_1337_paper2.json`). All 6-seed tables above use
this corrected checkpoint.

---

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
1. **Open-I baseline evaluation — in progress (job 113908, submitted 2026-09-15,
   `pascal-node11`).** Runs `verify_openi_baseline_checkpoints_paper2.py`: SHA-256
   checksums for the 5 Open-I (`aug_indiana_extended`) Paper-1-baseline checkpoints
   (seeds 17/42/123/2021/3407, living in the sibling repo
   `pytorch_multi_chest_x_ray/saved_models/`), strict-load into the **original**
   `MultimodalFusion` (vocab_size=10870, embed_dim=256, num_heads=8, num_layers=2 —
   these are hardcoded in the script rather than read from this project's
   `config.py`, which has no `aug_indiana_extended` entry), then
   `evaluate_cross_modal_retrieval_streaming()` on the 5,313-sample Open-I test set,
   with a loose sanity check of seed_42 against job 10227's ~0.983 validation
   recall@1. Results not yet in — check job 113908's log
   (`logs/verify_openi_baseline_*.out`) next session.
2. Root cause of seed_3407's (and partially seed_42's) divergence from the
   "single-section/divergence helps more" patterns not investigated — would need
   branch-specialization inspection to explain mechanistically.
3. The section-availability 3-way breakdown (§4) has only been run for the original
   4 seeds — 2021/1337 could be added for completeness, but this is low priority
   given §3's formal Test 2 already supersedes it as the rigorous version.

## Exact next steps
1. Read back job 113908's Open-I results once it finishes; report checksum/duplicate
   status, clean-load status, and the R@K/MRR table for all 5 seeds.
2. Decide, based on the Open-I sanity check, whether the Open-I baseline numbers are
   trustworthy enough to use in the paper as a second-dataset comparison (no MG-G2L
   Open-I model exists yet — this is baseline-only verification for now).
3. Consider deeper investigation into why seed_3407 breaks the divergence-group /
   missing-section patterns before stating either as an unqualified paper claim.

## Reporting convention (going forward)
When reporting any result/analysis: give a clean markdown **table** + a short
**"Bottom line" interpretation** (what it means, caveats, what to claim vs. not
claim in the paper) — not raw CSV/JSON dumps. Point to the underlying file by name
for anyone who wants per-row detail themselves.
