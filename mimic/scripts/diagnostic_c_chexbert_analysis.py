#!/usr/bin/env python3
"""
DIAGNOSTIC C (CPU-only statistics): MIMIC graded relevance under CheXbert labels.

  Part 2  labeler agreement, CheXpert-binary vs CheXbert-binary, per finding
          (agreement %, Cohen's kappa) + restricted-subset n under each.
  Part 3  6-seed mean +/- std graded relevance (restricted subset, both
          directions) under CheXbert labels, with the CheXpert delta alongside.
  Part 4  paired Wilcoxon (MG-G2L vs Paper 1, I->T nDCG@10 and P@5) per seed
          on the CheXbert per-query cache -- same call as
          extend_hypothesis_tests_6seed_paper2.run_test1_wilcoxon.

Writes aggregate-only files (no study_id) under mimic/results/*_chexbert_*.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
D = os.path.join(PROJECT_DIR, "mimic", "data")
R = os.path.join(PROJECT_DIR, "mimic", "results")

SEEDS = [17, 42, 123, 1337, 2021, 3407]
CHEXPERT_BIN = os.path.join(D, "test_labels_chexpert_binary.csv")
CHEXBERT_BIN = os.path.join(D, "test_labels_chexbert_binary.csv")


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def part2_agreement():
    cp = pd.read_csv(CHEXPERT_BIN, dtype={"study_id": str}).set_index("study_id")
    cb = pd.read_csv(CHEXBERT_BIN, dtype={"study_id": str}).set_index("study_id")
    assert list(cp.columns) == list(cb.columns)
    assert set(cp.index) == set(cb.index) and len(cp) == len(cb)
    cb = cb.loc[cp.index]
    findings = [c for c in cp.columns if c != "subject_id"]
    rows = []
    for f in findings:
        a, b = cp[f].values.astype(int), cb[f].values.astype(int)
        rows.append({"finding": f, "agreement_pct": 100 * float(np.mean(a == b)), "kappa": cohen_kappa(a, b),
                     "chexpert_pos": int(a.sum()), "chexbert_pos": int(b.sum())})
    pos_cols = [c for c in findings if c != "No Finding"]
    n_cp = int((cp[pos_cols] == 1).any(axis=1).sum())
    n_cb = int((cb[pos_cols] == 1).any(axis=1).sum())
    n_both = int(((cp[pos_cols] == 1).any(axis=1) & (cb[pos_cols] == 1).any(axis=1)).sum())
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, "labeler_agreement_chexpert_vs_chexbert.csv"), index=False)
    print("=== PART 2: LABELER AGREEMENT (MIMIC test) ===")
    print(f"rows={len(cp)}  restricted_n: CheXpert={n_cp}  CheXbert={n_cb}  in_both={n_both}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return set(cb.index[(cb[pos_cols] == 1).any(axis=1)])


def load_chexpert_all(kind):
    """kind in {'paper1_baseline', 'mg_g2l'} -- stitch the 3 existing CheXpert-label files into 6 seeds."""
    s42 = pd.read_csv(os.path.join(R, f"{kind}_graded_relevance.csv")); s42["seed"] = 42
    multi = pd.read_csv(os.path.join(R, f"{kind}_graded_relevance_multiseed.csv"))
    new = pd.read_csv(os.path.join(R, f"{kind}_graded_relevance_new_seeds.csv"))
    allv = pd.concat([s42, multi, new], ignore_index=True)
    assert sorted(allv["seed"].unique()) == sorted(SEEDS), sorted(allv["seed"].unique())
    return allv


def part3_graded():
    p1_cb = pd.read_csv(os.path.join(R, "paper1_baseline_graded_relevance_chexbert_6seed.csv"))
    mg_cb = pd.read_csv(os.path.join(R, "mg_g2l_graded_relevance_chexbert_6seed.csv"))
    p1_cp, mg_cp = load_chexpert_all("paper1_baseline"), load_chexpert_all("mg_g2l")
    rows = []
    for metric in ["nDCG@5", "nDCG@10", "mAP@10", "Precision@5"]:
        for direction in ["Image->Text", "Text->Image"]:
            def vals(df):
                v = df[(df.metric == metric) & (df.direction == direction)].sort_values("seed")
                assert len(v) == 6, (metric, direction, len(v))
                return v["restricted_subset_value"].values
            a, b = vals(p1_cb), vals(mg_cb)
            c, d = vals(p1_cp), vals(mg_cp)
            rows.append({"metric": metric, "direction": direction,
                         "paper1_chexbert_mean": a.mean(), "paper1_chexbert_std": a.std(ddof=1),
                         "mgg2l_chexbert_mean": b.mean(), "mgg2l_chexbert_std": b.std(ddof=1),
                         "delta_chexbert": b.mean() - a.mean(), "delta_chexpert": d.mean() - c.mean(),
                         "seeds_mg_gt_p1_chexbert": int(np.sum(b > a))})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, "graded_relevance_chexbert_vs_chexpert_6seed_summary.csv"), index=False)
    n_cb = int(p1_cb["restricted_subset_n"].iloc[0])
    print(f"\n=== PART 3: GRADED RELEVANCE, restricted subset (CheXbert n={n_cb}), 6-seed mean +/- std ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def part4_wilcoxon(restricted_ids):
    rows = []
    for seed in SEEDS:
        df = pd.read_csv(os.path.join(R, f"per_query_i2t_metrics_chexbert_seed_{seed}.csv"), dtype={"study_id": str})
        sub = df[df["study_id"].isin(restricted_ids)]
        assert len(sub) == len(restricted_ids), (seed, len(sub), len(restricted_ids))
        row = {"seed": seed, "n": len(sub)}
        for label, key in [("ndcg10", "ndcg10"), ("prec5", "prec5")]:
            p1, mg = sub[f"p1_{key}"].values, sub[f"mg_{key}"].values
            _, p = wilcoxon(mg, p1, zero_method="wilcox", alternative="two-sided")
            row[f"{label}_delta"] = float(np.mean(mg - p1))
            row[f"{label}_p"] = float(p)
            row[f"{label}_n_improved"] = int(np.sum(mg > p1))
            row[f"{label}_n_worse"] = int(np.sum(mg < p1))
        sig_pos = [row[f"{l}_delta"] > 0 and row[f"{l}_p"] < 0.05 for l in ("ndcg10", "prec5")]
        sig_neg = [row[f"{l}_delta"] < 0 and row[f"{l}_p"] < 0.05 for l in ("ndcg10", "prec5")]
        row["favours"] = "MG-G2L" if all(sig_pos) else "Paper 1" if all(sig_neg) else "mixed/ns"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R, "statistical_tests_i2t_chexbert_6seed.csv"), index=False)
    print("\n=== PART 4: WILCOXON (CheXbert labels, I->T, restricted subset) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    ids = part2_agreement()
    part3_graded()
    part4_wilcoxon(ids)
