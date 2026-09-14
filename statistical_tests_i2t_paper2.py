#!/usr/bin/env python3
"""
Pure statistics on the already-cached per_query_i2t_metrics_seed_{seed}.csv
files (seeds 42, 17, 123). No retraining, no recomputation of rankings --
only label lookups (test_labels_chexpert_binary.csv,
section_boundaries_test_paper2.csv, both already-existing static files) to
derive the restricted subset and section-availability groups.

TEST 1 (per seed): paired Wilcoxon signed-rank test, MG-G2L vs Paper 1,
restricted subset (n=8,968), for nDCG@10 and Precision@5 (I->T).

TEST 2 (per seed): per-query delta (MG-G2L - Paper 1), Mann-Whitney U test
comparing the delta distribution between "single_section"
(impression_only OR findings_only) and "both_sections" queries --
one-sided (alternative='greater'): are single_section deltas larger?
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 17, 123]

BINARY_LABELS_PATH = os.path.join(PROJECT_DIR, "test_labels_chexpert_binary.csv")
SECTION_BOUNDARIES_PATH = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")


def get_restricted_mask_and_groups():
    binary_df = pd.read_csv(BINARY_LABELS_PATH, dtype={"study_id": str}).set_index("study_id")
    finding_cols = [c for c in binary_df.columns if c not in ("subject_id",) and c != "No Finding"]
    assert len(finding_cols) == 13
    restricted_study_ids = set(binary_df.index[(binary_df[finding_cols] == 1).any(axis=1)])

    boundaries_df = pd.read_csv(SECTION_BOUNDARIES_PATH, dtype={"study_id": str}).set_index("study_id")

    def section_group(sid):
        hf = bool(boundaries_df.loc[sid, "has_findings"])
        hi = bool(boundaries_df.loc[sid, "has_impression"])
        if hf and hi:
            return "both_sections"
        elif hf or hi:
            return "single_section"
        else:
            return "neither"

    return restricted_study_ids, section_group


def main():
    restricted_study_ids, section_group = get_restricted_mask_and_groups()
    print(f"Restricted study_ids (>=1 confirmed positive finding): {len(restricted_study_ids)}")

    metrics = [("nDCG@10", "ndcg10"), ("Precision@5", "prec5")]

    test1_results = {}
    test2_results = {}

    for seed in SEEDS:
        csv_path = os.path.join(PROJECT_DIR, f"per_query_i2t_metrics_seed_{seed}.csv")
        df = pd.read_csv(csv_path, dtype={"study_id": str})
        df = df[df["study_id"].isin(restricted_study_ids)].copy()
        assert len(df) == len(restricted_study_ids), f"seed {seed}: expected {len(restricted_study_ids)} rows, got {len(df)}"

        df["section_group"] = df["study_id"].apply(section_group)
        assert (df["section_group"] != "neither").all(), f"seed {seed}: unexpected 'neither' rows in restricted subset"

        print(f"\n{'='*90}\nSEED {seed}  (n={len(df)})\n{'='*90}")

        test1_results[seed] = {}
        test2_results[seed] = {}

        for metric_label, metric_key in metrics:
            p1_col = f"p1_{metric_key}"
            mg_col = f"mg_{metric_key}"

            p1_vals = df[p1_col].values
            mg_vals = df[mg_col].values
            diffs = mg_vals - p1_vals

            n_improved = int(np.sum(diffs > 0))
            n_worse = int(np.sum(diffs < 0))
            n_tied = int(np.sum(diffs == 0))

            # Wilcoxon requires at least one non-zero difference; zero_method='wilcox' drops ties by default.
            if n_improved + n_worse == 0:
                stat, pval = float("nan"), float("nan")
            else:
                stat, pval = wilcoxon(mg_vals, p1_vals, zero_method="wilcox", alternative="two-sided")

            test1_results[seed][metric_label] = {
                "stat": stat, "pval": pval,
                "n_improved": n_improved, "n_worse": n_worse, "n_tied": n_tied,
            }

            print(f"\n--- TEST 1: {metric_label} (Wilcoxon signed-rank, MG-G2L vs Paper 1) ---")
            print(f"  statistic={stat:.2f}  p-value={pval:.3e}")
            print(f"  n_improved={n_improved}  n_worse={n_worse}  n_tied={n_tied}")
            print(f"  significant at p<0.05: {pval < 0.05}")

            # TEST 2: delta distribution, single_section vs both_sections
            df["_delta"] = diffs
            single_deltas = df.loc[df["section_group"] == "single_section", "_delta"].values
            both_deltas = df.loc[df["section_group"] == "both_sections", "_delta"].values

            u_stat, u_pval = mannwhitneyu(single_deltas, both_deltas, alternative="greater")
            median_single = float(np.median(single_deltas))
            median_both = float(np.median(both_deltas))

            test2_results[seed][metric_label] = {
                "u_stat": u_stat, "u_pval": u_pval,
                "n_single": len(single_deltas), "n_both": len(both_deltas),
                "median_single": median_single, "median_both": median_both,
            }

            print(f"\n--- TEST 2: {metric_label} (Mann-Whitney U, single_section delta > both_sections delta) ---")
            print(f"  n_single_section={len(single_deltas)}  n_both_sections={len(both_deltas)}")
            print(f"  median delta (single_section)={median_single:+.4f}  median delta (both_sections)={median_both:+.4f}")
            print(f"  U statistic={u_stat:.1f}  p-value={u_pval:.3e}")
            print(f"  significant at p<0.05 (one-sided, single > both): {u_pval < 0.05}")
            print(f"  direction matches hypothesis (single median > both median): {median_single > median_both}")

    # ---------------- Summary ----------------
    print("\n" + "=" * 100)
    print("SUMMARY: TEST 1 (overall improvement significance)")
    print("=" * 100)
    header = f"{'seed':<6} | {'metric':<12} | {'stat':<10} | {'p-value':<12} | {'sig p<0.05':<11} | {'n_improved':<11} | {'n_worse':<8} | {'n_tied':<7}"
    print(header)
    print("-" * len(header))
    for seed in SEEDS:
        for metric_label, _ in metrics:
            r = test1_results[seed][metric_label]
            print(f"{seed:<6} | {metric_label:<12} | {r['stat']:<10.2f} | {r['pval']:<12.3e} | {str(r['pval']<0.05):<11} | "
                  f"{r['n_improved']:<11} | {r['n_worse']:<8} | {r['n_tied']:<7}")

    all_test1_significant = all(test1_results[s][m]["pval"] < 0.05 for s in SEEDS for m, _ in metrics)
    print(f"\nAll 3 seeds significant at p<0.05 for BOTH metrics (Test 1): {all_test1_significant}")

    print("\n" + "=" * 100)
    print("SUMMARY: TEST 2 (single_section delta > both_sections delta)")
    print("=" * 100)
    header2 = f"{'seed':<6} | {'metric':<12} | {'U stat':<10} | {'p-value':<12} | {'sig p<0.05':<11} | {'direction OK':<12}"
    print(header2)
    print("-" * len(header2))
    for seed in SEEDS:
        for metric_label, _ in metrics:
            r = test2_results[seed][metric_label]
            direction_ok = r["median_single"] > r["median_both"]
            print(f"{seed:<6} | {metric_label:<12} | {r['u_stat']:<10.1f} | {r['u_pval']:<12.3e} | "
                  f"{str(r['u_pval']<0.05):<11} | {str(direction_ok):<12}")

    all_test2_significant = all(test2_results[s][m]["u_pval"] < 0.05 for s in SEEDS for m, _ in metrics)
    all_test2_direction_ok = all(
        test2_results[s][m]["median_single"] > test2_results[s][m]["median_both"] for s in SEEDS for m, _ in metrics
    )
    print(f"\nAll 3 seeds significant at p<0.05 (Test 2): {all_test2_significant}")
    print(f"All 3 seeds in expected direction (single_section > both_sections) (Test 2): {all_test2_direction_ok}")

    print("\n" + "=" * 100)
    print("COMBINED CROSS-SEED CONSISTENCY (not a formal meta-analysis -- simple sanity check)")
    print("=" * 100)
    test1_all_positive = all(test1_results[s][m]["n_improved"] > test1_results[s][m]["n_worse"] for s in SEEDS for m, _ in metrics)
    print(f"Test 1: MG-G2L improves more queries than it worsens, in ALL 3 seeds x 2 metrics: {test1_all_positive}")
    print(f"Test 2: single_section > both_sections delta, in ALL 3 seeds x 2 metrics: {all_test2_direction_ok}")
    print(f"\nOVERALL: both effects (Test 1 direction, Test 2 direction) consistent across all 3 seeds: "
          f"{test1_all_positive and all_test2_direction_ok}")


if __name__ == "__main__":
    main()
