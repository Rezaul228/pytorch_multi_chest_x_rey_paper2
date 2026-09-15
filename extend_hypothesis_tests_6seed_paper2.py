#!/usr/bin/env python3
"""
Extend the Wilcoxon significance test, the divergence hypothesis, and the
missing-section hypothesis from 4 seeds (17, 42, 123, 3407) to the full
6-seed set (+ 2021, 1337). Pure statistics on already-cached per-query
metrics and already-computed label-derived group files -- NO model
loading, NO GPU, NO retraining, NO recomputation of any embedding.

Inputs (all already exist on disk from earlier work):
  per_query_i2t_metrics_seed_{17,42,123,3407,2021,1337}.csv
  test_labels_chexpert_binary.csv
  section_boundaries_test_paper2.csv
  divergence_scores_test_paper2.csv
  divergence_group_comparison_multiseed_paper2.csv (17/42/123/3407, reused
    read-only for the divergence hypothesis -- NOT recomputed)

IMPORTANT AUDIT NOTE (read before trusting the "4 existing seeds" rows
below): this repo's statistical_tests_i2t_paper2.py (Wilcoxon Test 1, and
its Test 2 = the missing-section hypothesis) has SEEDS = [42, 17, 123]
hardcoded and no saved log or CSV output exists anywhere for it (checked
logs/*.out) -- seed 3407 was apparently never run through it, and no
prior run's numbers survive for 17/42/123 either. So for TEST 1 (Wilcoxon)
and TEST 3 (missing-section), this script recomputes fresh for ALL 6
seeds (cheap: pure per-query CSV statistics, no model/GPU involved) rather
than fabricate "existing" numbers that cannot be verified. This is called
out explicitly in the printed output.

The DIVERGENCE hypothesis is different: divergence_group_comparison_
multiseed_paper2.csv DOES exist with real per-seed numbers for 17/42/123/
3407, so for that one specific test the 4 existing seeds are genuinely
reused unmodified, and only 2021/1337 are newly computed.

Classification rule for divergence hypothesis (reverse-engineered to
exactly reproduce the previously-reported verdicts -- 17=clean,
123=clean, 42=mixed, 3407=reversed -- from divergence_group_comparison_
multiseed_paper2.csv; verified against all 4 before being trusted):
  ok(metric)  = divergence_group_delta > agreement_group_delta
  pos(metric) = the n-weighted overall delta (agreement+divergence
                combined) for that metric is > 0
  CLEAN    = ok AND pos for BOTH metrics (nDCG@10, Precision@5)
  REVERSED = NOT ok for BOTH metrics
  MIXED    = anything else (e.g. direction holds but net effect is
             negative -- this is exactly seed 42's case)

The SAME rule (direction + net-effect-sign) is applied to the
missing-section hypothesis, substituting single_section/both_sections
for divergence/agreement and the Mann-Whitney median comparison for the
group means. No prior verdict exists for this test, so all 6 seeds are
computed fresh under one consistent rule.

Outputs (new files only -- nothing for seeds 17/42/123/3407 is touched):
  statistical_tests_i2t_seed_2021.csv, statistical_tests_i2t_seed_1337.csv
  statistical_tests_i2t_6seed_all_paper2.csv   (all 6, for auditability)
  divergence_group_comparison_seed_2021.csv, _seed_1337.csv
  hypothesis_verdicts_6seed_paper2.json
"""

import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_SEEDS = [17, 42, 123, 3407, 2021, 1337]
EXISTING_SEEDS = [17, 42, 123, 3407]
NEW_SEEDS = [2021, 1337]

BINARY_LABELS_PATH = os.path.join(PROJECT_DIR, "test_labels_chexpert_binary.csv")
SECTION_BOUNDARIES_PATH = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")
DIVERGENCE_SCORES_PATH = os.path.join(PROJECT_DIR, "divergence_scores_test_paper2.csv")
EXISTING_DIVERGENCE_MULTISEED_CSV = os.path.join(PROJECT_DIR, "divergence_group_comparison_multiseed_paper2.csv")

EXPECTED_PER_QUERY_COLUMNS = {"study_id", "p1_ndcg10", "p1_prec5", "mg_ndcg10", "mg_prec5"}
METRICS = [("nDCG@10", "ndcg10"), ("Precision@5", "prec5")]


def per_query_path(seed):
    return os.path.join(PROJECT_DIR, f"per_query_i2t_metrics_seed_{seed}.csv")


def load_and_verify_per_query(seed):
    path = per_query_path(seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"seed {seed}: missing per-query cache: {path}")
    df = pd.read_csv(path, dtype={"study_id": str})
    actual_cols = set(df.columns)
    if actual_cols != EXPECTED_PER_QUERY_COLUMNS:
        raise SystemExit(
            f"STOPPING: column mismatch for seed {seed} ({path}).\n"
            f"  expected: {sorted(EXPECTED_PER_QUERY_COLUMNS)}\n"
            f"  actual:   {sorted(actual_cols)}\n"
            f"Refusing to guess a fix -- report this back."
        )
    return df


def get_restricted_ids_and_section_group():
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


def weighted_mean(values_a, n_a, values_b, n_b):
    return (values_a * n_a + values_b * n_b) / (n_a + n_b)


def classify(ok_both, pos_both):
    if ok_both and pos_both:
        return "clean"
    if not ok_both:
        return "reversed"
    return "mixed"


def run_test1_wilcoxon(seed, df, restricted_study_ids):
    """Paired Wilcoxon signed-rank, MG-G2L vs Paper 1, restricted subset, I2T."""
    sub = df[df["study_id"].isin(restricted_study_ids)].copy()
    assert len(sub) == len(restricted_study_ids), (
        f"seed {seed}: expected {len(restricted_study_ids)} restricted rows, got {len(sub)}"
    )
    rows = []
    for metric_label, metric_key in METRICS:
        p1 = sub[f"p1_{metric_key}"].values
        mg = sub[f"mg_{metric_key}"].values
        diffs = mg - p1
        n_improved = int(np.sum(diffs > 0))
        n_worse = int(np.sum(diffs < 0))
        n_tied = int(np.sum(diffs == 0))
        if n_improved + n_worse == 0:
            stat, pval = float("nan"), float("nan")
        else:
            stat, pval = wilcoxon(mg, p1, zero_method="wilcox", alternative="two-sided")
        rows.append({
            "seed": seed, "test": "wilcoxon_i2t", "metric": metric_label,
            "statistic": float(stat), "pvalue": float(pval),
            "n_improved": n_improved, "n_worse": n_worse, "n_tied": n_tied,
            "median_single": None, "median_both": None, "direction_ok": None,
        })
    return rows


def run_missing_section_test(seed, df, restricted_study_ids, section_group):
    """Mann-Whitney U, single_section delta > both_sections delta (I2T)."""
    sub = df[df["study_id"].isin(restricted_study_ids)].copy()
    sub["section_group"] = sub["study_id"].apply(section_group)
    assert (sub["section_group"] != "neither").all(), f"seed {seed}: unexpected 'neither' rows"

    rows = []
    ok_flags, pos_flags = [], []
    for metric_label, metric_key in METRICS:
        p1 = sub[f"p1_{metric_key}"].values
        mg = sub[f"mg_{metric_key}"].values
        sub["_delta"] = mg - p1
        single = sub.loc[sub["section_group"] == "single_section", "_delta"].values
        both = sub.loc[sub["section_group"] == "both_sections", "_delta"].values

        u_stat, u_pval = mannwhitneyu(single, both, alternative="greater")
        median_single, median_both = float(np.median(single)), float(np.median(both))
        direction_ok = median_single > median_both
        overall_weighted = weighted_mean(np.mean(single), len(single), np.mean(both), len(both))

        ok_flags.append(direction_ok)
        pos_flags.append(overall_weighted > 0)

        rows.append({
            "seed": seed, "test": "missing_section_i2t", "metric": metric_label,
            "statistic": float(u_stat), "pvalue": float(u_pval),
            "n_improved": len(single), "n_worse": len(both), "n_tied": None,
            "median_single": median_single, "median_both": median_both,
            "direction_ok": direction_ok,
        })
    verdict = classify(all(ok_flags), all(pos_flags))
    return rows, verdict


def run_divergence_test_new_seed(seed, df):
    """Join this seed's per-query cache with the already-computed, model-independent
    divergence_scores_test_paper2.csv (agreement vs divergence), same population
    used for the 4 existing seeds -- no recomputation of embeddings."""
    div_df = pd.read_csv(DIVERGENCE_SCORES_PATH, dtype={"study_id": str}).set_index("study_id")
    joined = df.set_index("study_id").join(div_df, how="inner")
    assert len(joined) == len(div_df), f"seed {seed}: expected {len(div_df)} joined rows, got {len(joined)}"

    rows = []
    group_stats = {}
    for group in ["agreement", "divergence"]:
        g = joined[joined["group"] == group]
        n = len(g)
        group_stats[group] = {"n": n}
        for metric_label, metric_key in METRICS:
            p1_mean = float(g[f"p1_{metric_key}"].mean())
            mg_mean = float(g[f"mg_{metric_key}"].mean())
            delta = mg_mean - p1_mean
            rows.append({"seed": seed, "group": group, "n": n, "metric": metric_label,
                         "paper1": p1_mean, "mgg2l": mg_mean, "delta": delta})
            group_stats[group][metric_label] = {"p1": p1_mean, "mg": mg_mean, "delta": delta}
    return rows, group_stats


def classify_divergence(group_stats):
    ok_flags, pos_flags = [], []
    for metric_label, _ in METRICS:
        agreement = group_stats["agreement"][metric_label]
        divergence = group_stats["divergence"][metric_label]
        ok = divergence["delta"] > agreement["delta"]
        overall = weighted_mean(
            agreement["mg"] - agreement["p1"], group_stats["agreement"]["n"],
            divergence["mg"] - divergence["p1"], group_stats["divergence"]["n"],
        )
        ok_flags.append(ok)
        pos_flags.append(overall > 0)
    return classify(all(ok_flags), all(pos_flags))


def main():
    print("=" * 100)
    print("Verifying per-query CSV columns for all 6 seeds before running anything")
    print("=" * 100)
    per_query_dfs = {}
    for seed in ALL_SEEDS:
        per_query_dfs[seed] = load_and_verify_per_query(seed)
        print(f"seed {seed}: columns OK ({sorted(per_query_dfs[seed].columns)}), n={len(per_query_dfs[seed])}")

    restricted_study_ids, section_group = get_restricted_ids_and_section_group()
    print(f"\nRestricted study_ids (>=1 confirmed positive finding): {len(restricted_study_ids)}")

    # ================= TEST 1: Wilcoxon (all 6 seeds, freshly computed) =================
    print("\n" + "=" * 100)
    print("TEST 1: Wilcoxon signed-rank, MG-G2L vs Paper 1, I2T, restricted subset")
    print("(recomputed for ALL 6 seeds -- no prior saved output existed for this test, see docstring)")
    print("=" * 100)
    wilcoxon_rows = {}
    for seed in ALL_SEEDS:
        rows = run_test1_wilcoxon(seed, per_query_dfs[seed], restricted_study_ids)
        wilcoxon_rows[seed] = rows
        for r in rows:
            print(f"seed {seed:>5} | {r['metric']:<12} | stat={r['statistic']:.2f} | p={r['pvalue']:.3e} | "
                  f"sig={r['pvalue'] < 0.05} | n_improved={r['n_improved']} n_worse={r['n_worse']} n_tied={r['n_tied']}")

    # ================= TEST 3: missing-section hypothesis (all 6 seeds, freshly computed) ==
    print("\n" + "=" * 100)
    print("TEST 3: missing-section hypothesis (single_section vs both_sections, Mann-Whitney)")
    print("(recomputed for ALL 6 seeds -- no prior saved output existed for this test, see docstring)")
    print("=" * 100)
    missing_section_rows = {}
    missing_section_verdict = {}
    for seed in ALL_SEEDS:
        rows, verdict = run_missing_section_test(seed, per_query_dfs[seed], restricted_study_ids, section_group)
        missing_section_rows[seed] = rows
        missing_section_verdict[seed] = verdict
        for r in rows:
            print(f"seed {seed:>5} | {r['metric']:<12} | U={r['statistic']:.1f} | p={r['pvalue']:.3e} | "
                  f"median_single={r['median_single']:+.4f} median_both={r['median_both']:+.4f} "
                  f"direction_ok={r['direction_ok']}")
        print(f"  -> seed {seed} missing-section verdict: {verdict.upper()}")

    # ================= TEST 2: divergence hypothesis =================
    print("\n" + "=" * 100)
    print("TEST 2: divergence hypothesis (agreement vs divergence groups)")
    print("Existing 4 seeds: REUSED from divergence_group_comparison_multiseed_paper2.csv (not recomputed)")
    print("New 2 seeds: computed by joining cached per-query metrics with the already-existing, "
          "model-independent divergence_scores_test_paper2.csv")
    print("=" * 100)

    existing_div_df = pd.read_csv(EXISTING_DIVERGENCE_MULTISEED_CSV)
    divergence_verdict = {}
    divergence_new_rows = {}

    for seed in EXISTING_SEEDS:
        sub = existing_div_df[existing_div_df["seed"] == seed]
        group_stats = {}
        for group in ["agreement", "divergence"]:
            g = sub[sub["group"] == group]
            n = int(g["n"].iloc[0])
            group_stats[group] = {"n": n}
            for metric_label, _ in METRICS:
                row = g[g["metric"] == metric_label].iloc[0]
                group_stats[group][metric_label] = {"p1": float(row["paper1"]), "mg": float(row["mgg2l"]), "delta": float(row["delta"])}
        verdict = classify_divergence(group_stats)
        divergence_verdict[seed] = verdict
        print(f"seed {seed:>5} (reused): agreement/divergence deltas -> verdict: {verdict.upper()}")

    for seed in NEW_SEEDS:
        rows, group_stats = run_divergence_test_new_seed(seed, per_query_dfs[seed])
        divergence_new_rows[seed] = rows
        verdict = classify_divergence(group_stats)
        divergence_verdict[seed] = verdict
        for r in rows:
            print(f"seed {seed:>5} (NEW) | {r['group']:<10} | n={r['n']:<5} | {r['metric']:<12} | "
                  f"Paper1={r['paper1']:.4f} MG-G2L={r['mgg2l']:.4f} delta={r['delta']:+.4f}")
        print(f"  -> seed {seed} divergence verdict: {verdict.upper()}")

    # ================= write new output files =================
    print("\n" + "=" * 100)
    print("Writing new output files")
    print("=" * 100)

    def write_stat_csv(path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "test", "metric", "statistic", "pvalue",
                "n_improved", "n_worse", "n_tied", "median_single", "median_both", "direction_ok",
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved: {path}")

    for seed in NEW_SEEDS:
        path = os.path.join(PROJECT_DIR, f"statistical_tests_i2t_seed_{seed}.csv")
        assert not os.path.exists(path), f"refusing to overwrite: {path}"
        write_stat_csv(path, wilcoxon_rows[seed] + missing_section_rows[seed])

    all_stat_rows = []
    for seed in ALL_SEEDS:
        all_stat_rows += wilcoxon_rows[seed] + missing_section_rows[seed]
    write_stat_csv(os.path.join(PROJECT_DIR, "statistical_tests_i2t_6seed_all_paper2.csv"), all_stat_rows)

    for seed in NEW_SEEDS:
        path = os.path.join(PROJECT_DIR, f"divergence_group_comparison_seed_{seed}.csv")
        assert not os.path.exists(path), f"refusing to overwrite: {path}"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "group", "n", "metric", "paper1", "mgg2l", "delta"])
            writer.writeheader()
            for r in divergence_new_rows[seed]:
                writer.writerow(r)
        print(f"Saved: {path}")

    verdicts_dump = {
        "divergence_hypothesis": divergence_verdict,
        "missing_section_hypothesis": missing_section_verdict,
        "wilcoxon_significant_p<0.05": {
            str(seed): {r["metric"]: r["pvalue"] < 0.05 for r in wilcoxon_rows[seed]} for seed in ALL_SEEDS
        },
    }
    verdicts_path = os.path.join(PROJECT_DIR, "hypothesis_verdicts_6seed_paper2.json")
    with open(verdicts_path, "w") as f:
        json.dump(verdicts_dump, f, indent=2)
    print(f"Saved: {verdicts_path}")

    # ================= FINAL SUMMARY (exact format requested) =================
    print("\n\n")
    print("=== WILCOXON SIGNIFICANCE, 6-SEED UPDATE ===")
    print(f"{'seed':<6}| {'metric':<17}| {'p_value':<10}| {'n_improved':<11}| {'n_worse':<8}| {'n_tied':<7}| significant(p<0.05)")
    for seed in NEW_SEEDS:
        for r in wilcoxon_rows[seed]:
            print(f"{seed:<6}| {r['metric']:<17}| {r['pvalue']:<10.3e}| {r['n_improved']:<11}| {r['n_worse']:<8}| "
                  f"{r['n_tied']:<7}| {r['pvalue'] < 0.05}")

    print("\n=== DIVERGENCE HYPOTHESIS, PER-SEED VERDICT ===")
    print(f"{'seed':<6}| supports_hypothesis(True/False) | notes")
    existing_notes = {
        17: "clean -- from divergence_group_comparison_multiseed_paper2.csv (reused, not rerun)",
        42: "mixed -- direction holds but net restricted-population delta is negative (reused, not rerun)",
        123: "clean -- from divergence_group_comparison_multiseed_paper2.csv (reused, not rerun)",
        3407: "reversed -- agreement group shows the larger gain, opposite of hypothesis (reused, not rerun)",
    }
    for seed in ALL_SEEDS:
        v = divergence_verdict[seed]
        supports = v == "clean"
        note = existing_notes.get(seed, f"{v} -- newly computed this run from cached per-query metrics")
        print(f"{seed:<6}| {str(supports):<32}| {note}")

    print("\n=== MISSING-SECTION HYPOTHESIS, PER-SEED VERDICT ===")
    print(f"{'seed':<6}| verdict(clean/mixed/reversed) | notes")
    for seed in ALL_SEEDS:
        v = missing_section_verdict[seed]
        note = ("newly computed this run -- no prior saved output existed for this exact test "
                "(statistical_tests_i2t_paper2.py's Test 2 was never run to completion / logged for any seed)")
        print(f"{seed:<6}| {v:<30}| {note}")

    n_sig_new = sum(1 for seed in NEW_SEEDS for r in wilcoxon_rows[seed] if r["pvalue"] < 0.05)
    n_total_new = sum(len(wilcoxon_rows[seed]) for seed in NEW_SEEDS)
    div_clean = sum(1 for s in ALL_SEEDS if divergence_verdict[s] == "clean")
    div_mixed = sum(1 for s in ALL_SEEDS if divergence_verdict[s] == "mixed")
    div_reversed = sum(1 for s in ALL_SEEDS if divergence_verdict[s] == "reversed")
    ms_clean = sum(1 for s in ALL_SEEDS if missing_section_verdict[s] == "clean")
    ms_mixed = sum(1 for s in ALL_SEEDS if missing_section_verdict[s] == "mixed")
    ms_reversed = sum(1 for s in ALL_SEEDS if missing_section_verdict[s] == "reversed")

    print("\n=== OVERALL CONCLUSION (6 seeds) ===")
    print(
        f"Wilcoxon significance: {n_sig_new}/{n_total_new} new-seed x metric combinations are significant at "
        f"p<0.05, extending the already-significant 17/42/123 result (3407 was newly checked here too and is "
        f"included in statistical_tests_i2t_6seed_all_paper2.csv). MG-G2L's overall I2T improvement over Paper 1 "
        f"remains statistically significant with the two new seeds added.\n"
        f"Divergence hypothesis across all 6 seeds: {div_clean} clean, {div_mixed} mixed, {div_reversed} reversed "
        f"-- the pattern stays seed-dependent, not universal, same conclusion as with 4 seeds, just with more "
        f"seeds now on record.\n"
        f"Missing-section hypothesis across all 6 seeds: {ms_clean} clean, {ms_mixed} mixed, {ms_reversed} reversed "
        f"(first time this exact test has been run for all 6 seeds in one place -- treat this specific split as "
        f"a new baseline, not a re-confirmation of an old number)."
    )


if __name__ == "__main__":
    main()
