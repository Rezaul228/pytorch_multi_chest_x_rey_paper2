#!/usr/bin/env python3
"""
DIAGNOSTIC B: compare Findings/Impression divergence-score distributions
across MIMIC, ReXGradient, and openi_sa test sets. CPU-only, pure statistics
on the already-computed per-study divergence_scores_test_*.csv files (each
produced by reusing mimic/scripts/compute_divergence_scores_paper2.py's
FINDING_KEYWORDS/keyword_present/threshold, unchanged).

Reads (per-study, gitignored -- study_id-linked, local-only):
  mimic/results/divergence_scores_test_paper2.csv          (existing, untouched)
  mimic/results/divergence_scores_test_control_rerun.csv   (control -- same function, new file)
  rexgradient/results/divergence_scores_test_rexgradient.csv
  openi/results/divergence_scores_test_openi.csv

Writes (aggregate only, no per-study identifiers -- safe to commit):
  mimic/results/divergence_dataset_comparison_diagnostic_b.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MIMIC_CSV = "mimic/results/divergence_scores_test_paper2.csv"
REX_CSV = "rexgradient/results/divergence_scores_test_rexgradient.csv"
OPENI_CSV = "openi/results/divergence_scores_test_openi.csv"
OUT_CSV = "mimic/results/divergence_dataset_comparison_diagnostic_b.csv"

# Mean Findings/Impression token counts (word-split), computed alongside each
# dataset's divergence script and hardcoded here since the token-length
# source rows (all both-sections rows, not just the restricted/scored subset)
# aren't part of the per-study divergence CSVs' schema.
MEAN_TOKENS = {
    "MIMIC": {"findings": 57.60, "impression": 22.87},
    "ReXGradient": {"findings": 33.13, "impression": 11.80},
    "openi_sa": {"findings": 31.07, "impression": 8.16},
}


def stats_row(name, df):
    s = df["divergence_score"].values
    return {
        "dataset": name,
        "n": len(s),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "P10": float(np.percentile(s, 10)),
        "P90": float(np.percentile(s, 90)),
        "pct_divergence": float((df["group"] == "divergence").mean() * 100),
    }


def main():
    mimic = pd.read_csv(MIMIC_CSV)
    rex = pd.read_csv(REX_CSV)
    openi = pd.read_csv(OPENI_CSV)

    rows = [stats_row("MIMIC", mimic), stats_row("ReXGradient", rex), stats_row("openi_sa", openi)]
    for r in rows:
        name = "MIMIC" if r["dataset"] == "MIMIC" else r["dataset"]
        r["mean_findings_tokens"] = MEAN_TOKENS.get(name, {}).get("findings")
        r["mean_impression_tokens"] = MEAN_TOKENS.get(name, {}).get("impression")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print("=== TABLE ===")
    print(out_df.to_string(index=False))

    print("\n=== KS TESTS ===")
    pairs = [("MIMIC", mimic, "ReXGradient", rex),
             ("MIMIC", mimic, "openi_sa", openi),
             ("ReXGradient", rex, "openi_sa", openi)]
    for n1, d1, n2, d2 in pairs:
        stat, p = ks_2samp(d1["divergence_score"].values, d2["divergence_score"].values)
        print(f"{n1}-{n2}: D={stat:.4f} p={p:.3e}")

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
