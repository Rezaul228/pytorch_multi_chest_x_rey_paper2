#!/usr/bin/env python3
"""
Findings/Impression divergence scoring for openi_sa's test split, reusing
mimic/scripts/compute_divergence_scores_paper2.py's FINDING_KEYWORDS,
keyword_present EXACTLY (imported, not redefined) -- same threshold
(agreement: divergence_score==0, divergence: divergence_score>=1), same
restriction (has_find AND has_imp AND >=1 confirmed-positive finding).

Labels: openi/data/test_labels_chexbert_binary.csv (built by
chexbert_step1_build_input_openi.py + the project's CheXbert checkpoint +
chexbert_step4_join_binarize_openi.py -- this dataset had no whole-report
CheXpert/CheXbert label file before this).
Text: openi_sa's own texts_test_openi_sa.csv (already-extracted
findings_text/impression_text columns -- no raw report re-parsing needed).
Section availability: openi/data/section_boundaries_test_openi_sa.csv
(has_findings/has_impression booleans).

96 impression-only (has_findings=0) rows are excluded here (n/a for
divergence -- there is no Findings section to diverge from), matching the
task's framing; they are NOT part of the 656-row both-sections population
this script scores.

Output: openi/results/divergence_scores_test_openi.csv
  columns: study_id, divergence_score, group  (restricted population only)
"""
import os
import sys
import csv

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
sys.path.insert(0, os.path.join(PROJECT_DIR, "mimic", "scripts"))
import compute_divergence_scores_paper2 as div  # noqa: E402  reused, unchanged

TEXTS = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/openi_sa/texts_test_openi_sa.csv"
BOUNDARIES = os.path.join(PROJECT_DIR, "openi", "data", "section_boundaries_test_openi_sa.csv")
BINARY_LABELS = os.path.join(PROJECT_DIR, "openi", "data", "test_labels_chexbert_binary.csv")
OUT_PATH = os.path.join(PROJECT_DIR, "openi", "results", "divergence_scores_test_openi.csv")


def main():
    texts = pd.read_csv(TEXTS, dtype={"study_id": str}).set_index("study_id")
    bd = pd.read_csv(BOUNDARIES, dtype={"study_id": str}).set_index("study_id")
    binary_df = pd.read_csv(BINARY_LABELS, dtype={"study_id": str}).set_index("study_id")
    finding_cols = [c for c in binary_df.columns if c not in ("subject_id",) and c != "No Finding"]
    assert len(finding_cols) == 13
    assert set(finding_cols) == set(div.FINDING_KEYWORDS.keys())

    rows_out, findings_lens, impression_lens = [], [], []
    n_impression_only = n_both = n_no_positive = 0

    for sid in bd.index:
        has_find = bool(bd.loc[sid, "has_findings"])
        has_imp = bool(bd.loc[sid, "has_impression"])
        if not has_imp:
            continue
        if not has_find:
            n_impression_only += 1
            continue
        n_both += 1

        findings_text = texts.loc[sid, "findings_text"]
        impression_text = texts.loc[sid, "impression_text"]
        findings_text = "" if pd.isna(findings_text) else str(findings_text)
        impression_text = "" if pd.isna(impression_text) else str(impression_text)
        findings_lens.append(len(findings_text.split()))
        impression_lens.append(len(impression_text.split()))

        if sid not in binary_df.index:
            continue
        positive_findings = [c for c in finding_cols if binary_df.loc[sid, c] == 1]
        if len(positive_findings) == 0:
            n_no_positive += 1
            continue

        findings_lower, impression_lower = findings_text.lower(), impression_text.lower()
        divergence_score = 0
        for finding in positive_findings:
            in_findings = div.keyword_present(findings_lower, div.FINDING_KEYWORDS[finding])
            in_impression = div.keyword_present(impression_lower, div.FINDING_KEYWORDS[finding])
            if in_findings != in_impression:
                divergence_score += 1
        group = "agreement" if divergence_score == 0 else "divergence"
        rows_out.append({"study_id": sid, "divergence_score": divergence_score, "group": group})

    print(f"openi_sa test total: {len(bd)}")
    print(f"impression-only (has_find=0, n/a for divergence): {n_impression_only}")
    print(f"both-sections rows: {n_both}")
    print(f"no positive finding (excluded): {n_no_positive}")
    print(f"scored (restricted, both sections, >=1 positive): {len(rows_out)}")
    print(f"agreement={sum(1 for r in rows_out if r['group']=='agreement')}  "
          f"divergence={sum(1 for r in rows_out if r['group']=='divergence')}")
    print(f"mean findings tokens: {sum(findings_lens)/len(findings_lens):.2f}")
    print(f"mean impression tokens: {sum(impression_lens)/len(impression_lens):.2f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["study_id", "divergence_score", "group"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
