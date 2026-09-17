#!/usr/bin/env python3
"""
Findings/Impression divergence scoring for ReXGradient's test split, reusing
mimic/scripts/compute_divergence_scores_paper2.py's FINDING_KEYWORDS,
keyword_present, and extract_text_from_report EXACTLY (imported, not
redefined) -- same threshold (agreement: divergence_score==0, divergence:
divergence_score>=1), same restriction (has_find AND has_imp AND >=1
confirmed-positive CheXpert/CheXbert finding).

Labels: rexgradient/data/test_labels_chexbert_binary.csv (already existed,
built via the project's own CheXbert pipeline -- same 13+No Finding CheXpert
category set/column order as MIMIC's binary label file).
Text: raw report .txt files via the reused extract_text_from_report (same
regex mimic_data_loader.extract_text_from_report uses).

Output: rexgradient/results/divergence_scores_test_rexgradient.csv
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

REX_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/rexgradient"
META_CSV = os.path.join(REX_DIR, "data", "rexgradient_metadata_mimic_layout.csv")
BINARY_LABELS = os.path.join(REX_DIR, "data", "test_labels_chexbert_binary.csv")
REP_DIR = "/home/abedin/Developments/download_ReXGradient-160K_python_file/raw_data_ReXGradient-160K/organized_data/reports"
OUT_PATH = os.path.join(PROJECT_DIR, "rexgradient", "results", "divergence_scores_test_rexgradient.csv")


def main():
    meta = pd.read_csv(META_CSV, dtype=str)
    test_meta = meta[meta["hybrid_split"] == "test"].set_index("study_id")
    binary_df = pd.read_csv(BINARY_LABELS, dtype={"study_id": str}).set_index("study_id")
    finding_cols = [c for c in binary_df.columns if c not in ("subject_id",) and c != "No Finding"]
    assert len(finding_cols) == 13
    assert set(finding_cols) == set(div.FINDING_KEYWORDS.keys())

    rows_out, findings_lens, impression_lens = [], [], []
    n_missing_report = n_missing_section = n_no_positive = 0

    for sid in test_meta.index:
        if sid not in binary_df.index:
            continue
        report_path = os.path.join(REP_DIR, test_meta.loc[sid, "report_file"])
        if not os.path.exists(report_path):
            n_missing_report += 1
            continue
        findings_text, impression_text = div.extract_text_from_report(report_path)
        if not findings_text or not impression_text:
            n_missing_section += 1
            continue
        findings_lens.append(len(findings_text.split()))
        impression_lens.append(len(impression_text.split()))

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

    print(f"ReXGradient test total: {len(test_meta)}")
    print(f"missing report file: {n_missing_report}, missing a section: {n_missing_section}")
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
