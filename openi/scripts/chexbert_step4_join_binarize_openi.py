#!/usr/bin/env python3
"""CheXbert Step 4 for openi_sa: prepend ids (by row order) to
labeled_reports.csv, reorder the 14 findings to the MIMIC column order,
save the raw file (source of truth, values 1.0/0.0/-1.0/blank) and the
binary file (1 iff raw == 1.0 else 0) with exactly the MIMIC binary header.
Mirrors rexgradient/scripts/chexbert_step4_join_binarize_rexgradient.py.
Note: openi_sa has no subject_id in its ids file; base_id (one report per
study here, val/test are not augmented) is used as a stand-in subject_id."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()

LABELED = os.path.join(PROJECT_DIR, "openi", "data", "chexbert_out", "labeled_reports.csv")
IDS = os.path.join(PROJECT_DIR, "openi", "data", "chexbert_input_test_ids.csv")
TEXT = os.path.join(PROJECT_DIR, "openi", "data", "chexbert_input_test.csv")
OUT_RAW = os.path.join(PROJECT_DIR, "openi", "data", "test_labels_chexbert.csv")
OUT_BIN = os.path.join(PROJECT_DIR, "openi", "data", "test_labels_chexbert_binary.csv")
MIMIC_BIN = os.path.join(PROJECT_DIR, "mimic", "data", "test_labels_chexpert_binary.csv")


def main():
    for p in (OUT_RAW, OUT_BIN):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite existing {p}")
    mimic_cols = pd.read_csv(MIMIC_BIN, nrows=0).columns.tolist()
    findings = mimic_cols[2:]
    lab = pd.read_csv(LABELED)
    ids = pd.read_csv(IDS, dtype=str)
    txt = pd.read_csv(TEXT)
    assert len(lab) == len(ids) == len(txt), (len(lab), len(ids), len(txt))
    assert (lab["Report Impression"].fillna("").astype(str).values == txt["Report Impression"].fillna("").astype(str).values).all()
    assert set(findings) == set(lab.columns) - {"Report Impression"}

    raw = pd.concat([
        ids[["base_id"]].rename(columns={"base_id": "subject_id"}).astype(str).reset_index(drop=True),
        ids[["study_id"]].reset_index(drop=True),
        lab[findings].reset_index(drop=True),
    ], axis=1)
    raw.to_csv(OUT_RAW, index=False)

    binary = raw.copy()
    binary[findings] = (raw[findings] == 1.0).astype(int)
    binary = binary[mimic_cols]
    binary.to_csv(OUT_BIN, index=False)

    assert pd.read_csv(OUT_BIN, nrows=0).columns.tolist() == mimic_cols
    n_any = int((binary[[c for c in findings if c != "No Finding"]].sum(axis=1) >= 1).sum())
    print("===== FINAL SUMMARY =====")
    print(f"rows={len(binary)}, columns_match_mimic={binary.columns.tolist() == mimic_cols}")
    print(f"n_with_ge1_positive_excluding_NoFinding={n_any}")
    print(f"outputs={OUT_RAW}, {OUT_BIN}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
