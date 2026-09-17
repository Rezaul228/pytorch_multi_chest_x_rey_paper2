#!/usr/bin/env python3
"""CheXbert Step 4 for MIMIC test: prepend ids (by row order) to
labeled_reports.csv, reorder the 14 findings to the existing MIMIC binary
column order, save the raw file (1.0/0.0/-1.0/blank) and the binary file
(1 iff raw == 1.0 else 0) with exactly test_labels_chexpert_binary.csv's
header. Mirrors the rexgradient/openi step4 scripts."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
D = os.path.join(PROJECT_DIR, "mimic", "data")

LABELED = os.path.join(D, "chexbert_out", "labeled_reports.csv")
IDS = os.path.join(D, "chexbert_input_test_ids.csv")
TEXT = os.path.join(D, "chexbert_input_test.csv")
OUT_RAW = os.path.join(D, "test_labels_chexbert.csv")
OUT_BIN = os.path.join(D, "test_labels_chexbert_binary.csv")
CHEXPERT_BIN = os.path.join(D, "test_labels_chexpert_binary.csv")


def main():
    for p in (OUT_RAW, OUT_BIN):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite existing {p}")
    cols = pd.read_csv(CHEXPERT_BIN, nrows=0).columns.tolist()
    findings = cols[2:]
    lab = pd.read_csv(LABELED)
    ids = pd.read_csv(IDS, dtype=str)
    txt = pd.read_csv(TEXT)
    assert len(lab) == len(ids) == len(txt), (len(lab), len(ids), len(txt))
    assert (lab["Report Impression"].fillna("").astype(str).values
            == txt["Report Impression"].fillna("").astype(str).values).all(), "row order mismatch"
    assert set(findings) == set(lab.columns) - {"Report Impression"}

    raw = pd.concat([ids[cols[:2]].reset_index(drop=True), lab[findings].reset_index(drop=True)], axis=1)
    raw.to_csv(OUT_RAW, index=False)
    binary = raw.copy()
    binary[findings] = (raw[findings] == 1.0).astype(int)
    binary = binary[cols]
    binary.to_csv(OUT_BIN, index=False)
    assert pd.read_csv(OUT_BIN, nrows=0).columns.tolist() == cols

    n_any = int((binary[[c for c in findings if c != "No Finding"]].sum(axis=1) >= 1).sum())
    raw_vals = pd.Series(raw[findings].values.ravel()).value_counts(dropna=False).to_dict()
    print("===== FINAL SUMMARY =====")
    print(f"rows={len(binary)}, columns_match_chexpert_binary={binary.columns.tolist() == cols}")
    print(f"raw_value_counts={ {('blank' if (isinstance(k, float) and np.isnan(k)) else k): v for k, v in raw_vals.items()} }")
    print(f"restricted_n_chexbert(>=1 positive excl. No Finding)={n_any}")
    print(f"outputs={OUT_RAW}, {OUT_BIN}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
