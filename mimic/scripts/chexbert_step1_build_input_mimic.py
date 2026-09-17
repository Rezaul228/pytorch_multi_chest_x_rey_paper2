#!/usr/bin/env python3
"""CheXbert Step 1 for the MIMIC-CXR TEST split (12,429 studies).
Mirrors rexgradient/openi chexbert_step1: 'Report Impression' = raw Findings
+ ' ' + raw Impression via the reused extract_text_from_report, one row per
test study_id, in test_labels_chexpert_binary.csv row order, plus a parallel
ids file (subject_id, study_id)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
sys.path.insert(0, os.path.join(PROJECT_DIR, "mimic", "scripts"))
import compute_divergence_scores_paper2 as div  # noqa: E402  reuse extract_text_from_report + raw-data paths

MIMIC_DATA_DIR = os.path.join(PROJECT_DIR, "mimic", "data")
CHEXPERT_BIN = os.path.join(MIMIC_DATA_DIR, "test_labels_chexpert_binary.csv")
OUT_TEXT = os.path.join(MIMIC_DATA_DIR, "chexbert_input_test.csv")
OUT_IDS = os.path.join(MIMIC_DATA_DIR, "chexbert_input_test_ids.csv")


def main():
    for p in (OUT_TEXT, OUT_IDS):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite existing {p}")
    ref = pd.read_csv(CHEXPERT_BIN, dtype={"study_id": str, "subject_id": str})
    meta = pd.read_csv(div.ORIGINAL_METADATA_CSV)
    meta = meta.set_index(meta["study_id"].astype(str))

    texts, empty, missing_meta, missing_file = [], 0, 0, 0
    for sid in ref["study_id"]:
        t = ""
        if sid not in meta.index:
            missing_meta += 1
        else:
            p = os.path.join(div.REPORTS_DIR, meta.loc[sid, "report_file"])
            if not os.path.exists(p):
                missing_file += 1
            else:
                f, i = div.extract_text_from_report(p)
                t = (f + " " + i).strip()
        empty += int(t == "")
        texts.append(t)

    pd.DataFrame({"Report Impression": texts}).to_csv(OUT_TEXT, index=False)
    ref[["subject_id", "study_id"]].to_csv(OUT_IDS, index=False)
    print("===== FINAL SUMMARY =====")
    print(f"test_studies={len(ref)}, rows_text={len(texts)}, unique_study_ids={ref['study_id'].nunique()}")
    print(f"empty_texts={empty}, missing_metadata={missing_meta}, missing_report_files={missing_file}, "
          f"mean_chars={sum(map(len, texts)) / max(len(texts), 1):.0f}")
    print(f"outputs={OUT_TEXT}, {OUT_IDS}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
