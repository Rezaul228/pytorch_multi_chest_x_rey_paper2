#!/usr/bin/env python3
"""CheXbert Step 1 for openi_sa's TEST split (752 studies, no augmentation).
Mirrors rexgradient/scripts/chexbert_step1_build_input_rexgradient.py's
input format exactly ('Report Impression' column + an ids file), but reads
text directly from openi_sa's own texts_test_openi_sa.csv (already-extracted
findings_text/impression_text) instead of re-parsing raw report files."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()

TEXTS = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/openi_sa/texts_test_openi_sa.csv"
OUT_TEXT = os.path.join(PROJECT_DIR, "openi", "data", "chexbert_input_test.csv")
OUT_IDS = os.path.join(PROJECT_DIR, "openi", "data", "chexbert_input_test_ids.csv")


def main():
    for p in (OUT_TEXT, OUT_IDS):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite existing {p}")
    t = pd.read_csv(TEXTS, dtype={"study_id": str})
    texts = []
    for r in t.itertuples(index=False):
        f = "" if pd.isna(r.findings_text) else str(r.findings_text)
        i = "" if pd.isna(r.impression_text) else str(r.impression_text)
        texts.append((f + " " + i).strip())
    os.makedirs(os.path.dirname(OUT_TEXT), exist_ok=True)
    pd.DataFrame({"Report Impression": texts}).to_csv(OUT_TEXT, index=False)
    pd.DataFrame({"study_id": t["study_id"], "base_id": t["base_id"]}).to_csv(OUT_IDS, index=False)
    print("===== FINAL SUMMARY =====")
    print(f"rows={len(texts)}, empty={sum(1 for x in texts if x=='')}, "
          f"mean_chars={sum(map(len, texts))/max(len(texts),1):.0f}")
    print(f"outputs={OUT_TEXT}, {OUT_IDS}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
