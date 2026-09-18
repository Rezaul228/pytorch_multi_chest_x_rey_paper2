#!/usr/bin/env python3
"""
URGENT CHECK (2026-09-18): confirm the MG-G2L section mechanism had real
section data during TRAINING for MIMIC and openi_sa -- i.e. the per-sample
fallback (has_find=False/has_imp=False) did NOT fire, on train AND val.

Replicates exactly what data_loader_v1_paper2.IndianaDataset.__getitem__ does:
  str(sample['study_ids']) in self.section_boundaries  ->  else fallback_count += 1
so the number printed here IS the fallback_count those training runs had.

Full scan of every train/val shard for both datasets. Report only.
"""
import csv
import glob
import os
import pickle
import sys

CASES = [
    # NOTE: the MIMIC CSVs sat at the repo root when the 2026-09 MG-G2L runs were
    # trained (that is the path their logs quote); STEP 1 moved them to mimic/data/.
    # Same files, verified by size/content -- only the location changed.
    ("MIMIC", "train", "mimic/data/section_boundaries_train_paper2.csv",
     "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/train"),
    ("MIMIC", "val", "mimic/data/section_boundaries_val_paper2.csv",
     "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/val"),
    ("openi_sa", "train", "openi/data/section_boundaries_train_openi_sa.csv",
     "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/openi_sa/train"),
    ("openi_sa", "val", "openi/data/section_boundaries_val_openi_sa.csv",
     "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/openi_sa/val"),
]
TRUE = ("1", "True", "true")


def main():
    print(f"{'dataset':<9} {'split':<6} {'shard_samples':>13} {'csv_entries':>11} {'fallback_count':>14} "
          f"{'has_find=1':>10} {'has_imp=1':>9} {'both=1':>7} {'mean_find_tok':>13}")
    bad = []
    for ds, sp, csvp, shdir in CASES:
        rows = list(csv.DictReader(open(csvp)))
        keys = {r["study_id"] for r in rows}
        n_seen = fb = 0
        for s in sorted(glob.glob(os.path.join(shdir, "*.pkl"))):
            with open(s, "rb") as f:
                sids = pickle.load(f)["study_ids"]
            for x in sids:
                n_seen += 1
                if str(x) not in keys:
                    fb += 1
        hf = sum(1 for r in rows if r["has_findings"] in TRUE)
        hi = sum(1 for r in rows if r["has_impression"] in TRUE)
        bo = sum(1 for r in rows if r["has_findings"] in TRUE and r["has_impression"] in TRUE)
        mft = sum(int(r["findings_token_count"]) for r in rows) / max(len(rows), 1)
        print(f"{ds:<9} {sp:<6} {n_seen:>13,} {len(keys):>11,} {fb:>14,} {hf:>10,} {hi:>9,} {bo:>7,} {mft:>13.1f}")
        if fb or bo == 0:
            bad.append(f"{ds}/{sp}: fallback={fb}, both_sections={bo}")
    print("\n===== FINAL SUMMARY (SECTION COVERAGE DURING TRAINING) =====")
    print("VERDICT=" + ("ACTIVE (fallback_count==0 on every train/val split)" if not bad else "INERT/PARTIAL"))
    for b in bad:
        print("  - " + b)
    print("===== END SUMMARY =====")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
