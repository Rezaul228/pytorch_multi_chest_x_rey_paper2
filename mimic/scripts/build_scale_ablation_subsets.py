#!/usr/bin/env python3
"""
MIMIC data-scale ablation, PART 1: fixed, reproducible id subsets shared by both
arms and all training seeds.

  train_ids_2553.csv   2,553 study_ids from the MIMIC TRAIN split
  train_ids_10000.csv  10,000 study_ids from the MIMIC TRAIN split
  val_ids_3000.csv     3,000 study_ids from the MIMIC VAL split (best-checkpoint selection)

Each draw: numpy RandomState(0).choice(SORTED unique study_ids, n, replace=False)
-- a fresh RandomState(0) per draw, ids sorted as strings first so the result does
not depend on CSV row order. The id universe is the existing section-boundary
CSV of the split, which job 115097 verified to be exactly the shard id set
(155,745 / 29,181 entries, fallback_count 0). The test split is never subsampled.

Output is MIMIC study_id-keyed -> mimic/data/scale_ablation/ is gitignored (DUA).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
D = os.path.join(P, "mimic", "data")
OUT = os.path.join(D, "scale_ablation")
DRAWS = [("train", 2553, "train_ids_2553.csv"), ("train", 10000, "train_ids_10000.csv"), ("val", 3000, "val_ids_3000.csv")]


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'subset':<22} {'split':<6} {'n':>6} {'unique':>7} {'has_find':>9} {'has_imp':>8} {'both':>6} {'imp_only':>9} {'find_only':>10} {'all in boundary CSV':>20}")
    for split, n, fn in DRAWS:
        out = os.path.join(OUT, fn)
        assert not os.path.exists(out), f"refusing to overwrite {out}"
        b = pd.read_csv(os.path.join(D, f"section_boundaries_{split}_paper2.csv"), dtype={"study_id": str})
        assert b.study_id.is_unique, f"{split}: duplicate study_ids in boundary CSV"
        universe = np.array(sorted(b.study_id.tolist()))
        pick = np.random.RandomState(0).choice(universe, size=n, replace=False)
        pd.DataFrame({"study_id": sorted(pick)}).to_csv(out, index=False)
        sub = b[b.study_id.isin(set(pick))]
        hf, hi = sub.has_findings.astype(int), sub.has_impression.astype(int)
        print(f"{fn:<22} {split:<6} {n:>6} {len(set(pick)):>7} {int(hf.sum()):>9} {int(hi.sum()):>8} "
              f"{int(((hf==1)&(hi==1)).sum()):>6} {int(((hf==0)&(hi==1)).sum()):>9} {int(((hf==1)&(hi==0)).sum()):>10} "
              f"{'yes' if len(sub)==n else 'NO':>20}")
    a = set(pd.read_csv(os.path.join(OUT, "train_ids_2553.csv"), dtype=str).study_id)
    c = set(pd.read_csv(os.path.join(OUT, "train_ids_10000.csv"), dtype=str).study_id)
    print(f"overlap(2553, 10000) = {len(a & c)} ids  (NESTED by construction: RandomState(0).choice(replace=False) takes the first n of one fixed permutation)")


if __name__ == "__main__":
    main()
