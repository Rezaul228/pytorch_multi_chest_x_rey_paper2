#!/usr/bin/env python3
"""
Test for the new section-aware IndianaDataset/IndianaDataLoader in
data_loader_v1_paper2.py (a copy of data_loader_v1.py -- the original is
untouched). Does NOT run any real training. Pulls 5 batches from the TRAIN
split at the real training batch size (256) and checks:
  - batch size and shapes of the new findings_token_count/has_find/has_imp
    tensors match the batch size
  - how many of the 5 batches had any fallback (study_id not found in the
    section-boundary dict) cases
  - for 2 study_ids actually present in a pulled batch, the values coming
    out of the DataLoader exactly match section_boundaries_train_paper2.csv
"""

import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from data_loader_v1_paper2 import IndianaDataLoader
import config

N_BATCHES = 5
BATCH_SIZE = config.get_default_batch_size()  # should be 256 for mimic_shards_hybrid_full_ori


def main():
    print(f"Config batch size: {BATCH_SIZE}")
    print(f"Dataset mode: {config.DATASET_MODE}")

    data_loader = IndianaDataLoader(
        batch_size=BATCH_SIZE,
        use_shards=True,
        shard_size=config.get_current_config()["shard_size"],
        shard_subfolder=config.DATASET_MODE,
    )
    # NOTE: capped at 2000 samples (comfortably more than N_BATCHES*BATCH_SIZE=1280)
    # instead of loading the full ~155,745-sample train split, to avoid loading
    # ~90GB of images into memory just for this lightweight, no-training test.
    max_samples_for_test = 2000
    data_loader.load_data(max_samples=max_samples_for_test, skip_processing=True)
    train_dataset = data_loader.get_data(max_samples=max_samples_for_test)

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Inferred split_name: {train_dataset.split_name}")
    print(f"Section boundaries loaded from: {train_dataset.section_boundaries_path}")
    print(f"Section boundaries dict size: {len(train_dataset.section_boundaries)}")

    train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("\n" + "=" * 90)
    print(f"Pulling {N_BATCHES} batches (batch_size={BATCH_SIZE})")
    print("=" * 90)

    batches_with_fallback = 0
    first_batch_study_ids = None
    first_batch_findings_tc = None
    first_batch_has_find = None
    first_batch_has_imp = None

    prev_fallback_count = train_dataset.fallback_count

    for i, batch in enumerate(train_loader):
        if i >= N_BATCHES:
            break

        images = batch["images"]
        captions = batch["captions"]
        study_ids = batch["study_ids"]
        findings_token_count = batch["findings_token_count"]
        has_find = batch["has_find"]
        has_imp = batch["has_imp"]

        cur_fallback_count = train_dataset.fallback_count
        batch_fallbacks = cur_fallback_count - prev_fallback_count
        prev_fallback_count = cur_fallback_count
        had_fallback = batch_fallbacks > 0
        batches_with_fallback += int(had_fallback)

        print(f"\nBatch {i}:")
        print(f"  batch size (images): {images.shape[0]}")
        print(f"  captions shape (token_ids -- same tensor, not duplicated): {tuple(captions.shape)}")
        print(f"  findings_token_count shape: {tuple(findings_token_count.shape)}")
        print(f"  has_find shape: {tuple(has_find.shape)}")
        print(f"  has_imp shape: {tuple(has_imp.shape)}")
        print(f"  fallback cases in this batch: {batch_fallbacks}  (had_fallback={had_fallback})")

        if i == 0:
            first_batch_study_ids = study_ids
            first_batch_findings_tc = findings_token_count
            first_batch_has_find = has_find
            first_batch_has_imp = has_imp

    print("\n" + "=" * 90)
    print(f"Batches with ANY fallback (of {N_BATCHES}): {batches_with_fallback}")
    print(f"Total cumulative fallback_count on dataset object: {train_dataset.fallback_count}")
    print("=" * 90)

    print("\nChecking for known study_ids 53957785 / 50225296 (these are TEST-split study_ids "
          "-- train/val/test are disjoint splits, so they will NOT appear in the train batches; "
          "picking 2 arbitrary study_ids actually present in batch 0 instead).")

    known_ids = {"53957785", "50225296"}
    present_known = [sid for sid in first_batch_study_ids if str(sid) in known_ids]
    print(f"  Known test study_ids found in batch 0: {present_known if present_known else 'none (as expected)'}")

    boundaries_df = pd.read_csv(
        os.path.join(PROJECT_DIR, "section_boundaries_train_paper2.csv"), dtype={"study_id": str}
    ).set_index("study_id")

    print("\n" + "=" * 90)
    print("Cross-check: 2 study_ids from batch 0 vs section_boundaries_train_paper2.csv")
    print("=" * 90)
    check_indices = [0, 1]
    all_match = True
    for idx in check_indices:
        sid = str(first_batch_study_ids[idx])
        dl_fc = int(first_batch_findings_tc[idx].item())
        dl_hf = bool(first_batch_has_find[idx].item())
        dl_hi = bool(first_batch_has_imp[idx].item())

        csv_row = boundaries_df.loc[sid]
        csv_fc = int(csv_row["findings_token_count"])
        csv_hf = bool(csv_row["has_findings"])
        csv_hi = bool(csv_row["has_impression"])

        match = (dl_fc == csv_fc) and (dl_hf == csv_hf) and (dl_hi == csv_hi)
        all_match = all_match and match
        print(f"  study_id={sid}")
        print(f"    DataLoader: findings_token_count={dl_fc}  has_find={dl_hf}  has_imp={dl_hi}")
        print(f"    CSV:        findings_token_count={csv_fc}  has_find={csv_hf}  has_imp={csv_hi}")
        print(f"    EXACT MATCH: {match}")

    print(f"\nAll checked study_ids match exactly: {all_match}")


if __name__ == "__main__":
    main()
