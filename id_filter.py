#!/usr/bin/env python3
"""
Optional study_id filter for the shard datasets (--train_ids / --val_ids).

Added 2026-09-19 for the MIMIC data-scale ablation. DEFAULT OFF: when no id file
is given every code path that existed before is executed unchanged. Kept
dependency-free (csv only) so it unit-tests in milliseconds.
"""
import csv


def load_id_filter(csv_path):
    """Read a CSV with a `study_id` column -> frozenset of str ids.
    Raises on a missing column, an empty file, or duplicate ids (a duplicated id
    would make 'loader kept exactly n samples' checks silently wrong)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "study_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: expected a 'study_id' column, got {reader.fieldnames}")
        ids = [str(row["study_id"]).strip() for row in reader]
    if not ids:
        raise ValueError(f"{csv_path}: no study_ids")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{csv_path}: {len(ids) - len(set(ids))} duplicate study_ids")
    return frozenset(ids)


def keep_sample(study_id, keep_ids):
    """True iff the sample should be kept. keep_ids=None means 'no filter'."""
    return keep_ids is None or str(study_id) in keep_ids


def assert_filter_satisfied(dataset_len, keep_ids, split_name):
    """After loading: the dataset must hold exactly the listed ids (study_ids are
    unique per split), otherwise some listed ids do not exist in that split."""
    if keep_ids is not None and dataset_len != len(keep_ids):
        raise RuntimeError(
            f"ID FILTER MISMATCH for split='{split_name}': id file lists {len(keep_ids)} study_ids "
            f"but the loader kept {dataset_len} samples. Ids from another split, or max_samples "
            f"cut the scan short (pass a sentinel larger than the split)."
        )
