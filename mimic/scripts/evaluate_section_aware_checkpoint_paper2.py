#!/usr/bin/env python3
"""
Re-evaluate the MG-G2L (section-aware) checkpoint from job 113710 using the
FIXED, section-aware evaluate_recall_k_batched() in train_retrieval_v2_paper2.py.

Inference only -- no training, no checkpoint modification. Loads the
already-saved checkpoint_resume.pth (epoch-50 weights + optimizer state) and
runs the corrected validation logic against:
  1. the val set (29,181 samples, section_boundaries_val_paper2.csv)
  2. the test set (12,429 samples, section_boundaries_test_paper2.csv --
     the same test set used for the Paper 1 baseline comparison)

Reports Recall@1/5/10 and MRR for both, and compares the test numbers
directly against the Paper 1 baseline (R@1 0.9928, R@5 0.9987, R@10 0.9995,
MRR 0.9955).
"""

import os
import sys

import torch
from torch.utils.data import DataLoader as TorchDataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()

from base_models_refactored_v1_paper2 import MultimodalFusion
from data_loader_v1_paper2 import IndianaDataLoader
from train_retrieval_v2_paper2 import EnhancedRetrievalTrainer
import config

EXPERIMENT_NAME = "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_42"
CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR, "saved_models", EXPERIMENT_NAME, "export", "checkpoint_resume.pth"
)

PAPER1_BASELINE = {"recall@1": 0.9928, "recall@5": 0.9987, "recall@10": 0.9995, "mrr": 0.9955}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    print(f"Loading checkpoint (inference only, no retraining): {CHECKPOINT_PATH}")

    vocab_size = config.get_vocab_size()
    embed_dim = config.get_embed_dim()
    cfg = config.get_current_config()
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]

    model = MultimodalFusion(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print(f"Checkpoint epoch field: {checkpoint['epoch']} (0-indexed -> epoch {checkpoint['epoch'] + 1} of training)")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"load_state_dict: missing={missing}  unexpected={unexpected}")
    model.eval()

    trainer = EnhancedRetrievalTrainer(
        model=model,
        learning_rate=config.get_default_learning_rate(),
        device=device,
        experiment_name=EXPERIMENT_NAME,
        model_save_path=None,  # never used -- no saving in this script
        viz_dir="/tmp/paper2_eval_section_aware_viz",
    )
    # _ensure_model_built() inside the trainer constructor re-inits nothing (model
    # weights already loaded above); it only runs one dummy no_grad forward pass.

    def build_dataset(split, num_samples):
        print(f"\nLoading {split} data (section_boundaries_{split}_paper2.csv, {num_samples} samples expected)...")
        data_loader = IndianaDataLoader(
            batch_size=config.get_default_batch_size(), use_shards=True,
            shard_size=cfg["shard_size"], shard_subfolder=config.DATASET_MODE,
        )
        data_loader.load_data(max_samples=None, skip_processing=True)
        if split == "val":
            dataset = data_loader.get_validation_data(num_samples=None)
        elif split == "test":
            dataset = data_loader.get_test_data(num_samples=None)
        else:
            raise ValueError(split)
        print(f"{split} dataset size: {len(dataset)}")
        print(f"Section boundaries loaded: {len(dataset.section_boundaries)}  fallback: {dataset.fallback_count}")
        return dataset

    val_dataset = build_dataset("val", 29181)
    test_dataset = build_dataset("test", 12429)

    val_loader = TorchDataLoader(val_dataset, batch_size=config.get_default_batch_size(), shuffle=False, num_workers=0)
    test_loader = TorchDataLoader(test_dataset, batch_size=config.get_default_batch_size(), shuffle=False, num_workers=0)

    print("\n" + "=" * 90)
    print("VAL SET -- section-aware (FIXED) evaluate_recall_k_batched()")
    print("=" * 90)
    val_recalls = trainer.evaluate_recall_k_batched(val_loader, k=[1, 5, 10])
    for k, v in val_recalls.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 90)
    print("TEST SET -- section-aware (FIXED) evaluate_recall_k_batched()")
    print("=" * 90)
    test_recalls = trainer.evaluate_recall_k_batched(test_loader, k=[1, 5, 10])
    for k, v in test_recalls.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 90)
    print("COMPARISON: MG-G2L (section-aware eval) TEST set vs Paper 1 baseline")
    print("=" * 90)
    header = f"{'metric':<12} | {'Paper 1 baseline':<18} | {'MG-G2L (section-aware)':<24} | {'delta':<10}"
    print(header)
    print("-" * len(header))
    for key, label in [("recall@1", "recall@1"), ("recall@5", "recall@5"), ("recall@10", "recall@10"), ("mrr", "mrr")]:
        baseline_val = PAPER1_BASELINE[key]
        new_val = test_recalls[key]
        delta = new_val - baseline_val
        print(f"{label:<12} | {baseline_val:<18.4f} | {new_val:<24.4f} | {delta:+.4f}")


if __name__ == "__main__":
    main()
