#!/usr/bin/env python3
"""
PROCESS 1 of the checkpoint save/resume verification (paper2).
Runs 3 real training steps, then saves a checkpoint AS IF this were epoch 3
(epoch index 2, satisfying (epoch+1) % 3 == 0), using
EnhancedRetrievalTrainer.save_checkpoint() from train_retrieval_v2_paper2.py.
Also writes a comparison snapshot to disk so a SEPARATE process
(test_checkpoint_resume_paper2.py) can verify the round-trip without
sharing any Python objects/memory with this process.
"""

import os
import sys

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from base_models_refactored_v1_paper2 import MultimodalFusion
from data_loader_v1_paper2 import IndianaDataLoader
from train_retrieval_v2_paper2 import EnhancedRetrievalTrainer
import config
from torch.utils.data import DataLoader as TorchDataLoader

EXPERIMENT_NAME = "checkpoint_resume_test_paper2"
SNAPSHOT_PATH = "/tmp/checkpoint_resume_verify_snapshot.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(123)

    batch_size = config.get_default_batch_size()
    learning_rate = config.get_default_learning_rate()

    print("Loading a small slice of real TRAIN data...")
    data_loader = IndianaDataLoader(
        batch_size=batch_size, use_shards=True,
        shard_size=config.get_current_config()["shard_size"],
        shard_subfolder=config.DATASET_MODE,
    )
    data_loader.load_data(max_samples=2000, skip_processing=True)
    train_dataset = data_loader.get_data(max_samples=2000)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    fusion_model = MultimodalFusion(
        vocab_size=config.get_vocab_size(), embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    ).to(device)

    model_save_path = os.path.join(PROJECT_DIR, "saved_models", EXPERIMENT_NAME, f"model_{EXPERIMENT_NAME}.pth")

    trainer = EnhancedRetrievalTrainer(
        model=fusion_model, learning_rate=learning_rate, device=device,
        experiment_name=EXPERIMENT_NAME, model_save_path=model_save_path,
        viz_dir="/tmp/paper2_checkpoint_test_viz",
    )

    print("\nRunning 3 real training steps...")
    step_count = 0
    for batch in train_loader:
        if step_count >= 3:
            break
        step_count += 1
        step_batch = {
            "images": batch["images"].to(device),
            "captions": batch["captions"].to(device),
            "findings_token_count": batch["findings_token_count"].to(device),
            "has_find": batch["has_find"].to(device),
            "has_imp": batch["has_imp"].to(device),
        }
        total_loss, *_ = trainer.train_step(step_batch)
        print(f"  step {step_count}: total_loss={total_loss:.6f}")

    print(f"\nSaving checkpoint AS IF this were epoch 3 (epoch index 2, satisfies (epoch+1)%3==0)...")
    trainer.save_checkpoint(2)

    checkpoint_path = trainer.get_resume_checkpoint_path()
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint exists on disk: {os.path.exists(checkpoint_path)}")

    export_dir = os.path.dirname(checkpoint_path)
    files_in_export_dir = sorted(os.listdir(export_dir))
    print(f"Files in export dir after save: {files_in_export_dir}")

    # Snapshot for the separate resume-verification process.
    snapshot = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()},
        "expected_resume_epoch": 3,  # saved_epoch=2 -> resume_epoch=2+1=3
        "checkpoint_path": checkpoint_path,
        "model_save_path": model_save_path,
    }
    torch.save(snapshot, SNAPSHOT_PATH)
    print(f"\nSnapshot for cross-process verification saved to: {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
