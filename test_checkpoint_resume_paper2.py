#!/usr/bin/env python3
"""
PROCESS 2 of the checkpoint save/resume verification (paper2). A genuinely
SEPARATE process from test_checkpoint_save_paper2.py -- shares no Python
objects/memory with it, only the checkpoint file on disk and a comparison
snapshot.

Instantiates a brand-new (freshly-initialized) trainer pointing at the same
experiment_name, calls load_checkpoint_if_exists() directly, and confirms:
  - it reports "Resuming from epoch N" with the correct N (3)
  - the reloaded model parameters are torch.equal to what process 1 saved
  - exactly ONE checkpoint-related file exists on disk (no duplicates)
"""

import os
import sys

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from base_models_refactored_v1_paper2 import MultimodalFusion
from train_retrieval_v2_paper2 import EnhancedRetrievalTrainer
import config

EXPERIMENT_NAME = "checkpoint_resume_test_paper2"
SNAPSHOT_PATH = "/tmp/checkpoint_resume_verify_snapshot.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    snapshot = torch.load(SNAPSHOT_PATH, map_location='cpu', weights_only=False)
    expected_resume_epoch = snapshot["expected_resume_epoch"]
    checkpoint_path = snapshot["checkpoint_path"]
    model_save_path = snapshot["model_save_path"]
    saved_model_state = snapshot["model_state_dict"]

    print(f"Expecting checkpoint at: {checkpoint_path}")
    print(f"Expected resume epoch: {expected_resume_epoch}")

    # Brand-new, freshly-initialized model/trainer -- NOT the same object as process 1.
    torch.manual_seed(999)
    fresh_model = MultimodalFusion(
        vocab_size=config.get_vocab_size(), embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    ).to(device)

    trainer = EnhancedRetrievalTrainer(
        model=fresh_model, learning_rate=config.get_default_learning_rate(), device=device,
        experiment_name=EXPERIMENT_NAME, model_save_path=model_save_path,
        viz_dir="/tmp/paper2_checkpoint_test_viz_resume",
    )

    # Sanity: BEFORE loading, the fresh model's weights should NOT already match
    # the saved ones (different random init / seed) -- proves the later match
    # is due to loading, not coincidence.
    pre_load_state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    any_pre_match = all(
        torch.equal(pre_load_state[k], saved_model_state[k]) for k in saved_model_state
    )
    print(f"\nSanity check -- fresh model matches saved model BEFORE loading (should be False): {any_pre_match}")

    print("\nCalling load_checkpoint_if_exists()...")
    resume_epoch = trainer.load_checkpoint_if_exists()

    print(f"\nReturned resume_epoch: {resume_epoch}")
    epoch_correct = (resume_epoch == expected_resume_epoch)
    print(f"Matches expected ({expected_resume_epoch}): {epoch_correct}")

    print("\nComparing reloaded model parameters against process 1's saved snapshot (torch.equal)...")
    loaded_state = trainer.model.state_dict()
    all_match = True
    mismatches = []
    for key in saved_model_state:
        match = torch.equal(loaded_state[key].detach().cpu(), saved_model_state[key])
        if not match:
            all_match = False
            mismatches.append(key)
    print(f"All {len(saved_model_state)} parameter tensors match exactly: {all_match}")
    if mismatches:
        print(f"Mismatched keys: {mismatches}")

    print("\nChecking export directory for exactly one checkpoint-related file...")
    export_dir = os.path.dirname(checkpoint_path)
    files_in_export_dir = sorted(os.listdir(export_dir))
    print(f"Files in export dir: {files_in_export_dir}")
    checkpoint_files = [f for f in files_in_export_dir if 'checkpoint' in f.lower()]
    single_checkpoint_file = len(checkpoint_files) == 1
    print(f"Checkpoint-related files found: {checkpoint_files}")
    print(f"Exactly one checkpoint file (no duplicates/versioned copies): {single_checkpoint_file}")

    print("\n" + "=" * 90)
    print("FINAL VERIFICATION RESULT")
    print("=" * 90)
    overall_pass = epoch_correct and all_match and single_checkpoint_file
    print(f"  Correct resume epoch reported: {epoch_correct}")
    print(f"  Model parameters exactly match (torch.equal): {all_match}")
    print(f"  Exactly one checkpoint file on disk: {single_checkpoint_file}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
