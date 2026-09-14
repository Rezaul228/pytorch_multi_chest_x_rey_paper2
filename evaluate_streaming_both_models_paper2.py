#!/usr/bin/env python3
"""
Run the SAME (now section-aware-capable) evaluate_cross_modal_retrieval_streaming()
from train_test_cross_modal_evaluation_v1_paper2.py on BOTH checkpoints, on the
same 12,429-sample test set, so both numbers come from one identical function.

Inference only -- no training, no checkpoint modification.

- MG-G2L (checkpoint_resume.pth, epoch 50): test set built via
  data_loader_v1_paper2.IndianaDataLoader, whose batches carry
  findings_token_count/has_find/has_imp -> the function's new section-aware
  branch fires automatically.
- Paper 1 baseline (model_weights.pth, original seed_42): test set built via
  the ORIGINAL data_loader_v1.IndianaDataLoader, whose batches never carry
  those fields -> the function's original (backward-compatible) branch fires
  automatically. This also re-confirms the Paper 1 baseline number using
  this exact paper2 file copy of the eval function.
"""

import os
import sys
import pickle

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from train_test_cross_modal_evaluation_v1_paper2 import evaluate_cross_modal_retrieval_streaming
import paths
import config

MGG2L_EXPERIMENT_NAME = "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_42"
MGG2L_CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR, "saved_models", MGG2L_EXPERIMENT_NAME, "export", "checkpoint_resume.pth"
)
PAPER1_MODEL_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42",
    "export", "model_weights.pth",
)
SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = 32  # matches test_full_evaluation.py exactly, for a fair re-confirmation


def load_tokenizer_from_metadata():
    metadata_path = paths.get_metadata_path(SHARD_SUBFOLDER)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get('tokenizer')
    if hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    return tokenizer


def build_test_dataset_paper2():
    from data_loader_v1_paper2 import IndianaDataLoader as IndianaDataLoaderPaper2
    dl = IndianaDataLoaderPaper2(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = load_tokenizer_from_metadata()
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"[paper2 loader] test dataset size: {len(test_dataset)}  "
          f"section_boundaries loaded: {len(test_dataset.section_boundaries)}  fallback: {test_dataset.fallback_count}")
    return test_dataset


def build_test_dataset_original():
    from data_loader_v1 import IndianaDataLoader as IndianaDataLoaderOriginal
    dl = IndianaDataLoaderOriginal(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = load_tokenizer_from_metadata()
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"[original loader] test dataset size: {len(test_dataset)}  "
          f"(no section fields -- original IndianaDataset)")
    return test_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = config.get_vocab_size()
    embed_dim = config.get_embed_dim()
    cfg = config.get_current_config()
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]

    # ---------- MG-G2L (section-aware path) ----------
    print("\n" + "=" * 90)
    print("MG-G2L checkpoint (epoch 50) -- section-aware path (via data_loader_v1_paper2)")
    print("=" * 90)
    from base_models_refactored_v1_paper2 import MultimodalFusion as MultimodalFusionPaper2

    mgg2l_model = MultimodalFusionPaper2(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    ).to(device)
    checkpoint = torch.load(MGG2L_CHECKPOINT_PATH, map_location=device, weights_only=False)
    print(f"Checkpoint epoch field: {checkpoint['epoch']} (0-indexed -> epoch {checkpoint['epoch'] + 1})")
    missing, unexpected = mgg2l_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"load_state_dict: missing={missing}  unexpected={unexpected}")
    mgg2l_model.eval()

    mgg2l_test_dataset = build_test_dataset_paper2()
    mgg2l_results = evaluate_cross_modal_retrieval_streaming(
        model=mgg2l_model, test_dataset=mgg2l_test_dataset, k_values=[1, 5, 10], batch_size=BATCH_SIZE,
    )

    # ---------- Paper 1 baseline (original, backward-compatible path) ----------
    print("\n" + "=" * 90)
    print("Paper 1 baseline checkpoint (original seed_42) -- backward-compatible path (via ORIGINAL data_loader_v1)")
    print("=" * 90)
    from base_models_refactored_v1 import MultimodalFusion as MultimodalFusionOriginal

    paper1_model = MultimodalFusionOriginal(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    ).to(device)
    state_dict = torch.load(PAPER1_MODEL_PATH, map_location=device, weights_only=True)
    missing, unexpected = paper1_model.load_state_dict(state_dict, strict=True)
    print(f"load_state_dict: missing={missing}  unexpected={unexpected}")
    paper1_model.eval()

    paper1_test_dataset = build_test_dataset_original()
    paper1_results = evaluate_cross_modal_retrieval_streaming(
        model=paper1_model, test_dataset=paper1_test_dataset, k_values=[1, 5, 10], batch_size=BATCH_SIZE,
    )

    # ---------- Side-by-side report ----------
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE: Paper 1 baseline vs MG-G2L, both via the IDENTICAL evaluate_cross_modal_retrieval_streaming()")
    print("=" * 100)
    rows = [
        ("I2T Recall@1", "i2t_recall@1"), ("I2T Recall@5", "i2t_recall@5"), ("I2T Recall@10", "i2t_recall@10"),
        ("I2T MRR", "i2t_mrr"),
        ("T2I Recall@1", "t2i_recall@1"), ("T2I Recall@5", "t2i_recall@5"), ("T2I Recall@10", "t2i_recall@10"),
        ("T2I MRR", "t2i_mrr"),
        ("Avg Recall@1", "avg_recall@1"), ("Avg Recall@5", "avg_recall@5"), ("Avg Recall@10", "avg_recall@10"),
        ("Avg MRR", "avg_mrr"),
    ]
    header = f"{'Metric':<16} | {'Paper 1 baseline':<18} | {'MG-G2L':<18} | {'delta':<10}"
    print(header)
    print("-" * len(header))
    for label, key in rows:
        p1 = paper1_results.get(key, float('nan'))
        mg = mgg2l_results.get(key, float('nan'))
        delta = mg - p1
        print(f"{label:<16} | {p1:<18.4f} | {mg:<18.4f} | {delta:+.4f}")

    print("\n" + "=" * 100)
    print("REPRODUCTION CHECK: does the Paper 1 number here match test_full_evaluation.py's earlier result?")
    print("=" * 100)
    earlier_i2t_r1 = 0.9928
    reproduced_i2t_r1 = paper1_results.get("i2t_recall@1", float('nan'))
    print(f"  Earlier (test_full_evaluation.py): I2T R@1 = {earlier_i2t_r1:.4f}")
    print(f"  Reproduced here (paper2 file, same underlying function): I2T R@1 = {reproduced_i2t_r1:.4f}")
    print(f"  Match: {abs(reproduced_i2t_r1 - earlier_i2t_r1) < 1e-4}")


if __name__ == "__main__":
    main()
