#!/usr/bin/env python3
"""
Quick check comparing MGG2L_paper2_seed_42_rerun against the original
MGG2L_paper2_seed_42:
  1. Run the official, corrected evaluate_cross_modal_retrieval_streaming()
     on the rerun's checkpoint (epoch 50), same test set/method as before.
  2. Direct tensor-level comparison of both checkpoints' state_dicts
     (torch.equal per parameter; max abs diff if not exact).

Inference only -- no training, no checkpoint modification, no
graded-relevance/divergence pipeline.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()

from train_test_cross_modal_evaluation_v1_paper2 import evaluate_cross_modal_retrieval_streaming
from base_models_refactored_v1_paper2 import MultimodalFusion
from data_loader_v1_paper2 import IndianaDataLoader
import config
import pickle

SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = 32

ORIGINAL_CKPT = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_42",
    "export", "checkpoint_resume.pth",
)
RERUN_CKPT = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_42_rerun",
    "export", "checkpoint_resume.pth",
)

# Official numbers already established for the original seed_42 (job 113714/113715).
ORIGINAL_RESULTS = {
    "i2t_recall@1": 0.9994, "i2t_recall@5": 1.0000, "i2t_recall@10": 1.0000, "i2t_mrr": 0.9996,
    "t2i_recall@1": 1.0000, "t2i_recall@5": 1.0000, "t2i_recall@10": 1.0000, "t2i_mrr": 1.0000,
}


def load_tokenizer_from_metadata():
    metadata_path = paths.get_metadata_path(SHARD_SUBFOLDER)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get("tokenizer")
    if hasattr(tokenizer, "word_index") and not hasattr(tokenizer, "word2idx"):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    return tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = config.get_vocab_size()
    embed_dim = config.get_embed_dim()
    cfg = config.get_current_config()
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]

    # ---------- Part 1: official streaming eval on the rerun checkpoint ----------
    print("\n" + "=" * 90)
    print("PART 1: Official evaluate_cross_modal_retrieval_streaming() on seed_42_rerun")
    print("=" * 90)

    rerun_model = MultimodalFusion(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    ).to(device)
    rerun_checkpoint = torch.load(RERUN_CKPT, map_location=device, weights_only=False)
    print(f"Checkpoint epoch field: {rerun_checkpoint['epoch']} (-> epoch {rerun_checkpoint['epoch'] + 1})")
    missing, unexpected = rerun_model.load_state_dict(rerun_checkpoint["model_state_dict"], strict=True)
    print(f"load_state_dict: missing={missing}  unexpected={unexpected}")
    rerun_model.eval()

    dl = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = load_tokenizer_from_metadata()
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"Test dataset size: {len(test_dataset)}  fallback: {test_dataset.fallback_count}")

    rerun_results = evaluate_cross_modal_retrieval_streaming(
        model=rerun_model, test_dataset=test_dataset, k_values=[1, 5, 10], batch_size=BATCH_SIZE,
    )

    del rerun_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("COMPARISON: original seed_42 vs seed_42_rerun (both via evaluate_cross_modal_retrieval_streaming)")
    print("=" * 100)
    header = f"{'metric':<14} | {'original':<10} | {'rerun':<10} | {'delta':<10}"
    print(header)
    print("-" * len(header))
    for key in ["i2t_recall@1", "i2t_recall@5", "i2t_recall@10", "i2t_mrr",
                "t2i_recall@1", "t2i_recall@5", "t2i_recall@10", "t2i_mrr"]:
        orig = ORIGINAL_RESULTS[key]
        rerun_val = rerun_results[key]
        print(f"{key:<14} | {orig:<10.4f} | {rerun_val:<10.4f} | {rerun_val - orig:<+10.4f}")

    # ---------- Part 2: tensor-level state_dict comparison ----------
    print("\n" + "=" * 90)
    print("PART 2: Direct tensor comparison, original seed_42 vs seed_42_rerun state_dicts")
    print("=" * 90)

    original_checkpoint = torch.load(ORIGINAL_CKPT, map_location="cpu", weights_only=False)
    original_state = original_checkpoint["model_state_dict"]
    rerun_state = rerun_checkpoint["model_state_dict"]

    assert set(original_state.keys()) == set(rerun_state.keys()), "state_dict key sets differ!"

    all_identical = True
    max_diff_overall = 0.0
    max_diff_param = None
    per_param_report = []

    for key in original_state:
        t1 = original_state[key].to(torch.float64)
        t2 = rerun_state[key].to(torch.float64)
        identical = torch.equal(original_state[key], rerun_state[key])
        diff = (t1 - t2).abs().max().item()
        if not identical:
            all_identical = False
        if diff > max_diff_overall:
            max_diff_overall = diff
            max_diff_param = key
        per_param_report.append((key, identical, diff))

    n_identical = sum(1 for _, ident, _ in per_param_report if ident)
    n_total = len(per_param_report)

    print(f"Total parameter tensors: {n_total}")
    print(f"Bit-identical tensors: {n_identical}")
    print(f"Non-identical tensors: {n_total - n_identical}")
    print(f"\nALL PARAMETERS BIT-IDENTICAL (torch.equal): {all_identical}")
    print(f"Max absolute difference across ALL parameters: {max_diff_overall:.10e}")
    print(f"  (in parameter: {max_diff_param})")

    if not all_identical:
        print("\nTop 10 most-different parameter tensors:")
        for key, ident, diff in sorted(per_param_report, key=lambda x: -x[2])[:10]:
            print(f"  {key}: max_diff={diff:.6e}  identical={ident}")


if __name__ == "__main__":
    main()
