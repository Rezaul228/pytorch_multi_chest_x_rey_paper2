#!/usr/bin/env python3
"""
Run the corrected, section-aware evaluate_cross_modal_retrieval_streaming()
(train_test_cross_modal_evaluation_v1_paper2.py) on the 3 newly-trained
MG-G2L checkpoints (seeds 17, 123, 3407), each against the same
12,429-sample test set, via data_loader_v1_paper2 (section fields attached).

Inference only -- no training, no checkpoint modification.
"""

import os
import sys
import pickle
import json

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from train_test_cross_modal_evaluation_v1_paper2 import evaluate_cross_modal_retrieval_streaming
from base_models_refactored_v1_paper2 import MultimodalFusion
from data_loader_v1_paper2 import IndianaDataLoader
import paths
import config

SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = 32
SEEDS = [17, 123, 3407]

CHECKPOINT_PATHS = {
    seed: os.path.join(
        PROJECT_DIR, "saved_models",
        f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_{seed}",
        "export", "checkpoint_resume.pth",
    )
    for seed in SEEDS
}

OUTPUT_JSON_PATH = os.path.join(PROJECT_DIR, "mgg2l_multiseed_streaming_results.json")


def load_tokenizer_from_metadata():
    metadata_path = paths.get_metadata_path(SHARD_SUBFOLDER)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get("tokenizer")
    if hasattr(tokenizer, "word_index") and not hasattr(tokenizer, "word2idx"):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    return tokenizer


def build_test_dataset():
    dl = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = load_tokenizer_from_metadata()
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"Test dataset size: {len(test_dataset)}  "
          f"section_boundaries: {len(test_dataset.section_boundaries)}  fallback: {test_dataset.fallback_count}")
    return test_dataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = config.get_vocab_size()
    embed_dim = config.get_embed_dim()
    cfg = config.get_current_config()
    num_heads = cfg["num_heads"]
    num_layers = cfg["num_layers"]

    all_results = {}

    for seed in SEEDS:
        ckpt_path = CHECKPOINT_PATHS[seed]
        print("\n" + "=" * 90)
        print(f"MG-G2L seed_{seed}: {ckpt_path}")
        print("=" * 90)
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        model = MultimodalFusion(
            vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
        ).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f"Checkpoint epoch field: {checkpoint['epoch']} (0-indexed -> epoch {checkpoint['epoch'] + 1})")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"load_state_dict: missing={missing}  unexpected={unexpected}")
        model.eval()

        test_dataset = build_test_dataset()
        results = evaluate_cross_modal_retrieval_streaming(
            model=model, test_dataset=test_dataset, k_values=[1, 5, 10], batch_size=BATCH_SIZE,
        )
        all_results[str(seed)] = results

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to: {OUTPUT_JSON_PATH}")

    print("\n" + "=" * 100)
    print("SUMMARY: MG-G2L corrected (section-aware) streaming eval, seeds 17/123/3407")
    print("=" * 100)
    header = f"{'seed':<8} | {'I2T R@1':<9} | {'I2T R@5':<9} | {'I2T R@10':<9} | {'I2T MRR':<9} | {'T2I R@1':<9} | {'T2I R@5':<9} | {'T2I R@10':<9} | {'T2I MRR':<9}"
    print(header)
    print("-" * len(header))
    for seed in SEEDS:
        r = all_results[str(seed)]
        print(f"{seed:<8} | {r['i2t_recall@1']:<9.4f} | {r['i2t_recall@5']:<9.4f} | {r['i2t_recall@10']:<9.4f} | "
              f"{r['i2t_mrr']:<9.4f} | {r['t2i_recall@1']:<9.4f} | {r['t2i_recall@5']:<9.4f} | {r['t2i_recall@10']:<9.4f} | {r['t2i_mrr']:<9.4f}")


if __name__ == "__main__":
    main()
