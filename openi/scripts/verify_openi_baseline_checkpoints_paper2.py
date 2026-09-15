#!/usr/bin/env python3
"""
Checksum + clean-load + Open-I test-set evaluation of the 5 Paper-1-baseline
aug_indiana_extended (Open-I) checkpoints (seeds 17/42/123/2021/3407), which
live in THIS project's saved_models/indiana_trained/.

Does NOT retrain and does NOT modify any model file. Imports the ORIGINAL
(non-paper2) MultimodalFusion / data_loader / evaluation code as-is.
"""

import os
import sys
import hashlib
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()

from base_models_refactored_v1 import MultimodalFusion  # noqa: E402
from data_loader_v1 import IndianaDataLoader  # noqa: E402
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming  # noqa: E402

SEEDS = [17, 42, 123, 2021, 3407]
DIRNAME_TMPL = (
    "aug_indiana_extended_17946_vo10870_to128_lr1e-4_weight_d_1e-5_b128_ep50_"
    "dualbr_sy65_main_loss20_ortho15__branch_seed_{seed}_v1"
)
CKPT_TMPL = os.path.join(
    PROJECT_DIR, "saved_models", "indiana_trained", DIRNAME_TMPL, "export", "model_weights.pth"
)

EXPECTED_VOCAB_SIZE = 10870
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2

REFERENCE_JOB_10227_SEED42_VAL_RECALL_AT_1 = 0.9830


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ckpt_paths = {s: CKPT_TMPL.format(seed=s) for s in SEEDS}

    print("=" * 70)
    print("STEP 1: SHA-256 checksums")
    print("=" * 70)
    checksums = {}
    for s in SEEDS:
        p = ckpt_paths[s]
        if not os.path.exists(p):
            raise FileNotFoundError(f"seed_{s}: checkpoint not found at {p}")
        h = sha256sum(p)
        checksums[s] = h
        print(f"seed_{s:>5}: {h}")

    print()
    seen = {}
    duplicate_seeds = set()
    for s in SEEDS:
        h = checksums[s]
        if h in seen:
            print(f"DUPLICATE: seed_{s} == seed_{seen[h]} (identical SHA-256)")
            duplicate_seeds.add(s)
            duplicate_seeds.add(seen[h])
        else:
            seen[h] = s
    if not duplicate_seeds:
        print("No duplicates: all 5 checksums are pairwise distinct.")

    distinct_seeds = [s for s in SEEDS if s not in duplicate_seeds]

    print()
    print("=" * 70)
    print("STEP 2: strict=True load into ORIGINAL MultimodalFusion")
    print(f"(vocab_size={EXPECTED_VOCAB_SIZE}, embed_dim={EMBED_DIM}, "
          f"num_heads={NUM_HEADS}, num_layers={NUM_LAYERS})")
    print("=" * 70)

    cleanly_loaded = {}
    for s in distinct_seeds:
        model = MultimodalFusion(
            vocab_size=EXPECTED_VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
        )
        state_dict = torch.load(ckpt_paths[s], map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"seed_{s:>5}: strict=True load OK -> 0 missing, 0 unexpected keys")
            model.eval()
            cleanly_loaded[s] = model
        except RuntimeError as e:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"seed_{s:>5}: STRICT LOAD FAILED")
            print(f"    missing_keys={missing}")
            print(f"    unexpected_keys={unexpected}")
            print(f"    error: {e}")

    print()
    print("=" * 70)
    print("STEP 3: evaluate_cross_modal_retrieval_streaming on Open-I TEST set")
    print("=" * 70)

    data_loader = IndianaDataLoader(
        batch_size=64, use_shards=True, shard_subfolder="aug_indiana_extended"
    )
    data_loader.load_data(max_samples=None, skip_processing=True)
    actual_vocab_size = len(data_loader.tokenizer.word2idx) + 1
    print(f"Tokenizer vocab_size (incl. pad) = {actual_vocab_size} "
          f"(expected {EXPECTED_VOCAB_SIZE})")
    assert actual_vocab_size == EXPECTED_VOCAB_SIZE, "vocab size mismatch vs config"

    test_dataset = data_loader.get_test_data(num_samples=None)
    print(f"Open-I test set size: {len(test_dataset)} samples (expected 5313)")

    all_results = {}
    for s, model in cleanly_loaded.items():
        print(f"\n--- seed_{s} ---")
        results = evaluate_cross_modal_retrieval_streaming(
            model=model,
            test_dataset=test_dataset,
            k_values=[1, 5, 10],
            batch_size=64,
            visualize=False,
            num_vis_examples=0,
            output_dir=None,
        )
        all_results[s] = results

    print()
    print("=" * 70)
    print("STEP 4: sanity check seed_42 against job 10227 VALIDATION reference")
    print(f"(reference recall@1 = {REFERENCE_JOB_10227_SEED42_VAL_RECALL_AT_1}, "
          f"validation set -- not test, loose sanity check only)")
    print("=" * 70)
    if 42 in all_results:
        i2t_r1 = all_results[42]["i2t_rank<=1"] / 100.0
        t2i_r1 = all_results[42]["t2i_rank<=1"] / 100.0
        print(f"seed_42 TEST i2t recall@1 = {i2t_r1:.4f}")
        print(f"seed_42 TEST t2i recall@1 = {t2i_r1:.4f}")
        print(f"reference (job 10227, VAL) recall@1 = "
              f"{REFERENCE_JOB_10227_SEED42_VAL_RECALL_AT_1:.4f}")
    else:
        print("seed_42 was not cleanly loaded/evaluated -- cannot sanity check.")

    print()
    print("=" * 70)
    print("STEP 5: summary table")
    print("=" * 70)
    header = (
        f"{'seed':>6} | {'distinct':>8} | {'clean_load':>10} | "
        f"{'I2T R@1':>8} {'I2T R@5':>8} {'I2T R@10':>9} {'I2T MRR':>8} | "
        f"{'T2I R@1':>8} {'T2I R@5':>8} {'T2I R@10':>9} {'T2I MRR':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in SEEDS:
        is_distinct = s in distinct_seeds
        is_clean = s in cleanly_loaded
        if s in all_results:
            r = all_results[s]
            row = (
                f"{s:>6} | {str(is_distinct):>8} | {str(is_clean):>10} | "
                f"{r['i2t_rank<=1']/100:>8.4f} {r['i2t_rank<=5']/100:>8.4f} "
                f"{r['i2t_rank<=10']/100:>9.4f} {r['i2t_mrr']:>8.4f} | "
                f"{r['t2i_rank<=1']/100:>8.4f} {r['t2i_rank<=5']/100:>8.4f} "
                f"{r['t2i_rank<=10']/100:>9.4f} {r['t2i_mrr']:>8.4f}"
            )
        else:
            row = f"{s:>6} | {str(is_distinct):>8} | {str(is_clean):>10} | (not evaluated)"
        print(row)

    out_path = os.path.join(
        PROJECT_DIR, "openi", "results",
        "openi_baseline_checkpoint_verification_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "checksums": checksums,
                "duplicate_seeds": sorted(duplicate_seeds),
                "distinct_seeds": distinct_seeds,
                "cleanly_loaded_seeds": sorted(cleanly_loaded.keys()),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
