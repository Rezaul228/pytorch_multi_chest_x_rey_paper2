#!/usr/bin/env python3
"""
Standalone sanity test for HierarchicalCoAttention.compute_section_pools()
in base_models_refactored_v1_paper2.py. NOT part of the model, NOT wired
into forward(). Does not modify any file.

Loads 3 real samples from the verified MIMIC test set (raw token ids
straight from the test shards -- no images), plus their real
findings_token_count from section_boundaries_test_paper2.csv, runs those
through a freshly-initialized (untrained) TextEncoder to get a text_tokens
tensor of the right shape, then calls ONLY compute_section_pools on it --
never HierarchicalCoAttention.forward(), and never the trained checkpoint --
since this test is about the pooling logic and its shapes/edge-case
handling, not about model correctness.
"""

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from base_models_refactored_v1_paper2 import TextEncoder, HierarchicalCoAttention
from config import get_vocab_size, get_embed_dim, get_current_config

SHARD_TEST_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/test"
SECTION_BOUNDARIES_CSV = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")

SAMPLE_STUDY_IDS = ["53957785", "50225296", "51065211"]  # findings+impr, impression-only, findings+impr


def load_captions_for_study_ids(study_ids_wanted):
    shard_paths = sorted(glob.glob(os.path.join(SHARD_TEST_DIR, "*.pkl")))
    found = {}
    for shard_path in shard_paths:
        if len(found) == len(study_ids_wanted):
            break
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        for sid, cap in zip(shard_data['study_ids'], shard_data['captions']):
            sid = str(sid)
            if sid in study_ids_wanted and sid not in found:
                found[sid] = np.asarray(cap)
    return found


def main():
    torch.manual_seed(0)

    print(f"Loading raw caption token ids for study_ids: {SAMPLE_STUDY_IDS}")
    captions_by_sid = load_captions_for_study_ids(set(SAMPLE_STUDY_IDS))
    for sid in SAMPLE_STUDY_IDS:
        assert sid in captions_by_sid, f"study_id {sid} not found in test shards"

    print("Loading findings_token_count from section_boundaries_test_paper2.csv (read-only)...")
    boundaries_df = pd.read_csv(SECTION_BOUNDARIES_CSV, dtype={'study_id': str}).set_index('study_id')

    token_ids_list = [captions_by_sid[sid] for sid in SAMPLE_STUDY_IDS]
    token_ids = torch.LongTensor(np.stack(token_ids_list, axis=0))  # (3, L)
    findings_token_count = torch.LongTensor(
        [int(boundaries_df.loc[sid, 'findings_token_count']) for sid in SAMPLE_STUDY_IDS]
    )

    real_lengths = (token_ids != 0).sum(dim=1)

    print("\nPer-sample summary:")
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        print(f"  study_id={sid}  findings_token_count={findings_token_count[i].item()}  "
              f"real_length={real_lengths[i].item()}  seq_len={token_ids.shape[1]}")

    embed_dim = get_embed_dim()
    num_heads = get_current_config()['num_heads']
    vocab_size = get_vocab_size()

    print(f"\nBuilding fresh (untrained, random-init) TextEncoder + HierarchicalCoAttention "
          f"(vocab_size={vocab_size}, embed_dim={embed_dim}, num_heads={num_heads})...")
    text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
    text_encoder.eval()

    hca = HierarchicalCoAttention(embed_dim=embed_dim, num_heads=num_heads)
    hca.eval()

    with torch.no_grad():
        text_tokens = text_encoder(token_ids, training=False)  # (3, L, D)
        print(f"text_tokens shape: {tuple(text_tokens.shape)}")

        result = hca.compute_section_pools(text_tokens, token_ids, findings_token_count)

        old_style_report = torch.mean(text_tokens, dim=1, keepdim=True)  # OLD unmasked behavior

    print("\n=== Output shapes ===")
    for key in ['global_text_find', 'global_text_imp', 'global_text_report']:
        print(f"  {key}: {tuple(result[key].shape)}")
    print(f"  has_find: {result['has_find'].tolist()}")
    print(f"  has_imp:  {result['has_imp'].tolist()}")

    print("\n=== Fallback sanity check: global_text_report vs OLD torch.mean(text_tokens, dim=1) ===")
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        new_vec = result['global_text_report'][i, 0]
        old_vec = old_style_report[i, 0]
        abs_diff = (new_vec - old_vec).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(new_vec.unsqueeze(0), old_vec.unsqueeze(0)).item()
        pad_frac = 1.0 - (real_lengths[i].item() / token_ids.shape[1])
        print(f"  study_id={sid}  (padding fraction={pad_frac:.2%})")
        print(f"    max abs diff: {max_abs_diff:.6f}   mean abs diff: {mean_abs_diff:.6f}   cosine sim: {cos_sim:.6f}")

    print("\n=== Per-sample section pool sanity ===")
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        print(f"  study_id={sid}: has_find={result['has_find'][i].item()}  has_imp={result['has_imp'][i].item()}  "
              f"find_vec_norm={result['global_text_find'][i,0].norm().item():.4f}  "
              f"imp_vec_norm={result['global_text_imp'][i,0].norm().item():.4f}  "
              f"report_vec_norm={result['global_text_report'][i,0].norm().item():.4f}")


if __name__ == "__main__":
    main()
