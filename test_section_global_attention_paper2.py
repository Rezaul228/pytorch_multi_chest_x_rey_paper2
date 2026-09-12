#!/usr/bin/env python3
"""
Standalone sanity test for HierarchicalCoAttention.compute_section_global_attention()
in base_models_refactored_v1_paper2.py. NOT part of the model, NOT wired
into forward(). Does not modify any file.

Same 3 real samples as the compute_section_pools test (53957785, 50225296,
51065211): loads their raw token ids + findings_token_count, runs
compute_section_pools() then compute_section_global_attention() on a
freshly-initialized (untrained) TextEncoder/ImageEncoder/HierarchicalCoAttention,
and checks:
  - output shapes and norms for G_find, G_imp, G_report
  - no NaN/Inf anywhere
  - G_find is EXACTLY a zero vector for study_id 50225296 (has_find=False)
  - G_report is bit-identical to independently re-running forward()'s
    original "Global Text -> Image" block by hand (same layers, using
    old-style torch.mean(text_tokens, dim=1, keepdim=True) as the query --
    NOT calling compute_section_global_attention's own helper, so this is a
    genuinely independent recomputation, not a tautology).
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

from base_models_refactored_v1_paper2 import TextEncoder, ImageEncoder, HierarchicalCoAttention
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
    batch_size = token_ids.shape[0]

    embed_dim = get_embed_dim()
    num_heads = get_current_config()['num_heads']
    vocab_size = get_vocab_size()

    print(f"\nBuilding fresh (untrained, random-init) TextEncoder + ImageEncoder + "
          f"HierarchicalCoAttention (vocab_size={vocab_size}, embed_dim={embed_dim}, num_heads={num_heads})...")
    text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
    text_encoder.eval()

    image_encoder = ImageEncoder(embed_dim=embed_dim)
    image_encoder.eval()

    hca = HierarchicalCoAttention(embed_dim=embed_dim, num_heads=num_heads)
    hca.eval()

    with torch.no_grad():
        text_tokens = text_encoder(token_ids, training=False)  # (3, L_text, D)
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        image_tokens = image_encoder(dummy_images, training=False)  # (3, L_img, D)
        print(f"text_tokens shape:  {tuple(text_tokens.shape)}")
        print(f"image_tokens shape: {tuple(image_tokens.shape)}")

        section_pools = hca.compute_section_pools(text_tokens, token_ids, findings_token_count)
        G_find, G_imp, G_report = hca.compute_section_global_attention(section_pools, image_tokens)

        # Independent re-derivation of forward()'s original "Global Text -> Image" block,
        # NOT calling hca._global_text_attend_to_image, to genuinely cross-check G_report.
        old_style_query = torch.mean(text_tokens, dim=1, keepdim=True)
        attended_ref, _ = hca.global_cross_attention2(query=old_style_query, key=image_tokens, value=image_tokens)
        gate_ref = torch.sigmoid(hca.global_text_gate_weights).view(1, 1, embed_dim)
        gated_ref = gate_ref * attended_ref + (1 - gate_ref) * old_style_query
        ref_out = hca.global_norm3(old_style_query + gated_ref)
        ref_out = hca.global_norm4(ref_out + hca.global_ffn2(ref_out))

    print("\n=== Output shapes ===")
    print(f"  G_find:   {tuple(G_find.shape)}")
    print(f"  G_imp:    {tuple(G_imp.shape)}")
    print(f"  G_report: {tuple(G_report.shape)}")

    print("\n=== NaN/Inf check ===")
    any_bad = False
    for name, tensor in [('G_find', G_find), ('G_imp', G_imp), ('G_report', G_report)]:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        print(f"  {name}: has_nan={has_nan}  has_inf={has_inf}")
        any_bad = any_bad or has_nan or has_inf
    print(f"  ALL CLEAN (no NaN/Inf anywhere): {not any_bad}")

    print("\n=== Per-sample norms ===")
    has_find = section_pools['has_find']
    has_imp = section_pools['has_imp']
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        print(f"  study_id={sid}  has_find={has_find[i].item()}  has_imp={has_imp[i].item()}  "
              f"|G_find|={G_find[i,0].norm().item():.4f}  |G_imp|={G_imp[i,0].norm().item():.4f}  "
              f"|G_report|={G_report[i,0].norm().item():.4f}")

    print("\n=== Zero-vector check for has_find=False (study_id 50225296) ===")
    idx_50225296 = SAMPLE_STUDY_IDS.index("50225296")
    assert not has_find[idx_50225296].item(), "expected has_find=False for study_id 50225296"
    g_find_vec = G_find[idx_50225296, 0]
    is_exact_zero = torch.equal(g_find_vec, torch.zeros_like(g_find_vec))
    max_abs_value = g_find_vec.abs().max().item()
    print(f"  G_find[50225296] is EXACTLY zero (torch.equal): {is_exact_zero}")
    print(f"  max abs value in that vector: {max_abs_value}")

    print("\n=== G_report bit-identical check vs independently re-derived forward()-block ===")
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        diff = (G_report[i, 0] - ref_out[i, 0]).abs()
        max_abs_diff = diff.max().item()
        is_exact = torch.equal(G_report[i, 0], ref_out[i, 0])
        print(f"  study_id={sid}: max abs diff={max_abs_diff:.10f}  torch.equal={is_exact}")


if __name__ == "__main__":
    main()
