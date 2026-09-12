#!/usr/bin/env python3
"""
Standalone sanity test for HierarchicalCoAttention.section_aware_feedback()
in base_models_refactored_v1_paper2.py. NOT part of the model, NOT wired
into forward(). Does not modify any file.

Same 3 real samples as the previous tests (53957785, 50225296, 51065211).

Checks:
  1. CRITICAL backward-compatibility: with has_find/has_imp forced to
     all-False ("no section info available"), section_aware_feedback's
     output must be torch.equal (bit-identical, not just close) to Paper
     1's original text_tokens + global_text_token.expand(...) -- verified
     against an INDEPENDENTLY hand-rederived copy of forward()'s original
     global block (not reusing any paper2 helper), not just against
     G_report internally.
  2. Per-position routing (Findings / Impression / padding) for a few
     example positions per sample, cross-checked against
     findings_token_count / real_length.
  3. Output shape, no NaN/Inf.
  4. For study_id 50225296 (has_find=False): mask_find is all-zero for
     every position.
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

SAMPLE_STUDY_IDS = ["53957785", "50225296", "51065211"]


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
    real_length = (token_ids != 0).sum(dim=1).to(torch.long)
    batch_size, seq_len = token_ids.shape

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
        text_tokens = text_encoder(token_ids, training=False)  # (B, L, D)
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        image_tokens = image_encoder(dummy_images, training=False)  # (B, L_img, D)

        section_pools = hca.compute_section_pools(text_tokens, token_ids, findings_token_count)
        has_find = section_pools['has_find']
        has_imp = section_pools['has_imp']
        G_find, G_imp, G_report = hca.compute_section_global_attention(section_pools, image_tokens)

        # --- Independent re-derivation of Paper 1's ORIGINAL global feedback,
        # not reusing any paper2 helper method, to genuinely cross-check step 3.
        old_style_query = torch.mean(text_tokens, dim=1, keepdim=True)
        attended_ref, _ = hca.global_cross_attention2(query=old_style_query, key=image_tokens, value=image_tokens)
        gate_ref = torch.sigmoid(hca.global_text_gate_weights).view(1, 1, embed_dim)
        gated_ref = gate_ref * attended_ref + (1 - gate_ref) * old_style_query
        ref_global_text_token = hca.global_norm3(old_style_query + gated_ref)
        ref_global_text_token = hca.global_norm4(ref_global_text_token + hca.global_ffn2(ref_global_text_token))
        original_output = text_tokens + ref_global_text_token.expand(-1, seq_len, -1)  # Paper 1's original line

        # --- Step 3: CRITICAL backward-compatibility test.
        has_find_none = torch.zeros_like(has_find)
        has_imp_none = torch.zeros_like(has_imp)
        backward_compat_output, mask_find_none, mask_imp_none = hca.section_aware_feedback(
            text_tokens, G_find, G_imp, G_report, findings_token_count, real_length,
            has_find_none, has_imp_none,
        )

        # --- Normal section-aware call (real has_find/has_imp) for the routing checks.
        output, mask_find, mask_imp = hca.section_aware_feedback(
            text_tokens, G_find, G_imp, G_report, findings_token_count, real_length,
            has_find, has_imp,
        )

    print("\n" + "=" * 70)
    print("STEP 3: CRITICAL BACKWARD-COMPATIBILITY CHECK")
    print("=" * 70)
    print("(has_find, has_imp both forced all-False -> masks must be all-zero everywhere)")
    all_masks_zero = (not mask_find_none.any().item()) and (not mask_imp_none.any().item())
    print(f"  mask_find all-zero: {not mask_find_none.any().item()}")
    print(f"  mask_imp  all-zero: {not mask_imp_none.any().item()}")

    is_exact = torch.equal(backward_compat_output, original_output)
    max_abs_diff = (backward_compat_output - original_output).abs().max().item()
    print(f"  torch.equal(section_aware_feedback(no-section-info), Paper1_original): {is_exact}")
    print(f"  max abs diff: {max_abs_diff}")
    if not (all_masks_zero and is_exact):
        print("  *** BACKWARD-COMPATIBILITY CHECK FAILED -- STOPPING, DO NOT PROCEED ***")
        return
    print("  BACKWARD-COMPATIBILITY CHECK PASSED.")

    print("\n" + "=" * 70)
    print("Output shape / NaN/Inf check")
    print("=" * 70)
    print(f"  output shape: {tuple(output.shape)}")
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    print(f"  has_nan={has_nan}  has_inf={has_inf}")

    print("\n" + "=" * 70)
    print("Per-position routing check (a few example positions per sample)")
    print("=" * 70)
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        fc = findings_token_count[i].item()
        rl = real_length[i].item()
        print(f"\n  study_id={sid}  findings_token_count={fc}  real_length={rl}  seq_len={seq_len}  "
              f"has_find={has_find[i].item()}  has_imp={has_imp[i].item()}")
        example_positions = sorted(set([0, max(fc - 1, 0), fc, min(fc + 1, seq_len - 1),
                                         max(rl - 1, 0), min(rl, seq_len - 1), seq_len - 1]))
        for j in example_positions:
            if j < fc and has_find[i]:
                route = "FINDINGS"
            elif fc <= j < rl and has_imp[i]:
                route = "IMPRESSION"
            elif j >= rl:
                route = "PADDING"
            else:
                route = "NEITHER (in real-token range but section missing)"
            mf = mask_find[i, j].item()
            mi = mask_imp[i, j].item()
            print(f"    pos {j:3d}: route={route:<32s}  mask_find={mf}  mask_imp={mi}")

    print("\n" + "=" * 70)
    print("study_id 50225296 (has_find=False): mask_find must be all-zero for every position")
    print("=" * 70)
    idx_50225296 = SAMPLE_STUDY_IDS.index("50225296")
    mask_find_this = mask_find[idx_50225296]
    print(f"  has_find={has_find[idx_50225296].item()}")
    print(f"  mask_find all-zero for all {seq_len} positions: {not mask_find_this.any().item()}")
    print(f"  mask_imp any True: {mask_imp[idx_50225296].any().item()} "
          f"(count={mask_imp[idx_50225296].sum().item()})")


if __name__ == "__main__":
    main()
