#!/usr/bin/env python3
"""
Integration test for the new optional-args HierarchicalCoAttention.forward()
in base_models_refactored_v1_paper2.py, using the REAL trained Paper 1
checkpoint (seed_42) and REAL data (images + captions) for 3 real test
samples (53957785, 50225296, 51065211). Does not modify any file.

PART A (backward compatibility, CRITICAL):
  Calls paper2's HierarchicalCoAttention.forward(image_tokens, text_tokens)
  -- i.e. WITHOUT the new optional args, exactly as Paper 1's code always
  does -- and compares it, torch.equal, against calling the ORIGINAL,
  untouched base_models_refactored_v1.py's HierarchicalCoAttention.forward()
  on the identical inputs, with the SAME trained checkpoint weights loaded
  into both. Tests the first co-attention layer of the synergy branch
  (synergy_branch.co_attn_layers[0]), which receives the shared
  image_tokens/text_tokens directly from MultimodalFusion's top-level
  image_encoder/text_encoder.

PART B (new path sanity check):
  Calls the SAME paper2 layer WITH the new args (real
  findings_token_count/has_find/has_imp/token_ids for these 3 samples).
  Reports output shape, NaN/Inf, and cosine similarity against Part A's
  output (expected: high but not exactly 1.0).
"""

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import base_models_refactored_v1 as orig_models
import base_models_refactored_v1_paper2 as paper2_models
from config import get_vocab_size, get_embed_dim, get_current_config

MODEL_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42",
    "export", "model_weights.pth",
)
SHARD_TEST_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/test"
SECTION_BOUNDARIES_CSV = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")

SAMPLE_STUDY_IDS = ["53957785", "50225296", "51065211"]


def load_samples_for_study_ids(study_ids_wanted):
    shard_paths = sorted(glob.glob(os.path.join(SHARD_TEST_DIR, "*.pkl")))
    found = {}
    for shard_path in shard_paths:
        if len(found) == len(study_ids_wanted):
            break
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        for sid, img, cap in zip(shard_data['study_ids'], shard_data['images'], shard_data['captions']):
            sid = str(sid)
            if sid in study_ids_wanted and sid not in found:
                found[sid] = (np.asarray(img), np.asarray(cap))
    return found


def build_model(module, vocab_size, embed_dim, num_heads, num_layers):
    model = module.MultimodalFusion(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    )
    return model


def main():
    print(f"Loading real images+captions for study_ids: {SAMPLE_STUDY_IDS}")
    samples = load_samples_for_study_ids(set(SAMPLE_STUDY_IDS))
    for sid in SAMPLE_STUDY_IDS:
        assert sid in samples, f"study_id {sid} not found in test shards"

    print("Loading findings_token_count from section_boundaries_test_paper2.csv (read-only)...")
    boundaries_df = pd.read_csv(SECTION_BOUNDARIES_CSV, dtype={'study_id': str}).set_index('study_id')

    images = np.stack([samples[sid][0] for sid in SAMPLE_STUDY_IDS], axis=0)
    captions = np.stack([samples[sid][1] for sid in SAMPLE_STUDY_IDS], axis=0)
    images_t = torch.FloatTensor(images)
    if images_t.shape[-1] == 3:
        images_t = images_t.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    token_ids = torch.LongTensor(captions)

    findings_token_count = torch.LongTensor(
        [int(boundaries_df.loc[sid, 'findings_token_count']) for sid in SAMPLE_STUDY_IDS]
    )
    real_length = (token_ids != 0).sum(dim=1)
    has_find = findings_token_count > 0
    has_imp = (real_length - findings_token_count) > 0

    vocab_size = get_vocab_size()
    embed_dim = get_embed_dim()
    cfg = get_current_config()
    num_heads = cfg['num_heads']
    num_layers = cfg['num_layers']

    print(f"\nLoading checkpoint into ORIGINAL (base_models_refactored_v1.py) model, strict=True...")
    orig_model = build_model(orig_models, vocab_size, embed_dim, num_heads, num_layers)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    missing, unexpected = orig_model.load_state_dict(state_dict, strict=True)
    print(f"  missing={missing}  unexpected={unexpected}")
    orig_model.eval()

    print(f"\nLoading SAME checkpoint into PAPER2 (base_models_refactored_v1_paper2.py) model, strict=False...")
    paper2_model = build_model(paper2_models, vocab_size, embed_dim, num_heads, num_layers)
    missing2, unexpected2 = paper2_model.load_state_dict(state_dict, strict=False)
    print(f"  missing={missing2}")
    print(f"  unexpected={unexpected2}")
    expected_missing = {
        'synergy_branch.co_attn_layers.0.section_find_gate_weights',
        'synergy_branch.co_attn_layers.0.section_imp_gate_weights',
        'synergy_branch.co_attn_layers.1.section_find_gate_weights',
        'synergy_branch.co_attn_layers.1.section_imp_gate_weights',
        'difference_branch.co_attn_layers.0.section_find_gate_weights',
        'difference_branch.co_attn_layers.0.section_imp_gate_weights',
        'difference_branch.co_attn_layers.1.section_find_gate_weights',
        'difference_branch.co_attn_layers.1.section_imp_gate_weights',
    }
    missing_set = set(missing2)
    only_expected_missing = missing_set.issubset(expected_missing) or missing_set == expected_missing
    print(f"  missing keys are ONLY the new section_*_gate_weights (nothing else): {only_expected_missing}")
    print(f"  no unexpected keys: {len(unexpected2) == 0}")
    paper2_model.eval()

    print("\nComputing shared image_tokens/text_tokens (from ORIGINAL model's encoders)...")
    with torch.no_grad():
        image_tokens = orig_model.image_encoder(images_t, training=False)
        text_tokens = orig_model.text_encoder(token_ids, training=False)
    print(f"  image_tokens: {tuple(image_tokens.shape)}   text_tokens: {tuple(text_tokens.shape)}")

    orig_layer = orig_model.synergy_branch.co_attn_layers[0]
    paper2_layer = paper2_model.synergy_branch.co_attn_layers[0]

    with torch.no_grad():
        orig_img_out, orig_txt_out = orig_layer(image_tokens, text_tokens)

        # PART A: paper2 layer called WITHOUT the new args (Paper 1's exact original call signature).
        partA_img_out, partA_txt_out = paper2_layer(image_tokens, text_tokens)

    print("\n" + "=" * 70)
    print("PART A: BACKWARD-COMPATIBILITY CHECK (real checkpoint, real data)")
    print("=" * 70)
    img_equal = torch.equal(orig_img_out, partA_img_out)
    txt_equal = torch.equal(orig_txt_out, partA_txt_out)
    print(f"  torch.equal(image_tokens output, ORIGINAL vs paper2-no-args): {img_equal}")
    print(f"  torch.equal(text_tokens  output, ORIGINAL vs paper2-no-args): {txt_equal}")
    img_max_diff = (orig_img_out - partA_img_out).abs().max().item()
    txt_max_diff = (orig_txt_out - partA_txt_out).abs().max().item()
    print(f"  max abs diff (image): {img_max_diff}")
    print(f"  max abs diff (text):  {txt_max_diff}")
    if not (img_equal and txt_equal):
        print("  *** PART A FAILED -- bit-identical match required, stopping. ***")
        return
    print("  PART A PASSED -- bit-identical to the original untouched module.")

    with torch.no_grad():
        partB_img_out, partB_txt_out = paper2_layer(
            image_tokens, text_tokens,
            findings_token_count=findings_token_count,
            has_find=has_find,
            has_imp=has_imp,
            token_ids=token_ids,
        )

    print("\n" + "=" * 70)
    print("PART B: NEW SECTION-AWARE PATH SANITY CHECK")
    print("=" * 70)
    print(f"  output shape (text): {tuple(partB_txt_out.shape)}")
    has_nan = torch.isnan(partB_txt_out).any().item()
    has_inf = torch.isinf(partB_txt_out).any().item()
    print(f"  has_nan={has_nan}  has_inf={has_inf}")

    # Image-side must stay identical between old and new paths.
    img_side_equal = torch.equal(partB_img_out, partA_img_out)
    print(f"  image-side output UNCHANGED between old and new paths (torch.equal): {img_side_equal}")

    print("\n  Per-sample cosine similarity: new-path text output vs Part A (old-path) text output")
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        old_vec = partA_txt_out[i].flatten()
        new_vec = partB_txt_out[i].flatten()
        cos_sim = F.cosine_similarity(old_vec.unsqueeze(0), new_vec.unsqueeze(0)).item()
        exact = torch.equal(old_vec, new_vec)
        print(f"    study_id={sid}  has_find={has_find[i].item()}  has_imp={has_imp[i].item()}  "
              f"cosine_sim={cos_sim:.6f}  torch.equal={exact}")


if __name__ == "__main__":
    main()
