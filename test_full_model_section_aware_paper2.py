#!/usr/bin/env python3
"""
Full-model integration test for the section-aware plumbing now threaded
through BranchEncoder.forward() and MultimodalFusion.forward() in
base_models_refactored_v1_paper2.py. Uses the REAL trained Paper 1
checkpoint (seed_42) and REAL data (images + captions) for 3 real test
samples (53957785, 50225296, 51065211). Does not modify any file.

PART A (full-model backward compatibility, CRITICAL):
  MultimodalFusion.forward() called WITHOUT the new args, compared
  torch.equal against the ORIGINAL, untouched base_models_refactored_v1.py's
  MultimodalFusion on identical inputs/checkpoint.

PART B (full-model new-path check, debug=True):
  MultimodalFusion.forward() called WITH the new args. Confirms (by
  capturing stdout) that all 4 co-attention layers
  (synergy_coattn_1, synergy_coattn_2, difference_coattn_1, difference_coattn_2)
  printed findings_token_count-is-not-None=True. Reports final_image_emb
  (expected bit-identical to Part A) and final_text_emb (expected high but
  not exact cosine similarity to Part A).
"""

import os
import sys
import glob
import pickle
import io
import contextlib

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
EXPECTED_LAYER_NAMES = [
    "synergy_coattn_1", "synergy_coattn_2",
    "difference_coattn_1", "difference_coattn_2",
]


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
    return module.MultimodalFusion(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    )


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
        images_t = images_t.permute(0, 3, 1, 2)
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

    print("\nLoading checkpoint into ORIGINAL model, strict=True...")
    orig_model = build_model(orig_models, vocab_size, embed_dim, num_heads, num_layers)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    missing, unexpected = orig_model.load_state_dict(state_dict, strict=True)
    print(f"  missing={missing}  unexpected={unexpected}")
    orig_model.eval()

    print("Loading SAME checkpoint into PAPER2 model, strict=False (only new section_*_gate_weights missing)...")
    paper2_model = build_model(paper2_models, vocab_size, embed_dim, num_heads, num_layers)
    missing2, unexpected2 = paper2_model.load_state_dict(state_dict, strict=False)
    print(f"  {len(missing2)} missing keys (expected: only the 8 new section_*_gate_weights), unexpected={unexpected2}")
    paper2_model.eval()

    with torch.no_grad():
        orig_final_img, orig_final_txt = orig_model((images_t, token_ids), training=False)
        partA_final_img, partA_final_txt = paper2_model((images_t, token_ids), training=False)

    print("\n" + "=" * 70)
    print("PART A: FULL-MODEL BACKWARD-COMPATIBILITY CHECK")
    print("=" * 70)
    img_equal = torch.equal(orig_final_img, partA_final_img)
    txt_equal = torch.equal(orig_final_txt, partA_final_txt)
    print(f"  torch.equal(final_image_emb, ORIGINAL vs paper2-no-args): {img_equal}")
    print(f"  torch.equal(final_text_emb,  ORIGINAL vs paper2-no-args): {txt_equal}")
    if not (img_equal and txt_equal):
        print("  *** PART A FAILED -- bit-identical match required, stopping. ***")
        return
    print("  PART A PASSED -- bit-identical to the original untouched module.")

    print("\n" + "=" * 70)
    print("PART B: FULL-MODEL NEW-PATH CHECK (debug=True)")
    print("=" * 70)
    debug_buffer = io.StringIO()
    with torch.no_grad():
        with contextlib.redirect_stdout(debug_buffer):
            partB_final_img, partB_final_txt = paper2_model(
                (images_t, token_ids), training=False,
                findings_token_count=findings_token_count,
                has_find=has_find, has_imp=has_imp,
                token_ids=token_ids, debug=True,
            )
    debug_output = debug_buffer.getvalue()
    print("  --- captured debug output ---")
    print(debug_output.rstrip())
    print("  --- end debug output ---")

    print("\n  4-layer debug confirmation:")
    all_present_and_true = True
    for layer_name in EXPECTED_LAYER_NAMES:
        expected_line = f"[DEBUG] HierarchicalCoAttention({layer_name}): findings_token_count is not None = True"
        present = expected_line in debug_output
        all_present_and_true = all_present_and_true and present
        print(f"    {layer_name}: {'FOUND' if present else 'MISSING'} -- \"{expected_line}\"")
    print(f"  ALL 4 layers confirmed receiving findings_token_count (not None): {all_present_and_true}")

    print("\n  Output shapes / NaN-Inf:")
    print(f"    final_image_emb: {tuple(partB_final_img.shape)}   final_text_emb: {tuple(partB_final_txt.shape)}")
    print(f"    has_nan (text)={torch.isnan(partB_final_txt).any().item()}  has_inf (text)={torch.isinf(partB_final_txt).any().item()}")

    print("\n  Image-side check (must be bit-identical to Part A -- image never touches section info):")
    img_side_equal = torch.equal(partB_final_img, partA_final_img)
    print(f"    torch.equal(final_image_emb, PartB vs PartA): {img_side_equal}")

    print("\n  Text-side check (expected: high but not exact cosine similarity to Part A):")
    for i, sid in enumerate(SAMPLE_STUDY_IDS):
        old_vec = partA_final_txt[i]
        new_vec = partB_final_txt[i]
        cos_sim = F.cosine_similarity(old_vec.unsqueeze(0), new_vec.unsqueeze(0)).item()
        exact = torch.equal(old_vec, new_vec)
        print(f"    study_id={sid}  has_find={has_find[i].item()}  has_imp={has_imp[i].item()}  "
              f"cosine_sim={cos_sim:.6f}  torch.equal={exact}")


if __name__ == "__main__":
    main()
