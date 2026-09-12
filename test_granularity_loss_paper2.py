#!/usr/bin/env python3
"""
Test for compute_granularity_loss() and the section-aware pooled outputs
now returned by BranchEncoder.forward()/MultimodalFusion.forward() in
base_models_refactored_v1_paper2.py, using the REAL trained Paper 1
checkpoint (seed_42) and REAL data for 16 real test samples (a larger
batch than the previous 3-sample tests, for a meaningful in-batch
negative pool). Does not modify any file, does not run any real training,
does not touch the main training loop.

Checks:
  - final_image_emb / final_text_emb contain no NaN/Inf (addendum check)
  - L_gran is finite and a reasonable positive number
  - n_find / n_imp match the actual has_find/has_imp distribution in the batch
  - gradients flow: L_gran.backward() gives non-zero gradients on
    section_find_gate_weights / section_imp_gate_weights (synergy branch
    layers only -- difference branch never receives section info, so its
    equivalent params should get NO gradient from this loss, checked too)
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

import base_models_refactored_v1_paper2 as paper2_models
from config import get_vocab_size, get_embed_dim, get_current_config

MODEL_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42",
    "export", "model_weights.pth",
)
SHARD_TEST_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/test"
SECTION_BOUNDARIES_CSV = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")

N_SAMPLES = 16


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


def main():
    boundaries_df_full = pd.read_csv(SECTION_BOUNDARIES_CSV, dtype={'study_id': str})
    sample_study_ids = boundaries_df_full['study_id'].tolist()[:N_SAMPLES]
    print(f"Using {len(sample_study_ids)} real study_ids: {sample_study_ids}")

    samples = load_samples_for_study_ids(set(sample_study_ids))
    for sid in sample_study_ids:
        assert sid in samples, f"study_id {sid} not found in test shards"

    boundaries_df = boundaries_df_full.set_index('study_id')

    images = np.stack([samples[sid][0] for sid in sample_study_ids], axis=0)
    captions = np.stack([samples[sid][1] for sid in sample_study_ids], axis=0)
    images_t = torch.FloatTensor(images)
    if images_t.shape[-1] == 3:
        images_t = images_t.permute(0, 3, 1, 2)
    token_ids = torch.LongTensor(captions)

    findings_token_count = torch.LongTensor(
        [int(boundaries_df.loc[sid, 'findings_token_count']) for sid in sample_study_ids]
    )
    real_length = (token_ids != 0).sum(dim=1)
    has_find = findings_token_count > 0
    has_imp = (real_length - findings_token_count) > 0

    print(f"has_find distribution: {has_find.tolist()}  (sum={has_find.sum().item()})")
    print(f"has_imp  distribution: {has_imp.tolist()}  (sum={has_imp.sum().item()})")

    vocab_size = get_vocab_size()
    embed_dim = get_embed_dim()
    cfg = get_current_config()
    num_heads = cfg['num_heads']
    num_layers = cfg['num_layers']

    print("\nLoading checkpoint into PAPER2 model, strict=False...")
    model = paper2_models.MultimodalFusion(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
    )
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  {len(missing)} missing keys (expected: only new section_*_gate_weights), unexpected={unexpected}")
    model.eval()  # eval mode for BatchNorm/Dropout determinism; does NOT disable autograd

    model.zero_grad()

    # NOTE: no torch.no_grad() here -- we need the graph intact for the .backward() check below.
    final_image_emb, final_text_emb, z_find, z_imp = model(
        (images_t, token_ids), training=False,
        findings_token_count=findings_token_count, has_find=has_find, has_imp=has_imp,
        token_ids=token_ids, debug=False,
    )

    print("\n" + "=" * 70)
    print("ADDENDUM: NaN/Inf check on main outputs")
    print("=" * 70)
    print(f"  final_image_emb: shape={tuple(final_image_emb.shape)}  "
          f"has_nan={torch.isnan(final_image_emb).any().item()}  has_inf={torch.isinf(final_image_emb).any().item()}")
    print(f"  final_text_emb:  shape={tuple(final_text_emb.shape)}  "
          f"has_nan={torch.isnan(final_text_emb).any().item()}  has_inf={torch.isinf(final_text_emb).any().item()}")

    L_gran, find_loss, imp_loss, n_find, n_imp = paper2_models.compute_granularity_loss(
        final_image_emb, z_find, z_imp, has_find, has_imp
    )

    print("\n" + "=" * 70)
    print("GRANULARITY LOSS")
    print("=" * 70)
    find_loss_val = find_loss.item() if torch.is_tensor(find_loss) else find_loss
    imp_loss_val = imp_loss.item() if torch.is_tensor(imp_loss) else imp_loss
    print(f"  find_loss: {find_loss_val}   (n_find={n_find}, expected sum(has_find)={has_find.sum().item()})")
    print(f"  imp_loss:  {imp_loss_val}   (n_imp={n_imp}, expected sum(has_imp)={has_imp.sum().item()})")
    print(f"  L_gran = find_loss + imp_loss = {L_gran.item()}")
    print(f"  L_gran finite: {torch.isfinite(L_gran).item()}")
    print(f"  (for reference: ln(n_find)={np.log(max(n_find,1)):.4f}, ln(n_imp)={np.log(max(n_imp,1)):.4f} "
          f"-- typical InfoNCE loss scale for these sub-batch sizes)")

    print("\n" + "=" * 70)
    print("GRADIENT FLOW CHECK")
    print("=" * 70)
    L_gran.backward()

    for branch_name in ['synergy_branch', 'difference_branch']:
        branch = getattr(model, branch_name)
        for i, layer in enumerate(branch.co_attn_layers):
            for pname in ['section_find_gate_weights', 'section_imp_gate_weights']:
                p = getattr(layer, pname)
                grad = p.grad
                if grad is None:
                    print(f"  {branch_name}.co_attn_layers[{i}].{pname}: grad=None")
                else:
                    grad_norm = grad.norm().item()
                    nonzero = grad_norm > 0
                    print(f"  {branch_name}.co_attn_layers[{i}].{pname}: grad_norm={grad_norm:.8f}  nonzero={nonzero}")

    print("\n  Expectation: synergy_branch layers should have non-zero gradients (they received "
          "section info); difference_branch layers should have grad=None (never received section "
          "info in this call, so autograd never touched those parameters).")


if __name__ == "__main__":
    main()
