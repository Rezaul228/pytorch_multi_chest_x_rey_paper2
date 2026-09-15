#!/usr/bin/env python3
"""
For seed_42 (Paper 1 baseline vs MG-G2L), compute per-query Image->Text
nDCG@10 and Precision@5 (reusing the exact same verified scoring pipeline
from paper2_graded_relevance_eval.py), then split by the
Findings/Impression "divergence" grouping from
compute_divergence_scores_paper2.py (agreement vs divergence), and compare
mean restricted nDCG@10 / Precision@5 per group, per model.

No per-query cache existed from the earlier graded-relevance runs (only
aggregate summary CSVs were saved), so this recomputes embeddings for both
checkpoints -- explicitly permitted by the task ("reuse cached results if
available, otherwise recompute using the same verified pipeline").

Inference only -- no training, no checkpoint modification.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
MIMIC_RESULTS_DIR = os.path.join(PROJECT_DIR, "mimic", "results")

import paper2_graded_relevance_eval as scoring  # reused, unchanged

DIVERGENCE_CSV_PATH = os.path.join(MIMIC_RESULTS_DIR, "divergence_scores_test_paper2.csv")
PAPER1_MODEL_PATH = scoring.DEFAULT_MODEL_PATH  # seed_42 baseline, default in the reused module
MGG2L_CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_42",
    "export", "checkpoint_resume.pth",
)


def compute_i2t_per_query_metrics(image_embeddings, text_embeddings, study_ids, device):
    """Reuses build_binary_matrix / compute_metrics_for_direction UNCHANGED; returns
    per-query i2t_ndcg10, i2t_prec5 arrays aligned to `study_ids`."""
    n = image_embeddings.shape[0]
    image_emb_t = torch.FloatTensor(image_embeddings).to(device)
    text_emb_t = torch.FloatTensor(text_embeddings).to(device)
    i2t_sim = torch.matmul(image_emb_t, text_emb_t.transpose(0, 1))

    B, _ = scoring.build_binary_matrix(study_ids, scoring.DEFAULT_BINARY_LABELS_PATH)
    B_t = torch.FloatTensor(B).to(device)
    R_full = torch.matmul(B_t, B_t.transpose(0, 1))
    R_full_np = R_full.cpu().numpy().astype(np.int8)
    del R_full, B_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ideal_sorted = -np.sort(-R_full_np, axis=1)
    ideal_top10 = ideal_sorted[:, :scoring.TOP_K]
    total_relevant = np.sum(R_full_np > 0, axis=1)

    _, i2t_ndcg10, _, i2t_prec5 = scoring.compute_metrics_for_direction(
        i2t_sim, R_full_np, ideal_top10, total_relevant, n
    )
    return i2t_ndcg10, i2t_prec5


def get_embeddings_for_model(model, test_dataset, device):
    return scoring.compute_embeddings(model, test_dataset, device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading divergence groups (from compute_divergence_scores_paper2.py)...")
    div_df = pd.read_csv(DIVERGENCE_CSV_PATH, dtype={"study_id": str}).set_index("study_id")
    print(f"Loaded {len(div_df)} scored study_ids "
          f"(agreement={sum(div_df['group']=='agreement')}, divergence={sum(div_df['group']=='divergence')})")

    # ---------- Paper 1 baseline (seed_42) ----------
    print("\n" + "=" * 90)
    print("Paper 1 baseline (seed_42)")
    print("=" * 90)
    p1_model = scoring.load_model(PAPER1_MODEL_PATH).to(device)
    test_dataset = scoring.load_test_dataset(scoring.DEFAULT_SHARD_SUBFOLDER)
    p1_img_emb, p1_txt_emb, p1_study_ids = get_embeddings_for_model(p1_model, test_dataset, device)
    p1_ndcg10, p1_prec5 = compute_i2t_per_query_metrics(p1_img_emb, p1_txt_emb, p1_study_ids, device)
    del p1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------- MG-G2L (seed_42) ----------
    print("\n" + "=" * 90)
    print("MG-G2L (seed_42, section-aware)")
    print("=" * 90)
    from base_models_refactored_v1_paper2 import MultimodalFusion as MultimodalFusionPaper2
    from data_loader_v1_paper2 import IndianaDataLoader as IndianaDataLoaderPaper2
    from config import get_vocab_size, get_embed_dim, get_current_config

    mg_model = MultimodalFusionPaper2(
        vocab_size=get_vocab_size(), embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"], num_layers=get_current_config()["num_layers"],
    )
    checkpoint = torch.load(MGG2L_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch field: {checkpoint['epoch']} (-> epoch {checkpoint['epoch'] + 1})")
    missing, unexpected = mg_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    mg_model.eval().to(device)

    mg_dl = IndianaDataLoaderPaper2(batch_size=scoring.BATCH_SIZE, use_shards=True, shard_subfolder=scoring.DEFAULT_SHARD_SUBFOLDER)
    mg_dl.tokenizer = scoring.load_tokenizer_from_metadata(scoring.DEFAULT_SHARD_SUBFOLDER)
    mg_dl.load_data(max_samples=None, skip_processing=True)
    mg_test_dataset = mg_dl.get_test_data(num_samples=None)
    print(f"MG-G2L test dataset size: {len(mg_test_dataset)}  "
          f"section_boundaries: {len(mg_test_dataset.section_boundaries)}  fallback: {mg_test_dataset.fallback_count}")

    from torch.utils.data import DataLoader
    loader = DataLoader(mg_test_dataset, batch_size=scoring.BATCH_SIZE, shuffle=False)
    all_image_emb, all_text_emb, all_study_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_images = batch["images"]
            batch_captions = batch["captions"]
            batch_study_ids = batch["study_ids"]
            findings_token_count = batch["findings_token_count"].to(device)
            has_find = batch["has_find"].to(device)
            has_imp = batch["has_imp"].to(device)
            if len(batch_images.shape) == 4 and batch_images.shape[-1] == 3:
                batch_images = batch_images.permute(0, 3, 1, 2)
            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)
            model_output = mg_model(
                (batch_images, batch_captions), training=False,
                findings_token_count=findings_token_count, has_find=has_find, has_imp=has_imp,
                token_ids=batch_captions,
            )
            image_emb, text_emb = model_output[0], model_output[1]
            all_image_emb.append(image_emb.cpu().numpy())
            all_text_emb.append(text_emb.cpu().numpy())
            if isinstance(batch_study_ids, torch.Tensor):
                all_study_ids.extend(int(sid.item()) for sid in batch_study_ids)
            else:
                all_study_ids.extend(int(sid) for sid in batch_study_ids)

    mg_img_emb = np.concatenate(all_image_emb, axis=0)
    mg_txt_emb = np.concatenate(all_text_emb, axis=0)
    mg_study_ids = np.array(all_study_ids, dtype=np.int64)
    del mg_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mg_ndcg10, mg_prec5 = compute_i2t_per_query_metrics(mg_img_emb, mg_txt_emb, mg_study_ids, device)

    # ---------- Join with divergence groups ----------
    assert list(p1_study_ids) == list(mg_study_ids), "study_id order mismatch between the two models' datasets"

    results_df = pd.DataFrame({
        "study_id": [str(s) for s in p1_study_ids],
        "p1_ndcg10": p1_ndcg10, "p1_prec5": p1_prec5,
        "mg_ndcg10": mg_ndcg10, "mg_prec5": mg_prec5,
    }).set_index("study_id")

    joined = results_df.join(div_df, how="inner")
    print(f"\nJoined rows (should match divergence CSV row count): {len(joined)}")

    print("\n" + "=" * 100)
    print("GROUP COMPARISON: Paper 1 baseline vs MG-G2L, restricted I2T nDCG@10 / Precision@5, by divergence group")
    print("=" * 100)
    header = f"{'group':<12} | {'n':<6} | {'metric':<12} | {'Paper 1':<10} | {'MG-G2L':<10} | {'delta':<10}"
    print(header)
    print("-" * len(header))
    for group in ["agreement", "divergence"]:
        sub = joined[joined["group"] == group]
        n = len(sub)
        for metric, p1_col, mg_col in [("nDCG@10", "p1_ndcg10", "mg_ndcg10"), ("Precision@5", "p1_prec5", "mg_prec5")]:
            p1_mean = sub[p1_col].mean()
            mg_mean = sub[mg_col].mean()
            delta = mg_mean - p1_mean
            print(f"{group:<12} | {n:<6} | {metric:<12} | {p1_mean:<10.4f} | {mg_mean:<10.4f} | {delta:<+10.4f}")


if __name__ == "__main__":
    main()
