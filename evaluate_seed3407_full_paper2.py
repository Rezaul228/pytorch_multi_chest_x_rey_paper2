#!/usr/bin/env python3
"""
Full evaluation of the newly-added Paper 1 baseline checkpoint for
seed_3407, matched against the already-trained MG-G2L seed_3407 checkpoint:
  1. Load Paper1 baseline seed_3407 into the ORIGINAL MultimodalFusion,
     strict=True sanity check.
  2. Official evaluate_cross_modal_retrieval_streaming() (no section args)
     on the 12,429-sample test set -> R@1/5/10/MRR both directions.
  3. Graded-relevance (I2T nDCG@10/Precision@5, restricted subset) for
     Paper1 baseline seed_3407, reusing the verified scoring pipeline.
  4. Graded-relevance for MG-G2L seed_3407 (not yet run through this
     pipeline -- checked first, confirmed absent).
  5. Compare both (R@K and graded-relevance) against the already-known
     MG-G2L seed_3407 numbers.
  6. Append seed_3407 rows to the multiseed CSVs and save a per-query cache
     (per_query_i2t_metrics_seed_3407.csv), consistent with seeds 42/17/123.

Inference only -- no training, no checkpoint modification.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming
import paper2_graded_relevance_eval as scoring  # reused, unchanged

SEED = 3407
PAPER1_MODEL_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_{SEED}",
    "export", "model_weights.pth",
)
MGG2L_CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_{SEED}",
    "export", "checkpoint_resume.pth",
)

# Already-known MG-G2L seed_3407 numbers (from the earlier multi-seed streaming eval, job 113730).
MGG2L_KNOWN_RESULTS = {
    "i2t_recall@1": 0.9998, "i2t_recall@5": 0.9999, "i2t_recall@10": 0.9999, "i2t_mrr": 0.9998,
    "t2i_recall@1": 1.0000, "t2i_recall@5": 1.0000, "t2i_recall@10": 1.0000, "t2i_mrr": 1.0000,
}

MULTISEED_PAPER1_CSV = os.path.join(PROJECT_DIR, "paper1_baseline_graded_relevance_multiseed.csv")
MULTISEED_MGG2L_CSV = os.path.join(PROJECT_DIR, "mg_g2l_graded_relevance_multiseed.csv")
PER_QUERY_CACHE_PATH = os.path.join(PROJECT_DIR, f"per_query_i2t_metrics_seed_{SEED}.csv")


def load_mgg2l_embeddings(checkpoint_path, device):
    from base_models_refactored_v1_paper2 import MultimodalFusion as MultimodalFusionPaper2
    from data_loader_v1_paper2 import IndianaDataLoader as IndianaDataLoaderPaper2
    from config import get_vocab_size, get_embed_dim, get_current_config

    model = MultimodalFusionPaper2(
        vocab_size=get_vocab_size(), embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"], num_layers=get_current_config()["num_layers"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  [MG-G2L] checkpoint epoch field: {checkpoint['epoch']} (-> epoch {checkpoint['epoch'] + 1})")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval().to(device)

    dl = IndianaDataLoaderPaper2(batch_size=scoring.BATCH_SIZE, use_shards=True, shard_subfolder=scoring.DEFAULT_SHARD_SUBFOLDER)
    dl.tokenizer = scoring.load_tokenizer_from_metadata(scoring.DEFAULT_SHARD_SUBFOLDER)
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"  [MG-G2L] test dataset size: {len(test_dataset)}  fallback: {test_dataset.fallback_count}")

    loader = DataLoader(test_dataset, batch_size=scoring.BATCH_SIZE, shuffle=False)
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
            model_output = model(
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

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (np.concatenate(all_image_emb, axis=0), np.concatenate(all_text_emb, axis=0),
            np.array(all_study_ids, dtype=np.int64))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- Step 2: load Paper1 baseline seed_3407, strict check ----------
    print("\n" + "=" * 90)
    print(f"STEP 2: Load Paper1 baseline seed_{SEED}, strict=True")
    print("=" * 90)
    p1_model = scoring.load_model(PAPER1_MODEL_PATH)  # asserts strict, prints param count
    p1_model = p1_model.to(device)

    # ---------- Step 3: official streaming eval ----------
    print("\n" + "=" * 90)
    print(f"STEP 3: evaluate_cross_modal_retrieval_streaming() on Paper1 baseline seed_{SEED}")
    print("=" * 90)
    p1_test_dataset = scoring.load_test_dataset(scoring.DEFAULT_SHARD_SUBFOLDER)
    p1_results = evaluate_cross_modal_retrieval_streaming(
        model=p1_model, test_dataset=p1_test_dataset, k_values=[1, 5, 10], batch_size=scoring.BATCH_SIZE,
    )

    # Keep embeddings for graded-relevance too (avoid recompute) -- but evaluate_cross_modal_retrieval_streaming
    # doesn't return them, so recompute once via scoring.compute_embeddings (same method, same dataset).
    p1_img_emb, p1_txt_emb, p1_study_ids = scoring.compute_embeddings(p1_model, p1_test_dataset, device)
    del p1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print(f"COMPARISON (R@K/MRR): Paper1 baseline seed_{SEED} vs MG-G2L seed_{SEED} (already known)")
    print("=" * 100)
    header = f"{'metric':<14} | {'Paper 1':<10} | {'MG-G2L':<10} | {'delta':<10}"
    print(header)
    print("-" * len(header))
    for key in ["i2t_recall@1", "i2t_recall@5", "i2t_recall@10", "i2t_mrr",
                "t2i_recall@1", "t2i_recall@5", "t2i_recall@10", "t2i_mrr"]:
        p1v = p1_results[key]
        mgv = MGG2L_KNOWN_RESULTS[key]
        print(f"{key:<14} | {p1v:<10.4f} | {mgv:<10.4f} | {mgv - p1v:<+10.4f}")

    # ---------- Step 4: MG-G2L seed_3407 embeddings (for graded relevance) ----------
    print("\n" + "=" * 90)
    print(f"STEP 4: MG-G2L seed_{SEED} embeddings (for graded relevance -- not yet cached)")
    print("=" * 90)
    mg_img_emb, mg_txt_emb, mg_study_ids = load_mgg2l_embeddings(MGG2L_CHECKPOINT_PATH, device)
    assert list(mg_study_ids) == list(p1_study_ids), "study_id order mismatch between Paper1 and MG-G2L datasets"

    # ---------- Graded relevance for both (reused scoring code) ----------
    print("\n" + "=" * 90)
    print("Graded relevance (reused build_binary_matrix / compute_metrics_for_direction)")
    print("=" * 90)
    n = len(p1_study_ids)
    B, _ = scoring.build_binary_matrix(p1_study_ids, scoring.DEFAULT_BINARY_LABELS_PATH)
    B_t = torch.FloatTensor(B).to(device)
    R_full = torch.matmul(B_t, B_t.transpose(0, 1))
    R_full_np = R_full.cpu().numpy().astype(np.int8)
    del R_full, B_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ideal_sorted = -np.sort(-R_full_np, axis=1)
    ideal_top10 = ideal_sorted[:, :scoring.TOP_K]
    total_relevant = np.sum(R_full_np > 0, axis=1)
    restricted_mask = total_relevant > 0
    restricted_n = int(np.sum(restricted_mask))
    print(f"Restricted subset (>=1 confirmed positive finding): {restricted_n} / {n}")

    p1_img_t = torch.FloatTensor(p1_img_emb).to(device)
    p1_txt_t = torch.FloatTensor(p1_txt_emb).to(device)
    p1_i2t_sim = torch.matmul(p1_img_t, p1_txt_t.transpose(0, 1))
    p1_t2i_sim = torch.matmul(p1_txt_t, p1_img_t.transpose(0, 1))
    p1_ndcg5, p1_ndcg10, p1_ap10, p1_prec5 = scoring.compute_metrics_for_direction(
        p1_i2t_sim, R_full_np, ideal_top10, total_relevant, n
    )
    p1_t2i_ndcg5, p1_t2i_ndcg10, p1_t2i_ap10, p1_t2i_prec5 = scoring.compute_metrics_for_direction(
        p1_t2i_sim, R_full_np, ideal_top10, total_relevant, n
    )
    del p1_img_t, p1_txt_t, p1_i2t_sim, p1_t2i_sim

    mg_img_t = torch.FloatTensor(mg_img_emb).to(device)
    mg_txt_t = torch.FloatTensor(mg_txt_emb).to(device)
    mg_i2t_sim = torch.matmul(mg_img_t, mg_txt_t.transpose(0, 1))
    mg_t2i_sim = torch.matmul(mg_txt_t, mg_img_t.transpose(0, 1))
    mg_ndcg5, mg_ndcg10, mg_ap10, mg_prec5 = scoring.compute_metrics_for_direction(
        mg_i2t_sim, R_full_np, ideal_top10, total_relevant, n
    )
    mg_t2i_ndcg5, mg_t2i_ndcg10, mg_t2i_ap10, mg_t2i_prec5 = scoring.compute_metrics_for_direction(
        mg_t2i_sim, R_full_np, ideal_top10, total_relevant, n
    )
    del mg_img_t, mg_txt_t, mg_i2t_sim, mg_t2i_sim
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def full_and_restricted(arr):
        return float(np.mean(arr)), float(np.mean(arr[restricted_mask]))

    def rows_for(i2t_arrs, t2i_arrs):
        rows = []
        for metric_name, i2t_arr, t2i_arr in [
            ("nDCG@5", i2t_arrs[0], t2i_arrs[0]), ("nDCG@10", i2t_arrs[1], t2i_arrs[1]),
            ("mAP@10", i2t_arrs[2], t2i_arrs[2]), ("Precision@5", i2t_arrs[3], t2i_arrs[3]),
        ]:
            for direction, arr in [("Image->Text", i2t_arr), ("Text->Image", t2i_arr)]:
                full_val, restricted_val = full_and_restricted(arr)
                rows.append({"seed": SEED, "metric": metric_name, "direction": direction,
                             "full_corpus_value": round(full_val, 4), "full_corpus_n": n,
                             "restricted_subset_value": round(restricted_val, 4), "restricted_subset_n": restricted_n})
        return rows

    p1_rows = rows_for((p1_ndcg5, p1_ndcg10, p1_ap10, p1_prec5), (p1_t2i_ndcg5, p1_t2i_ndcg10, p1_t2i_ap10, p1_t2i_prec5))
    mg_rows = rows_for((mg_ndcg5, mg_ndcg10, mg_ap10, mg_prec5), (mg_t2i_ndcg5, mg_t2i_ndcg10, mg_t2i_ap10, mg_t2i_prec5))

    print("\n" + "=" * 100)
    print(f"GRADED RELEVANCE COMPARISON (restricted subset, n={restricted_n}): Paper1 vs MG-G2L, seed_{SEED}")
    print("=" * 100)
    header2 = f"{'metric':<12} | {'direction':<12} | {'Paper 1':<10} | {'MG-G2L':<10} | {'delta':<10}"
    print(header2)
    print("-" * len(header2))
    for r_p1, r_mg in zip(p1_rows, mg_rows):
        delta = r_mg["restricted_subset_value"] - r_p1["restricted_subset_value"]
        print(f"{r_p1['metric']:<12} | {r_p1['direction']:<12} | {r_p1['restricted_subset_value']:<10.4f} | "
              f"{r_mg['restricted_subset_value']:<10.4f} | {delta:<+10.4f}")

    # ---------- Save: append to multiseed CSVs + per-query cache ----------
    def append_rows(csv_path, new_rows):
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "metric", "direction", "full_corpus_value", "full_corpus_n",
                "restricted_subset_value", "restricted_subset_n",
            ])
            if not file_exists:
                writer.writeheader()
            for r in new_rows:
                writer.writerow(r)

    append_rows(MULTISEED_PAPER1_CSV, p1_rows)
    append_rows(MULTISEED_MGG2L_CSV, mg_rows)
    print(f"\nAppended seed_{SEED} rows to: {MULTISEED_PAPER1_CSV}")
    print(f"Appended seed_{SEED} rows to: {MULTISEED_MGG2L_CSV}")

    per_query_df = pd.DataFrame({
        "study_id": [str(s) for s in p1_study_ids],
        "p1_ndcg10": p1_ndcg10, "p1_prec5": p1_prec5,
        "mg_ndcg10": mg_ndcg10, "mg_prec5": mg_prec5,
    })
    per_query_df.to_csv(PER_QUERY_CACHE_PATH, index=False)
    print(f"Saved per-query cache: {PER_QUERY_CACHE_PATH}")


if __name__ == "__main__":
    main()
