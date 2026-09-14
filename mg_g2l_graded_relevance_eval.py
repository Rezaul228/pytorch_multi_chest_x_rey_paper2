#!/usr/bin/env python3
"""
Graded-relevance retrieval evaluation for the MG-G2L checkpoint
(checkpoint_resume.pth, epoch 50), computed with the IDENTICAL scoring/
metric code used for paper1_baseline_graded_relevance.csv
(paper2_graded_relevance_eval.py) -- build_binary_matrix(), dcg(), and
compute_metrics_for_direction() are imported and reused UNCHANGED, not
reimplemented.

The ONLY thing that differs from the Paper 1 baseline script is how the
embeddings/similarity matrices are produced: this uses the MG-G2L
checkpoint via the paper2 model + paper2 DataLoader, passing
findings_token_count/has_find/has_imp/token_ids into the model call (the
same section-aware call verified in
train_test_cross_modal_evaluation_v1_paper2.py's
evaluate_cross_modal_retrieval_streaming()). Everything after the
similarity matrix -- relevance scoring, nDCG/mAP/Precision computation,
full-vs-restricted split -- is the same imported code.

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

# Reuse the Paper 1 baseline script's scoring/metric code UNCHANGED.
import paper2_graded_relevance_eval as baseline

from base_models_refactored_v1_paper2 import MultimodalFusion
from data_loader_v1_paper2 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config

MGG2L_EXPERIMENT_NAME = "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_42"
MGG2L_CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR, "saved_models", MGG2L_EXPERIMENT_NAME, "export", "checkpoint_resume.pth"
)
BINARY_LABELS_PATH = os.path.join(PROJECT_DIR, "test_labels_chexpert_binary.csv")
OUTPUT_CSV_PATH = os.path.join(PROJECT_DIR, "mg_g2l_graded_relevance.csv")
SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = baseline.BATCH_SIZE  # same batch size as the baseline script
TOP_K = baseline.TOP_K


def load_mgg2l_model(checkpoint_path, device):
    print(f"Loading MG-G2L model from: {checkpoint_path}")
    model = MultimodalFusion(
        vocab_size=get_vocab_size(),
        embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"],
        num_layers=get_current_config()["num_layers"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch field: {checkpoint['epoch']} (0-indexed -> epoch {checkpoint['epoch'] + 1})")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    return model.to(device)


def load_test_dataset_section_aware(shard_subfolder):
    print("Loading FULL test dataset via data_loader_v1_paper2 (section-aware fields attached)...")
    data_loader = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=shard_subfolder)
    data_loader.tokenizer = baseline.load_tokenizer_from_metadata(shard_subfolder)
    data_loader.load_data(max_samples=None, skip_processing=True)
    test_dataset = data_loader.get_test_data(num_samples=None)
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Section boundaries loaded: {len(test_dataset.section_boundaries)}  fallback: {test_dataset.fallback_count}")
    return test_dataset


def compute_embeddings_section_aware(model, test_dataset, device):
    """Same structure as baseline.compute_embeddings(), but passes
    findings_token_count/has_find/has_imp/token_ids into the model call --
    the ONLY change from the Paper 1 baseline script."""
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_image_emb = []
    all_text_emb = []
    all_study_ids = []

    sample_count = 0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            batch_images = batch["images"]
            batch_captions = batch["captions"]
            batch_study_ids = batch["study_ids"]
            findings_token_count = batch["findings_token_count"]
            has_find = batch["has_find"]
            has_imp = batch["has_imp"]

            if len(batch_images.shape) == 4 and batch_images.shape[-1] == 3:
                batch_images = batch_images.permute(0, 3, 1, 2)

            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)
            findings_token_count = findings_token_count.to(device)
            has_find = has_find.to(device)
            has_imp = has_imp.to(device)

            model_output = model(
                (batch_images, batch_captions),
                training=False,
                findings_token_count=findings_token_count,
                has_find=has_find,
                has_imp=has_imp,
                token_ids=batch_captions,  # same captions tensor -- not duplicated
            )
            image_emb, text_emb = model_output[0], model_output[1]

            all_image_emb.append(image_emb.cpu().numpy())
            all_text_emb.append(text_emb.cpu().numpy())

            if isinstance(batch_study_ids, torch.Tensor):
                study_ids_batch = [int(sid.item()) for sid in batch_study_ids]
            else:
                study_ids_batch = [int(sid) for sid in batch_study_ids]
            all_study_ids.extend(study_ids_batch)

            sample_count += len(batch_images)
            batch_count += 1
            if batch_count % 20 == 0:
                print(f"  Encoded {sample_count} samples in {batch_count} batches...")

    image_embeddings = np.concatenate(all_image_emb, axis=0)
    text_embeddings = np.concatenate(all_text_emb, axis=0)
    study_ids = np.array(all_study_ids, dtype=np.int64)

    print(f"Encoded {sample_count} samples total.")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")

    return image_embeddings, text_embeddings, study_ids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.path.exists(OUTPUT_CSV_PATH):
        raise FileExistsError(f"Refusing to overwrite existing file: {OUTPUT_CSV_PATH}")

    model = load_mgg2l_model(MGG2L_CHECKPOINT_PATH, device)
    test_dataset = load_test_dataset_section_aware(SHARD_SUBFOLDER)
    image_embeddings, text_embeddings, study_ids = compute_embeddings_section_aware(model, test_dataset, device)

    n = image_embeddings.shape[0]
    print(f"\nBuilding I2T and T2I similarity matrices ({n} x {n})...")
    image_emb_t = torch.FloatTensor(image_embeddings).to(device)
    text_emb_t = torch.FloatTensor(text_embeddings).to(device)

    i2t_sim = torch.matmul(image_emb_t, text_emb_t.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb_t, image_emb_t.transpose(0, 1))
    print("Similarity matrices computed.")

    # ---- Everything below this point reuses the Paper 1 baseline script's code UNCHANGED. ----

    print("\nBuilding graded-relevance matrix from CheXpert binary labels (reused code)...")
    B, finding_cols = baseline.build_binary_matrix(study_ids, BINARY_LABELS_PATH)
    B_t = torch.FloatTensor(B).to(device)

    R_full = torch.matmul(B_t, B_t.transpose(0, 1))
    R_full_np = R_full.cpu().numpy().astype(np.int8)
    del R_full, B_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Relevance matrix shape: {R_full_np.shape}, dtype: {R_full_np.dtype}")

    print("Computing ideal (best-case) rankings for IDCG normalization (reused code)...")
    ideal_sorted = -np.sort(-R_full_np, axis=1)
    ideal_top10 = ideal_sorted[:, :TOP_K]
    total_relevant = np.sum(R_full_np > 0, axis=1)
    del ideal_sorted

    restricted_mask = total_relevant > 0
    restricted_n = int(np.sum(restricted_mask))
    zero_pct = 100.0 * (1 - restricted_mask.mean())
    print(f"Restricted-subset n (>=1 confirmed positive finding, excl. 'No Finding'): {restricted_n} / {n}  (excluded {zero_pct:.2f}%)")

    print("\nComputing Image->Text metrics (reused compute_metrics_for_direction)...")
    i2t_ndcg5, i2t_ndcg10, i2t_ap10, i2t_prec5 = baseline.compute_metrics_for_direction(
        i2t_sim, R_full_np, ideal_top10, total_relevant, n
    )

    print("Computing Text->Image metrics (reused compute_metrics_for_direction)...")
    t2i_ndcg5, t2i_ndcg10, t2i_ap10, t2i_prec5 = baseline.compute_metrics_for_direction(
        t2i_sim, R_full_np, ideal_top10, total_relevant, n
    )

    def full_and_restricted(arr):
        full_val = float(np.mean(arr))
        restricted_val = float(np.mean(arr[restricted_mask]))
        return full_val, restricted_val

    rows_out = []
    for metric_name, i2t_arr, t2i_arr in [
        ("nDCG@5", i2t_ndcg5, t2i_ndcg5),
        ("nDCG@10", i2t_ndcg10, t2i_ndcg10),
        ("mAP@10", i2t_ap10, t2i_ap10),
        ("Precision@5", i2t_prec5, t2i_prec5),
    ]:
        for direction, arr in [("Image->Text", i2t_arr), ("Text->Image", t2i_arr)]:
            full_val, restricted_val = full_and_restricted(arr)
            rows_out.append({
                "metric": metric_name,
                "direction": direction,
                "full_corpus_value": round(full_val, 4),
                "full_corpus_n": n,
                "restricted_subset_value": round(restricted_val, 4),
                "restricted_subset_n": restricted_n,
            })

    print("\n" + "=" * 100)
    print(f"SUMMARY TABLE: MG-G2L graded-relevance metrics -- {MGG2L_CHECKPOINT_PATH}")
    print("=" * 100)
    header = f"{'metric':<14} | {'direction':<12} | {'full-corpus (n=' + str(n) + ')':<22} | {'restricted (n=' + str(restricted_n) + ')':<22}"
    print(header)
    print("-" * len(header))
    for r in rows_out:
        print(f"{r['metric']:<14} | {r['direction']:<12} | {r['full_corpus_value']:<22} | {r['restricted_subset_value']:<22}")

    # Exact same column format as paper1_baseline_graded_relevance.csv for direct merging.
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "direction", "full_corpus_value", "full_corpus_n",
            "restricted_subset_value", "restricted_subset_n",
        ])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"\nSaved summary table to: {OUTPUT_CSV_PATH}")
    print(f"File exists: {os.path.exists(OUTPUT_CSV_PATH)}")
    print(f"File size: {os.path.getsize(OUTPUT_CSV_PATH)} bytes")

    # ---- Side-by-side comparison against the Paper 1 baseline CSV ----
    baseline_csv_path = os.path.join(PROJECT_DIR, "paper1_baseline_graded_relevance.csv")
    baseline_df = pd.read_csv(baseline_csv_path).set_index(["metric", "direction"])
    mgg2l_df = pd.DataFrame(rows_out).set_index(["metric", "direction"])

    print("\n" + "=" * 120)
    print("SIDE-BY-SIDE COMPARISON: Paper 1 baseline vs MG-G2L")
    print("PRIMARY = restricted subset (n=8,968; scientifically meaningful). Full-corpus shown as secondary reference.")
    print("=" * 120)
    header2 = (f"{'metric':<14} | {'direction':<12} | {'P1 restricted':<14} | {'MGG2L restricted':<17} | "
               f"{'delta (restr.)':<15} | {'P1 full':<10} | {'MGG2L full':<11} | {'delta (full)':<13}")
    print(header2)
    print("-" * len(header2))
    for metric_name in ["nDCG@5", "nDCG@10", "mAP@10", "Precision@5"]:
        for direction in ["Image->Text", "Text->Image"]:
            p1_row = baseline_df.loc[(metric_name, direction)]
            mg_row = mgg2l_df.loc[(metric_name, direction)]
            p1_restr = p1_row["restricted_subset_value"]
            mg_restr = mg_row["restricted_subset_value"]
            delta_restr = mg_restr - p1_restr
            p1_full = p1_row["full_corpus_value"]
            mg_full = mg_row["full_corpus_value"]
            delta_full = mg_full - p1_full
            print(f"{metric_name:<14} | {direction:<12} | {p1_restr:<14.4f} | {mg_restr:<17.4f} | "
                  f"{delta_restr:<+15.4f} | {p1_full:<10.4f} | {mg_full:<11.4f} | {delta_full:<+13.4f}")

    # ---- Sanity check: 3 example queries, same style as the baseline script's --print-examples ----
    print("\n" + "=" * 70)
    print("3 EXAMPLE QUERIES (Image->Text) FOR MANUAL SANITY CHECK -- MG-G2L")
    print("=" * 70)
    topk_scores, topk_idx = torch.topk(i2t_sim, k=5, dim=1)
    topk_scores = topk_scores.cpu().numpy()
    topk_idx = topk_idx.cpu().numpy()
    rows_idx = np.arange(n)[:, None]
    rel_top5 = R_full_np[rows_idx, topk_idx]

    study_id_to_pos = {}
    for idx in range(n):
        pos_cols = [finding_cols[c] for c in range(len(finding_cols)) if B[idx, c] == 1]
        study_id_to_pos[study_ids[idx]] = pos_cols

    for i in range(3):
        q_sid = study_ids[i]
        q_pos = study_id_to_pos[q_sid]
        print(f"\nQuery {i}: study_id={q_sid}  positive_findings={q_pos}")
        print("  Top-5 candidates:")
        for rank in range(5):
            cand_pos_idx = topk_idx[i, rank]
            cand_sid = study_ids[cand_pos_idx]
            cand_score = topk_scores[i, rank]
            cand_rel = rel_top5[i, rank]
            cand_pos = study_id_to_pos[cand_sid]
            print(f"    rank {rank+1}: study_id={cand_sid}  sim={cand_score:.4f}  relevance={cand_rel}  positive_findings={cand_pos}")


if __name__ == "__main__":
    main()
