#!/usr/bin/env python3
"""
Graded-relevance retrieval evaluation (nDCG@5, nDCG@10, mAP@10, Precision@5)
for a MultimodalFusion checkpoint, using CheXpert binary labels
(test_labels_chexpert_binary.csv) as the graded relevance signal.

Computes both Image->Text and Text->Image directions, each reported two ways:
  - FULL: averaged over all test queries
  - RESTRICTED: averaged only over queries with >=1 confirmed positive
    finding (excluding "No Finding") -- i.e. excluding queries for which
    graded relevance is degenerate (no possible relevant candidate exists).

Read-only w.r.t. the model: no training, no checkpoint modification.

Usage:
    python3 paper2_graded_relevance_eval.py
    python3 paper2_graded_relevance_eval.py --output paper1_baseline_graded_relevance_v2.csv
    python3 paper2_graded_relevance_eval.py --model-path /path/to/model_weights.pth --force

Memory note: builds a dense (N x N) similarity matrix and a dense (N x N)
relevance matrix (int8) on GPU/CPU -- for the ~12.4K-sample MIMIC test set
this is a few hundred MB, fine on a single GPU node. Encoding all samples
dominates runtime (~5-10 min on a Volta GPU). Submit via sbatch rather than
running on the login node -- see submit_training_simple_v2.sh for the
SLURM header pattern used elsewhere in this project (partition=volta,
gres=gpu:1, mem=64G).
"""

import argparse
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
MIMIC_DATA_DIR = os.path.join(PROJECT_DIR, "mimic", "data")
MIMIC_RESULTS_DIR = os.path.join(PROJECT_DIR, "mimic", "results")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42",
    "export", "model_weights.pth",
)
DEFAULT_BINARY_LABELS_PATH = os.path.join(MIMIC_DATA_DIR, "test_labels_chexpert_binary.csv")
DEFAULT_OUTPUT_CSV_PATH = os.path.join(MIMIC_RESULTS_DIR, "paper1_baseline_graded_relevance.csv")
DEFAULT_SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = 64
TOP_K = 10


def load_tokenizer_from_metadata(shard_subfolder):
    metadata_path = paths.get_metadata_path(shard_subfolder)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get("tokenizer")
    if hasattr(tokenizer, "word_index") and not hasattr(tokenizer, "word2idx"):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    return tokenizer


def load_model(model_path):
    print(f"Loading model from: {model_path}")
    model = MultimodalFusion(
        vocab_size=get_vocab_size(),
        embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"],
        num_layers=get_current_config()["num_layers"],
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Total parameters: {total_params:,} ({total_params/1e6:.3f}M)")
    return model


def load_test_dataset(shard_subfolder):
    print("Loading FULL test dataset (no evaluation, no modification)...")
    data_loader = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=shard_subfolder)
    data_loader.tokenizer = load_tokenizer_from_metadata(shard_subfolder)
    data_loader.load_data(max_samples=None, skip_processing=True)
    test_dataset = data_loader.get_test_data(num_samples=None)
    print(f"Test dataset size: {len(test_dataset)}")
    return test_dataset


def compute_embeddings(model, test_dataset, device):
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

            if len(batch_images.shape) == 4 and batch_images.shape[-1] == 3:
                batch_images = batch_images.permute(0, 3, 1, 2)

            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)

            image_emb, text_emb = model((batch_images, batch_captions), training=False)

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


def build_binary_matrix(study_ids, binary_labels_path):
    df = pd.read_csv(binary_labels_path)
    df = df.set_index("study_id")

    finding_cols = [c for c in df.columns if c not in ("subject_id",) and c != "No Finding"]
    assert len(finding_cols) == 13, f"expected 13 non-'No Finding' columns, got {len(finding_cols)}: {finding_cols}"

    aligned = df.loc[study_ids, finding_cols]
    assert not aligned.isnull().values.any(), "Some study_ids in embeddings not found in binary label CSV, or NaNs present"

    B = aligned.values.astype(np.float32)
    return B, finding_cols


def dcg(rel_row, k):
    """Standard exponential-gain DCG: sum (2^rel - 1) / log2(rank + 1)."""
    rel_row = rel_row[:k]
    discounts = np.log2(np.arange(2, k + 2))
    gains = np.power(2.0, rel_row.astype(np.float64)) - 1.0
    return np.sum(gains / discounts)


def compute_metrics_for_direction(sim_matrix, R_full_np, ideal_top10, total_relevant, n, top_k=TOP_K):
    """
    Per-query nDCG@5, nDCG@10, mAP@10, Precision@5 for one retrieval direction.

    IDCG (for nDCG) and the relevant-count normalizer (for mAP) both use the
    TRUE best-possible ranking across the whole corpus (ideal_top10 /
    total_relevant), not just the retrieved top-k -- this is the standard/
    rigorous convention, not a top-k-only approximation.
    """
    topk_scores, topk_idx = torch.topk(sim_matrix, k=top_k, dim=1)
    topk_idx = topk_idx.cpu().numpy()

    rows = np.arange(n)[:, None]
    rel_topk = R_full_np[rows, topk_idx]

    ndcg5 = np.zeros(n)
    ndcg10 = np.zeros(n)
    ap10 = np.zeros(n)
    prec5 = np.zeros(n)

    for i in range(n):
        rel_i = rel_topk[i]
        ideal_i = ideal_top10[i]

        dcg5_i = dcg(rel_i, 5)
        idcg5_i = dcg(ideal_i, 5)
        ndcg5[i] = (dcg5_i / idcg5_i) if idcg5_i > 0 else 0.0

        dcg10_i = dcg(rel_i, 10)
        idcg10_i = dcg(ideal_i, 10)
        ndcg10[i] = (dcg10_i / idcg10_i) if idcg10_i > 0 else 0.0

        rel_bin = (rel_i > 0).astype(np.float64)
        prec5[i] = np.mean(rel_bin[:5])

        R_i = min(int(total_relevant[i]), top_k)
        if R_i > 0:
            precisions_at_k = np.cumsum(rel_bin) / np.arange(1, top_k + 1)
            ap10[i] = np.sum(precisions_at_k * rel_bin) / R_i
        else:
            ap10[i] = 0.0

    return ndcg5, ndcg10, ap10, prec5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                         help="Path to model_weights.pth (default: Paper 1 seed_42 checkpoint)")
    parser.add_argument("--shard-subfolder", default=DEFAULT_SHARD_SUBFOLDER,
                         help="paths.py shard subfolder to load test data from")
    parser.add_argument("--binary-labels", default=DEFAULT_BINARY_LABELS_PATH,
                         help="Path to test_labels_chexpert_binary.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV_PATH,
                         help="Where to save the summary CSV")
    parser.add_argument("--force", action="store_true",
                         help="Allow overwriting an existing output file (default: refuse)")
    parser.add_argument("--print-examples", type=int, default=0,
                         help="If >0, print this many example queries (I2T) for manual sanity check")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {args.output} (pass --force to override)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model_path)
    model = model.to(device)

    test_dataset = load_test_dataset(args.shard_subfolder)
    image_embeddings, text_embeddings, study_ids = compute_embeddings(model, test_dataset, device)

    n = image_embeddings.shape[0]
    print(f"\nBuilding I2T and T2I similarity matrices ({n} x {n})...")
    image_emb_t = torch.FloatTensor(image_embeddings).to(device)
    text_emb_t = torch.FloatTensor(text_embeddings).to(device)

    i2t_sim = torch.matmul(image_emb_t, text_emb_t.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb_t, image_emb_t.transpose(0, 1))
    print("Similarity matrices computed.")

    print("\nBuilding graded-relevance matrix from CheXpert binary labels...")
    B, finding_cols = build_binary_matrix(study_ids, args.binary_labels)
    B_t = torch.FloatTensor(B).to(device)

    R_full = torch.matmul(B_t, B_t.transpose(0, 1))
    R_full_np = R_full.cpu().numpy().astype(np.int8)
    del R_full, B_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Relevance matrix shape: {R_full_np.shape}, dtype: {R_full_np.dtype}")

    print("Computing ideal (best-case) rankings for IDCG normalization (shared across both directions)...")
    ideal_sorted = -np.sort(-R_full_np, axis=1)
    ideal_top10 = ideal_sorted[:, :TOP_K]
    total_relevant = np.sum(R_full_np > 0, axis=1)
    del ideal_sorted

    restricted_mask = total_relevant > 0
    restricted_n = int(np.sum(restricted_mask))
    zero_pct = 100.0 * (1 - restricted_mask.mean())
    print(f"Restricted-subset n (>=1 confirmed positive finding, excl. 'No Finding'): {restricted_n} / {n}  (excluded {zero_pct:.2f}%)")

    print("\nComputing Image->Text metrics...")
    i2t_ndcg5, i2t_ndcg10, i2t_ap10, i2t_prec5 = compute_metrics_for_direction(
        i2t_sim, R_full_np, ideal_top10, total_relevant, n
    )

    print("Computing Text->Image metrics...")
    t2i_ndcg5, t2i_ndcg10, t2i_ap10, t2i_prec5 = compute_metrics_for_direction(
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
    print(f"SUMMARY TABLE: Graded-relevance retrieval metrics -- {args.model_path}")
    print("=" * 100)
    header = f"{'metric':<14} | {'direction':<12} | {'full-corpus (n=' + str(n) + ')':<22} | {'restricted (n=' + str(restricted_n) + ')':<22}"
    print(header)
    print("-" * len(header))
    for r in rows_out:
        print(f"{r['metric']:<14} | {r['direction']:<12} | {r['full_corpus_value']:<22} | {r['restricted_subset_value']:<22}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "direction", "full_corpus_value", "full_corpus_n",
            "restricted_subset_value", "restricted_subset_n",
        ])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"\nSaved summary table to: {args.output}")
    print(f"File exists: {os.path.exists(args.output)}")
    print(f"File size: {os.path.getsize(args.output)} bytes")

    if args.print_examples > 0:
        print("\n" + "=" * 70)
        print(f"{args.print_examples} EXAMPLE QUERIES (Image->Text) FOR MANUAL SANITY CHECK")
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

        for i in range(args.print_examples):
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
