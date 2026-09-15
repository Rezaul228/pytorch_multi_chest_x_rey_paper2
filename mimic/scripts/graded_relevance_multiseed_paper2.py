#!/usr/bin/env python3
"""
Extend the graded-relevance evaluation (already verified for seed_42) to
seeds 17 and 123, for BOTH the Paper 1 baseline and the MG-G2L checkpoint,
on the same 12,429-sample test set.

Reuses the EXACT SAME scoring code as paper2_graded_relevance_eval.py
(build_binary_matrix, dcg, compute_metrics_for_direction) -- imported, not
reimplemented. Only the model-loading / embedding-generation differs:
  - Paper 1 baseline: base_models_refactored_v1.MultimodalFusion +
    data_loader_v1.IndianaDataLoader (original, no section fields) -- same
    as paper2_graded_relevance_eval.py's own default path.
  - MG-G2L: base_models_refactored_v1_paper2.MultimodalFusion +
    data_loader_v1_paper2.IndianaDataLoader (section fields attached),
    passing findings_token_count/has_find/has_imp/token_ids into the model
    call -- same as mg_g2l_graded_relevance_eval.py.

Inference only -- no training, no checkpoint modification.

Outputs:
  paper1_baseline_graded_relevance_multiseed.csv  (seeds 17, 123)
  mg_g2l_graded_relevance_multiseed.csv           (seeds 17, 123)
"""

import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
MIMIC_DATA_DIR = os.path.join(PROJECT_DIR, "mimic", "data")
MIMIC_RESULTS_DIR = os.path.join(PROJECT_DIR, "mimic", "results")

# Reuse the already-verified scoring/metric code UNCHANGED.
import paper2_graded_relevance_eval as scoring

BINARY_LABELS_PATH = os.path.join(MIMIC_DATA_DIR, "test_labels_chexpert_binary.csv")
SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = scoring.BATCH_SIZE
TOP_K = scoring.TOP_K
SEEDS = [17, 123]

PAPER1_MODEL_PATHS = {
    seed: os.path.join(
        PROJECT_DIR, "saved_models",
        f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_{seed}",
        "export", "model_weights.pth",
    )
    for seed in SEEDS
}
MGG2L_CHECKPOINT_PATHS = {
    seed: os.path.join(
        PROJECT_DIR, "saved_models",
        f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_{seed}",
        "export", "checkpoint_resume.pth",
    )
    for seed in SEEDS
}

PAPER1_OUTPUT_CSV = os.path.join(MIMIC_RESULTS_DIR, "paper1_baseline_graded_relevance_multiseed.csv")
MGG2L_OUTPUT_CSV = os.path.join(MIMIC_RESULTS_DIR, "mg_g2l_graded_relevance_multiseed.csv")


def compute_graded_relevance(image_embeddings, text_embeddings, study_ids, device):
    """Everything after embeddings are produced: reuses scoring module's code exactly."""
    n = image_embeddings.shape[0]
    image_emb_t = torch.FloatTensor(image_embeddings).to(device)
    text_emb_t = torch.FloatTensor(text_embeddings).to(device)

    i2t_sim = torch.matmul(image_emb_t, text_emb_t.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb_t, image_emb_t.transpose(0, 1))

    B, finding_cols = scoring.build_binary_matrix(study_ids, BINARY_LABELS_PATH)
    B_t = torch.FloatTensor(B).to(device)

    R_full = torch.matmul(B_t, B_t.transpose(0, 1))
    R_full_np = R_full.cpu().numpy().astype(np.int8)
    del R_full, B_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ideal_sorted = -np.sort(-R_full_np, axis=1)
    ideal_top10 = ideal_sorted[:, :TOP_K]
    total_relevant = np.sum(R_full_np > 0, axis=1)
    del ideal_sorted

    restricted_mask = total_relevant > 0
    restricted_n = int(np.sum(restricted_mask))

    i2t_ndcg5, i2t_ndcg10, i2t_ap10, i2t_prec5 = scoring.compute_metrics_for_direction(
        i2t_sim, R_full_np, ideal_top10, total_relevant, n
    )
    t2i_ndcg5, t2i_ndcg10, t2i_ap10, t2i_prec5 = scoring.compute_metrics_for_direction(
        t2i_sim, R_full_np, ideal_top10, total_relevant, n
    )

    def full_and_restricted(arr):
        return float(np.mean(arr)), float(np.mean(arr[restricted_mask]))

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
                "metric": metric_name, "direction": direction,
                "full_corpus_value": round(full_val, 4), "full_corpus_n": n,
                "restricted_subset_value": round(restricted_val, 4), "restricted_subset_n": restricted_n,
            })
    return rows_out


def load_paper1_baseline_embeddings(model_path, device):
    from base_models_refactored_v1 import MultimodalFusion
    from data_loader_v1 import IndianaDataLoader
    from config import get_vocab_size, get_embed_dim, get_current_config

    model = MultimodalFusion(
        vocab_size=get_vocab_size(), embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"], num_layers=get_current_config()["num_layers"],
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval().to(device)

    dl = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = scoring.load_tokenizer_from_metadata(SHARD_SUBFOLDER)
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"  [Paper1] test dataset size: {len(test_dataset)}")

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_image_emb, all_text_emb, all_study_ids = [], [], []
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
                all_study_ids.extend(int(sid.item()) for sid in batch_study_ids)
            else:
                all_study_ids.extend(int(sid) for sid in batch_study_ids)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (np.concatenate(all_image_emb, axis=0), np.concatenate(all_text_emb, axis=0),
            np.array(all_study_ids, dtype=np.int64))


def load_mgg2l_embeddings(checkpoint_path, device):
    from base_models_refactored_v1_paper2 import MultimodalFusion
    from data_loader_v1_paper2 import IndianaDataLoader
    from config import get_vocab_size, get_embed_dim, get_current_config

    model = MultimodalFusion(
        vocab_size=get_vocab_size(), embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"], num_layers=get_current_config()["num_layers"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  [MG-G2L] checkpoint epoch field: {checkpoint['epoch']} (-> epoch {checkpoint['epoch'] + 1})")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval().to(device)

    dl = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = scoring.load_tokenizer_from_metadata(SHARD_SUBFOLDER)
    dl.load_data(max_samples=None, skip_processing=True)
    test_dataset = dl.get_test_data(num_samples=None)
    print(f"  [MG-G2L] test dataset size: {len(test_dataset)}  section_boundaries: {len(test_dataset.section_boundaries)}  fallback: {test_dataset.fallback_count}")

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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


def write_csv(path, rows_by_seed):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "metric", "direction", "full_corpus_value", "full_corpus_n",
            "restricted_subset_value", "restricted_subset_n",
        ])
        writer.writeheader()
        for seed, rows in rows_by_seed.items():
            for r in rows:
                writer.writerow({"seed": seed, **r})


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for path in list(PAPER1_MODEL_PATHS.values()) + list(MGG2L_CHECKPOINT_PATHS.values()):
        assert os.path.exists(path), f"Missing file: {path}"

    paper1_rows_by_seed = {}
    mgg2l_rows_by_seed = {}

    for seed in SEEDS:
        print(f"\n{'='*90}\nSEED {seed} -- Paper 1 baseline\n{'='*90}")
        img_emb, txt_emb, study_ids = load_paper1_baseline_embeddings(PAPER1_MODEL_PATHS[seed], device)
        paper1_rows_by_seed[seed] = compute_graded_relevance(img_emb, txt_emb, study_ids, device)

        print(f"\n{'='*90}\nSEED {seed} -- MG-G2L (section-aware)\n{'='*90}")
        img_emb, txt_emb, study_ids = load_mgg2l_embeddings(MGG2L_CHECKPOINT_PATHS[seed], device)
        mgg2l_rows_by_seed[seed] = compute_graded_relevance(img_emb, txt_emb, study_ids, device)

    write_csv(PAPER1_OUTPUT_CSV, paper1_rows_by_seed)
    write_csv(MGG2L_OUTPUT_CSV, mgg2l_rows_by_seed)
    print(f"\nSaved: {PAPER1_OUTPUT_CSV}")
    print(f"Saved: {MGG2L_OUTPUT_CSV}")

    print("\n" + "=" * 120)
    print("3-SEED TABLE (seed 42 from prior single-seed runs; 17/123 from this job) -- RESTRICTED subset (n=8,968)")
    print("=" * 120)
    header = f"{'seed':<6} | {'metric':<12} | {'direction':<12} | {'P1 restricted':<14} | {'MGG2L restricted':<17} | {'delta':<10}"
    print(header)
    print("-" * len(header))
    for seed in SEEDS:
        for r_p1, r_mg in zip(paper1_rows_by_seed[seed], mgg2l_rows_by_seed[seed]):
            assert r_p1["metric"] == r_mg["metric"] and r_p1["direction"] == r_mg["direction"]
            delta = r_mg["restricted_subset_value"] - r_p1["restricted_subset_value"]
            print(f"{seed:<6} | {r_p1['metric']:<12} | {r_p1['direction']:<12} | "
                  f"{r_p1['restricted_subset_value']:<14.4f} | {r_mg['restricted_subset_value']:<17.4f} | {delta:<+10.4f}")


if __name__ == "__main__":
    main()
