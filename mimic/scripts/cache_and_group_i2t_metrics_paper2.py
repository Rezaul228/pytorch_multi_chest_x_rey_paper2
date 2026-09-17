#!/usr/bin/env python3
"""
For seeds 42, 17, 123: compute per-query Image->Text nDCG@10 and
Precision@5 for BOTH Paper 1 baseline and MG-G2L (reusing the exact same
verified scoring pipeline from paper2_graded_relevance_eval.py), SAVE them
to per-seed CSVs (so future re-grouping is free, unlike before), and then
split the restricted-subset (>=1 confirmed positive finding, n=8,968)
per-query results into THREE groups by section availability
(section_boundaries_test_paper2.csv):
  - both_sections:    has_find=True  AND has_imp=True
  - impression_only:  has_find=False AND has_imp=True
  - findings_only:    has_find=True  AND has_imp=False

The restricted-subset mask and ideal/total_relevant arrays depend only on
CheXpert labels (not on any model), so they're computed ONCE and reused
across all seeds/models -- no need to recompute per seed.

Inference only -- no training, no checkpoint modification.

Outputs:
  per_query_i2t_metrics_seed_{seed}.csv  (study_id, p1_ndcg10, p1_prec5, mg_ndcg10, mg_prec5)
    for seed in 42, 17, 123
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
MIMIC_DATA_DIR = os.path.join(PROJECT_DIR, "mimic", "data")
MIMIC_RESULTS_DIR = os.path.join(PROJECT_DIR, "mimic", "results")

import paper2_graded_relevance_eval as scoring  # reused, unchanged

# NEW: overridable via CLI, default to the exact original hardcoded values so
# an old invocation with no flags behaves identically to before.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--binary-labels", default=scoring.DEFAULT_BINARY_LABELS_PATH)
_parser.add_argument("--seeds", default="42,17,123", help="comma-separated seed list")
_parser.add_argument("--output-template", default="per_query_i2t_metrics_seed_{seed}.csv",
                      help="filename template, must contain {seed}")
_args, _ = _parser.parse_known_args()

SEEDS = [int(s) for s in _args.seeds.split(",")]
BINARY_LABELS_PATH_OVERRIDE = _args.binary_labels
OUTPUT_TEMPLATE = _args.output_template
SECTION_BOUNDARIES_PATH = os.path.join(MIMIC_DATA_DIR, "section_boundaries_test_paper2.csv")

PAPER1_MODEL_PATHS = {
    seed: os.path.join(
        PROJECT_DIR, "saved_models",
        f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_{seed}"
        + ("_rerun" if seed == 2021 else ""),  # v1_seed_2021 is a byte-duplicate of seed_123; the genuine retrain is _rerun
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

    # ---------- Build restricted mask / ideal_top10 / total_relevant ONCE (label-only, model-independent) ----------
    print("\nLoading Paper1 seed_42 to get canonical study_id ordering (label data is model-independent)...")
    ref_model = scoring.load_model(PAPER1_MODEL_PATHS[42]).to(device)
    ref_test_dataset = scoring.load_test_dataset(scoring.DEFAULT_SHARD_SUBFOLDER)
    _, _, study_ids = scoring.compute_embeddings(ref_model, ref_test_dataset, device)
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n = len(study_ids)
    B, _ = scoring.build_binary_matrix(study_ids, BINARY_LABELS_PATH_OVERRIDE)
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

    # ---------- Section-availability groups (section_boundaries_test_paper2.csv) ----------
    boundaries_df = pd.read_csv(SECTION_BOUNDARIES_PATH, dtype={"study_id": str}).set_index("study_id")
    study_id_strs = [str(s) for s in study_ids]

    def get_group(sid):
        if sid not in boundaries_df.index:
            return None
        hf = bool(boundaries_df.loc[sid, "has_findings"])
        hi = bool(boundaries_df.loc[sid, "has_impression"])
        if hf and hi:
            return "both_sections"
        elif (not hf) and hi:
            return "impression_only"
        elif hf and (not hi):
            return "findings_only"
        else:
            return "neither"  # not expected within restricted subset, but handled

    groups = np.array([get_group(sid) for sid in study_id_strs])

    all_per_seed_tables = {}

    for seed in SEEDS:
        print(f"\n{'='*90}\nSEED {seed}\n{'='*90}")

        print(f"Paper 1 baseline (seed_{seed})...")
        p1_model = scoring.load_model(PAPER1_MODEL_PATHS[seed]).to(device)
        p1_test_dataset = scoring.load_test_dataset(scoring.DEFAULT_SHARD_SUBFOLDER)
        p1_img_emb, p1_txt_emb, p1_study_ids = scoring.compute_embeddings(p1_model, p1_test_dataset, device)
        assert list(p1_study_ids) == list(study_ids), f"seed {seed}: Paper1 study_id order mismatch"
        del p1_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        p1_img_t = torch.FloatTensor(p1_img_emb).to(device)
        p1_txt_t = torch.FloatTensor(p1_txt_emb).to(device)
        p1_i2t_sim = torch.matmul(p1_img_t, p1_txt_t.transpose(0, 1))
        _, p1_ndcg10, _, p1_prec5 = scoring.compute_metrics_for_direction(
            p1_i2t_sim, R_full_np, ideal_top10, total_relevant, n
        )
        del p1_img_t, p1_txt_t, p1_i2t_sim
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"MG-G2L (seed_{seed}, section-aware)...")
        mg_img_emb, mg_txt_emb, mg_study_ids = load_mgg2l_embeddings(MGG2L_CHECKPOINT_PATHS[seed], device)
        assert list(mg_study_ids) == list(study_ids), f"seed {seed}: MG-G2L study_id order mismatch"

        mg_img_t = torch.FloatTensor(mg_img_emb).to(device)
        mg_txt_t = torch.FloatTensor(mg_txt_emb).to(device)
        mg_i2t_sim = torch.matmul(mg_img_t, mg_txt_t.transpose(0, 1))
        _, mg_ndcg10, _, mg_prec5 = scoring.compute_metrics_for_direction(
            mg_i2t_sim, R_full_np, ideal_top10, total_relevant, n
        )
        del mg_img_t, mg_txt_t, mg_i2t_sim
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Save per-query cache for this seed.
        per_query_df = pd.DataFrame({
            "study_id": study_id_strs,
            "p1_ndcg10": p1_ndcg10, "p1_prec5": p1_prec5,
            "mg_ndcg10": mg_ndcg10, "mg_prec5": mg_prec5,
        })
        out_path = os.path.join(MIMIC_RESULTS_DIR, OUTPUT_TEMPLATE.format(seed=seed))
        per_query_df.to_csv(out_path, index=False)
        print(f"Saved per-query cache: {out_path}")

        # 3-group breakdown, restricted subset only.
        rows = []
        for group_name in ["both_sections", "impression_only", "findings_only"]:
            mask = restricted_mask & (groups == group_name)
            gn = int(np.sum(mask))
            for metric, p1_arr, mg_arr in [("nDCG@10", p1_ndcg10, mg_ndcg10), ("Precision@5", p1_prec5, mg_prec5)]:
                p1_mean = float(np.mean(p1_arr[mask])) if gn > 0 else float("nan")
                mg_mean = float(np.mean(mg_arr[mask])) if gn > 0 else float("nan")
                rows.append({"group": group_name, "n": gn, "metric": metric,
                             "paper1": p1_mean, "mgg2l": mg_mean, "delta": mg_mean - p1_mean})
        all_per_seed_tables[seed] = rows

    print("\n" + "=" * 100)
    print("3-GROUP BREAKDOWN BY SECTION AVAILABILITY (restricted subset, I2T nDCG@10 / Precision@5)")
    print("=" * 100)
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        header = f"{'group':<17} | {'n':<6} | {'metric':<12} | {'Paper 1':<10} | {'MG-G2L':<10} | {'delta':<10}"
        print(header)
        print("-" * len(header))
        for r in all_per_seed_tables[seed]:
            print(f"{r['group']:<17} | {r['n']:<6} | {r['metric']:<12} | {r['paper1']:<10.4f} | {r['mgg2l']:<10.4f} | {r['delta']:<+10.4f}")


if __name__ == "__main__":
    main()
