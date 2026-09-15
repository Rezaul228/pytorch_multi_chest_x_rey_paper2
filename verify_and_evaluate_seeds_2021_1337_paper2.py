#!/usr/bin/env python3
"""
Checksum-verify + evaluate the two new MIMIC seeds (2021_rerun, 1337):

  - Paper 1 baseline: v1_seed_2021_rerun (correctly-retrained replacement for
    the byte-duplicate-of-seed_123 old v1_seed_2021), and v1_seed_1337
    (a brand-new baseline seed never in the study before).
  - MG-G2L (Paper 2): MGG2L_paper2_seed_2021, MGG2L_paper2_seed_1337
    (already trained, matched counterparts).

Does NOT retrain and does NOT modify any model file or any existing results
file for seeds 17/42/123/3407 -- all outputs here go to new, clearly-named
files. Given the seed_2021 mislabeling history, every checkpoint is
checksum-verified against ALL previously-confirmed-distinct checkpoints
(and against each other) BEFORE it is loaded or evaluated; a duplicate
STOPS that specific checkpoint instead of being used as evidence.

Steps (see docstring sections below): 1-2 checksum verification, 3 stop-on-
duplicate, 4 strict=True clean-load check, 5 Paper-1 R@K/MRR streaming eval,
6 MG-G2L (section-aware) R@K/MRR streaming eval, 7 Paper1-vs-MGG2L
comparison per seed, 8 graded-relevance (restricted nDCG@10/mAP@10/
Precision@5), 9 full report + proposed 5/6-seed multiseed update (labeled
PROPOSED, pending review), 10 new output files only.
"""

import csv
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming as eval_streaming_v1
from train_test_cross_modal_evaluation_v1_paper2 import evaluate_cross_modal_retrieval_streaming as eval_streaming_paper2
import paper2_graded_relevance_eval as scoring  # reused, unchanged

NEW_SEEDS = [2021, 1337]
EXISTING_SEEDS = [17, 42, 123, 3407]

V1_DIRNAME = {
    2021: "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_2021_rerun",
    1337: "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_1337",
    17: "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17",
    42: "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42",
    123: "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_123",
    3407: "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_3407",
}
MGG2L_DIRNAME = {
    seed: f"mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_MGG2L_paper2_seed_{seed}"
    for seed in NEW_SEEDS + EXISTING_SEEDS
}

def v1_weights_path(seed):
    return os.path.join(PROJECT_DIR, "saved_models", V1_DIRNAME[seed], "export", "model_weights.pth")

def mgg2l_weights_path(seed):
    return os.path.join(PROJECT_DIR, "saved_models", MGG2L_DIRNAME[seed], "export", "model_weights.pth")

def mgg2l_checkpoint_path(seed):
    return os.path.join(PROJECT_DIR, "saved_models", MGG2L_DIRNAME[seed], "export", "checkpoint_resume.pth")

OLD_MISLABELED_SEED2021_V1_PATH = os.path.join(
    "/home/abedin/Developments/mimic_ori_full_all_data/saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_2021",
    "export", "model_weights.pth",
)

# Previously verified/published numbers for the 4 already-confirmed seeds, sourced from:
#   Paper1 17/42/123: multi_seed_evaluation_12429_4seeds_results/all_results.json
#   Paper1 3407:      logs/eval_seed3407_full_113795.out (STEP 3 comparison table)
#   MG-G2L 17/123/3407: mgg2l_multiseed_streaming_results.json
#   MG-G2L 42:        logs/eval_streaming_both_113715.out (SIDE-BY-SIDE table)
# Kept as hardcoded, sourced constants (same practice as evaluate_seed3407_full_paper2.py's
# MGG2L_KNOWN_RESULTS) rather than recomputed, since recomputing would waste GPU time on
# already-settled results -- this job only adds 2021/1337.
EXISTING_RK_RESULTS = {
    "paper1": {
        17:   dict(i2t_r1=0.990989, i2t_r5=0.999759, i2t_r10=1.000000, i2t_mrr=0.994982,
                   t2i_r1=0.994207, t2i_r5=1.000000, t2i_r10=1.000000, t2i_mrr=0.997015),
        42:   dict(i2t_r1=0.992759, i2t_r5=0.998713, i2t_r10=0.999517, i2t_mrr=0.995500,
                   t2i_r1=1.000000, t2i_r5=1.000000, t2i_r10=1.000000, t2i_mrr=1.000000),
        123:  dict(i2t_r1=0.971920, i2t_r5=0.995655, i2t_r10=0.997747, i2t_mrr=0.982671,
                   t2i_r1=0.999115, t2i_r5=1.000000, t2i_r10=1.000000, t2i_mrr=0.999557),
        3407: dict(i2t_r1=0.9939,   i2t_r5=0.9986,   i2t_r10=0.9993,   i2t_mrr=0.9960,
                   t2i_r1=1.0000,   t2i_r5=1.0000,   t2i_r10=1.0000,   t2i_mrr=1.0000),
    },
    "mgg2l": {
        17:   dict(i2t_r1=0.998713, i2t_r5=0.999759, i2t_r10=1.000000, i2t_mrr=0.999247,
                   t2i_r1=1.000000, t2i_r5=1.000000, t2i_r10=1.000000, t2i_mrr=1.000000),
        42:   dict(i2t_r1=0.9994,   i2t_r5=1.0000,   i2t_r10=1.0000,   i2t_mrr=0.9996,
                   t2i_r1=1.0000,   t2i_r5=1.0000,   t2i_r10=1.0000,   t2i_mrr=1.0000),
        123:  dict(i2t_r1=0.999920, i2t_r5=1.000000, i2t_r10=1.000000, i2t_mrr=0.999946,
                   t2i_r1=1.000000, t2i_r5=1.000000, t2i_r10=1.000000, t2i_mrr=1.000000),
        3407: dict(i2t_r1=0.999759, i2t_r5=0.999920, i2t_r10=0.999920, i2t_mrr=0.999818,
                   t2i_r1=1.000000, t2i_r5=1.000000, t2i_r10=1.000000, t2i_mrr=1.000000),
    },
}
EXISTING_GRADED_PAPER1_CSVS = [
    (os.path.join(PROJECT_DIR, "paper1_baseline_graded_relevance.csv"), 42),        # no 'seed' column -> tag as 42
    (os.path.join(PROJECT_DIR, "paper1_baseline_graded_relevance_multiseed.csv"), None),  # has 'seed' column
]
EXISTING_GRADED_MGG2L_CSVS = [
    (os.path.join(PROJECT_DIR, "mg_g2l_graded_relevance.csv"), 42),
    (os.path.join(PROJECT_DIR, "mg_g2l_graded_relevance_multiseed.csv"), None),
]

OUT_CHECKSUM_JSON = os.path.join(PROJECT_DIR, "checksum_report_seeds_2021_1337_paper2.json")
OUT_FULL_RESULTS_JSON = os.path.join(PROJECT_DIR, "new_seeds_2021_1337_full_results_paper2.json")
OUT_PAPER1_GRADED_CSV = os.path.join(PROJECT_DIR, "paper1_baseline_graded_relevance_new_seeds.csv")
OUT_MGG2L_GRADED_CSV = os.path.join(PROJECT_DIR, "mg_g2l_graded_relevance_new_seeds.csv")


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_state_dict(path):
    """model_weights.pth is always a plain state_dict for both v1 and MG-G2L exports."""
    return torch.load(path, map_location="cpu", weights_only=True)


def state_dicts_equal(path_a, path_b):
    sa = load_raw_state_dict(path_a)
    sb = load_raw_state_dict(path_b)
    if set(sa.keys()) != set(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa.keys())


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =====================================================================
    # STEP 1-2: SHA-256 checksums for v1 baselines and MG-G2L checkpoints
    # =====================================================================
    print_header("STEP 1-2: SHA-256 checksums (v1 Paper-1 baselines)")
    v1_paths = {s: v1_weights_path(s) for s in EXISTING_SEEDS + NEW_SEEDS}
    v1_paths["2021_old_mislabeled"] = OLD_MISLABELED_SEED2021_V1_PATH
    v1_checksums = {}
    for label, p in v1_paths.items():
        assert os.path.exists(p), f"v1 checkpoint not found for {label}: {p}"
        h = sha256sum(p)
        v1_checksums[str(label)] = h
        print(f"v1 seed_{label:>20}: {h}")

    print_header("STEP 1-2: SHA-256 checksums (MG-G2L paper2 checkpoints)")
    mgg2l_paths = {s: mgg2l_weights_path(s) for s in EXISTING_SEEDS + NEW_SEEDS}
    mgg2l_checksums = {}
    for label, p in mgg2l_paths.items():
        assert os.path.exists(p), f"MG-G2L checkpoint not found for {label}: {p}"
        h = sha256sum(p)
        mgg2l_checksums[str(label)] = h
        print(f"MGG2L seed_{label:>18}: {h}")

    def find_duplicates(checksums):
        seen = {}
        dups = []
        for label, h in checksums.items():
            if h in seen:
                dups.append((label, seen[h]))
            else:
                seen[h] = label
        return dups

    print_header("STEP 3: duplicate check")
    v1_dups = find_duplicates(v1_checksums)
    mgg2l_dups = find_duplicates(mgg2l_checksums)
    print("v1 baseline duplicates:", v1_dups if v1_dups else "NONE")
    print("MG-G2L duplicates:     ", mgg2l_dups if mgg2l_dups else "NONE")

    # Re-confirm the OLD seed_2021 is still a seed_123 duplicate, and that the
    # rerun is genuinely different from both the old mislabeled file and 123.
    old_vs_123 = state_dicts_equal(OLD_MISLABELED_SEED2021_V1_PATH, v1_weights_path(123))
    old_vs_rerun = state_dicts_equal(OLD_MISLABELED_SEED2021_V1_PATH, v1_weights_path(2021))
    print(f"\nRe-confirmation: old seed_2021 == seed_123 (torch.equal on every tensor)? {old_vs_123}  "
          f"(expected True -- old file is still the known duplicate, excluded)")
    print(f"Re-confirmation: old seed_2021 == new seed_2021_rerun (torch.equal)?      {old_vs_rerun}  "
          f"(expected False -- rerun must be a genuinely new training run)")

    v1_dup_labels = {a for a, b in v1_dups} | {b for a, b in v1_dups}
    mgg2l_dup_labels = {a for a, b in mgg2l_dups} | {b for a, b in mgg2l_dups}

    v1_2021_distinct = "2021" not in v1_dup_labels
    v1_1337_distinct = "1337" not in v1_dup_labels
    mgg2l_2021_distinct = "2021" not in mgg2l_dup_labels
    mgg2l_1337_distinct = "1337" not in mgg2l_dup_labels

    print(f"\nv1_seed_2021_rerun distinct: {v1_2021_distinct}")
    print(f"v1_seed_1337 distinct:       {v1_1337_distinct}")
    print(f"MGG2L_seed_2021 distinct:    {mgg2l_2021_distinct}")
    print(f"MGG2L_seed_1337 distinct:    {mgg2l_1337_distinct}")

    if not v1_2021_distinct:
        print("STOPPING for v1_seed_2021_rerun: duplicate checksum found -- NOT proceeding to load/eval.")
    if not v1_1337_distinct:
        print("STOPPING for v1_seed_1337: duplicate checksum found -- NOT proceeding to load/eval.")
    if not mgg2l_2021_distinct:
        print("STOPPING for MGG2L_seed_2021: duplicate checksum found -- NOT proceeding to load/eval.")
    if not mgg2l_1337_distinct:
        print("STOPPING for MGG2L_seed_1337: duplicate checksum found -- NOT proceeding to load/eval.")

    checksum_report = {
        "v1_checksums": v1_checksums,
        "mgg2l_checksums": mgg2l_checksums,
        "v1_duplicates": v1_dups,
        "mgg2l_duplicates": mgg2l_dups,
        "old_seed2021_equals_seed123": old_vs_123,
        "old_seed2021_equals_new_rerun": old_vs_rerun,
        "distinct_flags": {
            "v1_seed_2021_rerun": v1_2021_distinct, "v1_seed_1337": v1_1337_distinct,
            "mgg2l_seed_2021": mgg2l_2021_distinct, "mgg2l_seed_1337": mgg2l_1337_distinct,
        },
    }
    with open(OUT_CHECKSUM_JSON, "w") as f:
        json.dump(checksum_report, f, indent=2)
    print(f"\nSaved checksum report: {OUT_CHECKSUM_JSON}")

    # =====================================================================
    # STEP 4-6: load + evaluate each confirmed-distinct checkpoint
    # =====================================================================
    v1_distinct = {2021: v1_2021_distinct, 1337: v1_1337_distinct}
    mgg2l_distinct = {2021: mgg2l_2021_distinct, 1337: mgg2l_1337_distinct}

    p1_rk_results = {}
    p1_embeddings = {}
    for seed in NEW_SEEDS:
        if not v1_distinct[seed]:
            continue
        print_header(f"STEP 4-5: Paper1 baseline seed_{seed} -- strict=True load + streaming eval")
        model = scoring.load_model(v1_weights_path(seed))  # asserts strict=True, 0 missing/unexpected
        model = model.to(device)
        test_dataset = scoring.load_test_dataset(scoring.DEFAULT_SHARD_SUBFOLDER)
        results = eval_streaming_v1(
            model=model, test_dataset=test_dataset, k_values=[1, 5, 10], batch_size=scoring.BATCH_SIZE,
        )
        p1_rk_results[seed] = results
        img_emb, txt_emb, study_ids = scoring.compute_embeddings(model, test_dataset, device)
        p1_embeddings[seed] = (img_emb, txt_emb, study_ids)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def load_mgg2l_embeddings_and_eval(seed):
        from base_models_refactored_v1_paper2 import MultimodalFusion as MultimodalFusionPaper2
        from data_loader_v1_paper2 import IndianaDataLoader as IndianaDataLoaderPaper2
        from config import get_vocab_size, get_embed_dim, get_current_config

        ckpt_path = mgg2l_checkpoint_path(seed)
        model = MultimodalFusionPaper2(
            vocab_size=get_vocab_size(), embed_dim=get_embed_dim(),
            num_heads=get_current_config()["num_heads"], num_layers=get_current_config()["num_layers"],
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"  [MG-G2L seed_{seed}] checkpoint epoch field: {checkpoint['epoch']} (-> epoch {checkpoint['epoch'] + 1})")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"  [MG-G2L seed_{seed}] strict=True load OK -> missing={missing}  unexpected={unexpected}")

        # Consistency check: checkpoint_resume.pth's weights must match the exported model_weights.pth.
        exported_state = load_raw_state_dict(mgg2l_weights_path(seed))
        consistent = all(
            k in exported_state and torch.equal(checkpoint["model_state_dict"][k], exported_state[k])
            for k in checkpoint["model_state_dict"].keys()
        )
        print(f"  [MG-G2L seed_{seed}] checkpoint_resume.pth matches export/model_weights.pth: {consistent}")

        model.eval().to(device)

        dl = IndianaDataLoaderPaper2(batch_size=scoring.BATCH_SIZE, use_shards=True, shard_subfolder=scoring.DEFAULT_SHARD_SUBFOLDER)
        dl.tokenizer = scoring.load_tokenizer_from_metadata(scoring.DEFAULT_SHARD_SUBFOLDER)
        dl.load_data(max_samples=None, skip_processing=True)
        test_dataset = dl.get_test_data(num_samples=None)
        print(f"  [MG-G2L seed_{seed}] test dataset size: {len(test_dataset)}  fallback: {test_dataset.fallback_count}")

        rk_results = eval_streaming_paper2(
            model=model, test_dataset=test_dataset, k_values=[1, 5, 10], batch_size=scoring.BATCH_SIZE,
        )

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
        embeddings = (np.concatenate(all_image_emb, axis=0), np.concatenate(all_text_emb, axis=0),
                      np.array(all_study_ids, dtype=np.int64))
        return rk_results, embeddings

    mg_rk_results = {}
    mg_embeddings = {}
    for seed in NEW_SEEDS:
        if not mgg2l_distinct[seed]:
            continue
        print_header(f"STEP 4-6: MG-G2L seed_{seed} -- strict=True load + section-aware streaming eval")
        rk_results, embeddings = load_mgg2l_embeddings_and_eval(seed)
        mg_rk_results[seed] = rk_results
        mg_embeddings[seed] = embeddings

    # =====================================================================
    # STEP 7: Paper1 vs MG-G2L R@K/MRR comparison, per seed
    # =====================================================================
    both_confirmed = {seed: (v1_distinct[seed] and mgg2l_distinct[seed]) for seed in NEW_SEEDS}
    for seed in NEW_SEEDS:
        if seed in p1_rk_results and seed in mg_rk_results:
            print_header(f"STEP 7: Paper1 vs MG-G2L R@K/MRR comparison -- seed_{seed}")
            p1r, mgr = p1_rk_results[seed], mg_rk_results[seed]
            header = f"{'metric':<14} | {'Paper 1':<10} | {'MG-G2L':<10} | {'delta':<10}"
            print(header)
            print("-" * len(header))
            for key in ["i2t_recall@1", "i2t_recall@5", "i2t_recall@10", "i2t_mrr",
                        "t2i_recall@1", "t2i_recall@5", "t2i_recall@10", "t2i_mrr"]:
                p1v, mgv = p1r[key], mgr[key]
                print(f"{key:<14} | {p1v:<10.4f} | {mgv:<10.4f} | {mgv - p1v:<+10.4f}")

    # =====================================================================
    # STEP 8: graded relevance (restricted subset), per seed
    # =====================================================================
    graded_rows_p1 = {}
    graded_rows_mg = {}
    per_query_caches = {}
    for seed in NEW_SEEDS:
        if seed not in p1_embeddings or seed not in mg_embeddings:
            continue
        print_header(f"STEP 8: graded relevance (restricted subset) -- seed_{seed}")
        p1_img, p1_txt, p1_sids = p1_embeddings[seed]
        mg_img, mg_txt, mg_sids = mg_embeddings[seed]
        assert list(p1_sids) == list(mg_sids), f"seed {seed}: study_id order mismatch between Paper1 and MG-G2L"

        n = len(p1_sids)
        B, _ = scoring.build_binary_matrix(p1_sids, scoring.DEFAULT_BINARY_LABELS_PATH)
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

        p1_img_t, p1_txt_t = torch.FloatTensor(p1_img).to(device), torch.FloatTensor(p1_txt).to(device)
        p1_i2t_sim = torch.matmul(p1_img_t, p1_txt_t.transpose(0, 1))
        p1_t2i_sim = torch.matmul(p1_txt_t, p1_img_t.transpose(0, 1))
        p1_i2t = scoring.compute_metrics_for_direction(p1_i2t_sim, R_full_np, ideal_top10, total_relevant, n)
        p1_t2i = scoring.compute_metrics_for_direction(p1_t2i_sim, R_full_np, ideal_top10, total_relevant, n)
        del p1_img_t, p1_txt_t, p1_i2t_sim, p1_t2i_sim

        mg_img_t, mg_txt_t = torch.FloatTensor(mg_img).to(device), torch.FloatTensor(mg_txt).to(device)
        mg_i2t_sim = torch.matmul(mg_img_t, mg_txt_t.transpose(0, 1))
        mg_t2i_sim = torch.matmul(mg_txt_t, mg_img_t.transpose(0, 1))
        mg_i2t = scoring.compute_metrics_for_direction(mg_i2t_sim, R_full_np, ideal_top10, total_relevant, n)
        mg_t2i = scoring.compute_metrics_for_direction(mg_t2i_sim, R_full_np, ideal_top10, total_relevant, n)
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
                    rows.append({"seed": seed, "metric": metric_name, "direction": direction,
                                 "full_corpus_value": round(full_val, 4), "full_corpus_n": n,
                                 "restricted_subset_value": round(restricted_val, 4), "restricted_subset_n": restricted_n})
            return rows

        p1_rows = rows_for(p1_i2t, p1_t2i)
        mg_rows = rows_for(mg_i2t, mg_t2i)
        graded_rows_p1[seed] = p1_rows
        graded_rows_mg[seed] = mg_rows

        header2 = f"{'metric':<12} | {'direction':<12} | {'Paper 1':<10} | {'MG-G2L':<10} | {'delta':<10}"
        print(header2)
        print("-" * len(header2))
        for r_p1, r_mg in zip(p1_rows, mg_rows):
            delta = r_mg["restricted_subset_value"] - r_p1["restricted_subset_value"]
            print(f"{r_p1['metric']:<12} | {r_p1['direction']:<12} | {r_p1['restricted_subset_value']:<10.4f} | "
                  f"{r_mg['restricted_subset_value']:<10.4f} | {delta:<+10.4f}")

        per_query_caches[seed] = pd.DataFrame({
            "study_id": [str(s) for s in p1_sids],
            "p1_ndcg10": p1_i2t[1], "p1_prec5": p1_i2t[3],
            "mg_ndcg10": mg_i2t[1], "mg_prec5": mg_i2t[3],
        })

    # =====================================================================
    # STEP 10: write NEW output files only (never touch existing seed files)
    # =====================================================================
    print_header("STEP 10: writing new output files")
    for seed, df in per_query_caches.items():
        out_path = os.path.join(PROJECT_DIR, f"per_query_i2t_metrics_seed_{seed}.csv")
        assert not os.path.exists(out_path), f"refusing to overwrite existing file: {out_path}"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    def write_graded_csv(path, rows_by_seed):
        all_rows = [r for seed in rows_by_seed for r in rows_by_seed[seed]]
        if not all_rows:
            print(f"Skipped (no confirmed seeds): {path}")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "metric", "direction", "full_corpus_value", "full_corpus_n",
                "restricted_subset_value", "restricted_subset_n",
            ])
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"Saved: {path}")

    write_graded_csv(OUT_PAPER1_GRADED_CSV, graded_rows_p1)
    write_graded_csv(OUT_MGG2L_GRADED_CSV, graded_rows_mg)

    # =====================================================================
    # STEP 9: full report per seed + proposed multiseed update (read-only
    # against existing files; existing files are never modified)
    # =====================================================================
    print_header("STEP 9: FULL PER-SEED REPORT")
    full_results_dump = {"checksum_report": checksum_report, "seeds": {}}
    for seed in NEW_SEEDS:
        print(f"\n--- seed_{seed} ---")
        print(f"  v1 baseline distinct & clean-loaded:   {seed in p1_rk_results}")
        print(f"  MG-G2L distinct & clean-loaded:        {seed in mg_rk_results}")
        seed_dump = {
            "v1_distinct": v1_distinct[seed], "mgg2l_distinct": mgg2l_distinct[seed],
            "paper1_rk": p1_rk_results.get(seed), "mgg2l_rk": mg_rk_results.get(seed),
            "paper1_graded_restricted": graded_rows_p1.get(seed), "mgg2l_graded_restricted": graded_rows_mg.get(seed),
        }
        full_results_dump["seeds"][seed] = seed_dump

    with open(OUT_FULL_RESULTS_JSON, "w") as f:
        json.dump(full_results_dump, f, indent=2, default=str)
    print(f"\nSaved full results dump: {OUT_FULL_RESULTS_JSON}")

    # ---- Proposed multiseed R@K/MRR update (PROPOSED, pending review) ----
    confirmed_new_seeds = [s for s in NEW_SEEDS if both_confirmed[s]]
    print_header(f"PROPOSED {4 + len(confirmed_new_seeds)}-SEED R@K/MRR UPDATE "
                 f"(existing 4: {EXISTING_SEEDS} + confirmed-distinct new: {confirmed_new_seeds}) "
                 f"-- PENDING REVIEW, no existing file modified")
    if not confirmed_new_seeds:
        print("No new seed passed both checksum checks -- no update to propose.")
    else:
        rk_keys = ["i2t_r1", "i2t_r5", "i2t_r10", "i2t_mrr", "t2i_r1", "t2i_r5", "t2i_r10", "t2i_mrr"]
        key_map = {"i2t_r1": "i2t_recall@1", "i2t_r5": "i2t_recall@5", "i2t_r10": "i2t_recall@10", "i2t_mrr": "i2t_mrr",
                   "t2i_r1": "t2i_recall@1", "t2i_r5": "t2i_recall@5", "t2i_r10": "t2i_recall@10", "t2i_mrr": "t2i_mrr"}
        for model_name, existing_dict, new_dict in [("Paper 1 baseline", EXISTING_RK_RESULTS["paper1"], p1_rk_results),
                                                      ("MG-G2L", EXISTING_RK_RESULTS["mgg2l"], mg_rk_results)]:
            print(f"\n{model_name}:")
            header = f"{'metric':<14} | {'mean':<10} | {'std':<10} | {'n_seeds':<8}"
            print(header)
            print("-" * len(header))
            for rk in rk_keys:
                values = [existing_dict[s][rk] for s in EXISTING_SEEDS]
                values += [new_dict[s][key_map[rk]] for s in confirmed_new_seeds]
                print(f"{rk:<14} | {np.mean(values):<10.4f} | {np.std(values):<10.4f} | {len(values):<8}")

        # ---- Proposed graded-relevance (restricted) update ----
        print_header(f"PROPOSED {4 + len(confirmed_new_seeds)}-SEED GRADED-RELEVANCE (restricted) UPDATE "
                     f"-- PENDING REVIEW, no existing file modified")

        def load_existing_graded(csv_specs):
            frames = []
            for path, fixed_seed in csv_specs:
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                if fixed_seed is not None:
                    df["seed"] = fixed_seed
                frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        for model_name, csv_specs, new_rows_by_seed in [
            ("Paper 1 baseline", EXISTING_GRADED_PAPER1_CSVS, graded_rows_p1),
            ("MG-G2L", EXISTING_GRADED_MGG2L_CSVS, graded_rows_mg),
        ]:
            existing_df = load_existing_graded(csv_specs)
            new_rows = [r for s in confirmed_new_seeds for r in new_rows_by_seed.get(s, [])]
            new_df = pd.DataFrame(new_rows)
            combined = pd.concat([existing_df, new_df], ignore_index=True) if len(new_df) else existing_df
            print(f"\n{model_name} (restricted_subset_value, mean +/- std across seeds "
                  f"{EXISTING_SEEDS + confirmed_new_seeds}):")
            header = f"{'metric':<14} | {'direction':<12} | {'mean':<10} | {'std':<10} | {'n_seeds':<8}"
            print(header)
            print("-" * len(header))
            if len(combined):
                for (metric, direction), grp in combined.groupby(["metric", "direction"]):
                    vals = grp["restricted_subset_value"].values
                    print(f"{metric:<14} | {direction:<12} | {np.mean(vals):<10.4f} | {np.std(vals):<10.4f} | {len(vals):<8}")

    print("\nDone. No existing per-seed result file was modified.")


if __name__ == "__main__":
    main()
