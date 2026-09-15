#!/usr/bin/env python3
"""
Laterality-confusion analysis for the Paper 1 baseline (seed_42 checkpoint),
using hard_negative_tags_test_paper2.csv (RIGHT/LEFT/BILATERAL/UNSPECIFIED
tags per (study_id, finding)) and the Image->Text retrieval rankings.

No cached similarity matrix / top-K rankings existed on disk from the
earlier nDCG/mAP evaluation (paper2_graded_relevance_eval.py only ever
saved the aggregate metrics table, not the raw similarity matrix or
per-query rankings) -- so this recomputes I2T embeddings/rankings using the
exact same method (same checkpoint, same test set, same
model((images, captions), training=False) call, cosine similarity via
normalized-embedding dot product). Read-only w.r.t. the model: no training,
no checkpoint modification. Does not touch DataLoader or SLURM job 113645.

For each of the 3 most RIGHT/LEFT-balanced findings (Pleural Effusion,
Atelectasis, Lung Opacity), and for each direction (RIGHT-query checking
LEFT confusion, and vice versa):
  - query subset = study_ids tagged with that laterality for that finding
  - for each query, take its top-10 retrieved candidates EXCLUDING its own
    self-match (top-11 by similarity, self removed, first 10 of the rest)
  - Confusion@10 = fraction of those 10 candidates tagged with the OPPOSITE
    laterality for the same finding
  - base_rate = (count of opposite-laterality study_ids for that finding) / (12429 - 1)
  - ratio = mean Confusion@10 / base_rate

Output: laterality_confusion_paper1_baseline.csv
  columns: finding, direction, n_queries, mean_confusion_at_10, base_rate, ratio
"""

import os
import sys
import pickle
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
MIMIC_DATA_DIR = os.path.join(PROJECT_DIR, "mimic", "data")
os.chdir(PROJECT_DIR)

from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config

MODEL_PATH = os.path.join(
    PROJECT_DIR, "saved_models",
    "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_42",
    "export", "model_weights.pth",
)
HARD_NEG_TAGS_PATH = os.path.join(MIMIC_DATA_DIR, "hard_negative_tags_test_paper2.csv")
OUTPUT_CSV_PATH = os.path.join(MIMIC_DATA_DIR, "laterality_confusion_paper1_baseline.csv")
SHARD_SUBFOLDER = "mimic_shards_hybrid_full_ori"
BATCH_SIZE = 64
TOP_K_PLUS_SELF = 11  # top-10 EXCLUDING self -> fetch 11 in case self is inside, then drop it

TARGET_FINDINGS = ['Pleural Effusion', 'Atelectasis', 'Lung Opacity']


def load_tokenizer_from_metadata():
    metadata_path = paths.get_metadata_path(SHARD_SUBFOLDER)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get('tokenizer')
    if hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    return tokenizer


def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    model = MultimodalFusion(
        vocab_size=get_vocab_size(),
        embed_dim=get_embed_dim(),
        num_heads=get_current_config()["num_heads"],
        num_layers=get_current_config()["num_layers"],
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"Checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    return model


def load_test_dataset():
    print("Loading FULL test dataset (no evaluation, no modification)...")
    data_loader = IndianaDataLoader(batch_size=BATCH_SIZE, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    data_loader.tokenizer = load_tokenizer_from_metadata()
    data_loader.load_data(max_samples=None, skip_processing=True)
    test_dataset = data_loader.get_test_data(num_samples=None)
    print(f"Test dataset size: {len(test_dataset)}")
    return test_dataset


def compute_embeddings(model, test_dataset, device):
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_image_emb, all_text_emb, all_study_ids = [], [], []
    sample_count = 0
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
                study_ids_batch = [str(int(sid.item())) for sid in batch_study_ids]
            else:
                study_ids_batch = [str(sid) for sid in batch_study_ids]
            all_study_ids.extend(study_ids_batch)

            sample_count += len(batch_images)
            if sample_count % 3200 == 0:
                print(f"  Encoded {sample_count} samples...")

    image_embeddings = np.concatenate(all_image_emb, axis=0)
    text_embeddings = np.concatenate(all_text_emb, axis=0)
    study_ids = np.array(all_study_ids)
    print(f"Encoded {sample_count} samples total.")
    return image_embeddings, text_embeddings, study_ids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model().to(device)
    test_dataset = load_test_dataset()
    image_embeddings, text_embeddings, study_ids = compute_embeddings(model, test_dataset, device)
    n = image_embeddings.shape[0]

    study_id_to_idx = {sid: i for i, sid in enumerate(study_ids)}

    print(f"\nBuilding I2T similarity matrix ({n} x {n})...")
    image_emb_t = torch.FloatTensor(image_embeddings).to(device)
    text_emb_t = torch.FloatTensor(text_embeddings).to(device)
    i2t_sim = torch.matmul(image_emb_t, text_emb_t.transpose(0, 1))

    print(f"Getting top-{TOP_K_PLUS_SELF} candidates per query...")
    _, topk_idx = torch.topk(i2t_sim, k=TOP_K_PLUS_SELF, dim=1)
    topk_idx = topk_idx.cpu().numpy()

    print("\nLoading hard_negative_tags_test_paper2.csv...")
    tags_df = pd.read_csv(HARD_NEG_TAGS_PATH, dtype={'study_id': str})

    rows_out = []

    for finding in TARGET_FINDINGS:
        sub = tags_df[tags_df['finding'] == finding]
        right_ids = set(sub.loc[sub['laterality_tag'] == 'RIGHT', 'study_id'])
        left_ids = set(sub.loc[sub['laterality_tag'] == 'LEFT', 'study_id'])

        print(f"\n{finding}: RIGHT={len(right_ids)}  LEFT={len(left_ids)}")

        for direction, query_ids, opposite_ids in [
            ('RIGHT-query', right_ids, left_ids),
            ('LEFT-query', left_ids, right_ids),
        ]:
            confusions = []
            n_queries_used = 0
            for sid in query_ids:
                if sid not in study_id_to_idx:
                    continue
                q_idx = study_id_to_idx[sid]
                candidates = [c for c in topk_idx[q_idx] if c != q_idx][:10]
                if len(candidates) < 10:
                    # extremely unlikely given ~99% R@1, but guard anyway
                    continue
                n_opposite_in_top10 = sum(1 for c in candidates if study_ids[c] in opposite_ids)
                confusions.append(n_opposite_in_top10 / 10.0)
                n_queries_used += 1

            mean_confusion = float(np.mean(confusions)) if confusions else float('nan')
            base_rate = len(opposite_ids) / (n - 1)
            ratio = (mean_confusion / base_rate) if base_rate > 0 else float('nan')

            rows_out.append({
                'finding': finding,
                'direction': direction,
                'n_queries': n_queries_used,
                'mean_confusion_at_10': round(mean_confusion, 6),
                'base_rate': round(base_rate, 6),
                'ratio': round(ratio, 4) if ratio == ratio else ratio,  # keep NaN as NaN
            })
            print(f"  {direction}: n_queries={n_queries_used}  mean_confusion@10={mean_confusion:.6f}  "
                  f"base_rate={base_rate:.6f}  ratio={ratio:.4f}")

    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['finding', 'direction', 'n_queries', 'mean_confusion_at_10', 'base_rate', 'ratio'])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"\nSaved: {OUTPUT_CSV_PATH}")

    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Laterality confusion, Paper 1 baseline (seed_42), MIMIC-CXR test set")
    print("=" * 100)
    header = f"{'finding':<18} | {'direction':<12} | {'n_queries':<10} | {'mean_conf@10':<13} | {'base_rate':<10} | {'ratio':<8}"
    print(header)
    print("-" * len(header))
    for r in rows_out:
        print(f"{r['finding']:<18} | {r['direction']:<12} | {r['n_queries']:<10} | "
              f"{r['mean_confusion_at_10']:<13} | {r['base_rate']:<10} | {r['ratio']:<8}")


if __name__ == "__main__":
    main()
