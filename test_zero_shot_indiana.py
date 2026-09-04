#!/usr/bin/env python3
"""
Zero-shot cross-dataset evaluation:
  Source model  = MIMIC-trained MultimodalFusion
  Target data   = indiana_shards (config.py indiana_shards settings)

Indiana captions are stored with the Indiana tokenizer. For a valid zero-shot
run they are decoded to text and re-encoded with the MIMIC tokenizer so token
IDs match the model embedding table.
"""

import os
import pickle
import re

import torch
from torch.utils.data import DataLoader, Dataset

import config
import paths
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming

SOURCE_DATASET = "mimic_shards"
TARGET_DATASET = "indiana_shards"
MODEL_PATH = (
    "/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v2/export/model_weights.pth"
)


def load_tokenizer(shard_subfolder):
    metadata_path = paths.get_metadata_path(shard_subfolder)
    print(f"Loading tokenizer from: {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get("tokenizer")
    if tokenizer is None:
        raise ValueError(f"No tokenizer in {metadata_path}")

    # Compatibility for EnhancedTokenizer / SimpleTokenizer
    if hasattr(tokenizer, "word_index") and not hasattr(tokenizer, "word2idx"):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    if hasattr(tokenizer, "word2idx") and not hasattr(tokenizer, "idx2word"):
        tokenizer.idx2word = {i: w for w, i in tokenizer.word2idx.items()}

    print(f"  {shard_subfolder} vocab size: {len(tokenizer.word2idx)}")
    return tokenizer


def decode_caption(token_ids, tokenizer):
    """Convert token ids back to a whitespace-joined text string."""
    idx2word = tokenizer.idx2word
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid == 0:  # pad
            continue
        word = idx2word.get(tid)
        if word is None or word in ("<pad>", "<unk>"):
            continue
        words.append(word)
    return " ".join(words)


def encode_with_source_tokenizer(text, tokenizer, maxlen):
    """Encode text with source (MIMIC) tokenizer, pad/truncate to maxlen."""
    if hasattr(tokenizer, "texts_to_sequences"):
        seqs = tokenizer.texts_to_sequences([text])
        seq = list(seqs[0]) if seqs else []
    else:
        # Fallback simple whitespace tokenize
        word2idx = tokenizer.word2idx
        oov = word2idx.get(getattr(tokenizer, "oov_token", "<unk>"), 1)
        seq = [word2idx.get(w, oov) for w in re.sub(r"[^\w\s]", " ", text).lower().split()]

    seq = seq[:maxlen]
    if len(seq) < maxlen:
        seq = seq + [0] * (maxlen - len(seq))
    return seq


class ZeroShotRemappedDataset(Dataset):
    """Wrap Indiana samples; remaps caption token IDs to the source tokenizer."""

    def __init__(self, base_dataset, target_tokenizer, source_tokenizer, source_maxlen):
        self.base_dataset = base_dataset
        self.target_tokenizer = target_tokenizer
        self.source_tokenizer = source_tokenizer
        self.source_maxlen = source_maxlen
        self._remap_stats = {"empty_decode": 0, "total": 0}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        caption_ids = sample["captions"].tolist()
        text = decode_caption(caption_ids, self.target_tokenizer)
        self._remap_stats["total"] += 1
        if not text.strip():
            self._remap_stats["empty_decode"] += 1

        remapped = encode_with_source_tokenizer(
            text, self.source_tokenizer, self.source_maxlen
        )
        return {
            "images": sample["images"],
            "captions": torch.LongTensor(remapped),
            "study_ids": sample["study_ids"],
        }


def print_results(results, label):
    print("\n" + "=" * 60)
    print(f"ZERO-SHOT RESULTS ({label})")
    print("=" * 60)
    print("Image-to-Text Retrieval:")
    print(f"   MRR: {results['i2t_mrr']:.4f}")
    print(f"   Mean Rank: {results['i2t_mean_rank']:.2f}")
    print(f"   Recall@1: {results['i2t_recall@1']:.4f}")
    print(f"   Recall@5: {results['i2t_recall@5']:.4f}")
    print(f"   Recall@10: {results['i2t_recall@10']:.4f}")

    print("\nText-to-Image Retrieval:")
    print(f"   MRR: {results['t2i_mrr']:.4f}")
    print(f"   Mean Rank: {results['t2i_mean_rank']:.2f}")
    print(f"   Recall@1: {results['t2i_recall@1']:.4f}")
    print(f"   Recall@5: {results['t2i_recall@5']:.4f}")
    print(f"   Recall@10: {results['t2i_recall@10']:.4f}")

    print("\nOverall Performance:")
    print(f"   Average MRR: {results['avg_mrr']:.4f}")
    print(f"   Average Recall@1: {results['avg_recall@1']:.4f}")
    print(f"   Average Recall@5: {results['avg_recall@5']:.4f}")
    print(f"   Average Recall@10: {results['avg_recall@10']:.4f}")
    print(f"\nSummary: {results['summary']}")


def test_zero_shot_indiana():
    print("Zero-shot test: MIMIC-trained model -> Indiana train set")
    print("=" * 60)

    # Model architecture must match SOURCE (MIMIC) training config
    config.switch_dataset(SOURCE_DATASET)
    source_cfg = config.get_current_config()
    source_maxlen = source_cfg["max_token_length"]
    print("\n[SOURCE / MODEL] MIMIC training config:")
    config.print_current_config()

    print(f"\nLoading model weights: {MODEL_PATH}")
    model = MultimodalFusion(
        vocab_size=source_cfg["vocab_size"],
        embed_dim=source_cfg["embed_dim"],
        num_heads=source_cfg["num_heads"],
        num_layers=source_cfg["num_layers"],
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Model loaded on {device} | params={sum(p.numel() for p in model.parameters()):,}")

    source_tokenizer = load_tokenizer(SOURCE_DATASET)

    # Target data = Indiana (images + captions from indiana_shards)
    config.switch_dataset(TARGET_DATASET)
    target_cfg = config.get_current_config()
    print("\n[TARGET / DATA] Indiana config:")
    config.print_current_config()
    print(f"Data path: {target_cfg['data_path']}")

    target_tokenizer = load_tokenizer(TARGET_DATASET)

    data_loader = IndianaDataLoader(
        batch_size=32,
        use_shards=True,
        shard_subfolder=TARGET_DATASET,
    )
    data_loader.tokenizer = target_tokenizer
    data_loader.load_data(max_samples=None, skip_processing=True)
    # Use Indiana train split for a larger eval set (test has only ~723 samples)
    base_eval = data_loader.get_data(max_samples=None)
    print(f"Indiana train samples: {len(base_eval)}")

    remapped_eval = ZeroShotRemappedDataset(
        base_dataset=base_eval,
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        source_maxlen=source_maxlen,
    )
    eval_loader = DataLoader(remapped_eval, batch_size=32, shuffle=False, num_workers=0)

    print("\nRunning zero-shot cross-modal evaluation on Indiana TRAIN...")
    print(f"  Captions remapped Indiana -> MIMIC tokenizer (maxlen={source_maxlen})")
    out_dir = "outputs/zero_shot_mimic_to_indiana_train/"
    os.makedirs(out_dir, exist_ok=True)

    results = evaluate_cross_modal_retrieval_streaming(
        model=model,
        test_dataset=eval_loader,
        k_values=[1, 5, 10],
        batch_size=32,
        visualize=False,  # avoid CHW imshow crash; metrics-only
        num_vis_examples=0,
        output_dir=out_dir,
    )

    print(
        f"\nRemap stats: {remapped_eval._remap_stats['total']} captions, "
        f"{remapped_eval._remap_stats['empty_decode']} empty after decode"
    )
    print_results(results, "MIMIC model on indiana_shards TRAIN")
    return results


if __name__ == "__main__":
    test_zero_shot_indiana()
