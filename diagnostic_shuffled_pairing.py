#!/usr/bin/env python3
"""
Diagnostic: does Recall@K depend on the model seeing the TRUE caption
during co-attention, or does it hold up under a WRONG pairing?

Run baseline eval first, then shuffled-pairing eval on the same test set.
"""

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader

import config
import paths
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming


def load_tokenizer_from_metadata():
    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get("tokenizer")
    if tokenizer is None:
        raise ValueError("No tokenizer found in metadata")
    if hasattr(tokenizer, "word_index") and not hasattr(tokenizer, "word2idx"):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    return tokenizer


def _prepare_batch(batch, device):
    if hasattr(batch, "keys"):
        batch_images = batch["images"]
        batch_captions = batch["captions"]
    else:
        batch_images, batch_captions, _ = batch

    if not isinstance(batch_images, torch.Tensor):
        batch_images = torch.FloatTensor(batch_images)
    if not isinstance(batch_captions, torch.Tensor):
        batch_captions = torch.LongTensor(batch_captions)

    if len(batch_images.shape) == 4 and batch_images.shape[-1] == 3:
        batch_images = batch_images.permute(0, 3, 1, 2)

    return batch_images.to(device), batch_captions.to(device)


def evaluate_with_shuffled_pairing(model, test_loader, k_values=(1, 5, 10), device="cuda", seed=0):
    """Encode with wrong (shuffled) image-caption pairs, then run standard gallery retrieval."""
    model.eval()
    torch.manual_seed(seed)
    all_image_emb, all_text_emb = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_images, batch_captions = _prepare_batch(batch, device)

            # Pair each image with a WRONG caption from the same batch.
            perm = torch.randperm(batch_captions.size(0), device=device)
            shuffled_captions = batch_captions[perm]

            image_emb, text_emb = model((batch_images, shuffled_captions), training=False)

            all_image_emb.append(image_emb.cpu())
            # Undo shuffle so text_emb[j] still indexes true text j in the gallery.
            inv_perm = torch.argsort(perm)
            all_text_emb.append(text_emb[inv_perm].cpu())

    image_emb = torch.cat(all_image_emb, dim=0)
    text_emb = torch.cat(all_text_emb, dim=0)
    n = image_emb.size(0)
    print(f"Shuffled-pairing embeddings: N={n}, dim={image_emb.size(1)}")

    results = {}
    for direction, sim_matrix, name in [
        ("i2t", torch.matmul(image_emb, text_emb.transpose(0, 1)), "Image-to-Text"),
        ("t2i", torch.matmul(text_emb, image_emb.transpose(0, 1)), "Text-to-Image"),
    ]:
        correct = torch.arange(sim_matrix.size(0))
        sim_np = sim_matrix.numpy()

        ranks = []
        for i in range(len(sim_np)):
            similarities = sim_np[i]
            correct_score = similarities[i]
            rank = 1 + np.sum(similarities > correct_score)
            tie_count = np.sum(np.isclose(similarities, correct_score)) - 1
            if tie_count > 0:
                rank = rank + tie_count / 2
            ranks.append(rank)
        ranks = np.array(ranks)

        results[f"{direction}_mrr"] = float(np.mean(1.0 / ranks))
        for k in k_values:
            _, topk = torch.topk(sim_matrix, k=k, dim=1)
            recall = torch.any(topk == correct.unsqueeze(1), dim=1).float().mean()
            results[f"{direction}_recall@{k}"] = float(recall)

        print(f"\n=== {name} under WRONG (shuffled) pairing ===")
        print(f"  MRR: {results[f'{direction}_mrr']:.4f}")
        for k in k_values:
            print(f"  Recall@{k}: {results[f'{direction}_recall@{k}']:.4f}")

    results["avg_mrr"] = (results["i2t_mrr"] + results["t2i_mrr"]) / 2
    for k in k_values:
        results[f"avg_recall@{k}"] = (
            results[f"i2t_recall@{k}"] + results[f"t2i_recall@{k}"]
        ) / 2

    print("\n=== Overall under WRONG (shuffled) pairing ===")
    print(f"  Average MRR: {results['avg_mrr']:.4f}")
    for k in k_values:
        print(f"  Average Recall@{k}: {results[f'avg_recall@{k}']:.4f}")

    chance_r10 = 10.0 / n
    print(f"\nChance-level Recall@10 (approx K/N): {chance_r10:.4f}")
    return results


def main():
    # Model was trained on MIMIC (vocab 10805, seq len 128)
    config.switch_dataset("mimic_shards")
    config.print_current_config()
    dataset_mode = config.DATASET_MODE
    model_path = (
        "/home/abedin/Developments/pytorch_multi_chest_x_ray1/"
        "saved_models/mimic_origi_vocab10805_to128_lr1e-4_b128_ep45_dualbr_v3/export/model_weights.pth"
    )
    batch_size = 32
    k_values = [1, 5, 10]

    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Device: {device}")
    print(f"Model loaded from: {model_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    data_loader = IndianaDataLoader(
        batch_size=batch_size,
        use_shards=True,
        shard_subfolder=dataset_mode,
    )
    data_loader.tokenizer = load_tokenizer_from_metadata()
    data_loader.load_data(max_samples=None, skip_processing=True)
    test_dataset = data_loader.get_test_data(num_samples=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")

    print("\n" + "=" * 60)
    print("BASELINE: true pairing (normal eval)")
    print("=" * 60)
    baseline = evaluate_cross_modal_retrieval_streaming(
        model=model,
        test_dataset=test_loader,
        k_values=k_values,
        batch_size=batch_size,
        visualize=False,
        num_vis_examples=0,
    )

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: wrong pairing (shuffled captions within each batch)")
    print("=" * 60)
    shuffled = evaluate_with_shuffled_pairing(
        model, test_loader, k_values=k_values, device=device, seed=0
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Shuffled':>12} {'Delta':>12}")
    print("-" * 60)
    for key in ["avg_mrr", "avg_recall@1", "avg_recall@5", "avg_recall@10"]:
        b = baseline[key]
        s = shuffled[key]
        print(f"{key:<20} {b:>12.4f} {s:>12.4f} {s - b:>+12.4f}")

    print("\nInterpretation:")
    print("  - Recall stays high  -> embeddings carry useful cross-modal info")
    print("  - Recall collapses   -> model may need true pairing at encode time")


if __name__ == "__main__":
    main()
