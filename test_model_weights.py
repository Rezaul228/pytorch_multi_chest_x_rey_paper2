#!/usr/bin/env python3
"""
Test Model Weights - Compare model_weights.pth vs model.pth performance
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import config
import paths
from data_loader_v1 import IndianaDataLoader
from base_models_refactored_v1 import MultimodalFusion

def load_model_weights(model_weights_path):
    """Load model weights only (not full checkpoint)"""
    print(f"📁 Loading model weights from: {model_weights_path}")
    
    # Load the weights
    weights = torch.load(model_weights_path, map_location='cpu')
    print(f"✅ Model weights loaded successfully")
    print(f"   Keys: {list(weights.keys()) if isinstance(weights, dict) else 'Not a dict'}")
    
    # Create model with the same architecture
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    # Load the weights
    model.load_state_dict(weights)
    model.eval()
    
    print(f"✅ Model architecture created and weights loaded")
    return model

def load_full_model_checkpoint(checkpoint_path):
    """Load the full model checkpoint for comparison"""
    print(f"📁 Loading full model checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Checkpoint loaded successfully")
    
    model_state_dict = checkpoint['model_state_dict']
    
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"✅ Model architecture reconstructed and weights loaded")
    return model

def evaluate_model_retrieval(model, data_loader, tokenizer, model_name, save_dir='model_comparison'):
    """Evaluate model retrieval performance"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🔍 Evaluating {model_name}...")
    
    # Get validation data
    val_dataset = data_loader.get_validation_data(num_samples=200)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Collect all embeddings and similarities
    all_image_embeddings = []
    all_text_embeddings = []
    all_similarities = []
    all_texts = []
    all_study_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 10:  # Process 10 batches (160 samples)
                break
                
            if isinstance(batch, dict):
                batch_images = batch['images']
                batch_texts = batch['captions']
                batch_study_ids = batch['study_ids']
            else:
                batch_images, batch_texts, batch_study_ids = batch
            
            # Convert images to BCHW format if needed
            if batch_images.shape[1] != 3:
                batch_images = batch_images.permute(0, 3, 1, 2)
            
            # Move to device
            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)
            
            # Get embeddings
            image_embeddings = model.image_encoder(batch_images)
            text_embeddings = model.text_encoder(batch_texts)
            
            # Average pool to get single vector per sample
            img_emb = image_embeddings.mean(dim=1)
            txt_emb = text_embeddings.mean(dim=1)
            
            all_image_embeddings.append(img_emb.cpu().numpy())
            all_text_embeddings.append(txt_emb.cpu().numpy())
            
            # Store texts and study IDs
            for i in range(len(batch_texts)):
                # Decode text
                text_tokens = batch_texts[i]
                words = []
                for token_id in text_tokens:
                    if token_id > 0:
                        token_id_int = int(token_id)
                        if hasattr(tokenizer, 'idx2word'):
                            word = tokenizer.idx2word.get(token_id_int, None)
                            if word is None:
                                word = tokenizer.idx2word.get(str(token_id_int), None)
                        else:
                            word = f"<{token_id_int}>"
                        if word and word not in ['<pad>', '<unk>']:
                            words.append(word)
                
                text = ' '.join(words[:30])  # First 30 words
                all_texts.append(text)
                all_study_ids.append(batch_study_ids[i] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[i]))
    
    # Combine all embeddings
    all_image_embeddings = np.vstack(all_image_embeddings)
    all_text_embeddings = np.vstack(all_text_embeddings)
    
    print(f"Collected {len(all_image_embeddings)} samples for {model_name}")
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(all_image_embeddings, all_text_embeddings)
    
    # Analyze diagonal (correct matches) vs off-diagonal (incorrect matches)
    diagonal_similarities = np.diag(similarity_matrix)
    off_diagonal_similarities = []
    
    for i in range(len(similarity_matrix)):
        row = similarity_matrix[i]
        # Remove the diagonal element
        off_diag = np.concatenate([row[:i], row[i+1:]])
        off_diagonal_similarities.extend(off_diag)
    
    # Calculate rank analysis
    ranks = []
    for i in range(len(similarity_matrix)):
        row = similarity_matrix[i]
        # Find rank of correct match
        sorted_indices = np.argsort(row)[::-1]  # Descending order
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
    
    # Calculate metrics
    correct_mean = np.mean(diagonal_similarities)
    incorrect_mean = np.mean(off_diagonal_similarities)
    separation = correct_mean - incorrect_mean
    recall_at_1 = sum(1 for r in ranks if r == 1) / len(ranks)
    recall_at_5 = sum(1 for r in ranks if r <= 5) / len(ranks)
    recall_at_10 = sum(1 for r in ranks if r <= 10) / len(ranks)
    
    results = {
        'model_name': model_name,
        'correct_similarities': diagonal_similarities,
        'incorrect_similarities': off_diagonal_similarities,
        'ranks': ranks,
        'correct_mean': correct_mean,
        'incorrect_mean': incorrect_mean,
        'separation': separation,
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'recall_at_10': recall_at_10,
        'similarity_matrix': similarity_matrix
    }
    
    print(f"\n📊 {model_name} RESULTS:")
    print(f"=" * 40)
    print(f"Correct match similarity mean: {correct_mean:.4f}")
    print(f"Incorrect match similarity mean: {incorrect_mean:.4f}")
    print(f"Separation gap: {separation:.4f}")
    print(f"Recall@1: {recall_at_1:.3f}")
    print(f"Recall@5: {recall_at_5:.3f}")
    print(f"Recall@10: {recall_at_10:.3f}")
    
    return results

def compare_models(results1, results2, save_dir='model_comparison'):
    """Compare two models and create visualization"""
    print(f"\n🔄 Comparing {results1['model_name']} vs {results2['model_name']}...")
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Comparison: {results1["model_name"]} vs {results2["model_name"]}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Similarity distribution comparison
    ax1.hist(results1['correct_similarities'], bins=30, alpha=0.7, 
             label=f'{results1["model_name"]} Correct', color='green', density=True)
    ax1.hist(results1['incorrect_similarities'], bins=30, alpha=0.7, 
             label=f'{results1["model_name"]} Incorrect', color='red', density=True)
    ax1.hist(results2['correct_similarities'], bins=30, alpha=0.5, 
             label=f'{results2["model_name"]} Correct', color='lightgreen', density=True)
    ax1.hist(results2['incorrect_similarities'], bins=30, alpha=0.5, 
             label=f'{results2["model_name"]} Incorrect', color='lightcoral', density=True)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('Similarity Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rank distribution comparison
    rank_counts1 = {}
    rank_counts2 = {}
    for rank in results1['ranks']:
        rank_counts1[rank] = rank_counts1.get(rank, 0) + 1
    for rank in results2['ranks']:
        rank_counts2[rank] = rank_counts2.get(rank, 0) + 1
    
    ranks1 = list(rank_counts1.keys())
    counts1 = list(rank_counts1.values())
    ranks2 = list(rank_counts2.keys())
    counts2 = list(rank_counts2.values())
    
    ax2.bar([r - 0.2 for r in ranks1], counts1, width=0.4, 
            label=results1['model_name'], alpha=0.7, color='blue')
    ax2.bar([r + 0.2 for r in ranks2], counts2, width=0.4, 
            label=results2['model_name'], alpha=0.7, color='orange')
    ax2.set_xlabel('Rank of Correct Match')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Rank Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics comparison
    metrics = ['Recall@1', 'Recall@5', 'Recall@10']
    values1 = [results1['recall_at_1'], results1['recall_at_5'], results1['recall_at_10']]
    values2 = [results2['recall_at_1'], results2['recall_at_5'], results2['recall_at_10']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, values1, width, label=results1['model_name'], alpha=0.7, color='blue')
    ax3.bar(x + width/2, values2, width, label=results2['model_name'], alpha=0.7, color='orange')
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Retrieval Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Separation gap comparison
    separations = [results1['separation'], results2['separation']]
    model_names = [results1['model_name'], results2['model_name']]
    colors = ['blue' if s > 0 else 'red' for s in separations]
    
    bars = ax4.bar(model_names, separations, color=colors, alpha=0.7)
    ax4.set_ylabel('Separation Gap')
    ax4.set_title('Separation Gap Comparison')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, sep in zip(bars, separations):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                f'{sep:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison saved to: {save_dir}/model_comparison.png")
    
    # Print comparison summary
    print(f"\n🔄 COMPARISON SUMMARY:")
    print(f"=" * 50)
    print(f"{'Metric':<20} {'Model Weights':<15} {'Full Checkpoint':<15} {'Difference':<15}")
    print(f"{'-'*65}")
    print(f"{'Recall@1':<20} {results2['recall_at_1']:<15.3f} {results1['recall_at_1']:<15.3f} {results2['recall_at_1'] - results1['recall_at_1']:<15.3f}")
    print(f"{'Recall@5':<20} {results2['recall_at_5']:<15.3f} {results1['recall_at_5']:<15.3f} {results2['recall_at_5'] - results1['recall_at_5']:<15.3f}")
    print(f"{'Recall@10':<20} {results2['recall_at_10']:<15.3f} {results1['recall_at_10']:<15.3f} {results2['recall_at_10'] - results1['recall_at_10']:<15.3f}")
    print(f"{'Separation Gap':<20} {results2['separation']:<15.4f} {results1['separation']:<15.4f} {results2['separation'] - results1['separation']:<15.4f}")
    print(f"{'Correct Mean':<20} {results2['correct_mean']:<15.4f} {results1['correct_mean']:<15.4f} {results2['correct_mean'] - results1['correct_mean']:<15.4f}")
    print(f"{'Incorrect Mean':<20} {results2['incorrect_mean']:<15.4f} {results1['incorrect_mean']:<15.4f} {results2['incorrect_mean'] - results1['incorrect_mean']:<15.4f}")

def main():
    """Main function to test and compare models"""
    print("🧪 MODEL WEIGHTS TESTING AND COMPARISON")
    print("=" * 60)
    
    # Configuration
    config.print_current_config()
    
    # Model paths
    model_weights_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model_weights.pth'
    full_checkpoint_path = 'saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model.pth'
    
    # Load models
    print(f"\n📁 Loading models...")
    model_weights = load_model_weights(model_weights_path)
    full_checkpoint = load_full_model_checkpoint(full_checkpoint_path)
    
    # Load validation data and tokenizer
    print(f"\n📁 Loading validation data and tokenizer...")
    data_loader = IndianaDataLoader(
        batch_size=32,
        use_shards=True,
        shard_subfolder=config.DATASET_MODE
    )
    
    # Load tokenizer from metadata
    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    if tokenizer is None:
        raise ValueError("No tokenizer found in metadata")
    
    # Add compatibility attributes
    if isinstance(tokenizer, dict):
        class DictTokenizer:
            def __init__(self, word2idx):
                self.word2idx = word2idx
                self.idx2word = {idx: word for word, idx in word2idx.items()}
        tokenizer = DictTokenizer(tokenizer)
    elif hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    
    data_loader.tokenizer = tokenizer
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Create save directory
    save_dir = 'model_comparison'
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate both models
    print(f"\n🔍 Evaluating models...")
    results_full = evaluate_model_retrieval(full_checkpoint, data_loader, tokenizer, 
                                          "Full Checkpoint", save_dir)
    results_weights = evaluate_model_retrieval(model_weights, data_loader, tokenizer, 
                                             "Model Weights", save_dir)
    
    # Compare models
    compare_models(results_full, results_weights, save_dir)
    
    print(f"\n✅ Model testing and comparison completed!")
    print(f"📁 Results saved to: {save_dir}/")
    
    # Determine which model is better
    if results_weights['recall_at_1'] > results_full['recall_at_1']:
        print(f"\n🏆 Model Weights performs better than Full Checkpoint!")
    elif results_full['recall_at_1'] > results_weights['recall_at_1']:
        print(f"\n🏆 Full Checkpoint performs better than Model Weights!")
    else:
        print(f"\n🤝 Both models perform similarly!")

if __name__ == "__main__":
    main()
