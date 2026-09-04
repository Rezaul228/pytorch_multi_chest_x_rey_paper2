#!/usr/bin/env python3
"""
Advanced Qualitative Analysis - Understanding Why Model Performs So Well
Creates insightful visualizations beyond basic retrieval examples
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import config
import paths
from data_loader_v1 import IndianaDataLoader
from base_models_refactored_v1 import MultimodalFusion

def load_full_model_checkpoint(checkpoint_path):
    """Load the full model checkpoint with architecture and weights"""
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

def analyze_retrieval_difficulty_distribution(model, data_loader, tokenizer, save_dir='advanced_qualitative_analysis'):
    """Analyze the distribution of retrieval difficulty - why is it so accurate?"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔍 Analyzing Retrieval Difficulty Distribution...")
    
    # Get validation data
    val_dataset = data_loader.get_validation_data(num_samples=500)  # Use more samples
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
            if batch_idx >= 20:  # Process 20 batches (320 samples)
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
    
    print(f"Collected {len(all_image_embeddings)} samples")
    
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
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Retrieval Difficulty Analysis: Why is the Model So Accurate?', fontsize=16, fontweight='bold')
    
    # 1. Similarity distribution comparison
    ax1.hist(diagonal_similarities, bins=50, alpha=0.7, label='Correct Matches', color='green', density=True)
    ax1.hist(off_diagonal_similarities, bins=50, alpha=0.7, label='Incorrect Matches', color='red', density=True)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('Similarity Distribution: Correct vs Incorrect Matches')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate separation
    correct_mean = np.mean(diagonal_similarities)
    incorrect_mean = np.mean(off_diagonal_similarities)
    separation = correct_mean - incorrect_mean
    ax1.axvline(correct_mean, color='green', linestyle='--', alpha=0.8, label=f'Correct Mean: {correct_mean:.3f}')
    ax1.axvline(incorrect_mean, color='red', linestyle='--', alpha=0.8, label=f'Incorrect Mean: {incorrect_mean:.3f}')
    ax1.text(0.05, 0.95, f'Separation: {separation:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontweight='bold')
    
    # 2. Rank analysis
    ranks = []
    for i in range(len(similarity_matrix)):
        row = similarity_matrix[i]
        # Find rank of correct match
        sorted_indices = np.argsort(row)[::-1]  # Descending order
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)
    
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    ranks_list = list(rank_counts.keys())
    counts_list = list(rank_counts.values())
    
    ax2.bar(ranks_list, counts_list, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Rank of Correct Match')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Rank Distribution of Correct Matches')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    recall_at_1 = sum(1 for r in ranks if r == 1) / len(ranks)
    recall_at_5 = sum(1 for r in ranks if r <= 5) / len(ranks)
    recall_at_10 = sum(1 for r in ranks if r <= 10) / len(ranks)
    
    ax2.text(0.05, 0.95, f'Recall@1: {recall_at_1:.3f}\nRecall@5: {recall_at_5:.3f}\nRecall@10: {recall_at_10:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7), fontweight='bold')
    
    # 3. Text length vs similarity
    text_lengths = [len(text.split()) for text in all_texts]
    ax3.scatter(text_lengths, diagonal_similarities, alpha=0.6, color='blue')
    ax3.set_xlabel('Text Length (words)')
    ax3.set_ylabel('Correct Match Similarity')
    ax3.set_title('Text Length vs Retrieval Similarity')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation
    correlation = np.corrcoef(text_lengths, diagonal_similarities)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7), fontweight='bold')
    
    # 4. Embedding space analysis
    # Sample a subset for t-SNE
    n_samples = min(200, len(all_image_embeddings))
    indices = np.random.choice(len(all_image_embeddings), n_samples, replace=False)
    
    img_sample = all_image_embeddings[indices]
    txt_sample = all_text_embeddings[indices]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    img_2d = tsne.fit_transform(img_sample)
    txt_2d = tsne.fit_transform(txt_sample)
    
    # Plot image and text embeddings
    ax4.scatter(img_2d[:, 0], img_2d[:, 1], alpha=0.6, label='Image Embeddings', color='red', s=30)
    ax4.scatter(txt_2d[:, 0], txt_2d[:, 1], alpha=0.6, label='Text Embeddings', color='blue', s=30)
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.set_title('Embedding Space Alignment (t-SNE)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'retrieval_difficulty_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Retrieval difficulty analysis saved to: {save_dir}/retrieval_difficulty_analysis.png")
    
    # Print insights
    print(f"\n🔍 RETRIEVAL ACCURACY INSIGHTS:")
    print(f"=" * 50)
    print(f"Correct match similarity mean: {correct_mean:.4f}")
    print(f"Incorrect match similarity mean: {incorrect_mean:.4f}")
    print(f"Separation gap: {separation:.4f}")
    print(f"Recall@1: {recall_at_1:.3f}")
    print(f"Recall@5: {recall_at_5:.3f}")
    print(f"Recall@10: {recall_at_10:.3f}")
    print(f"Text length vs similarity correlation: {correlation:.3f}")
    
    if separation > 0.1:
        print(f"✅ Large separation gap - model learns distinct representations")
    else:
        print(f"⚠️ Small separation gap - model may be overfitting")
    
    return {
        'correct_similarities': diagonal_similarities,
        'incorrect_similarities': off_diagonal_similarities,
        'ranks': ranks,
        'text_lengths': text_lengths,
        'separation': separation,
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'recall_at_10': recall_at_10
    }

def analyze_failure_cases(model, data_loader, tokenizer, save_dir='advanced_qualitative_analysis'):
    """Analyze where the model fails and why"""
    print("❌ Analyzing Failure Cases...")
    
    # Get validation data
    val_dataset = data_loader.get_validation_data(num_samples=200)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model.eval()
    device = next(model.parameters()).device
    
    failure_cases = []
    success_cases = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 10:  # Process 10 batches
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
            
            # Average pool
            img_emb = image_embeddings.mean(dim=1)
            txt_emb = text_embeddings.mean(dim=1)
            
            # Compute similarities
            similarities = torch.cosine_similarity(img_emb.unsqueeze(1), txt_emb.unsqueeze(0), dim=2)
            
            for i in range(len(batch_images)):
                # Find rank of correct match
                row = similarities[i].cpu().numpy()
                sorted_indices = np.argsort(row)[::-1]
                rank = np.where(sorted_indices == i)[0][0] + 1
                
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
                
                text = ' '.join(words[:30])
                study_id = batch_study_ids[i] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[i])
                
                case = {
                    'image': batch_images[i].cpu().numpy(),
                    'text': text,
                    'study_id': study_id,
                    'rank': rank,
                    'similarity': row[i]
                }
                
                if rank > 5:  # Failure case
                    failure_cases.append(case)
                else:  # Success case
                    success_cases.append(case)
    
    # Create failure case visualization
    if len(failure_cases) > 0:
        n_failures = min(8, len(failure_cases))
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Model Failure Cases Analysis', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i in range(n_failures):
            ax = axes[i]
            case = failure_cases[i]
            
            # Convert image for display
            image = case['image']
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            
            # Normalize image
            image = (image - image.min()) / (image.max() - image.min())
            
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Study: {case["study_id"]}\nRank: {case["rank"]}, Sim: {case["similarity"]:.3f}', 
                        fontsize=10, fontweight='bold', color='red')
            
            # Add text below image
            text = case['text'][:80] + '...' if len(case['text']) > 80 else case['text']
            ax.text(0.5, -0.2, text, transform=ax.transAxes, ha='center', va='top', 
                    fontsize=8, wrap=True)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_failures, 8):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'failure_cases_analysis.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Failure cases analysis saved to: {save_dir}/failure_cases_analysis.png")
        
        # Analyze failure patterns
        failure_ranks = [case['rank'] for case in failure_cases]
        failure_similarities = [case['similarity'] for case in failure_cases]
        
        print(f"\n❌ FAILURE CASE ANALYSIS:")
        print(f"=" * 40)
        print(f"Number of failure cases: {len(failure_cases)}")
        print(f"Average failure rank: {np.mean(failure_ranks):.1f}")
        print(f"Average failure similarity: {np.mean(failure_similarities):.4f}")
        print(f"Failure rate: {len(failure_cases) / (len(failure_cases) + len(success_cases)):.3f}")
    else:
        print("✅ No failure cases found in the sample!")

def analyze_embedding_quality(model, data_loader, tokenizer, save_dir='advanced_qualitative_analysis'):
    """Analyze the quality of learned embeddings"""
    print("🧠 Analyzing Embedding Quality...")
    
    # Get validation data
    val_dataset = data_loader.get_validation_data(num_samples=300)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Collect embeddings and metadata
    image_embeddings = []
    text_embeddings = []
    medical_categories = []
    text_lengths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 15:  # Process 15 batches
                break
                
            if isinstance(batch, dict):
                batch_images = batch['images']
                batch_texts = batch['captions']
            else:
                batch_images, batch_texts, _ = batch
            
            # Convert images to BCHW format if needed
            if batch_images.shape[1] != 3:
                batch_images = batch_images.permute(0, 3, 1, 2)
            
            # Move to device
            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)
            
            # Get embeddings
            img_emb = model.image_encoder(batch_images).mean(dim=1)
            txt_emb = model.text_encoder(batch_texts).mean(dim=1)
            
            image_embeddings.append(img_emb.cpu().numpy())
            text_embeddings.append(txt_emb.cpu().numpy())
            
            # Categorize texts
            for i in range(len(batch_texts)):
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
                            words.append(word.lower())
                
                text_str = ' '.join(words)
                text_lengths.append(len(words))
                
                # Categorize
                if any(word in text_str for word in ['pneumonia', 'consolidation', 'infiltrate']):
                    medical_categories.append('Pneumonia')
                elif any(word in text_str for word in ['effusion', 'pleural']):
                    medical_categories.append('Pleural Effusion')
                elif any(word in text_str for word in ['normal', 'clear', 'unremarkable']):
                    medical_categories.append('Normal')
                elif any(word in text_str for word in ['heart', 'cardiac', 'cardiomegaly']):
                    medical_categories.append('Cardiac')
                else:
                    medical_categories.append('Other')
    
    # Combine embeddings
    image_embeddings = np.vstack(image_embeddings)
    text_embeddings = np.vstack(text_embeddings)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Embedding Quality Analysis', fontsize=16, fontweight='bold')
    
    # 1. Embedding magnitude distribution
    img_norms = np.linalg.norm(image_embeddings, axis=1)
    txt_norms = np.linalg.norm(text_embeddings, axis=1)
    
    ax1.hist(img_norms, bins=30, alpha=0.7, label='Image Embeddings', color='red', density=True)
    ax1.hist(txt_norms, bins=30, alpha=0.7, label='Text Embeddings', color='blue', density=True)
    ax1.set_xlabel('Embedding Magnitude (L2 Norm)')
    ax1.set_ylabel('Density')
    ax1.set_title('Embedding Magnitude Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Intra-class vs inter-class similarity
    unique_categories = list(set(medical_categories))
    intra_class_similarities = []
    inter_class_similarities = []
    
    for i, cat1 in enumerate(unique_categories):
        indices1 = [j for j, cat in enumerate(medical_categories) if cat == cat1]
        indices2 = [j for j, cat in enumerate(medical_categories) if cat != cat1]
        
        if len(indices1) > 1:
            # Intra-class similarities
            for j in range(len(indices1)):
                for k in range(j+1, len(indices1)):
                    sim = cosine_similarity([image_embeddings[indices1[j]]], [image_embeddings[indices1[k]]])[0, 0]
                    intra_class_similarities.append(sim)
        
        # Inter-class similarities
        for idx1 in indices1[:5]:  # Sample to avoid too many comparisons
            for idx2 in indices2[:5]:
                sim = cosine_similarity([image_embeddings[idx1]], [image_embeddings[idx2]])[0, 0]
                inter_class_similarities.append(sim)
    
    ax2.hist(intra_class_similarities, bins=30, alpha=0.7, label='Intra-class', color='green', density=True)
    ax2.hist(inter_class_similarities, bins=30, alpha=0.7, label='Inter-class', color='orange', density=True)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('Intra-class vs Inter-class Similarities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Text length vs embedding quality
    ax3.scatter(text_lengths, img_norms, alpha=0.6, color='purple')
    ax3.set_xlabel('Text Length (words)')
    ax3.set_ylabel('Image Embedding Magnitude')
    ax3.set_title('Text Length vs Image Embedding Quality')
    ax3.grid(True, alpha=0.3)
    
    # 4. Category-wise embedding visualization (t-SNE)
    n_samples = min(150, len(image_embeddings))
    indices = np.random.choice(len(image_embeddings), n_samples, replace=False)
    
    img_sample = image_embeddings[indices]
    categories_sample = [medical_categories[i] for i in indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    img_2d = tsne.fit_transform(img_sample)
    
    unique_cats = list(set(categories_sample))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
    
    for i, cat in enumerate(unique_cats):
        mask = np.array(categories_sample) == cat
        ax4.scatter(img_2d[mask, 0], img_2d[mask, 1], 
                   c=[colors[i]], label=cat, alpha=0.7, s=50)
    
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.set_title('Medical Category Clustering in Embedding Space')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'embedding_quality_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Embedding quality analysis saved to: {save_dir}/embedding_quality_analysis.png")
    
    # Print insights
    print(f"\n🧠 EMBEDDING QUALITY INSIGHTS:")
    print(f"=" * 40)
    print(f"Image embedding magnitude mean: {np.mean(img_norms):.4f}")
    print(f"Text embedding magnitude mean: {np.mean(txt_norms):.4f}")
    print(f"Intra-class similarity mean: {np.mean(intra_class_similarities):.4f}")
    print(f"Inter-class similarity mean: {np.mean(inter_class_similarities):.4f}")
    print(f"Category separation: {np.mean(intra_class_similarities) - np.mean(inter_class_similarities):.4f}")

def main():
    """Main function to create advanced qualitative analysis"""
    print("🔬 ADVANCED QUALITATIVE ANALYSIS - Understanding Model Performance")
    print("=" * 70)
    
    # Configuration
    config.print_current_config()
    
    # Load full model checkpoint
    model_path = 'saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model.pth'
    print(f"📁 Loading full model from: {model_path}")
    model = load_full_model_checkpoint(model_path)
    print(f"✅ Full model loaded successfully")
    
    # Load validation data and tokenizer
    print("\n📁 Loading validation data and tokenizer...")
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
    save_dir = 'advanced_qualitative_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create all analyses
    try:
        retrieval_analysis = analyze_retrieval_difficulty_distribution(model, data_loader, tokenizer, save_dir)
        analyze_failure_cases(model, data_loader, tokenizer, save_dir)
        analyze_embedding_quality(model, data_loader, tokenizer, save_dir)
        
        print(f"\n✅ Advanced qualitative analysis completed!")
        print(f"📁 Results saved to: {save_dir}/")
        print(f"\n📋 Generated Files:")
        print(f"   - retrieval_difficulty_analysis.png: Why the model is so accurate")
        print(f"   - failure_cases_analysis.png: Where and why the model fails")
        print(f"   - embedding_quality_analysis.png: Quality of learned representations")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   - Separation gap: {retrieval_analysis['separation']:.4f}")
        print(f"   - Recall@1: {retrieval_analysis['recall_at_1']:.3f}")
        print(f"   - Recall@5: {retrieval_analysis['recall_at_5']:.3f}")
        print(f"   - Recall@10: {retrieval_analysis['recall_at_10']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
