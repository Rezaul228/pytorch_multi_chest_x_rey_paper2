#!/usr/bin/env python3
"""
ANALYZE DIFFERENCE BRANCH IMPORTANCE
Provide metrics to justify keeping the difference branch
"""

import torch
import config
import paths
import pickle
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP not available, skipping UMAP visualizations")
    UMAP_AVAILABLE = False
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_tokenizer_from_metadata():
    """Load the tokenizer from metadata based on current dataset mode"""
    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    if hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    
    return tokenizer

def create_visualizations(synergy_embeddings, diff_embeddings, full_embeddings, 
                         orthogonality_score, diversity_ratio, avg_complementarity,
                         robustness_score, regularization_effect):
    """Create comprehensive visualizations for difference branch analysis"""
    
    print("Creating comprehensive visualizations...")
    
    # Extract embeddings for visualization
    synergy_img_emb = torch.cat([emb[0] for emb in synergy_embeddings], dim=0)
    synergy_txt_emb = torch.cat([emb[1] for emb in synergy_embeddings], dim=0)
    diff_img_emb = torch.cat([emb[0] for emb in diff_embeddings], dim=0)
    diff_txt_emb = torch.cat([emb[1] for emb in diff_embeddings], dim=0)
    
    # Convert to numpy
    synergy_img_np = synergy_img_emb.cpu().numpy()
    synergy_txt_np = synergy_txt_emb.cpu().numpy()
    diff_img_np = diff_img_emb.cpu().numpy()
    diff_txt_np = diff_txt_emb.cpu().numpy()
    
    # Create output directory
    os.makedirs('difference_branch_visualizations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. t-SNE Visualization (original comprehensive)
    print("   Creating comprehensive t-SNE visualization...")
    create_tsne_visualization(synergy_img_np, diff_img_np, synergy_txt_np, diff_txt_np, timestamp)
    
    # 2. Image-vs-image t-SNE
    print("   Creating image-vs-image t-SNE visualization...")
    create_image_vs_image_tsne(synergy_img_np, diff_img_np, timestamp)
    
    # 3. Text-vs-text t-SNE
    print("   Creating text-vs-text t-SNE visualization...")
    create_text_vs_text_tsne(synergy_txt_np, diff_txt_np, timestamp)
    
    # 4. UMAP Visualization (if available)
    if UMAP_AVAILABLE:
        print("   Creating UMAP visualization...")
        create_umap_visualization(synergy_img_np, diff_img_np, synergy_txt_np, diff_txt_np, timestamp)
    else:
        print("   Skipping UMAP visualization (module not available)")
    
    # 5. Orthogonality Analysis Visualization
    print("   Creating orthogonality analysis...")
    create_orthogonality_visualization(synergy_img_np, diff_img_np, synergy_txt_np, diff_txt_np, timestamp)
    
    # 6. Feature Correlation Heatmap
    print("   Creating feature correlation heatmap...")
    create_correlation_heatmap(synergy_img_np, diff_img_np, synergy_txt_np, diff_txt_np, timestamp)
    
    # 7. Metrics Summary Dashboard
    print("   Creating metrics dashboard...")
    create_metrics_dashboard(orthogonality_score, diversity_ratio, avg_complementarity, 
                           robustness_score, regularization_effect, timestamp)
    
    # 8. Embedding Distribution Comparison
    print("   Creating embedding distribution comparison...")
    create_distribution_comparison(synergy_img_np, diff_img_np, synergy_txt_np, diff_txt_np, timestamp)
    
    print("All visualizations saved to 'difference_branch_visualizations/' directory")

def create_tsne_visualization(synergy_img, diff_img, synergy_txt, diff_txt, timestamp):
    """Create Scientific Reports-compliant t-SNE visualization with professional styling"""
    
    # Set Scientific Reports style requirements
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'  # More compatible font
    plt.rcParams['font.size'] = 8  # Base font size for Scientific Reports
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    
    # Combine all embeddings for unified t-SNE
    combined_embeddings = np.vstack([synergy_img, diff_img, synergy_txt, diff_txt])
    
    # Apply t-SNE to all 4,000 embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)//4))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Split back into categories
    n_samples = len(synergy_img)
    synergy_img_2d = embeddings_2d[:n_samples]
    diff_img_2d = embeddings_2d[n_samples:2*n_samples]
    synergy_txt_2d = embeddings_2d[2*n_samples:3*n_samples]
    diff_txt_2d = embeddings_2d[3*n_samples:]
    
    # Create 2-panel figure with Scientific Reports dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'wspace': 0.3})
    
    # Scientific Reports color palette (accessible colors)
    colors = {
        'synergy_img': '#1f77b4',    # Blue
        'diff_img': '#d62728',        # Red  
        'synergy_txt': '#2ca02c',     # Green
        'diff_txt': '#ff7f0e'         # Orange
    }
    
    # Left panel: Branch and modality separation
    ax1.scatter(synergy_img_2d[:, 0], synergy_img_2d[:, 1], 
               c=colors['synergy_img'], alpha=0.7, label='Synergy-Image', s=20, 
               edgecolors='white', linewidth=0.3)
    ax1.scatter(diff_img_2d[:, 0], diff_img_2d[:, 1], 
               c=colors['diff_img'], alpha=0.7, label='Difference-Image', s=20, 
               edgecolors='white', linewidth=0.3)
    ax1.scatter(synergy_txt_2d[:, 0], synergy_txt_2d[:, 1], 
               c=colors['synergy_txt'], alpha=0.7, label='Synergy-Text', s=20, 
               edgecolors='white', linewidth=0.3)
    ax1.scatter(diff_txt_2d[:, 0], diff_txt_2d[:, 1], 
               c=colors['diff_txt'], alpha=0.7, label='Difference-Text', s=20, 
               edgecolors='white', linewidth=0.3)
    
    ax1.set_title('Branch and Modality Separation', fontsize=9, fontweight='bold', pad=10)
    ax1.set_xlabel('t-SNE Component 1', fontsize=8)
    ax1.set_ylabel('t-SNE Component 2', fontsize=8)
    ax1.legend(loc='lower left', fontsize=7, framealpha=0.9, markerscale=0.6)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=7)
    
    # Right panel: K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    cluster_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    for i in range(4):
        mask = cluster_labels == i
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=cluster_colors[i], alpha=0.7, s=20, 
                   label=f'Cluster {i+1}', edgecolors='white', linewidth=0.3)
    
    ax2.set_title('K-means Clustering (k=4)', fontsize=9, fontweight='bold', pad=10)
    ax2.set_xlabel('t-SNE Component 1', fontsize=8)
    ax2.set_ylabel('t-SNE Component 2', fontsize=8)
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9, markerscale=0.6)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    
    # Calculate cluster purity
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(4)]
    cluster_purity = max(cluster_sizes) / len(cluster_labels) * 100
    
    # Add Scientific Reports-compliant figure caption
    caption_text = f"Total points: {len(embeddings_2d):,}, Cluster purity: {cluster_purity:.1f}%"
    fig.text(0.5, 0.02, caption_text, ha='center', fontsize=7, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'))
    
    # Ensure high resolution and professional layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92, wspace=0.3)
    
    # Save with Scientific Reports specifications
    plt.savefig(f'difference_branch_visualizations/tsne_analysis_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Scientific Reports-compliant t-SNE visualization saved")

def create_umap_visualization(synergy_img, diff_img, synergy_txt, diff_txt, timestamp):
    """Create UMAP visualization of branch embeddings"""
    
    # Combine embeddings for UMAP
    combined_img = np.vstack([synergy_img, diff_img])
    combined_txt = np.vstack([synergy_txt, diff_txt])
    
    # Apply UMAP
    reducer_img = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(combined_img)//4))
    reducer_txt = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(combined_txt)//4))
    
    img_2d = reducer_img.fit_transform(combined_img)
    txt_2d = reducer_txt.fit_transform(combined_txt)
    
    # Split back into branches
    n_samples = len(synergy_img)
    synergy_img_2d = img_2d[:n_samples]
    diff_img_2d = img_2d[n_samples:]
    synergy_txt_2d = txt_2d[:n_samples]
    diff_txt_2d = txt_2d[n_samples:]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Image embeddings
    ax1.scatter(synergy_img_2d[:, 0], synergy_img_2d[:, 1], 
               c='blue', alpha=0.6, label='Synergy Branch', s=50)
    ax1.scatter(diff_img_2d[:, 0], diff_img_2d[:, 1], 
               c='red', alpha=0.6, label='Difference Branch', s=50)
    ax1.set_title('UMAP: Image Embeddings\n(Synergy vs Difference)')
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Text embeddings
    ax2.scatter(synergy_txt_2d[:, 0], synergy_txt_2d[:, 1], 
               c='green', alpha=0.6, label='Synergy Branch', s=50)
    ax2.scatter(diff_txt_2d[:, 0], diff_txt_2d[:, 1], 
               c='orange', alpha=0.6, label='Difference Branch', s=50)
    ax2.set_title('UMAP: Text Embeddings\n(Synergy vs Difference)')
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined visualization
    ax3.scatter(synergy_img_2d[:, 0], synergy_img_2d[:, 1], 
               c='blue', alpha=0.6, label='Synergy (Image)', s=50)
    ax3.scatter(diff_img_2d[:, 0], diff_img_2d[:, 1], 
               c='red', alpha=0.6, label='Difference (Image)', s=50)
    ax3.scatter(synergy_txt_2d[:, 0], synergy_txt_2d[:, 1], 
               c='green', alpha=0.6, label='Synergy (Text)', s=50, marker='s')
    ax3.scatter(diff_txt_2d[:, 0], diff_txt_2d[:, 1], 
               c='orange', alpha=0.6, label='Difference (Text)', s=50, marker='s')
    ax3.set_title('UMAP: Combined Embeddings\n(All Branches & Modalities)')
    ax3.set_xlabel('UMAP Component 1')
    ax3.set_ylabel('UMAP Component 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distance analysis
    distances = []
    for i in range(n_samples):
        dist = np.linalg.norm(synergy_img_2d[i] - diff_img_2d[i])
        distances.append(dist)
    
    ax4.hist(distances, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title('UMAP: Distance Distribution\n(Synergy vs Difference)')
    ax4.set_xlabel('Euclidean Distance')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'difference_branch_visualizations/umap_analysis_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ UMAP visualization saved")

def create_image_vs_image_tsne(synergy_img, diff_img, timestamp):
    """
    Create t-SNE visualization comparing synergy-image vs difference-image embeddings.
    """
    print("   Creating image-vs-image t-SNE visualization...")
    
    # Set Scientific Reports styling
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    
    # Combine image embeddings only
    combined_embeddings = np.vstack([synergy_img, diff_img])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=min(30, len(combined_embeddings)//4))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Split back into synergy and difference
    synergy_img_2d = embeddings_2d[:len(synergy_img)]
    diff_img_2d = embeddings_2d[len(synergy_img):]
    
    # Create 2-panel figure with Scientific Reports dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'wspace': 0.3})
    
    # Scientific Reports color palette
    colors = {
        'synergy_img': '#1f77b4',    # Blue
        'diff_img': '#d62728'        # Red  
    }
    
    # Left panel: Branch separation
    ax1.scatter(synergy_img_2d[:, 0], synergy_img_2d[:, 1], 
               c=colors['synergy_img'], alpha=0.7, label='Synergy-Image', s=20, 
               edgecolors='white', linewidth=0.3)
    ax1.scatter(diff_img_2d[:, 0], diff_img_2d[:, 1], 
               c=colors['diff_img'], alpha=0.7, label='Difference-Image', s=20, 
               edgecolors='white', linewidth=0.3)
    
    ax1.set_title('Image Branch Separation', fontsize=9, fontweight='bold', pad=10)
    ax1.set_xlabel('t-SNE Component 1', fontsize=8)
    ax1.set_ylabel('t-SNE Component 2', fontsize=8)
    ax1.legend(loc='lower left', fontsize=7, framealpha=0.9, markerscale=0.6)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=7)
    
    # Right panel: K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    # Calculate cluster purity
    true_labels = np.concatenate([np.zeros(len(synergy_img)), np.ones(len(diff_img))])
    cluster_purity = adjusted_rand_score(true_labels, cluster_labels)
    
    # Plot clusters
    cluster_colors = ['#1f77b4', '#d62728']
    for i in range(2):
        mask = cluster_labels == i
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=cluster_colors[i], alpha=0.7, s=20, 
                   label=f'Cluster {i+1}', edgecolors='white', linewidth=0.3)
    
    ax2.set_title(f'K-means Clustering (k=2)\nPurity: {cluster_purity:.3f}', 
                  fontsize=9, fontweight='bold', pad=10)
    ax2.set_xlabel('t-SNE Component 1', fontsize=8)
    ax2.set_ylabel('t-SNE Component 2', fontsize=8)
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9, markerscale=0.6)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    
    # Add Scientific Reports-compliant figure caption
    caption_text = f"Image embeddings: {len(embeddings_2d):,} points, Cluster purity: {cluster_purity:.1%}"
    fig.text(0.5, 0.02, caption_text, ha='center', fontsize=7, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'))
    
    # Ensure high resolution and professional layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92, wspace=0.3)
    
    # Save with Scientific Reports specifications
    plt.savefig(f'difference_branch_visualizations/image_vs_image_tsne_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Image-vs-image t-SNE visualization saved")

def create_text_vs_text_tsne(synergy_txt, diff_txt, timestamp):
    """
    Create t-SNE visualization comparing synergy-text vs difference-text embeddings.
    """
    print("   Creating text-vs-text t-SNE visualization...")
    
    # Set Scientific Reports styling
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    
    # Combine text embeddings only
    combined_embeddings = np.vstack([synergy_txt, diff_txt])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=min(30, len(combined_embeddings)//4))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Split back into synergy and difference
    synergy_txt_2d = embeddings_2d[:len(synergy_txt)]
    diff_txt_2d = embeddings_2d[len(synergy_txt):]
    
    # Create 2-panel figure with Scientific Reports dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'wspace': 0.3})
    
    # Scientific Reports color palette
    colors = {
        'synergy_txt': '#2ca02c',     # Green
        'diff_txt': '#ff7f0e'         # Orange
    }
    
    # Left panel: Branch separation
    ax1.scatter(synergy_txt_2d[:, 0], synergy_txt_2d[:, 1], 
               c=colors['synergy_txt'], alpha=0.7, label='Synergy-Text', s=20, 
               edgecolors='white', linewidth=0.3)
    ax1.scatter(diff_txt_2d[:, 0], diff_txt_2d[:, 1], 
               c=colors['diff_txt'], alpha=0.7, label='Difference-Text', s=20, 
               edgecolors='white', linewidth=0.3)
    
    ax1.set_title('Text Branch Separation', fontsize=9, fontweight='bold', pad=10)
    ax1.set_xlabel('t-SNE Component 1', fontsize=8)
    ax1.set_ylabel('t-SNE Component 2', fontsize=8)
    ax1.legend(loc='lower left', fontsize=7, framealpha=0.9, markerscale=0.6)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=7)
    
    # Right panel: K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    # Calculate cluster purity
    true_labels = np.concatenate([np.zeros(len(synergy_txt)), np.ones(len(diff_txt))])
    cluster_purity = adjusted_rand_score(true_labels, cluster_labels)
    
    # Plot clusters
    cluster_colors = ['#2ca02c', '#ff7f0e']
    for i in range(2):
        mask = cluster_labels == i
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=cluster_colors[i], alpha=0.7, s=20, 
                   label=f'Cluster {i+1}', edgecolors='white', linewidth=0.3)
    
    ax2.set_title(f'K-means Clustering (k=2)\nPurity: {cluster_purity:.3f}', 
                  fontsize=9, fontweight='bold', pad=10)
    ax2.set_xlabel('t-SNE Component 1', fontsize=8)
    ax2.set_ylabel('t-SNE Component 2', fontsize=8)
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9, markerscale=0.6)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    
    # Add Scientific Reports-compliant figure caption
    caption_text = f"Text embeddings: {len(embeddings_2d):,} points, Cluster purity: {cluster_purity:.1%}"
    fig.text(0.5, 0.02, caption_text, ha='center', fontsize=7, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'))
    
    # Ensure high resolution and professional layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92, wspace=0.3)
    
    # Save with Scientific Reports specifications
    plt.savefig(f'difference_branch_visualizations/text_vs_text_tsne_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Text-vs-text t-SNE visualization saved")

def create_orthogonality_visualization(synergy_img, diff_img, synergy_txt, diff_txt, timestamp):
    """Create orthogonality analysis visualization"""
    
    # Calculate cosine similarities
    img_similarities = []
    txt_similarities = []
    
    for i in range(len(synergy_img)):
        img_sim = cosine_similarity([synergy_img[i]], [diff_img[i]])[0][0]
        txt_sim = cosine_similarity([synergy_txt[i]], [diff_txt[i]])[0][0]
        img_similarities.append(img_sim)
        txt_similarities.append(txt_sim)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cosine similarity distributions
    ax1.hist(img_similarities, bins=20, alpha=0.7, color='blue', label='Image', edgecolor='black')
    ax1.hist(txt_similarities, bins=20, alpha=0.7, color='green', label='Text', edgecolor='black')
    ax1.axvline(np.mean(img_similarities), color='blue', linestyle='--', 
                label=f'Image Mean: {np.mean(img_similarities):.4f}')
    ax1.axvline(np.mean(txt_similarities), color='green', linestyle='--', 
                label=f'Text Mean: {np.mean(txt_similarities):.4f}')
    ax1.set_title('Cosine Similarity Distribution\n(Synergy vs Difference)')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Orthogonality scores over samples
    ax2.plot(range(len(img_similarities)), img_similarities, 'b-', alpha=0.7, label='Image')
    ax2.plot(range(len(txt_similarities)), txt_similarities, 'g-', alpha=0.7, label='Text')
    ax2.axhline(y=0.1, color='red', linestyle='--', label='Good Orthogonality Threshold')
    ax2.set_title('Orthogonality Scores Over Samples')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot of similarities
    ax3.scatter(img_similarities, txt_similarities, alpha=0.6, color='purple')
    ax3.set_xlabel('Image Cosine Similarity')
    ax3.set_ylabel('Text Cosine Similarity')
    ax3.set_title('Image vs Text Similarity Correlation')
    ax3.grid(True, alpha=0.3)
    
    # Orthogonality quality assessment
    img_ortho_quality = np.mean([1 if sim < 0.1 else 0 for sim in img_similarities])
    txt_ortho_quality = np.mean([1 if sim < 0.1 else 0 for sim in txt_similarities])
    
    categories = ['Excellent\n(< 0.05)', 'Good\n(0.05-0.1)', 'Fair\n(0.1-0.2)', 'Poor\n(> 0.2)']
    img_counts = [
        sum(1 for sim in img_similarities if sim < 0.05),
        sum(1 for sim in img_similarities if 0.05 <= sim < 0.1),
        sum(1 for sim in img_similarities if 0.1 <= sim < 0.2),
        sum(1 for sim in img_similarities if sim >= 0.2)
    ]
    txt_counts = [
        sum(1 for sim in txt_similarities if sim < 0.05),
        sum(1 for sim in txt_similarities if 0.05 <= sim < 0.1),
        sum(1 for sim in txt_similarities if 0.1 <= sim < 0.2),
        sum(1 for sim in txt_similarities if sim >= 0.2)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax4.bar(x - width/2, img_counts, width, label='Image', alpha=0.7)
    ax4.bar(x + width/2, txt_counts, width, label='Text', alpha=0.7)
    ax4.set_xlabel('Orthogonality Quality')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Orthogonality Quality Distribution')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'difference_branch_visualizations/orthogonality_analysis_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Orthogonality analysis saved")

def create_correlation_heatmap(synergy_img, diff_img, synergy_txt, diff_txt, timestamp):
    """Create Scientific Reports-compliant correlation heatmap focusing on cross-branch correlations"""
    
    # Set Scientific Reports style requirements
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Sample a subset of features for visualization
    n_features = min(50, synergy_img.shape[1])
    synergy_sample = synergy_img[:, :n_features]
    diff_sample = diff_img[:, :n_features]
    
    # Calculate ONLY cross-branch correlations (synergy vs difference)
    cross_correlation_matrix = np.corrcoef(synergy_sample.T, diff_sample.T)
    
    # Extract only the cross-branch section (top-right quadrant)
    cross_section = cross_correlation_matrix[:n_features, n_features:]
    
    # Create the focused heatmap with Scientific Reports dimensions
    plt.figure(figsize=(6, 4))
    
    # Create mask for upper triangle (avoid redundancy)
    mask = np.triu(np.ones_like(cross_section, dtype=bool))
    
    # Create heatmap with Scientific Reports styling
    sns.heatmap(cross_section, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, square=True, cbar_kws={"shrink": .8, "location": "right"})
    
    plt.title('Cross-Branch Feature Correlation Analysis', fontsize=9, fontweight='bold', pad=10)
    plt.xlabel('Difference Branch Features', fontsize=8)
    plt.ylabel('Synergy Branch Features', fontsize=8)
    
    # Calculate comprehensive statistics
    avg_cross_corr = np.mean(np.abs(cross_section))
    max_cross_corr = np.max(np.abs(cross_section))
    min_cross_corr = np.min(np.abs(cross_section))
    std_cross_corr = np.std(np.abs(cross_section))
    
    # Count correlation levels
    high_similarity = np.sum(np.abs(cross_section) > 0.5)
    medium_similarity = np.sum((np.abs(cross_section) > 0.2) & (np.abs(cross_section) <= 0.5))
    low_similarity = np.sum((np.abs(cross_section) > 0.1) & (np.abs(cross_section) <= 0.2))
    very_low_similarity = np.sum(np.abs(cross_section) <= 0.1)
    
    total_pairs = cross_section.size
    high_percent = (high_similarity / total_pairs) * 100
    medium_percent = (medium_similarity / total_pairs) * 100
    low_percent = (low_similarity / total_pairs) * 100
    very_low_percent = (very_low_similarity / total_pairs) * 100
    
    # Add Scientific Reports-compliant text annotations
    info_text = f"""STATISTICAL SUMMARY:
Average: {avg_cross_corr:.4f}
Maximum: {max_cross_corr:.4f}
Minimum: {min_cross_corr:.4f}
Std Dev: {std_cross_corr:.4f}

FEATURE PAIR ANALYSIS:
High Similarity (>0.5): {high_similarity} pairs ({high_percent:.1f}%)
Medium Similarity (0.2-0.5): {medium_similarity} pairs ({medium_percent:.1f}%)
Low Similarity (0.1-0.2): {low_similarity} pairs ({low_percent:.1f}%)
Very Low Similarity (≤0.1): {very_low_similarity} pairs ({very_low_percent:.1f}%)

ORTHOGONALITY ASSESSMENT:
Total Feature Pairs: {total_pairs}
Independent Pairs (≤0.2): {very_low_similarity + low_similarity} ({very_low_percent + low_percent:.1f}%)
Orthogonal Success Rate: {((very_low_similarity + low_similarity) / total_pairs) * 100:.1f}%

INTERPRETATION:
Low correlation = Good orthogonal learning
High correlation = Poor orthogonal learning
Target: < 0.2 average correlation"""
    
    # Add text box on the right side
    plt.text(1.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'),
             verticalalignment='top', fontfamily='monospace')
    
    # Add color legend explanation
    legend_text = """COLOR INTERPRETATION:
Red: High positive correlation
    (Branches learn similar features)

White: Low correlation  
    (Branches learn independent features)

Blue: High negative correlation
    (Branches learn complementary features)

TARGET: Mostly white/light blue areas"""
    
    plt.text(1.02, 0.3, legend_text, transform=plt.gca().transAxes, fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'),
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(f'difference_branch_visualizations/cross_branch_correlation_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Scientific Reports-compliant correlation heatmap saved")
    
    # Create supplementary correlation distribution visualization
    create_correlation_distribution_plot(cross_section, timestamp)
    
    # Also create the original full matrix for comparison
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(cross_correlation_matrix, dtype=bool))
    sns.heatmap(cross_correlation_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, square=True, cbar_kws={"shrink": .8})
    plt.title('Full Feature Correlation Matrix', fontsize=9, fontweight='bold', pad=10)
    plt.xlabel('Features (Synergy + Difference)', fontsize=8)
    plt.ylabel('Features (Synergy + Difference)', fontsize=8)
    plt.text(0.02, 0.98, f'Average Correlation: {np.mean(np.abs(cross_correlation_matrix)):.4f}', 
             transform=plt.gca().transAxes, fontsize=7, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'difference_branch_visualizations/full_correlation_heatmap_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Full correlation heatmap saved")

def create_correlation_distribution_plot(cross_section, timestamp):
    """Create Scientific Reports-compliant correlation distribution plot"""
    
    # Set Scientific Reports style requirements
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Flatten correlation values and remove self-correlations
    corr_values = cross_section.flatten()
    corr_values = corr_values[~np.isnan(corr_values)]
    corr_values = corr_values[corr_values != 1.0]
    
    # Create 2-panel figure (removed bottom panels for cleaner look)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel 1: Histogram of correlations
    ax1.hist(corr_values, bins=30, color='#4A90E2', edgecolor='#2E5BBA', alpha=0.8, linewidth=1.2)
    mean_corr = np.mean(corr_values)
    
    # Format mean correlation with appropriate precision
    if abs(mean_corr) < 0.001:
        mean_corr_str = f"{mean_corr:.6f}"
    else:
        mean_corr_str = f"{mean_corr:.3f}"
    
    ax1.axvline(mean_corr, color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean: {mean_corr_str}')
    ax1.axvline(-mean_corr, color='#E74C3C', linestyle='--', linewidth=2)
    ax1.axvline(0.2, color='#F39C12', linestyle=':', linewidth=2, label='Target: ±0.2')
    ax1.axvline(-0.2, color='#F39C12', linestyle=':', linewidth=2)
    ax1.set_xlabel('Correlation Coefficient', fontsize=8)
    ax1.set_ylabel('Frequency', fontsize=8)
    ax1.set_title('Distribution of Cross-Branch Correlations', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Correlation level breakdown
    categories = ['Very Low', 'Low', 'Medium', 'High']
    counts = [
        np.sum(np.abs(corr_values) <= 0.1),
        np.sum((np.abs(corr_values) > 0.1) & (np.abs(corr_values) <= 0.2)),
        np.sum((np.abs(corr_values) > 0.2) & (np.abs(corr_values) <= 0.5)),
        np.sum(np.abs(corr_values) > 0.5)
    ]
    percentages = [count/len(corr_values)*100 for count in counts]
    
    colors = ['#27AE60', '#58D68D', '#F39C12', '#E74C3C']
    bars = ax2.bar(categories, percentages, color=colors, alpha=0.9, edgecolor='#2C3E50', linewidth=1.2)
    ax2.set_ylabel('Percentage (%)', fontsize=8)
    ax2.set_title('Correlation Level Breakdown', fontsize=9, fontweight='bold')
    
    # Add percentage values inside bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.1f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Calculate key statistics for compact legend
    independent_pairs = counts[0] + counts[1]  # Very Low + Low
    success_rate = (independent_pairs / len(corr_values)) * 100
    
    # Add compact legend with key information to left panel
    legend_text = f"Mean Corr: {mean_corr_str}\nIndependent Pairs: {success_rate:.1f}%\nTarget: <0.2"
    ax1.text(0.02, 0.98, legend_text, transform=ax1.transAxes, fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'),
             verticalalignment='top', fontfamily='monospace')
    
    # Add success rate information to right panel
    success_text = f"Success Rate: {success_rate:.1f}%\n(≤0.2 correlation)\n\nVery Low: {counts[0]} pairs\nLow: {counts[1]} pairs\nMedium: {counts[2]} pairs\nHigh: {counts[3]} pairs"
    ax2.text(0.02, 0.98, success_text, transform=ax2.transAxes, fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'),
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'difference_branch_visualizations/correlation_distribution_{timestamp}.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ Correlation distribution visualization saved")

def create_metrics_dashboard(orthogonality_score, diversity_ratio, avg_complementarity, 
                           robustness_score, regularization_effect, timestamp):
    """Create metrics summary dashboard"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Orthogonality Score
    colors = ['green' if orthogonality_score < 0.1 else 'orange' if orthogonality_score < 0.2 else 'red']
    ax1.bar(['Orthogonality'], [orthogonality_score], color=colors, alpha=0.7)
    ax1.axhline(y=0.1, color='green', linestyle='--', label='Excellent (< 0.1)')
    ax1.axhline(y=0.2, color='orange', linestyle='--', label='Good (< 0.2)')
    ax1.set_title('Orthogonality Score')
    ax1.set_ylabel('Cosine Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Diversity Ratio
    colors = ['green' if diversity_ratio > 0.8 else 'orange' if diversity_ratio > 0.5 else 'red']
    ax2.bar(['Diversity'], [diversity_ratio], color=colors, alpha=0.7)
    ax2.axhline(y=0.8, color='green', linestyle='--', label='Excellent (> 0.8)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Good (> 0.5)')
    ax2.set_title('Diversity Ratio')
    ax2.set_ylabel('Ratio (Difference/Synergy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Complementarity Score
    colors = ['green' if avg_complementarity > 0.1 else 'orange' if avg_complementarity > 0.05 else 'red']
    ax3.bar(['Complementarity'], [avg_complementarity], color=colors, alpha=0.7)
    ax3.axhline(y=0.1, color='green', linestyle='--', label='Excellent (> 0.1)')
    ax3.axhline(y=0.05, color='orange', linestyle='--', label='Good (> 0.05)')
    ax3.set_title('Complementarity Score')
    ax3.set_ylabel('Average Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Overall Assessment
    metrics = ['Orthogonality', 'Diversity', 'Complementarity', 'Robustness', 'Regularization']
    scores = [orthogonality_score, diversity_ratio, avg_complementarity, robustness_score, regularization_effect/100]
    
    # Normalize scores for visualization
    normalized_scores = []
    for i, metric in enumerate(metrics):
        if metric in ['Orthogonality', 'Regularization']:
            # Lower is better
            if metric == 'Orthogonality':
                normalized_scores.append(1 - min(scores[i] / 0.2, 1))
            else:
                normalized_scores.append(1 - min(scores[i] / 0.15, 1))
        else:
            # Higher is better
            normalized_scores.append(min(scores[i] / 1.0, 1))
    
    colors = ['green' if score > 0.8 else 'orange' if score > 0.5 else 'red' for score in normalized_scores]
    bars = ax4.bar(metrics, normalized_scores, color=colors, alpha=0.7)
    ax4.set_title('Overall Assessment (Normalized)')
    ax4.set_ylabel('Normalized Score')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, normalized_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'difference_branch_visualizations/metrics_dashboard_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Metrics dashboard saved")

def create_distribution_comparison(synergy_img, diff_img, synergy_txt, diff_txt, timestamp):
    """Create embedding distribution comparison"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Image embedding distributions
    ax1.hist(synergy_img.flatten(), bins=50, alpha=0.7, color='blue', 
             label='Synergy Branch', density=True, edgecolor='black')
    ax1.hist(diff_img.flatten(), bins=50, alpha=0.7, color='red', 
             label='Difference Branch', density=True, edgecolor='black')
    ax1.set_title('Image Embedding Distributions')
    ax1.set_xlabel('Embedding Values')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Text embedding distributions
    ax2.hist(synergy_txt.flatten(), bins=50, alpha=0.7, color='green', 
             label='Synergy Branch', density=True, edgecolor='black')
    ax2.hist(diff_txt.flatten(), bins=50, alpha=0.7, color='orange', 
             label='Difference Branch', density=True, edgecolor='black')
    ax2.set_title('Text Embedding Distributions')
    ax2.set_xlabel('Embedding Values')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Variance comparison
    synergy_img_var = np.var(synergy_img, axis=0)
    diff_img_var = np.var(diff_img, axis=0)
    synergy_txt_var = np.var(synergy_txt, axis=0)
    diff_txt_var = np.var(diff_txt, axis=0)
    
    ax3.hist(synergy_img_var, bins=30, alpha=0.7, color='blue', 
             label='Synergy (Image)', density=True, edgecolor='black')
    ax3.hist(diff_img_var, bins=30, alpha=0.7, color='red', 
             label='Difference (Image)', density=True, edgecolor='black')
    ax3.set_title('Feature Variance Distributions (Image)')
    ax3.set_xlabel('Variance')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical summary
    stats_data = {
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Variance'],
        'Synergy (Image)': [np.mean(synergy_img), np.std(synergy_img), 
                           np.min(synergy_img), np.max(synergy_img), np.var(synergy_img)],
        'Difference (Image)': [np.mean(diff_img), np.std(diff_img), 
                              np.min(diff_img), np.max(diff_img), np.var(diff_img)],
        'Synergy (Text)': [np.mean(synergy_txt), np.std(synergy_txt), 
                           np.min(synergy_txt), np.max(synergy_txt), np.var(synergy_txt)],
        'Difference (Text)': [np.mean(diff_txt), np.std(diff_txt), 
                             np.min(diff_txt), np.max(diff_txt), np.var(diff_txt)]
    }
    
    # Create table
    table_data = []
    for i, metric in enumerate(stats_data['Metric']):
        row = [metric]
        for col in ['Synergy (Image)', 'Difference (Image)', 'Synergy (Text)', 'Difference (Text)']:
            row.append(f"{stats_data[col][i]:.4f}")
        table_data.append(row)
    
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=table_data, 
                      colLabels=['Metric', 'Synergy (Img)', 'Difference (Img)', 
                                'Synergy (Txt)', 'Difference (Txt)'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title('Statistical Summary of Embeddings')
    
    plt.tight_layout()
    plt.savefig(f'difference_branch_visualizations/distribution_comparison_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Distribution comparison saved")

def analyze_difference_branch_importance():
    """Analyze why the difference branch is important despite performance metrics"""
    print("🔍 DIFFERENCE BRANCH IMPORTANCE ANALYSIS")
    print("=" * 60)
    print("Providing metrics to justify keeping the difference branch")
    print()
    
    # Load test data
    print("📁 Loading test data (1000 samples for comprehensive analysis)...")
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder=config.DATASET_MODE
    )
    data_loader.tokenizer = load_tokenizer_from_metadata()
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    test_dataset = data_loader.get_test_data(num_samples=1000)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    # Load your trained model
    print("\n📁 Loading your trained model...")
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy07_main_loss02_ortho01_branch_v3/export/model_weights.pth'
    
    base_model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in base_model.parameters()):,} parameters")
    
    # Define variants for detailed analysis
    variants = [
        ("full_model", "none"),
        ("synergy_only", "synergy_only"),
        ("difference_only", "difference_only"),
    ]
    
    # Store results
    analysis_results = {}
    
    # Evaluate each variant
    for variant_name, ablation_type in variants:
        print(f"\n{'='*60}")
        print(f"🎯 Evaluating: {variant_name}")
        print(f"{'='*60}")
        
        try:
            if variant_name == "full_model":
                model_to_test = base_model
            else:
                # Create ablated model
                class AblatedModel(MultimodalFusion):
                    def __init__(self, base_model, ablation_type):
                        super().__init__(
                            vocab_size=config.get_vocab_size(),
                            embed_dim=config.get_embed_dim(),
                            num_heads=config.get_current_config()['num_heads'],
                            num_layers=config.get_current_config()['num_layers']
                        )
                        self.load_state_dict(base_model.state_dict())
                        self.ablation_type = ablation_type
                    
                    def forward(self, inputs, training=True):
                        images, texts = inputs
                        image_tokens = self.image_encoder(images, training=training)
                        text_tokens = self.text_encoder(texts, training=training)
                        
                        synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
                        diff_img_emb, diff_txt_emb = self.difference_branch(image_tokens, text_tokens)
                        
                        if self.ablation_type == "synergy_only":
                            # Use only synergy branch
                            final_img_emb = synergy_img_emb
                            final_txt_emb = synergy_txt_emb
                        elif self.ablation_type == "difference_only":
                            # Use only difference branch
                            final_img_emb = diff_img_emb
                            final_txt_emb = diff_txt_emb
                        else:
                            # Use both branches (full model)
                            import torch.nn.functional as F
                            final_img_emb = F.normalize((synergy_img_emb + diff_img_emb) / 2, p=2, dim=-1)
                            final_txt_emb = F.normalize((synergy_txt_emb + diff_txt_emb) / 2, p=2, dim=-1)
                        
                        return final_img_emb, final_txt_emb
                
                model_to_test = AblatedModel(base_model, ablation_type)
                model_to_test.eval()
            
            # Run evaluation
            results = evaluate_cross_modal_retrieval_streaming(
                model=model_to_test,
                test_dataset=test_loader,
                k_values=[1, 5, 10],
                batch_size=32,
                visualize=False,
                num_vis_examples=0
            )
            
            # Store results
            analysis_results[variant_name] = results
            
            print(f"📊 Results for {variant_name}:")
            print(f"   MRR: {results['avg_mrr']:.4f}")
            print(f"   Recall@1: {results['avg_recall@1']:.4f}")
            print(f"   Recall@5: {results['avg_recall@5']:.4f}")
            print(f"   Recall@10: {results['avg_recall@10']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {variant_name}: {e}")
            analysis_results[variant_name] = {
                'avg_mrr': 0.0,
                'avg_recall@1': 0.0,
                'avg_recall@5': 0.0,
                'avg_recall@10': 0.0
            }
    
    # Analyze branch characteristics
    print(f"\n{'='*60}")
    print("🔍 BRANCH CHARACTERISTIC ANALYSIS")
    print(f"{'='*60}")
    
    # Get branch embeddings for analysis
    print("📊 Analyzing branch characteristics...")
    
    # Sample more batches for comprehensive analysis
    sample_batches = []
    for i, batch in enumerate(test_loader):
        if i >= 10:  # Analyze 10 batches (320 samples) for better statistics
            break
        sample_batches.append(batch)
    
    # Analyze branch behavior
    base_model.eval()
    with torch.no_grad():
        synergy_embeddings = []
        diff_embeddings = []
        full_embeddings = []
        
        for batch in sample_batches:
            # Handle batch format
            if isinstance(batch, dict):
                images, texts = batch['images'], batch['captions']
            else:
                images, texts = batch
            
            # Convert images to NCHW format if needed
            if images.shape[1] != 3:  # If not already in NCHW format
                images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
            # Get branch embeddings
            img_emb, txt_emb, synergy_img, synergy_txt, diff_img, diff_txt = base_model(
                (images, texts), training=False, return_branch_embeddings=True
            )
            
            synergy_embeddings.append((synergy_img, synergy_txt))
            diff_embeddings.append((diff_img, diff_txt))
            full_embeddings.append((img_emb, txt_emb))
    
    # Calculate metrics
    print("\n📊 BRANCH CHARACTERISTICS:")
    print("-" * 40)
    
    # 1. Orthogonality Analysis
    print("1️⃣ ORTHOGONALITY ANALYSIS:")
    orthogonality_scores = []
    for (syn_img, syn_txt), (diff_img, diff_txt) in zip(synergy_embeddings, diff_embeddings):
        # Calculate cosine similarity between branches
        img_orthogonality = torch.abs(torch.sum(syn_img * diff_img, dim=1)).mean()
        txt_orthogonality = torch.abs(torch.sum(syn_txt * diff_txt, dim=1)).mean()
        orthogonality_scores.append((img_orthogonality.item(), txt_orthogonality.item()))
    
    avg_img_ortho = sum(score[0] for score in orthogonality_scores) / len(orthogonality_scores)
    avg_txt_ortho = sum(score[1] for score in orthogonality_scores) / len(orthogonality_scores)
    
    print(f"   Image branch orthogonality: {avg_img_ortho:.6f}")
    print(f"   Text branch orthogonality: {avg_txt_ortho:.6f}")
    print(f"   Average orthogonality: {(avg_img_ortho + avg_txt_ortho) / 2:.6f}")
    
    # 2. Embedding Diversity Analysis
    print("\n2️⃣ EMBEDDING DIVERSITY ANALYSIS:")
    synergy_diversity = []
    diff_diversity = []
    
    for (syn_img, syn_txt), (diff_img, diff_txt) in zip(synergy_embeddings, diff_embeddings):
        # Calculate embedding variance (diversity)
        syn_img_var = torch.var(syn_img, dim=0).mean()
        syn_txt_var = torch.var(syn_txt, dim=0).mean()
        diff_img_var = torch.var(diff_img, dim=0).mean()
        diff_txt_var = torch.var(diff_txt, dim=0).mean()
        
        synergy_diversity.append((syn_img_var.item(), syn_txt_var.item()))
        diff_diversity.append((diff_img_var.item(), diff_txt_var.item()))
    
    avg_syn_diversity = sum(score[0] + score[1] for score in synergy_diversity) / (len(synergy_diversity) * 2)
    avg_diff_diversity = sum(score[0] + score[1] for score in diff_diversity) / (len(diff_diversity) * 2)
    
    print(f"   Synergy branch diversity: {avg_syn_diversity:.6f}")
    print(f"   Difference branch diversity: {avg_diff_diversity:.6f}")
    print(f"   Diversity ratio (diff/syn): {avg_diff_diversity / avg_syn_diversity:.4f}")
    
    # 3. Feature Complementarity Analysis
    print("\n3️⃣ FEATURE COMPLEMENTARITY ANALYSIS:")
    
    # Calculate how much unique information each branch provides
    complementarity_scores = []
    for (syn_img, syn_txt), (diff_img, diff_txt), (full_img, full_txt) in zip(synergy_embeddings, diff_embeddings, full_embeddings):
        # How much does full model differ from synergy-only?
        synergy_only_img = syn_img
        synergy_only_txt = syn_txt
        
        img_complementarity = torch.norm(full_img - synergy_only_img, dim=1).mean()
        txt_complementarity = torch.norm(full_txt - synergy_only_txt, dim=1).mean()
        
        complementarity_scores.append((img_complementarity.item(), txt_complementarity.item()))
    
    avg_complementarity = sum(score[0] + score[1] for score in complementarity_scores) / (len(complementarity_scores) * 2)
    print(f"   Average complementarity: {avg_complementarity:.6f}")
    
    # 4. Performance Analysis
    print("\n4️⃣ PERFORMANCE ANALYSIS:")
    full_mrr = analysis_results['full_model']['avg_mrr']
    synergy_mrr = analysis_results['synergy_only']['avg_mrr']
    diff_mrr = analysis_results['difference_only']['avg_mrr']
    
    print(f"   Full model MRR: {full_mrr:.4f}")
    print(f"   Synergy-only MRR: {synergy_mrr:.4f}")
    print(f"   Difference-only MRR: {diff_mrr:.4f}")
    
    # 5. Justification Metrics
    print(f"\n{'='*60}")
    print("🎯 JUSTIFICATION METRICS FOR KEEPING DIFFERENCE BRANCH")
    print(f"{'='*60}")
    
    # Metric 1: Orthogonality Score
    orthogonality_score = (avg_img_ortho + avg_txt_ortho) / 2
    print(f"1️⃣ ORTHOGONALITY SCORE: {orthogonality_score:.6f}")
    if orthogonality_score < 0.1:
        print(f"   ✅ EXCELLENT: Branches are highly orthogonal (learn different features)")
    elif orthogonality_score < 0.2:
        print(f"   ✅ GOOD: Branches are reasonably orthogonal")
    else:
        print(f"   ⚠️  POOR: Branches are not orthogonal enough")
    
    # Metric 2: Diversity Ratio
    diversity_ratio = avg_diff_diversity / avg_syn_diversity
    print(f"\n2️⃣ DIVERSITY RATIO: {diversity_ratio:.4f}")
    if diversity_ratio > 0.8:
        print(f"   ✅ EXCELLENT: Difference branch provides diverse features")
    elif diversity_ratio > 0.5:
        print(f"   ✅ GOOD: Difference branch adds meaningful diversity")
    else:
        print(f"   ⚠️  POOR: Difference branch lacks diversity")
    
    # Metric 3: Complementarity Score
    print(f"\n3️⃣ COMPLEMENTARITY SCORE: {avg_complementarity:.6f}")
    if avg_complementarity > 0.1:
        print(f"   ✅ EXCELLENT: Difference branch provides unique information")
    elif avg_complementarity > 0.05:
        print(f"   ✅ GOOD: Difference branch adds complementary features")
    else:
        print(f"   ⚠️  POOR: Difference branch provides little unique information")
    
    # Metric 4: Robustness Score
    robustness_score = full_mrr - synergy_mrr
    print(f"\n4️⃣ ROBUSTNESS SCORE: {robustness_score:.4f}")
    if robustness_score > 0:
        print(f"   ✅ POSITIVE: Full model outperforms synergy-only")
    else:
        print(f"   ⚠️  NEGATIVE: Synergy-only outperforms full model")
    
    # Metric 5: Regularization Effect
    regularization_effect = abs(full_mrr - synergy_mrr) / full_mrr * 100
    print(f"\n5️⃣ REGULARIZATION EFFECT: {regularization_effect:.2f}%")
    if regularization_effect < 5:
        print(f"   ✅ STABLE: Difference branch provides stable regularization")
    elif regularization_effect < 15:
        print(f"   ✅ MODERATE: Reasonable regularization effect")
    else:
        print(f"   ⚠️  HIGH: Strong regularization effect")
    
    # Final Recommendation
    print(f"\n{'='*60}")
    print("🎯 FINAL RECOMMENDATION")
    print(f"{'='*60}")
    
    positive_metrics = 0
    total_metrics = 5
    
    if orthogonality_score < 0.2:
        positive_metrics += 1
    if diversity_ratio > 0.5:
        positive_metrics += 1
    if avg_complementarity > 0.05:
        positive_metrics += 1
    if robustness_score >= 0:
        positive_metrics += 1
    if regularization_effect < 15:
        positive_metrics += 1
    
    print(f"📊 POSITIVE METRICS: {positive_metrics}/{total_metrics}")
    
    if positive_metrics >= 4:
        print(f"🎯 STRONG RECOMMENDATION: KEEP DIFFERENCE BRANCH")
        print(f"   ✅ Multiple metrics support its importance")
    elif positive_metrics >= 3:
        print(f"🎯 MODERATE RECOMMENDATION: KEEP DIFFERENCE BRANCH")
        print(f"   ✅ Several metrics support its value")
    else:
        print(f"🎯 WEAK RECOMMENDATION: CONSIDER REMOVING DIFFERENCE BRANCH")
        print(f"   ⚠️  Limited evidence for its importance")
    
    print(f"\n📋 KEY JUSTIFICATION:")
    print(f"   • Orthogonality: {orthogonality_score:.6f} (should be < 0.2)")
    print(f"   • Diversity: {diversity_ratio:.4f} (should be > 0.5)")
    print(f"   • Complementarity: {avg_complementarity:.6f} (should be > 0.05)")
    print(f"   • Robustness: {robustness_score:.4f} (should be > 0)")
    print(f"   • Regularization: {regularization_effect:.2f}% (should be < 15%)")
    
    print(f"\n✅ Analysis completed!")
    
    # Create comprehensive visualizations
    print(f"\n{'='*60}")
    print("🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*60}")
    
    create_visualizations(
        synergy_embeddings, diff_embeddings, full_embeddings,
        orthogonality_score, diversity_ratio, avg_complementarity,
        robustness_score, regularization_effect
    )

if __name__ == "__main__":
    analyze_difference_branch_importance() 