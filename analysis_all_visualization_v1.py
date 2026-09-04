#!/usr/bin/env python3
"""
Comprehensive Visualization Module for MIMIC-CXR Multimodal Retrieval - PyTorch Version
========================================================================================
All visualization functions consolidated from across the project.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import textwrap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Patch
from collections import defaultdict, Counter
import random
import os
from datetime import datetime

plt.switch_backend('Agg')

try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False
    print("Seaborn not available. Some visualizations will use basic matplotlib.")

# ============================================================================
# RETRIEVAL VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_retrieval_examples(model, test_data, num_examples=3, k=3, output_dir=None):
    """Visualize retrieval examples with Scientific Reports publication standards"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(output_dir):
            if filename.startswith("retrieval_example_") and filename.endswith(".png"):
                old_file_path = os.path.join(output_dir, filename)
                os.remove(old_file_path)
        
        print(f"Visualizations will be saved to {output_dir}")
    
    # Set Scientific Reports font and styling
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7
    
    # Scientific Reports color palette
    colors = {
        'blue': '#1f77b4',
        'red': '#d62728', 
        'green': '#2ca02c',
        'orange': '#ff7f0e',
        'white': '#ffffff',
        'black': '#000000',
        'gray': '#cccccc'
    }
    
    # Move model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Convert data to PyTorch tensors if needed
        if isinstance(test_data['images'], np.ndarray):
            images = torch.FloatTensor(test_data['images'])
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        else:
            images = test_data['images']
            
        if isinstance(test_data['captions'], np.ndarray):
            captions = torch.LongTensor(test_data['captions'])
        else:
            captions = test_data['captions']
        
        # Move to device if model is on GPU
        device = next(model.parameters()).device
        images = images.to(device)
        captions = captions.to(device)
        
        image_emb, text_emb = model((images, captions), training=False)
    
    # Compute similarity matrices using PyTorch
    i2t_sim = torch.matmul(image_emb, text_emb.transpose(0, 1)).cpu().numpy()
    t2i_sim = torch.matmul(text_emb, image_emb.transpose(0, 1)).cpu().numpy()
    
    tokenizer = test_data.get('tokenizer', None)
    
    num_examples = min(num_examples, len(test_data['images']))
    example_indices = random.sample(range(len(test_data['images'])), num_examples)
    
    for idx in example_indices:
        # Create figure optimized for single-column width (25% larger)
        fig = plt.figure(figsize=(9.0, 6.0), facecolor=colors['white'])
        
        # Create title with Scientific Reports formatting
        title_ax = fig.add_axes([0.1, 0.92, 0.8, 0.06])
        title_ax.text(0.5, 0.5, f"Bidirectional Chest X-ray Image–Report Retrieval Results", 
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color=colors['black'],
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['white'], 
                             edgecolor=colors['gray'], linewidth=1))
        title_ax.axis('off')
        
        # Create two horizontal boxes layout with increased height and proper spacing
        # Top box for Image→Text retrieval (increased height and moved down)
        top_box_ax = fig.add_axes([0.05, 0.50, 0.9, 0.40])
        top_box_ax.set_facecolor(colors['white'])
        top_box_ax.spines['top'].set_visible(True)
        top_box_ax.spines['bottom'].set_visible(True)
        top_box_ax.spines['left'].set_visible(True)
        top_box_ax.spines['right'].set_visible(True)
        top_box_ax.spines['top'].set_color('#888888')  # Light gray
        top_box_ax.spines['bottom'].set_color('#888888')  # Light gray
        top_box_ax.spines['left'].set_color('#888888')  # Light gray
        top_box_ax.spines['right'].set_color('#888888')  # Light gray
        top_box_ax.spines['top'].set_linewidth(1.5)  # Thinner lines
        top_box_ax.spines['bottom'].set_linewidth(1.5)  # Thinner lines
        top_box_ax.spines['left'].set_linewidth(1.5)  # Thinner lines
        top_box_ax.spines['right'].set_linewidth(1.5)  # Thinner lines
        top_box_ax.set_xticks([])
        top_box_ax.set_yticks([])
        
        # Bottom box for Text→Image retrieval (increased height and moved up)
        bottom_box_ax = fig.add_axes([0.05, 0.05, 0.9, 0.40])
        bottom_box_ax.set_facecolor(colors['white'])
        bottom_box_ax.spines['top'].set_visible(True)
        bottom_box_ax.spines['bottom'].set_visible(True)
        bottom_box_ax.spines['left'].set_visible(True)
        bottom_box_ax.spines['right'].set_visible(True)
        bottom_box_ax.spines['top'].set_color('#888888')  # Light gray
        bottom_box_ax.spines['bottom'].set_color('#888888')  # Light gray
        bottom_box_ax.spines['left'].set_color('#888888')  # Light gray
        bottom_box_ax.spines['right'].set_color('#888888')  # Light gray
        bottom_box_ax.spines['top'].set_linewidth(1.5)  # Thinner lines
        bottom_box_ax.spines['bottom'].set_linewidth(1.5)  # Thinner lines
        bottom_box_ax.spines['left'].set_linewidth(1.5)  # Thinner lines
        bottom_box_ax.spines['right'].set_linewidth(1.5)  # Thinner lines
        bottom_box_ax.set_xticks([])
        bottom_box_ax.set_yticks([])
        
        # TOP BOX: Query Image + 3 Text Results
        # Query Image (left side of top box) - moved down and away from right boundary
        ax_query_img = fig.add_axes([0.08, 0.52, 0.1875, 0.375])
        query_image = test_data['images'][idx]
        ax_query_img.imshow(query_image, cmap='gray')
        ax_query_img.set_title("Query Image", fontsize=6, fontweight='bold', color=colors['black'])
        ax_query_img.axis('off')
        ax_query_img.set_aspect('equal')
        
        # Add professional border to query image
        for spine in ax_query_img.spines.values():
            spine.set_color(colors['black'])
            spine.set_linewidth(2)
        
        # Get text retrieval results
        sim_scores_i2t = i2t_sim[idx]
        top_k_indices_i2t = np.argsort(sim_scores_i2t)[::-1][:k]
        
        # Rank 1 Text Box (middle of top box) - moved left away from right boundary
        ax_text1 = fig.add_axes([0.28, 0.55, 0.207, 0.30])
        ax_text1.axis('off')
        rank1_caption = decode_caption(test_data['captions'][top_k_indices_i2t[0]], tokenizer)
        rank1_score = sim_scores_i2t[top_k_indices_i2t[0]]
        is_correct1 = top_k_indices_i2t[0] == idx
        
        # Professional text wrapping
        import textwrap
        wrapped_text = textwrap.fill(rank1_caption, width=25)
        
        # Rank and score label
        ax_text1.text(0.5, 0.98, f"Rank 1\nScore: {rank1_score:.2f}",
                     ha='center', va='top', fontsize=6, fontweight='bold',
                     color=colors['green'] if is_correct1 else colors['black'],
                     transform=ax_text1.transAxes)
        
        # Text content with professional background
        text_box_props = dict(boxstyle="round,pad=0.3", 
                            facecolor=colors['white'], 
                            edgecolor=colors['gray'], 
                            linewidth=1, alpha=0.9)
        ax_text1.text(0.5, 0.5, wrapped_text,
                     ha='center', va='center', fontsize=7,
                     bbox=text_box_props,
                     transform=ax_text1.transAxes, clip_on=True, wrap=True)
        
        # Rank 2 Text Box - moved left away from right boundary
        ax_text2 = fig.add_axes([0.51, 0.55, 0.207, 0.30])
        ax_text2.axis('off')
        rank2_caption = decode_caption(test_data['captions'][top_k_indices_i2t[1]], tokenizer)
        rank2_score = sim_scores_i2t[top_k_indices_i2t[1]]
        is_correct2 = top_k_indices_i2t[1] == idx
        
        wrapped_text = textwrap.fill(rank2_caption, width=25)
        
        ax_text2.text(0.5, 0.98, f"Rank 2\nScore: {rank2_score:.2f}",
                     ha='center', va='top', fontsize=6, fontweight='bold',
                     color=colors['green'] if is_correct2 else colors['black'],
                     transform=ax_text2.transAxes)
        
        ax_text2.text(0.5, 0.5, wrapped_text,
                     ha='center', va='center', fontsize=7,
                     bbox=text_box_props,
                     transform=ax_text2.transAxes, clip_on=True, wrap=True)
        
        # Rank 3 Text Box - moved left away from right boundary
        ax_text3 = fig.add_axes([0.74, 0.55, 0.207, 0.30])
        ax_text3.axis('off')
        rank3_caption = decode_caption(test_data['captions'][top_k_indices_i2t[2]], tokenizer)
        rank3_score = sim_scores_i2t[top_k_indices_i2t[2]]
        is_correct3 = top_k_indices_i2t[2] == idx
        
        wrapped_text = textwrap.fill(rank3_caption, width=25)
        
        ax_text3.text(0.5, 0.98, f"Rank 3\nScore: {rank3_score:.2f}",
                     ha='center', va='top', fontsize=6, fontweight='bold',
                     color=colors['green'] if is_correct3 else colors['black'],
                     transform=ax_text3.transAxes)
        
        ax_text3.text(0.5, 0.5, wrapped_text,
                     ha='center', va='center', fontsize=7,
                     bbox=text_box_props,
                     transform=ax_text3.transAxes, clip_on=True, wrap=True)
        
        # BOTTOM BOX: Query Text + 3 Image Results
        # Query Text (left side of bottom box) - moved further left
        ax_query_text = fig.add_axes([0.05, 0.08, 0.207, 0.30])
        ax_query_text.axis('off')
        query_caption = decode_caption(test_data['captions'][idx], tokenizer)
        wrapped_query = textwrap.fill(query_caption, width=25)
        
        # Query Text label
        ax_query_text.text(0.5, 0.98, "Query Text",
                          ha='center', va='top', fontsize=6, fontweight='bold',
                          color=colors['blue'], transform=ax_query_text.transAxes)
        
        # Query text content with professional background
        text_box_props = dict(boxstyle="round,pad=0.3", 
                            facecolor=colors['white'], 
                            edgecolor=colors['gray'], 
                            linewidth=1, alpha=0.9)
        ax_query_text.text(0.5, 0.5, wrapped_query, 
                          ha='center', va='center', fontsize=7, bbox=text_box_props,
                          transform=ax_query_text.transAxes, clip_on=True, wrap=True)
        
        # Get image retrieval results
        sim_scores_t2i = t2i_sim[idx]
        top_k_indices_t2i = np.argsort(sim_scores_t2i)[::-1][:k]
        
        # Rank 1 Image - moved further left
        ax_img1 = fig.add_axes([0.25, 0.08, 0.2025, 0.3375])
        rank1_img = test_data['images'][top_k_indices_t2i[0]]
        rank1_score = sim_scores_t2i[top_k_indices_t2i[0]]
        is_correct_img1 = top_k_indices_t2i[0] == idx
        
        ax_img1.imshow(rank1_img, cmap='gray')
        ax_img1.set_title(f"Rank 1\nScore: {rank1_score:.2f}", 
                         fontsize=6, fontweight='bold',
                         color=colors['green'] if is_correct_img1 else colors['black'])
        ax_img1.axis('off')
        ax_img1.set_aspect('equal')
        
        # Add professional border to retrieved images
        for spine in ax_img1.spines.values():
            spine.set_color(colors['black'])
            spine.set_linewidth(1)
        
        # Rank 2 Image - moved further left
        ax_img2 = fig.add_axes([0.48, 0.08, 0.2025, 0.3375])
        rank2_img = test_data['images'][top_k_indices_t2i[1]]
        rank2_score = sim_scores_t2i[top_k_indices_t2i[1]]
        is_correct_img2 = top_k_indices_t2i[1] == idx
        
        ax_img2.imshow(rank2_img, cmap='gray')
        ax_img2.set_title(f"Rank 2\nScore: {rank2_score:.2f}", 
                         fontsize=6, fontweight='bold',
                         color=colors['green'] if is_correct_img2 else colors['black'])
        ax_img2.axis('off')
        ax_img2.set_aspect('equal')
        
        # Add professional border to retrieved images
        for spine in ax_img2.spines.values():
            spine.set_color(colors['black'])
            spine.set_linewidth(1)
        
        # Rank 3 Image - moved further left
        ax_img3 = fig.add_axes([0.71, 0.08, 0.2025, 0.3375])
        rank3_img = test_data['images'][top_k_indices_t2i[2]]
        rank3_score = sim_scores_t2i[top_k_indices_t2i[2]]
        is_correct_img3 = top_k_indices_t2i[2] == idx
        
        ax_img3.imshow(rank3_img, cmap='gray')
        ax_img3.set_title(f"Rank 3\nScore: {rank3_score:.2f}", 
                         fontsize=6, fontweight='bold',
                         color=colors['green'] if is_correct_img3 else colors['black'])
        ax_img3.axis('off')
        ax_img3.set_aspect('equal')
        
        # Add professional border to retrieved images
        for spine in ax_img3.spines.values():
            spine.set_color(colors['black'])
            spine.set_linewidth(1)
        

        
        # Add global boundary with professional styling
        fig.patch.set_edgecolor(colors['black'])
        fig.patch.set_linewidth(1)
        
        # Save with Scientific Reports specifications
        if output_dir:
            filename = os.path.join(output_dir, f"retrieval_example_{idx}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor=colors['white'], edgecolor=colors['black'],
                       format='png', transparent=False)
            print(f"Saved Scientific Reports compliant visualization to {filename}")
        plt.close()

# Removed old helper functions - functionality integrated into main visualize_retrieval_examples function

def decode_caption(caption_seq, tokenizer=None):
    """Decode tokenized caption back to text"""
    if tokenizer is None:
        return f"Caption sequence: {caption_seq[:10]}..."
    
    try:
        # Handle both numpy arrays and torch tensors
        if isinstance(caption_seq, torch.Tensor):
            caption_seq = caption_seq.cpu().numpy()
        
        valid_tokens = caption_seq[caption_seq > 0]
        
        # Handle different tokenizer formats
        reverse_word_index = {}
        
        if hasattr(tokenizer, 'word_index'):
            # Standard Keras tokenizer
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        elif hasattr(tokenizer, 'word2idx'):
            # Custom tokenizer with word2idx
            reverse_word_index = {v: k for k, v in tokenizer.word2idx.items()}
        elif isinstance(tokenizer, dict):
            # Tokenizer is a dictionary
            if 'word_index' in tokenizer:
                reverse_word_index = {v: k for k, v in tokenizer['word_index'].items()}
            elif 'word2idx' in tokenizer:
                reverse_word_index = {v: k for k, v in tokenizer['word2idx'].items()}
            else:
                # Try to use the dictionary directly as word_index
                reverse_word_index = {v: k for k, v in tokenizer.items()}
        else:
            # Fallback: try to access word_index attribute
            try:
                reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
            except:
                return f"TOKENIZER_ERROR: Unsupported tokenizer format"
        
        # Add padding token
        reverse_word_index[0] = '<PAD>'
        
        words = []
        for token_id in valid_tokens:
            word = reverse_word_index.get(token_id, f"<UNK_{token_id}>")
            words.append(word)
        
        return " ".join(words)
    except Exception as e:
        return f"DECODE_ERROR: {str(e)[:50]}"

# ============================================================================
# TRAINING VISUALIZATION FUNCTIONS
# ============================================================================

class TrainingVisualizer:
    """Training progress visualization - PyTorch version"""
    
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        self.history = defaultdict(list)
        os.makedirs(save_dir, exist_ok=True)
    
    def update_history(self, epoch_metrics):
        """Update training history with new epoch metrics"""
        for key, value in epoch_metrics.items():
            self.history[key].append(value)
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.history:
            print("No training history to plot!")
            return
        
        plt.figure(figsize=(16, 6))
        
        # Find the correct loss key that exists in history
        loss_key = None
        for key in ['total_loss', 'loss']:
            if key in self.history and len(self.history[key]) > 0:
                loss_key = key
                break
        
        if loss_key is None:
            print("No loss data found in training history!")
            print(f"Available keys: {list(self.history.keys())}")
            return
        
        epochs = list(range(1, len(self.history[loss_key]) + 1))
        
        plt.subplot(1, 2, 1)
        
        plt.plot(epochs, self.history[loss_key], 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if len(epochs) > 0:
            last_loss = self.history[loss_key][-1]
            plt.annotate(f'{last_loss:.4f}',
                        xy=(epochs[-1], last_loss),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.subplot(1, 2, 2)
        
        recall_keys = [k for k in self.history.keys() if 'recall' in k.lower()]
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for i, key in enumerate(recall_keys):
            if len(self.history[key]) > 0:  # Only plot if data exists
                color = colors[i % len(colors)]
                plt.plot(epochs, self.history[key],
                        color=color, linewidth=2, marker='s', markersize=3, label=key)
                
                if len(epochs) > 0:
                    last_recall = self.history[key][-1]
                    plt.annotate(f'{last_recall:.3f}',
                                xy=(epochs[-1], last_recall),
                                xytext=(10, 5), textcoords='offset points',
                                fontsize=9, color=color)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.title('Validation Recall', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/training_progress_{timestamp}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress saved to: {self.save_dir}/training_progress_{timestamp}.png")

    def plot_dual_branch_losses(self):
        """Plot synergy loss and orthogonal loss with simplified comparison"""
        if not self.history:
            print("No training history to plot!")
            return
        
        # Check if we have the required loss data
        has_synergy = 'synergy_loss' in self.history and len(self.history['synergy_loss']) > 0
        has_orthogonal = 'orthogonal_loss' in self.history and len(self.history['orthogonal_loss']) > 0
        
        if not (has_synergy and has_orthogonal):
            print("Required loss data not found!")
            print(f"Available keys: {list(self.history.keys())}")
            return
        
        epochs = list(range(1, len(self.history['synergy_loss']) + 1))
        
        plt.figure(figsize=(12, 5))
        
        # Add main title
        plt.suptitle('Dual Branch Architecture Loss Analysis', fontsize=14, fontweight='bold', y=0.98)
        
        # Plot 1: Synergy Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['synergy_loss'], 'g-', linewidth=2, marker='o', markersize=4, label='Synergy Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Synergy Branch Loss', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Orthogonal Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['orthogonal_loss'], 'b-', linewidth=2, marker='s', markersize=4, label='Orthogonal Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Orthogonal Regularization Loss', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.save_dir}/dual_branch_losses_{timestamp}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Dual branch losses saved to: {self.save_dir}/dual_branch_losses_{timestamp}.png")

    def plot_similarity_matrix(self, similarity_matrix, title="Similarity Matrix", k=20):
        """Plot similarity matrix heatmap with simplified view"""
        try:
            # Handle PyTorch tensors
            if isinstance(similarity_matrix, torch.Tensor):
                similarity_matrix = similarity_matrix.cpu().numpy()
                
            # Ensure similarity_matrix is a numpy array
            if not isinstance(similarity_matrix, np.ndarray):
                print(f"Error: similarity_matrix must be torch.Tensor or numpy.ndarray, got {type(similarity_matrix)}")
                return
            
            # Check if matrix is empty or has invalid values
            if similarity_matrix.size == 0:
                print("Error: similarity_matrix is empty")
                return
            
            if np.isnan(similarity_matrix).any() or np.isinf(similarity_matrix).any():
                print("Error: similarity_matrix contains NaN or Inf values")
                return
            
            # Limit k to matrix dimensions
            k = min(k, similarity_matrix.shape[0], similarity_matrix.shape[1])
            if k <= 0:
                print("Error: k must be positive")
                return
            
            sim_subset = similarity_matrix[:k, :k]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"similarity_matrix_{timestamp}.png"
            
            plt.figure(figsize=(10, 8))
            
            # Add main title
            plt.suptitle('Cross-Modal Similarity Matrix Analysis', fontsize=14, fontweight='bold', y=0.98)
        
            # Main similarity matrix heatmap
            if seaborn_available:
                sns.heatmap(sim_subset, annot=False, cmap='viridis', square=True,
                                   cbar_kws={'label': 'Cosine Similarity Score'})
            else:
                im = plt.imshow(sim_subset, cmap='viridis', aspect='auto')
                plt.colorbar(im, label='Cosine Similarity Score')
        
            plt.title(f'{title}\n(Matrix Size: {k}x{k})', fontsize=12, fontweight='bold', pad=20)
            plt.xlabel('Text Index', fontsize=12)
            plt.ylabel('Image Index', fontsize=12)
        
            # Add diagonal markers for correct matches
            for i in range(min(k, sim_subset.shape[0])):
                plt.plot(i + 0.5, i + 0.5, 'ro', markersize=6, markeredgecolor='white', markeredgewidth=1)
        
            # Add explanation text
            plt.text(0.02, 0.98, 'Red dots = Correct matches\n'
                    'Yellow = High similarity\n'
                    'Blue = Low similarity',
                transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top')
        
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        
            plt.savefig(f'{self.save_dir}/{filename}', dpi=200, bbox_inches='tight')
            plt.close()
        
            print(f"Similarity matrix saved to: {self.save_dir}/{filename}")

        except Exception as e:
            print(f"Error plotting similarity matrix: {e}")
            print(f"Matrix shape: {similarity_matrix.shape if hasattr(similarity_matrix, 'shape') else 'unknown'}")
            print(f"Matrix type: {type(similarity_matrix)}")

    def create_comprehensive_analysis(self, model, val_loader, epoch=None):
        """Create comprehensive training analysis - PyTorch version"""
        print(f"Creating comprehensive analysis for epoch {epoch}...")
        
        model.eval()
        
        # Collect a small sample from validation loader for visualization
        sample_images = []
        sample_texts = []
        sample_count = 0
        max_samples = 100  # Limit for visualization
        
        try:
            for batch in val_loader:
                if sample_count >= max_samples:
                    break
                
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch_images = batch['images']
                    batch_texts = batch['captions']
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_images, batch_texts = batch
                else:
                    print(f"Unexpected batch format: {type(batch)}")
                    continue
                
                # Convert images to BCHW format if needed
                if len(batch_images.shape) == 4 and batch_images.shape[1] != 3:
                    batch_images = batch_images.permute(0, 3, 1, 2)
                
                sample_images.append(batch_images)
                sample_texts.append(batch_texts)
                sample_count += len(batch_images)
                
                if sample_count >= max_samples:
                    break
            
            if len(sample_images) == 0:
                print("No validation data available for analysis")
                return
            
            # Concatenate samples
            images = torch.cat(sample_images, dim=0)[:max_samples]
            texts = torch.cat(sample_texts, dim=0)[:max_samples]
            
            print(f"Processing {len(images)} samples for similarity matrix...")
            
            with torch.no_grad():
                # Move to device if model is on GPU
                device = next(model.parameters()).device
                images = images.to(device)
                texts = texts.to(device)
                
                # Get embeddings
                image_emb, text_emb = model((images, texts), training=False)
                
                # Normalize embeddings for cosine similarity
                image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=1)
                text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(image_emb, text_emb.transpose(0, 1))
            
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Similarity matrix range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
            
            # Plot similarity matrix
            self.plot_similarity_matrix(similarity_matrix, f"Similarity Matrix - Epoch {epoch}")
            
        except Exception as e:
            print(f"Error creating comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping visualization generation")

# ============================================================================
# ATTENTION VISUALIZATION FUNCTIONS
# ============================================================================

def extract_attention_weights(model, images, texts, layer_idx=0):
    """Extract attention weights from hierarchical co-attention layers - PyTorch version"""
    model.eval()
    
    with torch.no_grad():
        # Convert data to PyTorch tensors if needed
        if isinstance(images, np.ndarray):
            images = torch.FloatTensor(images)
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
                
        if isinstance(texts, np.ndarray):
            texts = torch.LongTensor(texts)
        
        # Move to device if model is on GPU
        device = next(model.parameters()).device
        images = images.to(device)
        texts = texts.to(device)
        
        image_tokens = model.image_encoder(images, training=False)
        text_tokens = model.text_encoder(texts, training=False)
        
        synergy_layer = model.synergy_branch.co_attn_layers[layer_idx]
        
        # For PyTorch, we need to handle attention extraction differently
        # Since PyTorch MultiheadAttention doesn't return attention weights by default,
        # we'll approximate the attention visualization
        
        # Get the attention outputs
        i2t_output = synergy_layer.cross_attention1(
            query=image_tokens.transpose(0, 1),  # PyTorch expects (seq_len, batch, features)
            key=text_tokens.transpose(0, 1),
            value=text_tokens.transpose(0, 1)
        )[0].transpose(0, 1)  # Convert back to (batch, seq_len, features)
        
        t2i_output = synergy_layer.cross_attention2(
            query=text_tokens.transpose(0, 1),
            key=image_tokens.transpose(0, 1),
            value=image_tokens.transpose(0, 1)
        )[0].transpose(0, 1)
        
        # Approximate attention weights using dot product similarity
        # This is a simplified version - for exact attention weights, 
        # you'd need to modify the model to return attention weights
        i2t_weights = torch.matmul(
            torch.nn.functional.normalize(image_tokens, dim=-1),
            torch.nn.functional.normalize(text_tokens, dim=-1).transpose(1, 2)
        )
        
        t2i_weights = torch.matmul(
            torch.nn.functional.normalize(text_tokens, dim=-1),
            torch.nn.functional.normalize(image_tokens, dim=-1).transpose(1, 2)
        )
    
    return {
        'image_tokens': image_tokens.cpu().numpy(),
        'text_tokens': text_tokens.cpu().numpy(),
        'i2t_weights': i2t_weights.cpu().numpy(),
        't2i_weights': t2i_weights.cpu().numpy(),
    }

def visualize_image_attention(model, image, text_tokens, sample_idx=0, tokenizer=None, save_dir='attention_analysis'):
    """Visualize what parts of image are attended to by text tokens - PyTorch version"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure inputs are in the right format
    if len(image.shape) == 3:
        image = image[None]  # Add batch dimension
    if len(text_tokens.shape) == 1:
        text_tokens = text_tokens[None]  # Add batch dimension
    
    attention_weights = extract_attention_weights(model, image, text_tokens)
    
    t2i_attn = attention_weights['t2i_weights'][sample_idx].mean(axis=0)
    
    num_patches = t2i_attn.shape[1]
    patch_grid_size = int(np.sqrt(num_patches))
    
    if tokenizer:
        word_ids = text_tokens[sample_idx][text_tokens[sample_idx] > 0]
        words = []
        for token_id in word_ids:
            for word, idx in tokenizer.word_index.items():
                if idx == token_id:
                    words.append(word)
                    break
            else:
                words.append(f"<{token_id}>")
    else:
        words = [f"token_{i}" for i in range(len(text_tokens[sample_idx]))]
    
    # Handle tensor conversion
    if isinstance(text_tokens, torch.Tensor):
        text_tokens_np = text_tokens.cpu().numpy()
    else:
        text_tokens_np = text_tokens
    
    valid_tokens = len([t for t in text_tokens_np[sample_idx] if t > 0])
    top_words_idx = np.argsort(t2i_attn[:valid_tokens].max(axis=1))[-6:]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Attention Visualization', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Convert image for visualization if needed
    if isinstance(image, torch.Tensor):
        image_np = image[sample_idx].cpu().numpy()
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)  # Convert from CHW to HWC
    else:
        image_np = image[sample_idx]
    
    for i, token_idx in enumerate(top_words_idx):
        ax = axes[i]
        
        token_attention = t2i_attn[token_idx]
        attention_map = token_attention.reshape(patch_grid_size, patch_grid_size)
        
        scale_factor = 224 // patch_grid_size
        attention_resized = np.kron(attention_map, np.ones((scale_factor, scale_factor)))
        
        ax.imshow(image_np, alpha=0.7)
        im = ax.imshow(attention_resized, alpha=0.6, cmap='Reds')
        
        word = words[token_idx] if token_idx < len(words) else f"token_{token_idx}"
        ax.set_title(f'Word: "{word}"\nMax Attention: {token_attention.max():.3f}', 
                    fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'image_attention_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Image attention visualization saved to: {save_path}")

# ============================================================================
# ABLATION STUDY VISUALIZATION FUNCTIONS
# ============================================================================

def create_ablation_performance_comparison(results, save_dir, timestamp):
    """Create performance comparison charts for ablation study"""
    # Filter out specific variants to exclude
    excluded_variants = ['no_local_attention', 'single_coattention_layer']
    variants = [v for v in results.keys() if v not in excluded_variants]
    
    # Show Average MRR, Recall@1, and Recall@5
    metrics = ['avg_mrr', 'avg_recall@1', 'avg_recall@5']
    metric_labels = ['Average MRR', 'Recall@1', 'Recall@5']
    
    # Create single row with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(variants)))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        values = []
        labels = []
        
        for variant in variants:
            if metric in results[variant]:
                values.append(results[variant][metric])
                # Create shorter, more readable labels
                short_label = variant.replace('_', ' ').title()
                if short_label == 'Full Model':
                    short_label = 'Full Model'
                elif 'No ' in short_label:
                    short_label = short_label.replace('No ', 'No\n')
                labels.append(short_label)
        
        bars = ax.bar(labels, values, color=colors[:len(values)])
        
        ax.set_title(label, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Score', fontsize=14)
        
        # Improve x-axis label formatting
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add value labels on top of bars with better positioning
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # Position label above the bar with some padding
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Fix x-axis label positioning to align with bars
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, ha='right')
        
        # Set y-axis limits with some padding
        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.2)
        ax.grid(True, alpha=0.3)
        
        # Add subtle borders to bars
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=3.0)
    
    save_path = os.path.join(save_dir, f'ablation_performance_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation performance comparison saved to: {save_path}")

def create_ablation_contribution_analysis(results, save_dir, timestamp):
    """Create contribution analysis for ablation study"""
    # Check for baseline key - could be 'full_model' or 'Baseline'
    baseline_key = None
    if 'full_model' in results:
        baseline_key = 'full_model'
    elif 'Baseline' in results:
        baseline_key = 'Baseline'
    else:
        print("No baseline model results found for contribution analysis")
        return
    
    baseline_mrr = results[baseline_key].get('avg_mrr', 0)
    
    contributions = {}
    for variant, result in results.items():
        if variant != baseline_key and 'avg_mrr' in result:
            contribution = baseline_mrr - result['avg_mrr']
            contributions[variant] = contribution
    
    if not contributions:
        print("No valid ablation variants found for contribution analysis")
        return
    
    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Contribution magnitude plot
    variants, values = zip(*sorted_contributions)
    colors = ['red' if v > 0 else 'blue' for v in values]
    
    # Create shorter, more readable labels for horizontal bar chart
    short_labels = []
    for variant in variants:
        short_label = variant.replace('_', ' ').title()
        if 'No ' in short_label:
            short_label = short_label.replace('No ', 'No\n')
        short_labels.append(short_label)
    
    bars = ax1.barh(short_labels, values, color=colors, alpha=0.7)
    ax1.set_xlabel('MRR Contribution (Baseline - Ablated Model)', fontsize=14)
    ax1.set_title('Component Contribution Analysis', fontsize=16, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=12)
    
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax1.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left' if width >= 0 else 'right', va='center', 
                fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Performance comparison
    all_variants = [baseline_key] + list(variants)
    all_mrrs = [baseline_mrr] + [results[v]['avg_mrr'] for v in variants]
    
    # Create shorter labels for vertical bar chart
    short_all_labels = []
    for variant in all_variants:
        short_label = variant.replace('_', ' ').title()
        if short_label == 'Full Model':
            short_label = 'Full Model'
        elif 'No ' in short_label:
            short_label = short_label.replace('No ', 'No\n')
        short_all_labels.append(short_label)
    
    bars2 = ax2.bar(range(len(all_variants)), all_mrrs, 
                   color=['green'] + ['lightcoral'] * (len(all_variants) - 1))
    
    ax2.set_xlabel('Model Variant', fontsize=14)
    ax2.set_ylabel('Average MRR', fontsize=14)
    ax2.set_title('Performance Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(range(len(all_variants)))
    ax2.set_xticklabels(short_all_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=12)
    
    for bar, value in zip(bars2, all_mrrs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add subtle borders to bars
    for bar in bars2:
        bar.set_edgecolor('black')
        bar.set_linewidth(0.5)
    
    plt.tight_layout(pad=3.0)
    
    save_path = os.path.join(save_dir, f'ablation_contribution_analysis_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation contribution analysis saved to: {save_path}")

def create_failure_rank_distribution_plot(failure_cases, success_cases, save_dir, timestamp):
    """Create rank distribution plot for failure analysis"""
    plt.figure(figsize=(14, 8))
    
    # Extract ranks
    failure_ranks = [case.get('rank', 0) for case in failure_cases]
    success_ranks = [case.get('rank', 0) for case in success_cases]
    
    # Create bins
    max_rank = max(max(failure_ranks) if failure_ranks else 0, 
                   max(success_ranks) if success_ranks else 0)
    bins = np.logspace(0, np.log10(max(max_rank, 10)), 20)
    
    plt.subplot(1, 2, 1)
    plt.hist([failure_ranks, success_ranks], bins=bins, alpha=0.7, 
             label=['Failures', 'Successes'], color=['red', 'green'])
    plt.xscale('log')
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Rank Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    
    if failure_ranks:
        failure_ranks_sorted = np.sort(failure_ranks)
        failure_cumsum = np.arange(1, len(failure_ranks_sorted) + 1) / len(failure_ranks_sorted)
        plt.plot(failure_ranks_sorted, failure_cumsum, 'r-', linewidth=2, label='Failures')
    
    if success_ranks:
        success_ranks_sorted = np.sort(success_ranks)
        success_cumsum = np.arange(1, len(success_ranks_sorted) + 1) / len(success_ranks_sorted)
        plt.plot(success_ranks_sorted, success_cumsum, 'g-', linewidth=2, label='Successes')
    
    plt.xscale('log')
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Rank Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'failure_rank_distribution_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Failure rank distribution saved to: {save_path}")

def create_failure_type_distribution_plot(failure_types, save_dir, timestamp):
    """Create failure type distribution plot"""
    if not failure_types:
        print("No failure types provided for visualization")
        return
    
    type_counts = Counter(failure_types)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    types, counts = zip(*type_counts.most_common())
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    
    bars = ax1.bar(types, counts, color=colors)
    ax1.set_xlabel('Failure Type', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Failure Type Distribution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=types, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Failure Type Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'failure_type_distribution_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Failure type distribution saved to: {save_path}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_training_visualizer(save_dir='visualizations'):
    """Create a training visualizer instance"""
    return TrainingVisualizer(save_dir)

def setup_visualization_backend():
    """Setup matplotlib backend for non-interactive use"""
    plt.switch_backend('Agg')
    print("Visualization backend set to 'Agg' for non-interactive use") 