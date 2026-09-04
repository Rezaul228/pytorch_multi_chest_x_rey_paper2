#!/usr/bin/env python3
"""
Similarity Progression Visualization
Shows how similarity between image and text embeddings changes with each component addition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import gc
from datetime import datetime
import config
import paths
from data_loader_v1 import IndianaDataLoader
from base_models_refactored_v1 import MultimodalFusion

def load_model_weights_only(checkpoint_path):
    """Load only the model weights without full checkpoint"""
    print(f"📁 Loading model weights from: {checkpoint_path}")
    
    # Load the weights directly
    model_state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with the same architecture
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    # Load the weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model

class BaselineModel(nn.Module):
    """Baseline model with only encoders and basic pooling"""
    def __init__(self, original_model):
        super().__init__()
        self.image_encoder = original_model.image_encoder
        self.text_encoder = original_model.text_encoder
        
        # Simple projection layers (no attention)
        self.image_proj = nn.Linear(256, 128)  # embed_dim = 256
        self.text_proj = nn.Linear(256, 128)   # embed_dim = 256
    
    def forward(self, inputs):
        images, texts = inputs
        
        # Get token embeddings
        image_tokens = self.image_encoder(images)
        text_tokens = self.text_encoder(texts)
        
        # Simple mean pooling
        image_emb = torch.mean(image_tokens, dim=1)
        text_emb = torch.mean(text_tokens, dim=1)
        
        # Project to final dimension
        image_emb = self.image_proj(image_emb)
        text_emb = self.text_proj(text_emb)
        
        # L2 normalize
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        return image_emb, text_emb

class LocalAttentionModel(nn.Module):
    """Model with only local attention (no global attention)"""
    def __init__(self, original_model):
        super().__init__()
        self.image_encoder = original_model.image_encoder
        self.text_encoder = original_model.text_encoder
        
        # Copy only local attention components
        self.local_cross_attention1 = original_model.synergy_branch.co_attn_layers[0].cross_attention1
        self.local_cross_attention2 = original_model.synergy_branch.co_attn_layers[0].cross_attention2
        self.local_norm1 = original_model.synergy_branch.co_attn_layers[0].norm1
        self.local_norm2 = original_model.synergy_branch.co_attn_layers[0].norm2
        self.local_norm3 = original_model.synergy_branch.co_attn_layers[0].norm3
        self.local_norm4 = original_model.synergy_branch.co_attn_layers[0].norm4
        self.local_ffn1 = original_model.synergy_branch.co_attn_layers[0].ffn1
        self.local_ffn2 = original_model.synergy_branch.co_attn_layers[0].ffn2
        self.local_image_gate_weights = original_model.synergy_branch.co_attn_layers[0].local_image_gate_weights
        self.local_text_gate_weights = original_model.synergy_branch.co_attn_layers[0].local_text_gate_weights
        
        # Projection layers
        self.image_proj = nn.Linear(256, 128)  # embed_dim = 256
        self.text_proj = nn.Linear(256, 128)   # embed_dim = 256
    
    def forward(self, inputs):
        images, texts = inputs
        
        # Get token embeddings
        image_tokens = self.image_encoder(images)
        text_tokens = self.text_encoder(texts)
        
        # Local cross-attention: Image -> Text
        attended_image, _ = self.local_cross_attention1(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens
        )
        
        local_image_gate = torch.sigmoid(self.local_image_gate_weights).view(1, 1, -1)
        gated_image = local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
        
        image_tokens = self.local_norm1(image_tokens + gated_image)
        image_tokens = self.local_norm2(image_tokens + self.local_ffn1(image_tokens))
        
        # Local cross-attention: Text -> Image
        attended_text, _ = self.local_cross_attention2(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens
        )
        
        local_text_gate = torch.sigmoid(self.local_text_gate_weights).view(1, 1, -1)
        gated_text = local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
        
        text_tokens = self.local_norm3(text_tokens + gated_text)
        text_tokens = self.local_norm4(text_tokens + self.local_ffn2(text_tokens))
        
        # Simple mean pooling (no global attention)
        image_emb = torch.mean(image_tokens, dim=1)
        text_emb = torch.mean(text_tokens, dim=1)
        
        # Project to final dimension
        image_emb = self.image_proj(image_emb)
        text_emb = self.text_proj(text_emb)
        
        # L2 normalize
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        return image_emb, text_emb

class LocalPlusGlobalModel(nn.Module):
    """Model with local attention + global attention (no feedback)"""
    def __init__(self, original_model):
        super().__init__()
        self.image_encoder = original_model.image_encoder
        self.text_encoder = original_model.text_encoder
        
        # Copy local attention components
        self.local_cross_attention1 = original_model.synergy_branch.co_attn_layers[0].cross_attention1
        self.local_cross_attention2 = original_model.synergy_branch.co_attn_layers[0].cross_attention2
        self.local_norm1 = original_model.synergy_branch.co_attn_layers[0].norm1
        self.local_norm2 = original_model.synergy_branch.co_attn_layers[0].norm2
        self.local_norm3 = original_model.synergy_branch.co_attn_layers[0].norm3
        self.local_norm4 = original_model.synergy_branch.co_attn_layers[0].norm4
        self.local_ffn1 = original_model.synergy_branch.co_attn_layers[0].ffn1
        self.local_ffn2 = original_model.synergy_branch.co_attn_layers[0].ffn2
        self.local_image_gate_weights = original_model.synergy_branch.co_attn_layers[0].local_image_gate_weights
        self.local_text_gate_weights = original_model.synergy_branch.co_attn_layers[0].local_text_gate_weights
        
        # Copy global attention components
        self.global_cross_attention1 = original_model.synergy_branch.co_attn_layers[0].global_cross_attention1
        self.global_cross_attention2 = original_model.synergy_branch.co_attn_layers[0].global_cross_attention2
        self.global_norm1 = original_model.synergy_branch.co_attn_layers[0].global_norm1
        self.global_norm2 = original_model.synergy_branch.co_attn_layers[0].global_norm2
        self.global_norm3 = original_model.synergy_branch.co_attn_layers[0].global_norm3
        self.global_norm4 = original_model.synergy_branch.co_attn_layers[0].global_norm4
        self.global_ffn1 = original_model.synergy_branch.co_attn_layers[0].global_ffn1
        self.global_ffn2 = original_model.synergy_branch.co_attn_layers[0].global_ffn2
        self.global_image_gate_weights = original_model.synergy_branch.co_attn_layers[0].global_image_gate_weights
        self.global_text_gate_weights = original_model.synergy_branch.co_attn_layers[0].global_text_gate_weights
        
        # Projection layers
        self.image_proj = nn.Linear(256, 128)  # embed_dim = 256
        self.text_proj = nn.Linear(256, 128)   # embed_dim = 256
    
    def forward(self, inputs):
        images, texts = inputs
        
        # Get token embeddings
        image_tokens = self.image_encoder(images)
        text_tokens = self.text_encoder(texts)
        
        # Local attention processing
        attended_image, _ = self.local_cross_attention1(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens
        )
        
        local_image_gate = torch.sigmoid(self.local_image_gate_weights).view(1, 1, -1)
        gated_image = local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
        
        image_tokens = self.local_norm1(image_tokens + gated_image)
        image_tokens = self.local_norm2(image_tokens + self.local_ffn1(image_tokens))
        
        attended_text, _ = self.local_cross_attention2(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens
        )
        
        local_text_gate = torch.sigmoid(self.local_text_gate_weights).view(1, 1, -1)
        gated_text = local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
        
        text_tokens = self.local_norm3(text_tokens + gated_text)
        text_tokens = self.local_norm4(text_tokens + self.local_ffn2(text_tokens))
        
        # Global attention processing (separate from local)
        global_image_token = torch.mean(image_tokens, dim=1, keepdim=True)
        global_text_token = torch.mean(text_tokens, dim=1, keepdim=True)
        
        attended_global_image, _ = self.global_cross_attention1(
            query=global_image_token,
            key=text_tokens,
            value=text_tokens
        )
        
        global_image_gate = torch.sigmoid(self.global_image_gate_weights).view(1, 1, -1)
        gated_global_image = global_image_gate * attended_global_image + (1 - global_image_gate) * global_image_token
        
        global_image_token = self.global_norm1(global_image_token + gated_global_image)
        global_image_token = self.global_norm2(global_image_token + self.global_ffn1(global_image_token))
        
        attended_global_text, _ = self.global_cross_attention2(
            query=global_text_token,
            key=image_tokens,
            value=image_tokens
        )
        
        global_text_gate = torch.sigmoid(self.global_text_gate_weights).view(1, 1, -1)
        gated_global_text = global_text_gate * attended_global_text + (1 - global_text_gate) * global_text_token
        
        global_text_token = self.global_norm3(global_text_token + gated_global_text)
        global_text_token = self.global_norm4(global_text_token + self.global_ffn2(global_text_token))
        
        # Use global tokens for final embedding (no feedback to local)
        image_emb = global_image_token.squeeze(1)
        text_emb = global_text_token.squeeze(1)
        
        # Project to final dimension
        image_emb = self.image_proj(image_emb)
        text_emb = self.text_proj(text_emb)
        
        # L2 normalize
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        return image_emb, text_emb

class LocalGlobalFeedbackModel(nn.Module):
    """Model with local attention + global attention + feedback (no dual branch)"""
    def __init__(self, original_model):
        super().__init__()
        self.image_encoder = original_model.image_encoder
        self.text_encoder = original_model.text_encoder
        
        # Copy all attention components from synergy branch
        self.synergy_branch = original_model.synergy_branch
        
        # Projection layers
        self.image_proj = nn.Linear(256, 128)  # embed_dim = 256
        self.text_proj = nn.Linear(256, 128)   # embed_dim = 256
    
    def forward(self, inputs):
        images, texts = inputs
        
        # Get token embeddings
        image_tokens = self.image_encoder(images)
        text_tokens = self.text_encoder(texts)
        
        # Process through synergy branch (includes local + global + feedback)
        synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
        
        # Project to final dimension
        image_emb = self.image_proj(synergy_img_emb)
        text_emb = self.text_proj(synergy_txt_emb)
        
        # L2 normalize
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        return image_emb, text_emb

def create_similarity_progression_visualization(model, data_loader, tokenizer, save_dir='similarity_progression'):
    """Create visualization showing similarity progression with each component"""
    print("🔬 Creating Similarity Progression Visualization...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get one sample
    val_dataset = data_loader.get_validation_data(num_samples=1)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    device = next(model.parameters()).device
    
    # Process one sample
    batch = next(iter(val_loader))
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
    
    study_id = batch_study_ids[0] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[0])
    
    # Decode text
    text_tokens = batch_texts[0]
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
    
    text_description = ' '.join(words[:25])
    
    # Create different model variants
    model_variants = {
        'Baseline': BaselineModel(model),
        'Local Attention': LocalAttentionModel(model),
        'Local + Global': LocalPlusGlobalModel(model),
        'Local + Global + Feedback': LocalGlobalFeedbackModel(model),
        'Full Model': model
    }
    
    # Calculate similarities for each variant
    similarities = {}
    contributions = {}
    
    print(f"\n📊 Similarity Progression Analysis for Study ID: {study_id}")
    print(f"Text: {text_description}")
    print("=" * 80)
    
    baseline_similarity = None
    
    for variant_name, variant_model in model_variants.items():
        variant_model.eval()
        
        with torch.no_grad():
            # Get embeddings
            image_emb, text_emb = variant_model((batch_images, batch_texts))
            
            # Calculate similarity
            similarity = torch.cosine_similarity(image_emb, text_emb, dim=1).item()
            similarities[variant_name] = similarity
            
            # Calculate contribution
            if baseline_similarity is None:
                baseline_similarity = similarity
                contribution = 0.0
            else:
                contribution = similarity - baseline_similarity
            
            contributions[variant_name] = contribution
            
            print(f"{variant_name:25s}: {similarity:.4f} (Contribution: {contribution:+.4f})")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Image-Text Similarity Progression Analysis\nStudy ID: {study_id}', 
                fontsize=18, fontweight='bold')
    
    # Plot 1: Similarity scores progression
    ax1 = plt.subplot(2, 2, 1)
    variant_names = list(similarities.keys())
    sim_values = list(similarities.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#2E8B57']
    
    bars1 = ax1.bar(variant_names, sim_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_title('Image-Text Similarity Scores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.3, 1.0)
    
    # Add value labels on bars
    for bar, val in zip(bars1, sim_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.05), 
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 2: Component contributions
    ax2 = plt.subplot(2, 2, 2)
    contrib_values = list(contributions.values())
    contrib_colors = ['#FF6B6B'] + ['#4ECDC4', '#45B7D1', '#96CEB4', '#2E8B57'][1:]
    
    bars2 = ax2.bar(variant_names, contrib_values, color=contrib_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_title('Component Contributions to Similarity', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Similarity Improvement', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars2, contrib_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.05), 
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Similarity progression line chart
    ax3 = plt.subplot(2, 2, 3)
    x_positions = range(len(variant_names))
    ax3.plot(x_positions, sim_values, marker='o', linewidth=3, markersize=8, color='#2E8B57')
    ax3.fill_between(x_positions, sim_values, alpha=0.3, color='#2E8B57')
    ax3.set_title('Similarity Progression', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Model Components', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(variant_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on points
    for i, val in enumerate(sim_values):
        ax3.annotate(f'{val:.3f}', (i, val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
    
    # Plot 4: Component impact analysis
    ax4 = plt.subplot(2, 2, 4)
    
    # Calculate individual component impacts
    component_impacts = {
        'Local Attention': contributions['Local Attention'],
        'Global Attention': contributions['Local + Global'] - contributions['Local Attention'],
        'Feedback': contributions['Local + Global + Feedback'] - contributions['Local + Global'],
        'Dual Branch': contributions['Full Model'] - contributions['Local + Global + Feedback']
    }
    
    impact_names = list(component_impacts.keys())
    impact_values = list(component_impacts.values())
    impact_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#2E8B57']
    
    bars4 = ax4.bar(impact_names, impact_values, color=impact_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_title('Individual Component Impact', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Similarity Impact', fontsize=12, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars4, impact_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.05), 
                f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarity_progression_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed results table
    results_text = f"""
Image-Text Similarity Progression Analysis Results
===============================================
Study ID: {study_id}
Text: {text_description}

Similarity Scores by Component:
"""
    
    for variant_name, similarity in similarities.items():
        contribution = contributions[variant_name]
        results_text += f"{variant_name:25s}: {similarity:.4f} (Contribution: {contribution:+.4f})\n"
    
    results_text += f"""
Individual Component Impacts:
- Local Attention: {component_impacts['Local Attention']:+.4f}
- Global Attention: {component_impacts['Global Attention']:+.4f}
- Feedback Mechanism: {component_impacts['Feedback']:+.4f}
- Dual Branch Architecture: {component_impacts['Dual Branch']:+.4f}

Key Insights:
- Baseline similarity: {baseline_similarity:.4f}
- Final similarity: {similarities['Full Model']:.4f}
- Total improvement: {contributions['Full Model']:+.4f}
- Dual branch provides {component_impacts['Dual Branch']:+.4f} improvement alone
"""
    
    # Save results to file
    with open(os.path.join(save_dir, 'similarity_progression_results.txt'), 'w') as f:
        f.write(results_text)
    
    print(f"\n✅ Similarity progression analysis saved to: {save_dir}/")
    print(f"📁 Files generated:")
    print(f"   - similarity_progression_analysis.png: Comprehensive visualization")
    print(f"   - similarity_progression_results.txt: Detailed results")
    
    return similarities, contributions, component_impacts

def main():
    """Main function to create similarity progression visualization"""
    print("🔬 IMAGE-TEXT SIMILARITY PROGRESSION VISUALIZATION")
    print("=" * 60)
    
    # Configuration
    config.print_current_config()
    
    # Load model weights only
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model_weights.pth'
    print(f"📁 Loading model from: {model_path}")
    model = load_model_weights_only(model_path)
    print(f"✅ Model loaded successfully")
    
    # Load validation data and tokenizer
    print("\n📁 Loading validation data and tokenizer...")
    data_loader = IndianaDataLoader(
        batch_size=1,
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
    save_dir = 'similarity_progression'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create analysis
    try:
        similarities, contributions, component_impacts = create_similarity_progression_visualization(model, data_loader, tokenizer, save_dir)
        
        print(f"\n✅ Similarity progression analysis completed!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()


















