#!/usr/bin/env python3
"""
Medical Term Attention Minimal Visualization
Clean visualization with no text overlays and more attention patches
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

def extract_attention_weights(model, images, texts, layer_idx=0):
    """Extract attention weights from the model"""
    model.eval()
    
    with torch.no_grad():
        # Get token embeddings
        image_tokens = model.image_encoder(images)
        text_tokens = model.text_encoder(texts)
        
        # Process through synergy branch
        synergy_branch = model.synergy_branch
        
        # Get attention from the first co-attention layer
        if layer_idx < len(synergy_branch.co_attn_layers):
            co_attn_layer = synergy_branch.co_attn_layers[layer_idx]
            
            # Extract attention weights
            _, img_to_txt_attn = co_attn_layer.cross_attention1(
                query=image_tokens,
                key=text_tokens,
                value=text_tokens,
                need_weights=True
            )
            
            _, txt_to_img_attn = co_attn_layer.cross_attention2(
                query=text_tokens,
                key=image_tokens,
                value=image_tokens,
                need_weights=True
            )
            
            return img_to_txt_attn, txt_to_img_attn, image_tokens, text_tokens
    
    return None, None, image_tokens, text_tokens

def find_reports_with_medical_terms(data_loader, tokenizer, target_terms, num_samples_per_term=4):
    """Find reports containing specific medical terms"""
    print(f"🔍 Searching for reports containing: {target_terms}")
    
    # Get validation data
    val_dataset = data_loader.get_validation_data(num_samples=3000)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    found_reports = {term: [] for term in target_terms}
    seen_study_ids = set()
    
    for batch in val_loader:
        if isinstance(batch, dict):
            batch_texts = batch['captions']
            batch_study_ids = batch['study_ids']
        else:
            _, batch_texts, batch_study_ids = batch
        
        study_id = batch_study_ids[0] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[0])
        
        # Skip if we've already seen this study
        if study_id in seen_study_ids:
            continue
        
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
        
        text_description = ' '.join(words).lower()
        
        # Check for target terms
        for term in target_terms:
            if (term.lower() in text_description and 
                len(found_reports[term]) < num_samples_per_term and
                study_id not in seen_study_ids):
                
                found_reports[term].append({
                    'study_id': study_id,
                    'text': text_description,
                    'words': words,
                    'batch': batch
                })
                seen_study_ids.add(study_id)
                print(f"  ✅ Found '{term}' in study: {study_id}")
                break
    
    return found_reports

def create_minimal_attention_visualization(model, found_reports, tokenizer, save_dir='medical_term_attention_minimal'):
    """Create minimal visualization with no text and more attention patches"""
    print("🔬 Creating Minimal Medical Term Attention Visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    
    for term, reports in found_reports.items():
        if not reports:
            print(f"  ⚠️ No reports found for '{term}'")
            continue
            
        print(f"  📊 Processing '{term}' - {len(reports)} reports found")
        
        # Create 4x3 grid visualization for this term (12 samples)
        fig = plt.figure(figsize=(30, 16))
        fig.suptitle(f'Medical Term: "{term.upper()}" - Ultra-Intensive Analysis (12 Samples)', fontsize=20, fontweight='bold')
        
        # Process up to 12 reports
        num_reports = min(len(reports), 12)
        
        for i in range(num_reports):
            report = reports[i]
            batch = report['batch']
            study_id = report['study_id']
            words = report['words']
            
            # Process batch
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
            
            # Get original image
            image_np = batch_images[0].cpu().numpy()
            if len(image_np.shape) == 3 and image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            
            # Normalize image
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            
            # Extract attention weights
            img_to_txt_attn, txt_to_img_attn, image_tokens, text_tokens = extract_attention_weights(
                model, batch_images, batch_texts, layer_idx=0
            )
            
            # Create subplot (4x3 grid for 12 samples)
            ax = plt.subplot(3, 4, i+1)
            
            if img_to_txt_attn is not None and txt_to_img_attn is not None:
                # Find the target word in the text
                target_word_idx = None
                for j, word in enumerate(words):
                    if term.lower() in word.lower():
                        target_word_idx = j
                        break
                
                if target_word_idx is not None:
                    # Get attention for the target word
                    word_attention = txt_to_img_attn[0, target_word_idx, :]
                    
                    # Reshape to spatial dimensions (14x14)
                    attention_2d = word_attention.view(14, 14)
                    
                    # Resize attention to image size
                    attention_resized = torch.nn.functional.interpolate(
                        attention_2d.unsqueeze(0).unsqueeze(0), 
                        size=(image_np.shape[0], image_np.shape[1]), 
                        mode='bilinear', align_corners=False
                    ).squeeze()
                    
                    # Show X-ray image clearly
                    ax.imshow(image_np, cmap='gray', alpha=1.0)
                    
                    # Create mask for high attention regions (top 10% instead of 5%)
                    attention_np = attention_resized.cpu().numpy()
                    
                    # Find high attention threshold (top 10% of attention values)
                    attention_flat = attention_np.flatten()
                    threshold = np.percentile(attention_flat, 90)  # Top 10%
                    
                    # Create binary mask for high attention regions
                    high_attention_mask = attention_np >= threshold
                    
                    # Create deeper blue overlay only for high attention regions
                    overlay = np.zeros((*attention_np.shape, 4))  # RGBA
                    overlay[high_attention_mask, 0] = 0.1  # Low red channel
                    overlay[high_attention_mask, 1] = 0.2  # Low green channel  
                    overlay[high_attention_mask, 2] = 0.9  # Deeper blue channel
                    overlay[high_attention_mask, 3] = 0.4  # More transparent (40% opacity)
                    
                    # Show only high attention patches
                    ax.imshow(overlay)
                    
                    # Find top 10 attention patches for highlighting (instead of 5)
                    attention_flat = attention_resized.flatten()
                    top_10_indices = torch.topk(attention_flat, k=10).indices
                    
                    # Highlight top 10 attention patches with cyan rectangles
                    for idx in top_10_indices:
                        y = idx // attention_resized.shape[1]
                        x = idx % attention_resized.shape[1]
                        # Draw rectangle around high attention patch
                        rect = plt.Rectangle((x-20, y-20), 40, 40, linewidth=2, 
                                           edgecolor='cyan', facecolor='none')
                        ax.add_patch(rect)
                    
                    # Calculate attention statistics
                    max_attn = attention_resized.max().item()
                    mean_attn = attention_resized.mean().item()
                    high_attn_count = np.sum(high_attention_mask)
                    total_pixels = attention_np.size
                    high_attn_percentage = (high_attn_count / total_pixels) * 100
                    
                    # Simple title with just study ID
                    ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
                    
                    print(f"    📍 Study {study_id}: Word '{term}' - Max: {max_attn:.4f}, High attention: {high_attn_percentage:.1f}%")
                else:
                    # Word not found in text
                    ax.imshow(image_np, cmap='gray', alpha=1.0)
                    ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
            else:
                # No attention available
                ax.imshow(image_np, cmap='gray', alpha=1.0)
                ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
            
            ax.axis('off')
        
        # No additional text or explanations
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'minimal_medical_term_{term}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved minimal visualization for '{term}'")

def main():
    """Main function to create minimal medical term attention visualization"""
    print("🔬 MINIMAL MEDICAL TERM ATTENTION VISUALIZATION")
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
    
    # Define target medical terms
    #target_terms = ['cardiac', 'opacity', 'pleural', 'costophrenic', 'bilateral', 
    #               'effusion', 'consolidation', 'infiltrate', 'pneumothorax', 'atelectasis']

    target_terms = ['cardiac', 'opacity', 'pleural', 'costophrenic', 'bilateral', 
                   'effusion', 'consolidation', 'infiltrate', 'pneumothorax', 'atelectasis']
    
    # Find reports containing these terms
    found_reports = find_reports_with_medical_terms(data_loader, tokenizer, target_terms, num_samples_per_term=12)
    
    # Create save directory
    save_dir = 'medical_term_attention_minimal'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visualizations
    try:
        create_minimal_attention_visualization(model, found_reports, tokenizer, save_dir)
        
        print(f"\n✅ Minimal medical term attention visualization completed!")
        print(f"📁 Results saved to: {save_dir}/")
        print(f"\n📋 Generated Files:")
        for term in target_terms:
            if found_reports[term]:
                print(f"   - minimal_medical_term_{term}.png: Minimal attention visualization for '{term}'")
        
        print(f"\n🎯 KEY FEATURES:")
        print(f"   - No text overlays on images")
        print(f"   - X-ray images clearly visible")
        print(f"   - Only high attention patches shown in deeper blue (40% opacity)")
        print(f"   - Top 10 attention patches highlighted with cyan rectangles")
        print(f"   - Ultra-intensive analysis: 12 samples per medical term")
        print(f"   - 4x3 grid layout for comprehensive verification")
        print(f"   - Clean, minimal design")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()

