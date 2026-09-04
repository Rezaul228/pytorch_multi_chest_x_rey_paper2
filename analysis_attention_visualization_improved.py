#!/usr/bin/env python3
"""
Improved Attention Visualization with Full Model Loading
Shows distinct attention patterns for each medical word with individual normalization.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import re
from datetime import datetime
import config
import paths
from data_loader_v1 import IndianaDataLoader, SimpleTokenizer
from base_models_refactored_v1 import MultimodalFusion

def load_full_model_checkpoint(checkpoint_path):
    """Load the full model checkpoint with architecture and weights"""
    print(f"📁 Loading full model checkpoint from: {checkpoint_path}")
    
    # Load the full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Checkpoint loaded successfully")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   Experiment: {checkpoint.get('experiment_name', 'Unknown')}")
    
    # Extract model state dict
    model_state_dict = checkpoint['model_state_dict']
    
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
    
    print(f"✅ Model architecture reconstructed and weights loaded")
    return model

def extract_attention_weights_from_full_model(model, images, texts, layer_idx=0):
    """Extract attention weights using the full model's internal components"""
    model.eval()
    with torch.no_grad():
        # Convert data to PyTorch tensors if needed
        if isinstance(images, np.ndarray):
            images = torch.FloatTensor(images)
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        
        if isinstance(texts, np.ndarray):
            texts = torch.LongTensor(texts)
        
        # Move to device
        device = next(model.parameters()).device
        images = images.to(device)
        texts = texts.to(device)
        
        # Access the model's internal components
        print(f"[DEBUG] Extracting attention from model components...")
        
        # Get image and text tokens from encoders
        image_tokens = model.image_encoder(images)
        text_tokens = model.text_encoder(texts)
        
        print(f"[DEBUG] Image tokens shape: {image_tokens.shape}")
        print(f"[DEBUG] Text tokens shape: {text_tokens.shape}")
        
        # Access the synergy branch's co-attention layers
        synergy_layer = model.synergy_branch.co_attn_layers[layer_idx]
        
        # Extract attention weights from the cross-attention layers
        # The model uses batch_first=True, so we don't need to transpose
        
        # Get the attention outputs and weights for I2T
        i2t_output, i2t_weights = synergy_layer.cross_attention1(
            query=image_tokens,  # (batch, seq_len, embed_dim)
            key=text_tokens,     # (batch, seq_len, embed_dim)
            value=text_tokens,   # (batch, seq_len, embed_dim)
            need_weights=True    # Return attention weights
        )
        
        # Get the attention outputs and weights for T2I
        t2i_output, t2i_weights = synergy_layer.cross_attention2(
            query=text_tokens,   # (batch, seq_len, embed_dim)
            key=image_tokens,    # (batch, seq_len, embed_dim)
            value=image_tokens,  # (batch, seq_len, embed_dim)
            need_weights=True    # Return attention weights
        )
        
        print(f"[DEBUG] I2T attention weights shape: {i2t_weights.shape}")
        print(f"[DEBUG] T2I attention weights shape: {t2i_weights.shape}")
        print(f"[DEBUG] I2T attention weights type: {type(i2t_weights)}")
        print(f"[DEBUG] T2I attention weights type: {type(t2i_weights)}")
        
        # Check if attention weights are averaged across heads already
        print(f"[DEBUG] I2T attention weights min/max: {i2t_weights.min():.4f} / {i2t_weights.max():.4f}")
        print(f"[DEBUG] T2I attention weights min/max: {t2i_weights.min():.4f} / {t2i_weights.max():.4f}")
        
        return {
            'image_tokens': image_tokens.cpu().numpy(),
            'text_tokens': text_tokens.cpu().numpy(),
            'i2t_weights': i2t_weights.cpu().numpy(),
            't2i_weights': t2i_weights.cpu().numpy(),
            'i2t_output': i2t_output.cpu().numpy(),
            't2i_output': t2i_output.cpu().numpy(),
        }

def visualize_attention_with_improved_contrast(model, image, text_tokens, tokenizer, sample_idx=0, save_dir='attention_analysis_improved'):
    """Visualize attention with improved contrast and individual normalization"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure inputs are in the right format
    if len(image.shape) == 3:
        image = image[None]  # Add batch dimension
    if len(text_tokens.shape) == 1:
        text_tokens = text_tokens[None]  # Add batch dimension
    
    # Extract attention weights using the full model
    attention_weights = extract_attention_weights_from_full_model(model, image, text_tokens)
    
    # Get text-to-image attention weights
    t2i_weights = attention_weights['t2i_weights'][sample_idx]  # Shape: (text_len, image_len)
    print(f"[DEBUG] T2I attention weights shape: {t2i_weights.shape}")
    
    # The attention weights should already be (text_len, image_len) - no need to average
    # If they have an extra dimension (like num_heads), we need to handle it properly
    if t2i_weights.ndim == 3:  # Shape: (num_heads, text_len, image_len) - average over heads
        t2i_attn = t2i_weights.mean(axis=0)  # Shape: (text_len, image_len)
        print(f"[DEBUG] Averaged over heads - T2I attention shape: {t2i_attn.shape}")
    elif t2i_weights.ndim == 2:  # Shape: (text_len, image_len) - already correct
        t2i_attn = t2i_weights
        print(f"[DEBUG] No averaging needed - T2I attention shape: {t2i_attn.shape}")
    else:
        print(f"[DEBUG] Unexpected T2I weights shape: {t2i_weights.shape}")
        t2i_attn = t2i_weights
        print(f"[DEBUG] Using as-is - T2I attention shape: {t2i_attn.shape}")
    
    # Get text tokens for visualization
    if tokenizer:
        word_ids = text_tokens[sample_idx][text_tokens[sample_idx] > 0]
        words = []
        for token_id in word_ids:
            token_id_int = int(token_id)  # Ensure it's an integer
            word = None
            if hasattr(tokenizer, 'idx2word'):
                word = tokenizer.idx2word.get(token_id_int, None)
                if word is None:
                    word = tokenizer.idx2word.get(str(token_id_int), None)
            elif hasattr(tokenizer, 'index_word'):
                word = tokenizer.index_word.get(token_id_int, None)
                if word is None:
                    tokenizer.index_word.get(str(token_id_int), None)
            elif hasattr(tokenizer, 'word_index'):
                # Create reverse mapping from word_index
                if not hasattr(tokenizer, '_idx2word_cache'):
                    tokenizer._idx2word_cache = {idx: word for word, idx in tokenizer.word_index.items()}
                word = tokenizer._idx2word_cache.get(token_id_int, None)
            
            if word is None:
                word = f"<{token_id_int}>"
            words.append(word)
    else:
        words = [f"token_{i}" for i in range(len(text_tokens[sample_idx]))]
    
    # Handle tensor conversion
    if isinstance(text_tokens, torch.Tensor):
        text_tokens_np = text_tokens.cpu().numpy()
    else:
        text_tokens_np = text_tokens
    
    valid_tokens = len([t for t in text_tokens_np[sample_idx] if t > 0])
    print(f"[DEBUG] Valid tokens: {valid_tokens}")
    print(f"[DEBUG] Sample words: {words[:10]}")  # Show first 10 words
    
    # Filter for medical words only and get their indices
    medical_word_indices = []
    medical_words_found = []
    for i, word in enumerate(words[:valid_tokens]):
        if is_medical_word(word):
            medical_word_indices.append(i)
            medical_words_found.append(word)
    
    print(f"[DEBUG] Medical words found: {medical_words_found}")
    print(f"[DEBUG] Medical word indices: {medical_word_indices}")
    
    if len(medical_word_indices) == 0:
        print("❌ No medical words found in this sample")
        return
    
    # Get top medical words by attention score
    if len(medical_word_indices) > 0 and t2i_attn.shape[0] > 0:
        if t2i_attn.ndim == 2:
            # Get attention scores for medical words only
            medical_attention_scores = []
            for idx in medical_word_indices:
                if idx < t2i_attn.shape[0]:
                    max_attention = t2i_attn[idx].max()
                    medical_attention_scores.append((idx, max_attention, words[idx]))
            
            # Sort by attention score and take top 6
            medical_attention_scores.sort(key=lambda x: x[1], reverse=True)
            top_medical_words = medical_attention_scores[:6]
            
            if len(top_medical_words) == 0:
                print("❌ No medical words with valid attention scores")
                return
                
            top_words_idx = [item[0] for item in top_medical_words]
            
        elif t2i_attn.ndim == 1:
            print("⚠️ t2i_attn is 1D, cannot select top words by attention. Skipping per-word visualization.")
            return
        else:
            print(f"❌ Unexpected t2i_attn shape: {t2i_attn.shape}")
            return
    else:
        print("❌ No valid medical words or attention weights available")
        return
    
    # Convert image for visualization
    if isinstance(image, torch.Tensor):
        image_np = image[sample_idx].cpu().numpy()
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)  # Convert from CHW to HWC
    else:
        image_np = image[sample_idx]
    
    # Create visualization with improved contrast
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Medical Attention Visualization - Individual Normalization', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Different colormaps for each word to make them visually distinct
    colormaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys']
    
    for i, token_idx in enumerate(top_words_idx):
        if i >= 6:
            break
            
        ax = axes[i]
        
        # Get attention scores for this token
        token_attention = t2i_attn[token_idx]
        
        # Reshape attention scores to image grid
        num_patches = len(token_attention)
        patch_grid_size = int(np.sqrt(num_patches))
        
        if patch_grid_size * patch_grid_size == num_patches:
            attention_map = token_attention.reshape(patch_grid_size, patch_grid_size)
        else:
            attention_map = token_attention[:patch_grid_size * patch_grid_size].reshape(patch_grid_size, patch_grid_size)
        
        # CRITICAL: Normalize each attention map individually to show relative patterns
        attention_min = attention_map.min()
        attention_max = attention_map.max()
        attention_range = attention_max - attention_min
        
        if attention_range > 0:
            # Normalize to 0-1 range
            attention_normalized = (attention_map - attention_min) / attention_range
            # Apply contrast enhancement (square root to enhance low values)
            attention_enhanced = np.sqrt(attention_normalized)
        else:
            attention_enhanced = attention_map
        
        # Resize attention map to image size
        scale_factor = 224 // patch_grid_size
        attention_resized = np.kron(attention_enhanced, np.ones((scale_factor, scale_factor)))
        
        # Create visualization
        ax.imshow(image_np, alpha=0.8)
        im = ax.imshow(attention_resized, alpha=0.4, cmap=colormaps[i % len(colormaps)])
        
        word = words[token_idx] if token_idx < len(words) else f"token_{token_idx}"
        ax.set_title(f'Medical Word: "{word}"\nRange: {attention_min:.5f} - {attention_max:.5f}', 
                    fontweight='bold', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
        
        print(f"[DEBUG] Word '{word}': min={attention_min:.5f}, max={attention_max:.5f}, range={attention_range:.5f}")
    
    # Hide unused subplots
    for i in range(len(top_words_idx), 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'improved_medical_attention_{timestamp}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Improved medical attention visualization saved to: {save_path}")
    print(f"📊 Medical Attention Summary:")
    print(f"   Total words: {len(words)}")
    print(f"   Valid tokens: {valid_tokens}")
    print(f"   Medical words found: {len(medical_words_found)}")
    print(f"   Medical words: {medical_words_found}")
    print(f"   Top medical attention words:")
    for i, token_idx in enumerate(top_words_idx[:3]):
        word = words[token_idx] if token_idx < len(words) else f"token_{token_idx}"
        max_attn = t2i_attn[token_idx].max()
        min_attn = t2i_attn[token_idx].min()
        print(f"     '{word}': {min_attn:.5f} - {max_attn:.5f} (range: {max_attn-min_attn:.5f})")

def is_medical_word(word):
    """Determine if a word is medical based on common patterns and characteristics."""
    if not word or len(word) < 2:
        return False
    
    word_lower = word.lower()
    
    # Only filter out the most common non-medical words
    non_medical_words = {
        # Articles and basic connectors
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
        # Basic pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'this', 'that', 'these', 'those',
        # Common time/location words
        'here', 'there', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom',
        'today', 'yesterday', 'tomorrow', 'now', 'then', 'before', 'after', 'during', 'while',
        # Common punctuation and special tokens
        'unk', '<unk>', 'pad', '<pad>', 'start', '<start>', 'end', '<end>'
    }
    
    # Filter out basic non-medical words
    if word_lower in non_medical_words:
        return False
    
    # Medical word patterns (keep these) - more inclusive
    medical_patterns = [
        # Anatomical terms
        r'.*lung.*', r'.*heart.*', r'.*chest.*', r'.*rib.*', r'.*spine.*', r'.*vertebra.*',
        r'.*aorta.*', r'.*artery.*', r'.*vein.*', r'.*bronchus.*', r'.*trachea.*', r'.*diaphragm.*',
        r'.*pleura.*', r'.*mediastinum.*', r'.*hilum.*', r'.*fissure.*', r'.*stomach.*', r'.*abdomen.*',
        r'.*pelvis.*', r'.*thorax.*', r'.*clavicle.*',
        # Pathological terms
        r'.*pneumonia.*', r'.*edema.*', r'.*effusion.*', r'.*atelectasis.*', r'.*consolidation.*',
        r'.*nodule.*', r'.*mass.*', r'.*lesion.*', r'.*opacity.*', r'.*infiltrate.*', r'.*fibrosis.*',
        r'.*emphysema.*', r'.*bronchiectasis.*', r'.*pneumothorax.*', r'.*hemothorax.*',
        r'.*pleural.*', r'.*pulmonary.*', r'.*cardiac.*', r'.*vascular.*',
        # Medical descriptors and positions
        r'.*bilateral.*', r'.*unilateral.*', r'.*right.*', r'.*left.*', r'.*upper.*', r'.*lower.*',
        r'.*anterior.*', r'.*posterior.*', r'.*lateral.*', r'.*medial.*', r'.*apical.*', r'.*basal.*',
        r'.*central.*', r'.*peripheral.*', r'.*diffuse.*', r'.*focal.*', r'.*multifocal.*',
        r'.*bibasilar.*', r'.*basilar.*', r'.*hilar.*', r'.*perihilar.*', r'.*subpleural.*',
        # Medical conditions and findings
        r'.*normal.*', r'.*abnormal.*', r'.*mild.*', r'.*moderate.*', r'.*severe.*', r'.*prominent.*',
        r'.*enlarged.*', r'.*increased.*', r'.*decreased.*', r'.*absent.*', r'.*present.*',
        r'.*clear.*', r'.*obscured.*', r'.*visible.*', r'.*unchanged.*', r'.*stable.*',
        r'.*improved.*', r'.*worsened.*',
        # Medical procedures and equipment
        r'.*tube.*', r'.*picc.*', r'.*catheter.*', r'.*line.*', r'.*wire.*', r'.*stent.*',
        r'.*radiograph.*', r'.*xray.*', r'.*x-ray.*', r'.*ct.*', r'.*scan.*', r'.*exam.*',
        r'.*portable.*', r'.*erect.*', r'.*supine.*', r'.*lateral.*', r'.*oblique.*',
        # Medical findings and structures
        r'.*calcification.*', r'.*calcified.*', r'.*density.*', r'.*shadow.*', r'.*air.*',
        r'.*fluid.*', r'.*blood.*', r'.*tissue.*', r'.*bone.*', r'.*soft.*', r'.*structures.*',
        r'.*contours.*', r'.*margins.*', r'.*borders.*',
        # Medical measurements and sizes
        r'.*size.*', r'.*volume.*', r'.*small.*', r'.*large.*', r'.*tiny.*', r'.*huge.*',
        r'.*cm.*', r'.*mm.*', r'.*inch.*', r'.*diameter.*', r'.*width.*', r'.*length.*',
        # Medical abbreviations and terms
        r'.*cxr.*', r'.*pa.*', r'.*ap.*', r'.*lat.*', r'.*obl.*', r'.*decub.*',
        r'.*copd.*', r'.*chf.*', r'.*pe.*', r'.*dvt.*', r'.*tb.*', r'.*covid.*',
        r'.*svc.*', r'.*ivc.*', r'.*ng.*', r'.*og.*', r'.*ett.*', r'.*trach.*',
        # Medical positions and directions
        r'.*position.*', r'.*positioning.*', r'.*placement.*', r'.*location.*',
        r'.*coiled.*', r'.*layering.*', r'.*crowding.*', r'.*congestion.*',
        # Results and findings
        r'.*results.*', r'.*findings.*', r'.*impression.*', r'.*conclusion.*',
        r'.*obtained.*', r'.*since.*', r'.*preceding.*', r'.*day.*', r'.*exam.*'
    ]
    
    # Check if word matches any medical pattern
    for pattern in medical_patterns:
        if re.match(pattern, word_lower):
            return True
    
    # Check if word contains medical substrings - more inclusive
    medical_substrings = [
        'lung', 'heart', 'chest', 'rib', 'spine', 'aorta', 'artery', 'vein', 'bronchus',
        'pneumonia', 'edema', 'effusion', 'atelectasis', 'nodule', 'mass', 'lesion',
        'opacity', 'infiltrate', 'fibrosis', 'emphysema', 'pneumothorax', 'pleural',
        'pulmonary', 'cardiac', 'vascular', 'bilateral', 'unilateral', 'anterior',
        'posterior', 'lateral', 'medial', 'apical', 'basal', 'central', 'peripheral',
        'diffuse', 'focal', 'multifocal', 'normal', 'abnormal', 'mild', 'moderate',
        'severe', 'prominent', 'enlarged', 'increased', 'decreased', 'absent', 'present',
        'clear', 'obscured', 'visible', 'calcification', 'calcified', 'density', 'shadow',
        'air', 'fluid', 'blood', 'tissue', 'bone', 'soft', 'tube', 'picc', 'catheter',
        'line', 'radiograph', 'xray', 'portable', 'erect', 'hilar', 'mediastinal',
        'bibasilar', 'size', 'volume', 'position', 'unchanged', 'stable', 'improved',
        'worsened', 'structures', 'contours', 'margins', 'stomach', 'abdomen', 'thorax',
        'clavicle', 'diaphragm', 'trachea', 'mediastinum', 'fissure', 'congestion',
        'crowding', 'coiled', 'layering', 'svc', 'ivc', 'ng', 'og', 'ett', 'trach',
        'copd', 'chf', 'pe', 'dvt'
    ]
    
    for substring in medical_substrings:
        if substring in word_lower:
            return True
    
    # If word ends with medical suffixes, likely medical
    medical_suffixes = ['itis', 'osis', 'oma', 'ic', 'al', 'ar', 'ary', 'ous', 'ism']
    for suffix in medical_suffixes:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return True
    
    # If word starts with medical prefixes, likely medical
    medical_prefixes = ['hyper', 'hypo', 'endo', 'exo', 'peri', 'para', 'inter', 'intra']
    for prefix in medical_prefixes:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            return True
    
    return False

def main():
    """Main function to run improved attention visualization"""
    print("🔍 IMPROVED ATTENTION VISUALIZATION WITH INDIVIDUAL NORMALIZATION")
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
        word2idx = tokenizer
        idx2word = {idx: word for word, idx in tokenizer.items()}
    # Add compatibility attributes
    if isinstance(tokenizer, dict):
        # Create a simple wrapper for dictionary-based tokenizer
        class DictTokenizer:
            def __init__(self, word2idx):
                self.word2idx = word2idx
                self.idx2word = {idx: word for word, idx in word2idx.items()}
        tokenizer = DictTokenizer(tokenizer)
    elif hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    data_loader.load_data(max_samples=None, skip_processing=True)
    val_dataset = data_loader.get_validation_data(num_samples=None)
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"✅ Validation data loaded: {len(val_dataset)} samples")
    
    # Test improved attention visualization on a few samples
    model.eval()
    device = next(model.parameters()).device
    sample_count = 0
    max_samples = 2  # Process 2 samples for comparison
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if sample_count >= max_samples:
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
            
            # Process each sample in the batch
            for i in range(len(batch_images)):
                if sample_count >= max_samples:
                    break
                
                # Get single sample
                image = batch_images[i:i+1]
                text = batch_texts[i:i+1]
                study_id = batch_study_ids[i] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[i])
                
                print(f"\n📋 Sample {sample_count + 1}: Study ID {study_id}")
                
                # Create improved attention visualization
                sample_save_dir = f'attention_analysis_improved/sample_{sample_count + 1}_study_{study_id}'
                visualize_attention_with_improved_contrast(
                    model, image, text, tokenizer,
                    sample_idx=0, save_dir=sample_save_dir
                )
                
                sample_count += 1
    
    print(f"\n✅ Improved attention visualization complete!")
    print(f"📁 Results saved to: attention_analysis_improved/")

if __name__ == "__main__":
    main()
