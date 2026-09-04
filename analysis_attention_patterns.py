#!/usr/bin/env python3
"""
Analyze attention patterns to understand why all medical words show similar attention
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
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

def extract_attention_weights_from_full_model(model, images, texts, layer_idx=0):
    """Extract attention weights using the full model's internal components"""
    model.eval()
    with torch.no_grad():
        if isinstance(images, np.ndarray):
            images = torch.FloatTensor(images)
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
        
        if isinstance(texts, np.ndarray):
            texts = torch.LongTensor(texts)
        
        device = next(model.parameters()).device
        images = images.to(device)
        texts = texts.to(device)
        
        print(f"[DEBUG] Extracting attention from model components...")
        
        image_tokens = model.image_encoder(images)
        text_tokens = model.text_encoder(texts)
        
        print(f"[DEBUG] Image tokens shape: {image_tokens.shape}")
        print(f"[DEBUG] Text tokens shape: {text_tokens.shape}")
        
        synergy_layer = model.synergy_branch.co_attn_layers[layer_idx]
        
        i2t_output, i2t_weights = synergy_layer.cross_attention1(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            need_weights=True
        )
        
        t2i_output, t2i_weights = synergy_layer.cross_attention2(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=True
        )
        
        print(f"[DEBUG] T2I attention weights shape: {t2i_weights.shape}")
        print(f"[DEBUG] T2I attention weights min/max: {t2i_weights.min():.6f} / {t2i_weights.max():.6f}")
        
        return {
            'image_tokens': image_tokens.cpu().numpy(),
            'text_tokens': text_tokens.cpu().numpy(),
            'i2t_weights': i2t_weights.cpu().numpy(),
            't2i_weights': t2i_weights.cpu().numpy(),
        }

def analyze_attention_patterns(model, image, text_tokens, tokenizer, sample_idx=0):
    """Analyze attention patterns in detail"""
    
    # Ensure inputs are in the right format
    if len(image.shape) == 3:
        image = image[None]
    if len(text_tokens.shape) == 1:
        text_tokens = text_tokens[None]
    
    # Extract attention weights
    attention_weights = extract_attention_weights_from_full_model(model, image, text_tokens)
    t2i_weights = attention_weights['t2i_weights'][sample_idx]  # Shape: (text_len, image_len)
    
    print(f"[ANALYSIS] T2I attention weights shape: {t2i_weights.shape}")
    
    # Get text tokens for analysis
    if tokenizer:
        word_ids = text_tokens[sample_idx][text_tokens[sample_idx] > 0]
        words = []
        for token_id in word_ids:
            token_id_int = int(token_id)
            word = None
            if hasattr(tokenizer, 'idx2word'):
                word = tokenizer.idx2word.get(token_id_int, None)
                if word is None:
                    word = tokenizer.idx2word.get(str(token_id_int), None)
            elif hasattr(tokenizer, 'word_index'):
                if not hasattr(tokenizer, '_idx2word_cache'):
                    tokenizer._idx2word_cache = {idx: word for word, idx in tokenizer.word_index.items()}
                word = tokenizer._idx2word_cache.get(token_id_int, None)
            
            if word is None:
                word = f"<{token_id_int}>"
            words.append(word)
    else:
        words = [f"token_{i}" for i in range(len(text_tokens[sample_idx]))]
    
    valid_tokens = len([t for t in text_tokens[sample_idx] if t > 0])
    print(f"[ANALYSIS] Valid tokens: {valid_tokens}")
    print(f"[ANALYSIS] Sample words: {words[:10]}")
    
    # Filter for medical words
    medical_word_indices = []
    medical_words_found = []
    for i, word in enumerate(words[:valid_tokens]):
        if is_medical_word(word):
            medical_word_indices.append(i)
            medical_words_found.append(word)
    
    print(f"[ANALYSIS] Medical words found: {medical_words_found}")
    
    if len(medical_word_indices) == 0:
        print("❌ No medical words found in this sample")
        return
    
    # Analyze attention patterns for top medical words
    medical_attention_scores = []
    for idx in medical_word_indices:
        if idx < t2i_weights.shape[0]:
            max_attention = t2i_weights[idx].max()
            min_attention = t2i_weights[idx].min()
            mean_attention = t2i_weights[idx].mean()
            std_attention = t2i_weights[idx].std()
            medical_attention_scores.append((idx, max_attention, min_attention, mean_attention, std_attention, words[idx]))
    
    # Sort by max attention and take top 6
    medical_attention_scores.sort(key=lambda x: x[1], reverse=True)
    top_medical_words = medical_attention_scores[:6]
    
    print(f"\n🔍 DETAILED ATTENTION ANALYSIS:")
    print(f"=" * 60)
    
    for i, (idx, max_att, min_att, mean_att, std_att, word) in enumerate(top_medical_words):
        print(f"\n📊 Word {i+1}: '{word}' (token {idx})")
        print(f"   Max attention: {max_att:.6f}")
        print(f"   Min attention: {min_att:.6f}")
        print(f"   Mean attention: {mean_att:.6f}")
        print(f"   Std deviation: {std_att:.6f}")
        print(f"   Range: {max_att - min_att:.6f}")
        print(f"   Coefficient of variation: {std_att/mean_att:.4f}")
        
        # Analyze spatial distribution
        attention_map = t2i_weights[idx]
        num_patches = len(attention_map)
        patch_grid_size = int(np.sqrt(num_patches))
        
        if patch_grid_size * patch_grid_size == num_patches:
            attention_2d = attention_map.reshape(patch_grid_size, patch_grid_size)
            
            # Analyze different regions
            h, w = attention_2d.shape
            top_region = attention_2d[:h//3, :].mean()
            middle_region = attention_2d[h//3:2*h//3, :].mean()
            bottom_region = attention_2d[2*h//3:, :].mean()
            
            left_region = attention_2d[:, :w//3].mean()
            center_region = attention_2d[:, w//3:2*w//3].mean()
            right_region = attention_2d[:, 2*w//3:].mean()
            
            print(f"   Spatial analysis:")
            print(f"     Top region: {top_region:.6f}")
            print(f"     Middle region: {middle_region:.6f}")
            print(f"     Bottom region: {bottom_region:.6f}")
            print(f"     Left region: {left_region:.6f}")
            print(f"     Center region: {center_region:.6f}")
            print(f"     Right region: {right_region:.6f}")
            
            # Check if attention is uniform
            attention_flat = attention_2d.flatten()
            attention_sorted = np.sort(attention_flat)
            top_10_percent = attention_sorted[-len(attention_sorted)//10:].mean()
            bottom_10_percent = attention_sorted[:len(attention_sorted)//10].mean()
            attention_ratio = top_10_percent / bottom_10_percent if bottom_10_percent > 0 else float('inf')
            
            print(f"   Attention concentration:")
            print(f"     Top 10% mean: {top_10_percent:.6f}")
            print(f"     Bottom 10% mean: {bottom_10_percent:.6f}")
            print(f"     Concentration ratio: {attention_ratio:.2f}")
            
            if attention_ratio < 1.5:
                print(f"     ⚠️  WARNING: Very uniform attention (ratio < 1.5)")
            elif attention_ratio < 2.0:
                print(f"     ⚠️  CAUTION: Low attention concentration (ratio < 2.0)")
            else:
                print(f"     ✅ Good attention concentration")
    
    # Compare attention patterns between words
    print(f"\n🔄 ATTENTION PATTERN COMPARISON:")
    print(f"=" * 60)
    
    if len(top_medical_words) >= 2:
        word1_idx, word1_max, _, _, _, word1 = top_medical_words[0]
        word2_idx, word2_max, _, _, _, word2 = top_medical_words[1]
        
        attention1 = t2i_weights[word1_idx]
        attention2 = t2i_weights[word2_idx]
        
        # Calculate correlation between attention patterns
        correlation = np.corrcoef(attention1, attention2)[0, 1]
        print(f"Correlation between '{word1}' and '{word2}': {correlation:.4f}")
        
        if correlation > 0.95:
            print(f"⚠️  WARNING: Very high correlation - attention patterns are nearly identical!")
        elif correlation > 0.8:
            print(f"⚠️  CAUTION: High correlation - attention patterns are very similar")
        else:
            print(f"✅ Good: Attention patterns are distinct")
        
        # Calculate difference in attention patterns
        attention_diff = np.abs(attention1 - attention2)
        max_diff = attention_diff.max()
        mean_diff = attention_diff.mean()
        
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 0.0001:
            print(f"⚠️  WARNING: Attention patterns are virtually identical!")
        elif max_diff < 0.001:
            print(f"⚠️  CAUTION: Very small differences in attention patterns")
        else:
            print(f"✅ Good: Meaningful differences in attention patterns")

def is_medical_word(word):
    """Determine if a word is medical based on common patterns and characteristics."""
    if not word or len(word) < 2:
        return False
    
    word_lower = word.lower()
    
    # Only filter out the most common non-medical words
    non_medical_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'this', 'that', 'these', 'those',
        'here', 'there', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom',
        'today', 'yesterday', 'tomorrow', 'now', 'then', 'before', 'after', 'during', 'while',
        'unk', '<unk>', 'pad', '<pad>', 'start', '<start>', 'end', '<end>'
    }
    
    if word_lower in non_medical_words:
        return False
    
    # Medical word patterns
    medical_patterns = [
        r'.*lung.*', r'.*heart.*', r'.*chest.*', r'.*rib.*', r'.*spine.*', r'.*vertebra.*',
        r'.*aorta.*', r'.*artery.*', r'.*vein.*', r'.*bronchus.*', r'.*trachea.*', r'.*diaphragm.*',
        r'.*pleura.*', r'.*mediastinum.*', r'.*hilum.*', r'.*fissure.*', r'.*stomach.*', r'.*abdomen.*',
        r'.*pelvis.*', r'.*thorax.*', r'.*clavicle.*',
        r'.*pneumonia.*', r'.*edema.*', r'.*effusion.*', r'.*atelectasis.*', r'.*consolidation.*',
        r'.*nodule.*', r'.*mass.*', r'.*lesion.*', r'.*opacity.*', r'.*infiltrate.*', r'.*fibrosis.*',
        r'.*emphysema.*', r'.*bronchiectasis.*', r'.*pneumothorax.*', r'.*hemothorax.*',
        r'.*pleural.*', r'.*pulmonary.*', r'.*cardiac.*', r'.*vascular.*',
        r'.*bilateral.*', r'.*unilateral.*', r'.*right.*', r'.*left.*', r'.*upper.*', r'.*lower.*',
        r'.*anterior.*', r'.*posterior.*', r'.*lateral.*', r'.*medial.*', r'.*apical.*', r'.*basal.*',
        r'.*central.*', r'.*peripheral.*', r'.*diffuse.*', r'.*focal.*', r'.*multifocal.*',
        r'.*bibasilar.*', r'.*basilar.*', r'.*hilar.*', r'.*perihilar.*', r'.*subpleural.*',
        r'.*normal.*', r'.*abnormal.*', r'.*mild.*', r'.*moderate.*', r'.*severe.*', r'.*prominent.*',
        r'.*enlarged.*', r'.*increased.*', r'.*decreased.*', r'.*absent.*', r'.*present.*',
        r'.*clear.*', r'.*obscured.*', r'.*visible.*', r'.*unchanged.*', r'.*stable.*',
        r'.*improved.*', r'.*worsened.*',
        r'.*tube.*', r'.*picc.*', r'.*catheter.*', r'.*line.*', r'.*wire.*', r'.*stent.*',
        r'.*radiograph.*', r'.*xray.*', r'.*x-ray.*', r'.*ct.*', r'.*scan.*', r'.*exam.*',
        r'.*portable.*', r'.*erect.*', r'.*supine.*', r'.*lateral.*', r'.*oblique.*',
        r'.*calcification.*', r'.*calcified.*', r'.*density.*', r'.*shadow.*', r'.*air.*',
        r'.*fluid.*', r'.*blood.*', r'.*tissue.*', r'.*bone.*', r'.*soft.*', r'.*structures.*',
        r'.*contours.*', r'.*margins.*', r'.*borders.*',
        r'.*size.*', r'.*volume.*', r'.*small.*', r'.*large.*', r'.*tiny.*', r'.*huge.*',
        r'.*cm.*', r'.*mm.*', r'.*inch.*', r'.*diameter.*', r'.*width.*', r'.*length.*',
        r'.*cxr.*', r'.*pa.*', r'.*ap.*', r'.*lat.*', r'.*obl.*', r'.*decub.*',
        r'.*copd.*', r'.*chf.*', r'.*pe.*', r'.*dvt.*', r'.*tb.*', r'.*covid.*',
        r'.*svc.*', r'.*ivc.*', r'.*ng.*', r'.*og.*', r'.*ett.*', r'.*trach.*',
        r'.*position.*', r'.*positioning.*', r'.*placement.*', r'.*location.*',
        r'.*coiled.*', r'.*layering.*', r'.*crowding.*', r'.*congestion.*',
        r'.*results.*', r'.*findings.*', r'.*impression.*', r'.*conclusion.*',
        r'.*obtained.*', r'.*since.*', r'.*preceding.*', r'.*day.*', r'.*exam.*'
    ]
    
    import re
    for pattern in medical_patterns:
        if re.match(pattern, word_lower):
            return True
    
    # Check if word contains medical substrings
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
    
    return False

def main():
    """Main function to analyze attention patterns"""
    print("🔍 ATTENTION PATTERN ANALYSIS")
    print("=" * 50)
    
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
    val_dataset = data_loader.get_validation_data(num_samples=None)
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"✅ Validation data loaded: {len(val_dataset)} samples")
    
    # Analyze attention patterns on one sample
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
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
            
            # Analyze first sample
            image = batch_images[0:1]
            text = batch_texts[0:1]
            study_id = batch_study_ids[0] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[0])
            
            print(f"\n📋 Analyzing Sample: Study ID {study_id}")
            
            analyze_attention_patterns(model, image, text, tokenizer, sample_idx=0)
            break  # Only analyze one sample
    
    print(f"\n✅ Attention pattern analysis complete!")

if __name__ == "__main__":
    main()
