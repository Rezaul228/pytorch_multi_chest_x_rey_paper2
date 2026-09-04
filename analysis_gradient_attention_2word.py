"""
Medical Term Attention Minimal Visualization
--------------------------------------------
Clean visualization with no text overlays and more attention patches.
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


# ===============================================================
# Utility Functions
# ===============================================================

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


def find_reports_with_medical_term(data_loader, tokenizer, target_term, num_samples=12):
    """Find reports containing a specific medical term (can be 1 or 2 words)"""
    print(f"🔍 Searching for reports containing: '{target_term}' (need {num_samples} samples)")

    val_dataset = data_loader.get_validation_data(num_samples=5000)  # Search more samples
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

    found_reports = []
    seen_study_ids = set()

    # Split target term into words for more flexible matching
    target_words = target_term.lower().split()
    print(f"   Target words: {target_words}")

    for batch in val_loader:
        if isinstance(batch, dict):
            batch_texts = batch['captions']
            batch_study_ids = batch['study_ids']
        else:
            _, batch_texts, batch_study_ids = batch

        study_id = batch_study_ids[0] if isinstance(batch_study_ids, str) else str(batch_study_ids[0])
        if study_id in seen_study_ids:
            continue

        # Decode text
        text_tokens = batch_texts[0]
        words = []
        for token_id in text_tokens:
            if token_id > 0:
                token_id_int = int(token_id)
                if hasattr(tokenizer, 'idx2word'):
                    word = tokenizer.idx2word.get(token_id_int, tokenizer.idx2word.get(str(token_id_int), None))
                else:
                    word = f"<{token_id_int}>"
                if word and word not in ['<pad>', '<unk>']:
                    words.append(word)
        text_description = ' '.join(words).lower()

        # Check for target term - all words must be present
        if (all(word in text_description for word in target_words) and
                len(found_reports) < num_samples and
                study_id not in seen_study_ids):
            found_reports.append({
                'study_id': study_id,
                'text': text_description,
                'words': words,
                'batch': batch
            })
            seen_study_ids.add(study_id)
            print(f" ✅ Found '{target_term}' in study: {study_id} ({len(found_reports)}/{num_samples})")

    return found_reports

def find_reports_with_medical_terms(data_loader, tokenizer, target_terms, num_samples_per_term=4):
    """Find reports containing specific medical terms"""
    print(f"🔍 Searching for reports containing: {target_terms}")

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

        study_id = batch_study_ids[0] if isinstance(batch_study_ids, str) else str(batch_study_ids[0])
        if study_id in seen_study_ids:
            continue

        # Decode text
        text_tokens = batch_texts[0]
        words = []
        for token_id in text_tokens:
            if token_id > 0:
                token_id_int = int(token_id)
                if hasattr(tokenizer, 'idx2word'):
                    word = tokenizer.idx2word.get(token_id_int, tokenizer.idx2word.get(str(token_id_int), None))
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
                print(f" ✅ Found '{term}' in study: {study_id}")
                break

    return found_reports


def create_single_term_visualization(model, found_reports, tokenizer, target_term, save_dir='medical_term_2word_analysis'):
    """Create visualization for a single medical term with 12 samples"""
    print(f"🔬 Creating Visualization for Medical Term: '{target_term.upper()}'...")
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    if not found_reports:
        print(f" ⚠️ No reports found for '{target_term}'")
        return

    print(f" 📊 Processing '{target_term}' - {len(found_reports)} reports found")

    # Create 3x4 grid for 12 samples
    fig = plt.figure(figsize=(30, 16))
    fig.suptitle(f'Medical Term: "{target_term.upper()}" - Gradient Distance Analysis (12 Samples)',
                 fontsize=20, fontweight='bold')

    num_reports = min(len(found_reports), 12)
    for i in range(num_reports):
        report = found_reports[i]
        batch = report['batch']
        study_id = report['study_id']
        words = report['words']

        if isinstance(batch, dict):
            batch_images = batch['images']
            batch_texts = batch['captions']
        else:
            batch_images, batch_texts, _ = batch

        if batch_images.shape[1] != 3:
            batch_images = batch_images.permute(0, 3, 1, 2)

        batch_images = batch_images.to(device)
        batch_texts = batch_texts.to(device)

        # Normalize image
        image_np = batch_images[0].cpu().numpy()
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Extract attention weights
        img_to_txt_attn, txt_to_img_attn, _, _ = extract_attention_weights(model, batch_images, batch_texts, layer_idx=0)

        # Create 3x4 grid for 12 samples
        ax = plt.subplot(3, 4, i + 1)

        if img_to_txt_attn is not None and txt_to_img_attn is not None:
            target_word_idx = None
            target_words = target_term.lower().split()
            
            # Find the first word that matches any of the target words
            for j, word in enumerate(words):
                if any(target_word in word.lower() for target_word in target_words):
                    target_word_idx = j
                    break

            if target_word_idx is not None:
                word_attention = txt_to_img_attn[0, target_word_idx, :]
                attention_2d = word_attention.view(14, 14)
                attention_resized = F.interpolate(
                    attention_2d.unsqueeze(0).unsqueeze(0),
                    size=(image_np.shape[0], image_np.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

                attention_np = attention_resized.cpu().numpy()
                attention_flat = attention_np.flatten()
                threshold = np.percentile(attention_flat, 90)
                high_attention_mask = attention_np >= threshold
                top_10_indices = torch.topk(attention_resized.flatten(), k=10).indices

                ax.imshow(image_np, cmap='gray', alpha=1.0)

                h, w = image_np.shape[:2]
                attention_overlay = np.zeros((h, w, 4))

                for y in range(h):
                    for x in range(w):
                        if high_attention_mask[y, x]:
                            min_distance = float('inf')
                            for idx in top_10_indices:
                                box_y = idx // attention_resized.shape[1]
                                box_x = idx % attention_resized.shape[1]
                                box_y_scaled = int(box_y * h / attention_resized.shape[0])
                                box_x_scaled = int(box_x * w / attention_resized.shape[1])
                                distance = np.sqrt((y - box_y_scaled) ** 2 + (x - box_x_scaled) ** 2)
                                min_distance = min(min_distance, distance)

                            attention_overlay[y, x, 0] = 0.1
                            attention_overlay[y, x, 1] = 0.2
                            attention_overlay[y, x, 2] = 0.9
                            max_distance = 100
                            if min_distance <= 20:
                                attention_overlay[y, x, 3] = 0.8
                            else:
                                fade_factor = min(1.0, min_distance / max_distance)
                                attention_overlay[y, x, 3] = 0.2 + 0.15 * (1.0 - fade_factor)

                ax.imshow(attention_overlay)

                for idx in top_10_indices:
                    y = idx // attention_resized.shape[1]
                    x = idx % attention_resized.shape[1]
                    rect = plt.Rectangle((x - 20, y - 20), 40, 40, linewidth=2,
                                         edgecolor='cyan', facecolor='none')
                    ax.add_patch(rect)

                ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
            else:
                ax.imshow(image_np, cmap='gray', alpha=1.0)
                ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
        else:
            ax.imshow(image_np, cmap='gray', alpha=1.0)
            ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'minimal_medical_term_{target_term}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✅ Saved visualization for '{target_term}' with {num_reports} samples")


def create_minimal_attention_visualization(model, found_reports, tokenizer, save_dir='medical_term_2word_analysis'):
    """Create minimal visualization with no text and more attention patches"""
    print("🔬 Creating Minimal Medical Term Attention Visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    for term, reports in found_reports.items():
        if not reports:
            print(f" ⚠️ No reports found for '{term}'")
            continue

        print(f" 📊 Processing '{term}' - {len(reports)} reports found")

        # Create 2x2 grid visualization
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Medical Term: \"{term.upper()}\" - Gradient Distance Analysis (4 Samples)',
                     fontsize=20, fontweight='bold')

        num_reports = min(len(reports), 4)
        for i in range(num_reports):
            report = reports[i]
            batch = report['batch']
            study_id = report['study_id']
            words = report['words']

            if isinstance(batch, dict):
                batch_images = batch['images']
                batch_texts = batch['captions']
            else:
                batch_images, batch_texts, _ = batch

            if batch_images.shape[1] != 3:
                batch_images = batch_images.permute(0, 3, 1, 2)

            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)

            # Normalize image
            image_np = batch_images[0].cpu().numpy()
            if len(image_np.shape) == 3 and image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

            # Extract attention weights
            img_to_txt_attn, txt_to_img_attn, _, _ = extract_attention_weights(model, batch_images, batch_texts, layer_idx=0)

            ax = plt.subplot(2, 2, i + 1)

            if img_to_txt_attn is not None and txt_to_img_attn is not None:
                target_word_idx = None
                for j, word in enumerate(words):
                    if term.lower() in word.lower():
                        target_word_idx = j
                        break

                if target_word_idx is not None:
                    word_attention = txt_to_img_attn[0, target_word_idx, :]
                    attention_2d = word_attention.view(14, 14)
                    attention_resized = F.interpolate(
                        attention_2d.unsqueeze(0).unsqueeze(0),
                        size=(image_np.shape[0], image_np.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    attention_np = attention_resized.cpu().numpy()
                    attention_flat = attention_np.flatten()
                    threshold = np.percentile(attention_flat, 90)
                    high_attention_mask = attention_np >= threshold
                    top_10_indices = torch.topk(attention_resized.flatten(), k=10).indices

                    ax.imshow(image_np, cmap='gray', alpha=1.0)

                    h, w = image_np.shape[:2]
                    attention_overlay = np.zeros((h, w, 4))

                    for y in range(h):
                        for x in range(w):
                            if high_attention_mask[y, x]:
                                min_distance = float('inf')
                                for idx in top_10_indices:
                                    box_y = idx // attention_resized.shape[1]
                                    box_x = idx % attention_resized.shape[1]
                                    box_y_scaled = int(box_y * h / attention_resized.shape[0])
                                    box_x_scaled = int(box_x * w / attention_resized.shape[1])
                                    distance = np.sqrt((y - box_y_scaled) ** 2 + (x - box_x_scaled) ** 2)
                                    min_distance = min(min_distance, distance)

                                attention_overlay[y, x, 0] = 0.1
                                attention_overlay[y, x, 1] = 0.2
                                attention_overlay[y, x, 2] = 0.9
                                max_distance = 100
                                if min_distance <= 20:
                                    attention_overlay[y, x, 3] = 0.8
                                else:
                                    fade_factor = min(1.0, min_distance / max_distance)
                                    attention_overlay[y, x, 3] = 0.2 + 0.15 * (1.0 - fade_factor)

                    ax.imshow(attention_overlay)

                    for idx in top_10_indices:
                        y = idx // attention_resized.shape[1]
                        x = idx % attention_resized.shape[1]
                        rect = plt.Rectangle((x - 20, y - 20), 40, 40, linewidth=2,
                                             edgecolor='cyan', facecolor='none')
                        ax.add_patch(rect)

                    ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
                else:
                    ax.imshow(image_np, cmap='gray', alpha=1.0)
                    ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
            else:
                ax.imshow(image_np, cmap='gray', alpha=1.0)
                ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')

            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'minimal_medical_term_{term}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f" ✅ Saved minimal visualization for '{term}'")


# ===============================================================
# Main Execution
# ===============================================================

def main():
    """Main function to create minimal medical term attention visualization"""
    print("🔬 GRADIENT DISTANCE ATTENTION VISUALIZATION")
    print("=" * 60)

    config.print_current_config()

    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/' \
                 'mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_' \
                 'main_loss20_ortho15__branch_v1_seed_17/export/model_weights.pth'
    print(f"📁 Loading model from: {model_path}")
    model = load_model_weights_only(model_path)

    print("\n📁 Loading validation data and tokenizer...")
    data_loader = IndianaDataLoader(batch_size=1, use_shards=True,
                                    shard_subfolder=config.DATASET_MODE)

    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    tokenizer = metadata.get('tokenizer')

    if tokenizer is None:
        raise ValueError("No tokenizer found in metadata")

    # Handle tokenizer compatibility
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

    # Available medical terms (can be 1 or 2 words)
    available_terms = ['cardiac', 'opacity', 'pleural', 'costophrenic', 'bilateral',
                      'effusion', 'consolidation', 'infiltrate', 'pneumothorax', 'atelectasis',
                      'left effusion', 'right effusion', 'cardiac enlargement', 'left infiltrate',
                      'right infiltrate', 'bilateral infiltrate', 'pleural effusion', 'costophrenic angle']
    
    # Check if target term is provided
    import sys
    target_term = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_term is None:
        print(f"\n📋 Available medical terms: {', '.join(available_terms)}")
        print("Usage: python analysis_gradient_attention_2word.py <medical_term>")
        print("Example: python analysis_gradient_attention_2word.py cardiac")
        print("Example: python analysis_gradient_attention_2word.py 'left effusion'")
        print("Example: python analysis_gradient_attention_2word.py 'cardiac enlargement'")
        return
    
    # Check if target term is valid (either exact match or contains valid words)
    target_words = target_term.lower().split()
    valid_terms = ['cardiac', 'opacity', 'pleural', 'costophrenic', 'bilateral',
                  'effusion', 'consolidation', 'infiltrate', 'pneumothorax', 'atelectasis',
                  'left', 'right', 'enlargement', 'angle']
    
    if not all(word in valid_terms for word in target_words):
        print(f"❌ Error: '{target_term}' contains invalid words.")
        print(f"Valid words: {', '.join(valid_terms)}")
        return

    # Find reports for the specific target term
    found_reports = find_reports_with_medical_term(data_loader, tokenizer, target_term, num_samples=12)

    save_dir = 'medical_term_2word_analysis'
    os.makedirs(save_dir, exist_ok=True)

    try:
        create_single_term_visualization(model, found_reports, tokenizer, target_term, save_dir)
        print(f"\n✅ Visualization completed for '{target_term}'!")
        print(f"📁 Results saved to: {save_dir}/")
        print(f" - minimal_medical_term_{target_term}.png: 12 samples for '{target_term}'")

        print("\n🎯 KEY FEATURES:")
        print(" - X-ray image at full brightness everywhere")
        print(" - Blue attention pixels: Deep blue inside bounding boxes (80% opacity)")
        print(" - Light blue outside bounding boxes (20-35% opacity)")
        print(" - Distance-based fading (farther = lighter)")
        print(" - Top 10 attention patches highlighted in cyan")
        print(" - 12 samples in 3x4 grid layout")

    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()



'''
#!/usr/bin/env python3
"""
Medical Term Attention Minimal Visualization
Clean visualization with no text overlays and more attention patches

ONLY CHANGE: activated pixels are colored via a heatmap (blue→green→yellow→red)
like your example, while all logic/opacity/boxes remain unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pickle
import gc
from datetime import datetime
import config
import paths
from data_loader_v1 import IndianaDataLoader
from base_models_refactored_v1 import MultimodalFusion

# -------------------------
# Heatmap color configuration
# 'jet' matches your screenshot look closely; 'turbo' is smoother/perceptual.
COLORMAP_NAME = 'jet'   # options: 'jet', 'turbo', 'inferno', 'plasma'
# -------------------------

def load_model_weights_only(checkpoint_path):
    """Load only the model weights without full checkpoint"""
    print(f"📁 Loading model weights from: {checkpoint_path}")
    model_state_dict = torch.load(checkpoint_path, map_location='cpu')
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f"✅ Model loaded successfully")
    return model

def extract_attention_weights(model, images, texts, layer_idx=0):
    """Extract attention weights from the model"""
    model.eval()
    with torch.no_grad():
        image_tokens = model.image_encoder(images)
        text_tokens  = model.text_encoder(texts)
        synergy_branch = model.synergy_branch
        if layer_idx < len(synergy_branch.co_attn_layers):
            co_attn_layer = synergy_branch.co_attn_layers[layer_idx]
            _, img_to_txt_attn = co_attn_layer.cross_attention1(
                query=image_tokens, key=text_tokens, value=text_tokens, need_weights=True
            )
            _, txt_to_img_attn = co_attn_layer.cross_attention2(
                query=text_tokens, key=image_tokens, value=image_tokens, need_weights=True
            )
            return img_to_txt_attn, txt_to_img_attn, image_tokens, text_tokens
    return None, None, image_tokens, text_tokens

def find_reports_with_medical_terms(data_loader, tokenizer, target_terms, num_samples_per_term=4):
    """Find reports containing specific medical terms"""
    print(f"🔍 Searching for reports containing: {target_terms}")
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
        if study_id in seen_study_ids:
            continue
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

def create_minimal_attention_visualization(model, found_reports, tokenizer, target_term, save_dir='medical_term_2word_analysis'):
    """Create minimal visualization for a specific medical term with 12 samples"""
    print(f"🔬 Creating Heatmap Visualization for Medical Term: '{target_term.upper()}'...")
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    cmap = cm.get_cmap(COLORMAP_NAME)

    if target_term not in found_reports or not found_reports[target_term]:
        print(f"  ⚠️ No reports found for '{target_term}'")
        return
    
    reports = found_reports[target_term]
    print(f"  📊 Processing '{target_term}' - {len(reports)} reports found")

    fig = plt.figure(figsize=(30, 16))
    fig.suptitle(f'Medical Term: "{target_term.upper()}" - Heatmap-Colored Activations (12 Samples)', fontsize=20, fontweight='bold')
    num_reports = min(len(reports), 12)

    for i in range(num_reports):
        report = reports[i]
        batch = report['batch']
        study_id = report['study_id']
        words = report['words']

        if isinstance(batch, dict):
            batch_images = batch['images']
            batch_texts  = batch['captions']
        else:
            batch_images, batch_texts, _ = batch

        # Ensure BCHW
        if batch_images.shape[1] != 3:
            batch_images = batch_images.permute(0, 3, 1, 2)

        batch_images = batch_images.to(device)
        batch_texts  = batch_texts.to(device)

        image_np = batch_images[0].cpu().numpy()
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)

        img_to_txt_attn, txt_to_img_attn, image_tokens, text_tokens = extract_attention_weights(
            model, batch_images, batch_texts, layer_idx=0
        )

        # Create 4x3 grid for 12 samples
        ax = plt.subplot(3, 4, i+1)

        if img_to_txt_attn is not None and txt_to_img_attn is not None:
            target_word_idx = None
            target_words = target_term.lower().split()
            
            # Find the first word that matches any of the target words
            for j, word in enumerate(words):
                if any(target_word in word.lower() for target_word in target_words):
                    target_word_idx = j
                    break

            if target_word_idx is not None:
                word_attention = txt_to_img_attn[0, target_word_idx, :]

                # reshape to 14x14 (adjust if your grid differs)
                attention_2d = word_attention.view(14, 14)

                # resize to image size
                attention_resized = F.interpolate(
                    attention_2d.unsqueeze(0).unsqueeze(0),
                    size=(image_np.shape[0], image_np.shape[1]),
                    mode='bilinear', align_corners=False
                ).squeeze()

                attention_np = attention_resized.cpu().numpy()
                attention_flat = attention_np.flatten()
                threshold = np.percentile(attention_flat, 90)  # Top 10%
                high_attention_mask = attention_np >= threshold

                top_10_indices = torch.topk(attention_resized.flatten(), k=10).indices

                ax.imshow(image_np, cmap='gray', alpha=1.0)

                h, w = image_np.shape[:2]
                attention_overlay = np.zeros((h, w, 4), dtype=np.float32)

                # ----- NEW: color from colormap based on normalized attention within the high-attention set
                ha_vals = attention_np[high_attention_mask]
                a_min = float(ha_vals.min()) if ha_vals.size > 0 else float(attention_np.min())
                a_max = float(ha_vals.max()) if ha_vals.size > 0 else float(attention_np.max())
                a_den = max(a_max - a_min, 1e-6)

                for y in range(h):
                    # only iterate columns that have any high-attention; keeps logic identical, just color change
                    for x in range(w):
                        if high_attention_mask[y, x]:
                            # Normalize attention value into [0,1] for colormap
                            a_norm = (attention_np[y, x] - a_min) / a_den
                            r, g, b, _ = cmap(a_norm)  # RGBA 0..1

                            attention_overlay[y, x, 0] = r
                            attention_overlay[y, x, 1] = g
                            attention_overlay[y, x, 2] = b

                            # Opacity logic unchanged (distance-based fade to emphasize boxed areas)
                            min_distance = float('inf')
                            for idx in top_10_indices:
                                box_y = idx // attention_resized.shape[1]
                                box_x = idx %  attention_resized.shape[1]
                                box_y_scaled = int(box_y * h / attention_resized.shape[0])
                                box_x_scaled = int(box_x * w / attention_resized.shape[1])
                                distance = np.sqrt((y - box_y_scaled)**2 + (x - box_x_scaled)**2)
                                min_distance = min(min_distance, distance)

                            max_distance = 100
                            if min_distance <= 20:
                                attention_overlay[y, x, 3] = 0.8
                            else:
                                fade_factor = min(1.0, min_distance / max_distance)
                                attention_overlay[y, x, 3] = 0.3 + 0.2 * (1.0 - fade_factor)
                # ----- END NEW

                ax.imshow(attention_overlay)

                # draw top-10 rectangles (unchanged)
                for idx in top_10_indices:
                    y = idx // attention_resized.shape[1]
                    x = idx %  attention_resized.shape[1]
                    rect = plt.Rectangle((x-20, y-20), 40, 40, linewidth=2,
                                         edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
            else:
                ax.imshow(image_np, cmap='gray', alpha=1.0)
                ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')
        else:
            ax.imshow(image_np, cmap='gray', alpha=1.0)
            ax.set_title(f'Study: {study_id}', fontsize=14, fontweight='bold')

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'minimal_medical_term_{target_term}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved visualization for '{target_term}' with {num_reports} samples")

def main(target_term=None):
    print("🔬 GRADIENT DISTANCE ATTENTION VISUALIZATION (Heatmap Coloring)")
    print("=" * 60)
    config.print_current_config()

    # Available medical terms
    available_terms = ['cardiac', 'opacity', 'pleural', 'costophrenic', 'bilateral',
                      'effusion', 'consolidation', 'infiltrate', 'pneumothorax', 'atelectasis']
    
    if target_term is None:
        print(f"\n📋 Available medical terms: {', '.join(available_terms)}")
        print("Usage: python analysis_gradient_attention.py <medical_term>")
        print("Example: python analysis_gradient_attention.py cardiac")
        return
    
    if target_term.lower() not in available_terms:
        print(f"❌ Error: '{target_term}' is not a valid medical term.")
        print(f"Available terms: {', '.join(available_terms)}")
        return

    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model_weights.pth'
    print(f"📁 Loading model from: {model_path}")
    model = load_model_weights_only(model_path)
    print(f"✅ Model loaded successfully")

    print("\n📁 Loading validation data and tokenizer...")
    data_loader = IndianaDataLoader(batch_size=1, use_shards=True, shard_subfolder=config.DATASET_MODE)

    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    tokenizer = metadata.get('tokenizer')
    if tokenizer is None:
        raise ValueError("No tokenizer found in metadata")

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

    # Find reports for the specific target term
    found_reports = find_reports_with_medical_terms(data_loader, tokenizer, [target_term], num_samples_per_term=12)

    save_dir = 'gradient_attention_analysis'
    os.makedirs(save_dir, exist_ok=True)

    try:
        create_minimal_attention_visualization(model, found_reports, tokenizer, target_term, save_dir)
        print(f"\n✅ Visualization completed for '{target_term}'! Results in: {save_dir}/")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    import sys
    target_term = sys.argv[1] if len(sys.argv) > 1 else None
    main(target_term)
'''