#!/usr/bin/env python3
"""
Simple script to show text report and corresponding image side by side for a given study ID
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import config
import paths
from data_loader_v1 import IndianaDataLoader

def load_tokenizer():
    """Load tokenizer from metadata"""
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
    
    return tokenizer

def find_study_by_id(data_loader, target_study_id):
    """Find a specific study by ID"""
    print(f"🔍 Searching for study ID: {target_study_id}")
    
    # Get validation data
    val_dataset = data_loader.get_validation_data(num_samples=5000)  # Search more samples
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for batch_idx, batch in enumerate(val_loader):
        if isinstance(batch, dict):
            batch_texts = batch['captions']
            batch_study_ids = batch['study_ids']
            batch_images = batch['images']
        else:
            batch_images, batch_texts, batch_study_ids = batch
        
        study_id = batch_study_ids[0] if isinstance(batch_study_ids, torch.Tensor) else str(batch_study_ids[0])
        
        if str(study_id) == str(target_study_id):
            print(f"✅ Found study {study_id} at batch {batch_idx}")
            return batch_images, batch_texts, study_id
    
    print(f"❌ Study ID {target_study_id} not found in validation data")
    return None, None, None

def decode_text_tokens(text_tokens, tokenizer):
    """Decode text tokens to words"""
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
    
    return ' '.join(words)

def show_study_report(study_id, save_image=True):
    """Show text report and corresponding image side by side"""
    print(f"📋 STUDY REPORT VISUALIZATION")
    print(f"Study ID: {study_id}")
    print("=" * 50)
    
    # Load data loader and tokenizer
    data_loader = IndianaDataLoader(
        batch_size=1,
        use_shards=True,
        shard_subfolder=config.DATASET_MODE
    )
    
    tokenizer = load_tokenizer()
    data_loader.tokenizer = tokenizer
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Find the study
    batch_images, batch_texts, found_study_id = find_study_by_id(data_loader, study_id)
    
    if batch_images is None:
        print(f"❌ Could not find study {study_id}")
        return
    
    # Get image and text
    image = batch_images[0]
    text_tokens = batch_texts[0]
    
    # Convert image to numpy
    if image.shape[0] == 3:  # CHW format
        image_np = image.permute(1, 2, 0).numpy()
    else:  # HWC format
        image_np = image.numpy()
    
    # Normalize image
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    # Decode text
    text_report = decode_text_tokens(text_tokens, tokenizer)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Show image
    ax1.imshow(image_np, cmap='gray')
    ax1.set_title(f'Chest X-ray Image\nStudy ID: {found_study_id}', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Show text report
    ax2.text(0.05, 0.95, f'Study ID: {found_study_id}', 
             transform=ax2.transAxes, fontsize=18, fontweight='bold', 
             verticalalignment='top')
    
    ax2.text(0.05, 0.90, 'Text Report:', 
             transform=ax2.transAxes, fontsize=16, fontweight='bold', 
             verticalalignment='top')
    
    # Wrap text for better display
    words = text_report.split()
    lines = []
    current_line = ""
    max_chars_per_line = 50  # Reduced for bigger font
    
    for word in words:
        if len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    # Display text
    y_position = 0.85
    for line in lines:
        ax2.text(0.05, y_position, line, 
                 transform=ax2.transAxes, fontsize=14, 
                 verticalalignment='top', fontfamily='monospace')
        y_position -= 0.04  # Increased spacing for bigger font
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_image:
        # Create output directory
        output_dir = 'study_reports'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        output_path = os.path.join(output_dir, f'study_{study_id}_report.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved to: {output_path}")
    
    plt.show()
    
    # Print text report to console
    print(f"\n📝 TEXT REPORT:")
    print("-" * 50)
    print(text_report)
    print("-" * 50)
    
    return text_report, image_np

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analysis_show_study_report.py <study_id>")
        print("Example: python analysis_show_study_report.py 53134267")
        return
    
    study_id = sys.argv[1]
    show_study_report(study_id, save_image=True)

if __name__ == "__main__":
    main()
