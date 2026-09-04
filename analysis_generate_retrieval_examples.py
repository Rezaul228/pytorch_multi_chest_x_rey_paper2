#!/usr/bin/env python3
"""
Generate retrieval examples with visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Add current directory to path for imports
sys.path.append('.')

from analysis_all_visualization_v1 import visualize_retrieval_examples
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config
import paths

def load_tokenizer_from_metadata():
    """Load the tokenizer from metadata based on current dataset mode"""
    shard_subfolder = "indiana_shards"  # Use the current dataset mode
    metadata_path = paths.get_metadata_path(shard_subfolder)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    if hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    
    return tokenizer

def load_trained_model(model_path):
    """Load the trained model from the specified path"""
    print(f"🤖 Loading trained model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model with same architecture
    model = MultimodalFusion(
        vocab_size=get_vocab_size(),
        embed_dim=get_embed_dim(),
        num_heads=get_current_config()['num_heads'],
        num_layers=get_current_config()['num_layers']
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {next(model.parameters()).device}")
    
    return model

def load_test_data(num_samples=200):
    """Load test data for visualization"""
    print(f"📊 Loading test data ({num_samples} samples)...")
    
    # Initialize data loader
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder="mimic_shards"
    )
    
    # Load tokenizer
    data_loader.tokenizer = load_tokenizer_from_metadata()
    
    # Load test data
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Get test dataset
    test_dataset = data_loader.get_test_data(num_samples=num_samples)
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    return test_dataset, data_loader

def generate_retrieval_examples():
    """Generate retrieval examples with visualization"""
    print("🚀 GENERATING RETRIEVAL EXAMPLES")
    print("=" * 50)
    
    # Model path
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_ori_web_vo10805_to128_lr1e-4_b128_ep50_dualbr_total_loss_diff_ratio_0.1__branch_v2/export/model_weights.pth'
    
    try:
        # Load trained model
        model = load_trained_model(model_path)
        
        # Load test data (200 samples)
        test_dataset, data_loader = load_test_data(num_samples=300)
        
        print(f"\n📊 Creating retrieval visualizations for 3 examples from {len(test_dataset)} samples...")
        
        # Convert dataset to the format expected by visualization function
        test_data = {
            'images': [],
            'captions': [],
            'tokenizer': data_loader.tokenizer
        }
        
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            test_data['images'].append(sample['images'].numpy())
            test_data['captions'].append(sample['captions'].numpy())
        
        test_data['images'] = np.array(test_data['images'])
        test_data['captions'] = np.array(test_data['captions'])
        
        # Generate visualizations
        visualize_retrieval_examples(
            model=model,
            test_data=test_data,
            num_examples=3,  # Generate 3 examples
            k=3,  # Show top 3 results
            output_dir="retrieval_examples"
        )
        
        print(f"\n✅ Retrieval examples generated!")
        print("📁 Check the 'retrieval_examples' directory for results")
        
        # List generated files
        if os.path.exists("retrieval_examples"):
            files = os.listdir("retrieval_examples")
            png_files = [f for f in files if f.endswith('.png')]
            print(f"\n📊 Generated {len(png_files)} visualization files:")
            
            for i, file in enumerate(png_files):
                file_path = os.path.join("retrieval_examples", file)
                file_size = os.path.getsize(file_path) / 1024
                print(f"   {i+1}. {file} ({file_size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during visualization generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🔍 RETRIEVAL EXAMPLES GENERATION")
    print("=" * 50)
    print("Generating Scientific Reports compliant retrieval visualizations")
    print()
    
    # Generate examples
    success = generate_retrieval_examples()
    
    if success:
        print("\n🎉 SUCCESS! Retrieval examples generated!")
        print("\n📋 FEATURES GENERATED:")
        print("-" * 30)
        print("✅ 3 publication-ready figures")
        print("✅ Scientific Reports formatting")
        print("✅ High-resolution output (300 DPI)")
        print("✅ Professional styling and borders")
        print()
        print("🎯 Ready for paper submission!")
    else:
        print("\n❌ Generation failed. Check error messages above.")

if __name__ == "__main__":
    main() 