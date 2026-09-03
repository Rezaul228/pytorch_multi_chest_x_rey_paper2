#!/usr/bin/env python3
"""
Test script for trained model using cross-modal evaluation
Automatically adapts to dataset mode from config.py
"""

import torch
import config
import paths
import pickle
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming

def load_tokenizer_from_metadata():
    """Load the tokenizer from metadata based on current dataset mode"""
    # Get metadata path from current dataset configuration
    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    
    print(f"📁 Loading tokenizer from: {metadata_path}")
    print(f"🔄 Dataset mode: {shard_subfolder}")
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        tokenizer = metadata.get('tokenizer')
        if tokenizer is None:
            raise ValueError("No tokenizer found in metadata")
        
        # Add compatibility attributes for EnhancedTokenizer
        if hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
            tokenizer.word2idx = tokenizer.word_index
            tokenizer.idx2word = tokenizer.index_word
        
        print(f"✅ Loaded {shard_subfolder} tokenizer with {len(tokenizer.word2idx)} words")
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error loading {shard_subfolder} tokenizer: {e}")
        raise

def test_trained_model():
    print("🧪 Testing Trained Model")
    print("=" * 50)
    
    # Last tested MIMIC model (vocab 10805, seq len 128, dual BR, 45 epochs)
    config.switch_dataset("mimic_shards")
    config.print_current_config()
    
    # Model configuration - use current dataset mode
    dataset_mode = config.DATASET_MODE
    model_path = (
        '/home/abedin/Developments/pytorch_multi_chest_x_ray1/'
        'saved_models/mimic_origi_vocab10805_to128_lr1e-4_b128_ep45_dualbr_v3/export/model_weights.pth'
    )
    
    print(f"📁 Loading model from: {model_path}")
    
    # Create model with current config
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f"⚠️  Model not found at {model_path}")
        print(f"💡 Please train a model first or check the model path")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load test data using current dataset configuration
    print(f"\n📁 Loading {dataset_mode} test data...")
    
    try:
        data_loader = IndianaDataLoader(
            batch_size=32, 
            use_shards=True, 
            shard_subfolder=dataset_mode
        )
        
        # Load the tokenizer for current dataset
        data_loader.tokenizer = load_tokenizer_from_metadata()
        
        data_loader.load_data(max_samples=None, skip_processing=True)
        
        # Get test dataset
        test_dataset = data_loader.get_test_data(num_samples=None)
        
        print(f"📊 {dataset_mode} Test dataset loaded: {len(test_dataset)} samples")
        
        # Create DataLoader for streaming evaluation
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        print(f"\n🎯 Running cross-modal evaluation on {dataset_mode} test data...")
        print("=" * 50)
        
        # Run evaluation
        results = evaluate_cross_modal_retrieval_streaming(
            model=model,
            test_dataset=test_loader,
            k_values=[1, 5, 10],
            batch_size=32,
            visualize=True,
            num_vis_examples=5,
            output_dir=f"outputs/test_results_{dataset_mode}/"
        )
        
        print("\n" + "=" * 50)
        print(f"🎯 FINAL TEST RESULTS ({dataset_mode.upper()} Test Data)")
        print("=" * 50)
        
        # Print detailed results
        print(f"📊 Image-to-Text Retrieval:")
        print(f"   MRR: {results['i2t_mrr']:.4f}")
        print(f"   Mean Rank: {results['i2t_mean_rank']:.2f}")
        print(f"   Recall@1: {results['i2t_recall@1']:.4f}")
        print(f"   Recall@5: {results['i2t_recall@5']:.4f}")
        print(f"   Recall@10: {results['i2t_recall@10']:.4f}")
        
        print(f"\n📊 Text-to-Image Retrieval:")
        print(f"   MRR: {results['t2i_mrr']:.4f}")
        print(f"   Mean Rank: {results['t2i_mean_rank']:.2f}")
        print(f"   Recall@1: {results['t2i_recall@1']:.4f}")
        print(f"   Recall@5: {results['t2i_recall@5']:.4f}")
        print(f"   Recall@10: {results['t2i_recall@10']:.4f}")
        
        print(f"\n📊 Overall Performance:")
        print(f"   Average MRR: {results['avg_mrr']:.4f}")
        print(f"   Average Recall@1: {results['avg_recall@1']:.4f}")
        print(f"   Average Recall@5: {results['avg_recall@5']:.4f}")
        print(f"   Average Recall@10: {results['avg_recall@10']:.4f}")
        
        print(f"\n📋 Summary: {results['summary']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_trained_model() 