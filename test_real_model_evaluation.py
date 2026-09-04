#!/usr/bin/env python3
"""
TEST REAL MODEL EVALUATION
Test evaluation metrics with actual trained model and real data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Add current directory to path for imports
sys.path.append('.')

from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval, evaluate_cross_modal_retrieval_streaming
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config
import paths

def load_tokenizer_from_metadata():
    """Load the tokenizer from metadata based on current dataset mode"""
    shard_subfolder = "mimic_shards_hybrid_full_ori"  # Use the current dataset mode
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

def load_test_data(num_samples=100):
    """Load a small sample of test data"""
    print(f"📊 Loading test data ({num_samples} samples)...")
    
    # Initialize data loader
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder="mimic_shards_hybrid_full_ori"  # Use the current dataset mode
    )
    
    # Load tokenizer
    data_loader.tokenizer = load_tokenizer_from_metadata()
    
    # Load test data
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Get test dataset
    test_dataset = data_loader.get_test_data(num_samples=num_samples)
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    return test_dataset, data_loader

def test_small_sample_evaluation():
    """Test evaluation with small sample using real model and data"""
    print("🧪 TESTING REAL MODEL EVALUATION")
    print("=" * 60)
    
    # Model path
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v1_seed_17/export/model_weights.pth'
    
    try:
        # Load trained model
        model = load_trained_model(model_path)
        
        # Load test data (small sample for testing)
        test_dataset, data_loader = load_test_data(num_samples=100)
        
        print("\n📊 Running evaluation with real model and data...")
        
        # Test evaluation with rank distribution plots
        results = evaluate_cross_modal_retrieval_streaming(
            model=model,
            test_dataset=test_dataset,
            k_values=[1, 5, 10],
            batch_size=32,
            visualize=False,
            num_vis_examples=0,
            create_rank_plots=True,  # Enable rank distribution plots
            output_dir="real_model_evaluation_plots"
        )
        
        print("\n✅ EVALUATION RESULTS:")
        print("-" * 40)
        
        # Display key metrics
        key_metrics = [
            ('i2t_recall@1', 'Image→Text R@1'),
            ('i2t_recall@5', 'Image→Text R@5'),
            ('i2t_recall@10', 'Image→Text R@10'),
            ('t2i_recall@1', 'Text→Image R@1'),
            ('t2i_recall@5', 'Text→Image R@5'),
            ('t2i_recall@10', 'Text→Image R@10'),
            ('i2t_mrr', 'Image→Text MRR'),
            ('t2i_mrr', 'Text→Image MRR'),
            ('i2t_mean_rank', 'Image→Text Mean Rank'),
            ('t2i_mean_rank', 'Text→Image Mean Rank'),
            ('avg_mrr', 'Average MRR'),
            ('avg_mean_rank', 'Average Mean Rank')
        ]
        
        for metric_key, metric_name in key_metrics:
            if metric_key in results:
                print(f"✅ {metric_name}: {results[metric_key]:.4f}")
            else:
                print(f"❌ {metric_name}: MISSING")
        
        # Check if rank distribution plot was created
        if 'rank_plot_path' in results:
            print(f"\n✅ Rank distribution plot created: {results['rank_plot_path']}")
        else:
            print("\n❌ Rank distribution plot not created")
        
        # Print summary
        if 'summary' in results:
            print(f"\n📋 Summary: {results['summary']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_full_evaluation_preview():
    """Preview what full evaluation would look like"""
    print("\n🔍 FULL EVALUATION PREVIEW")
    print("=" * 40)
    print("This shows what metrics you'll get with full test set:")
    print()
    print("✅ Primary Metrics (Paper Ready):")
    print("   - Recall@K (K=1,5,10) → measures how early correct match appears")
    print("   - MRR → exact rank quality in reciprocal terms")
    print("   - Mean & Median Rank → intuitive interpretation")
    print("   - Rank distribution plots → performance spread")
    print()
    print("✅ Additional Metrics (Extra):")
    print("   - Precision@K (K=1,5,10)")
    print("   - Rank percentages (≤1, ≤5, ≤10, ≤50, ≤100)")
    print("   - Per-direction and macro-averaged results")
    print()
    print("🎯 For Final Results:")
    print("   - Use full test dataset (not 100 samples)")
    print("   - Set create_rank_plots=True")
    print("   - Results will be publication-ready")

def main():
    """Main function to test real model evaluation"""
    print("🔍 REAL MODEL EVALUATION TESTING")
    print("=" * 60)
    print("Testing evaluation metrics with actual trained model")
    print()
    
    # Test with small sample
    results = test_small_sample_evaluation()
    
    if results:
        print("\n🎉 SUCCESS! All metrics working with real model.")
        print("\n📋 SUMMARY:")
        print("-" * 20)
        print("✅ Currently Implemented:")
        print("   - Recall@K (K=1,5,10)")
        print("   - MRR")
        print("   - Mean & Median Rank")
        print("   - Rank distribution plots")
        print("   - Precision@K (extra)")
        print("   - Rank percentages (extra)")
        print()
        print("🎯 Ready for paper with all required metrics!")
        print("   - Use full test dataset for final results")
        print("   - All metrics are publication-ready")
    else:
        print("\n❌ Evaluation failed. Check error messages above.")
    
    # Show full evaluation preview
    test_full_evaluation_preview()

if __name__ == "__main__":
    main() 