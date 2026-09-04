#!/usr/bin/env python3
"""
RUN FULL EVALUATION
Run evaluation on entire test set with trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Add current directory to path for imports
sys.path.append('.')

from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming
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

def load_full_test_data():
    """Load the entire test dataset"""
    print(f"📊 Loading FULL test dataset...")
    
    # Initialize data loader
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder="indiana_shards"
    )
    
    # Load tokenizer
    data_loader.tokenizer = load_tokenizer_from_metadata()
    
    # Load test data
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Get FULL test dataset (no sample limit)
    test_dataset = data_loader.get_test_data(num_samples=None)
    
    print(f"✅ FULL test data loaded: {len(test_dataset)} samples")
    
    return test_dataset, data_loader

def run_full_evaluation():
    """Run evaluation on entire test set"""
    print("🚀 RUNNING FULL EVALUATION ON ENTIRE TEST SET")
    print("=" * 70)
    
    # Model path
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v2/export/model_weights.pth'
    
    try:
        # Load trained model
        model = load_trained_model(model_path)
        
        # Load FULL test data
        test_dataset, data_loader = load_full_test_data()
        
        print(f"\n📊 Running evaluation on {len(test_dataset)} test samples...")
        print("   This may take several minutes depending on dataset size...")
        
        # Run evaluation with rank distribution plots
        results = evaluate_cross_modal_retrieval_streaming(
            model=model,
            test_dataset=test_dataset,
            k_values=[1, 5, 10],
            batch_size=32,
            visualize=False,
            num_vis_examples=0,
            create_rank_plots=True,  # Enable rank distribution plots
            output_dir="full_evaluation_results"
        )
        
        print("\n" + "=" * 70)
        print("🎉 FULL EVALUATION COMPLETE!")
        print("=" * 70)
        
        print("\n📊 FINAL RESULTS:")
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
        
        # Save results to file
        results_file = "full_evaluation_results/evaluation_results.txt"
        os.makedirs("full_evaluation_results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write("FULL EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test samples: {len(test_dataset)}\n")
            f.write(f"Model: {model_path}\n\n")
            
            for metric_key, metric_name in key_metrics:
                if metric_key in results:
                    f.write(f"{metric_name}: {results[metric_key]:.4f}\n")
            
            if 'summary' in results:
                f.write(f"\nSummary: {results['summary']}\n")
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during full evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🔍 FULL TEST SET EVALUATION")
    print("=" * 70)
    print("Running evaluation on entire test dataset")
    print("This will provide final results for your paper")
    print()
    
    # Run full evaluation
    results = run_full_evaluation()
    
    if results:
        print("\n🎯 SUCCESS! Full evaluation complete.")
        print("\n📋 PAPER READY RESULTS:")
        print("-" * 30)
        print("✅ All metrics calculated on full test set")
        print("✅ Rank distribution plots generated")
        print("✅ Results saved to file")
        print("✅ Ready for paper submission")
        print()
        print("📊 Use these results in your paper:")
        print("   - Recall@K values (both directions)")
        print("   - MRR values (both directions)")
        print("   - Mean/Median rank values")
        print("   - Rank distribution plots as figures")
    else:
        print("\n❌ Full evaluation failed. Check error messages above.")

if __name__ == "__main__":
    main() 