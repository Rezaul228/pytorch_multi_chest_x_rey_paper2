#!/usr/bin/env python3
"""
Evaluate all trained models with different seeds and generate paper-ready results.
"""

import os
import json
import pandas as pd
from datetime import datetime
import argparse
import sys
import pickle

# Add current directory to path for imports
sys.path.append('.')

from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config
import torch

def load_tokenizer_from_metadata():
    """Load the tokenizer from metadata based on current dataset mode"""
    import paths
    shard_subfolder = "mimic_shards_hybrid_full_ori"
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
    
    return model

def load_test_data(num_samples=10000):
    """Load test dataset with specified number of samples"""
    print(f"📊 Loading test dataset with {num_samples} samples...")
    
    # Initialize data loader
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder="mimic_shards_hybrid_full_ori"
    )
    
    # Load tokenizer
    data_loader.tokenizer = load_tokenizer_from_metadata()
    
    # Load test data
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Get test dataset with specified number of samples
    test_dataset = data_loader.get_test_data(num_samples=num_samples)
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    return test_dataset, data_loader

def evaluate_model(model_path, config_name, output_dir, seed, num_samples=10000):
    """Evaluate a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating model with seed {seed}")
    print(f"Model: {model_path}")
    print(f"Test samples: {num_samples}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        model = load_trained_model(model_path)
        
        # Load test data
        test_dataset, data_loader = load_test_data(num_samples)
        
        # Run evaluation
        results = evaluate_cross_modal_retrieval_streaming(
            model=model,
            test_dataset=test_dataset,
            k_values=[1, 5, 10],
            batch_size=32,
            visualize=False,
            num_vis_examples=0,
            create_rank_plots=True,
            output_dir=output_dir
        )
        
        print(f"✅ Evaluation completed for seed {seed}")
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed for seed {seed}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_metrics_from_results(results):
    """Extract metrics from evaluation results."""
    if not results:
        return None
    
    metrics = {}
    try:
        # Extract Image-to-Text metrics
        metrics['i2t_recall_at_1'] = results.get('i2t_recall@1', 0)
        metrics['i2t_recall_at_5'] = results.get('i2t_recall@5', 0)
        metrics['i2t_recall_at_10'] = results.get('i2t_recall@10', 0)
        metrics['i2t_mrr'] = results.get('i2t_mrr', 0)
        metrics['i2t_mean_rank'] = results.get('i2t_mean_rank', 0)
        metrics['i2t_median_rank'] = results.get('i2t_median_rank', 0)
        
        # Extract Text-to-Image metrics
        metrics['t2i_recall_at_1'] = results.get('t2i_recall@1', 0)
        metrics['t2i_recall_at_5'] = results.get('t2i_recall@5', 0)
        metrics['t2i_recall_at_10'] = results.get('t2i_recall@10', 0)
        metrics['t2i_mrr'] = results.get('t2i_mrr', 0)
        metrics['t2i_mean_rank'] = results.get('t2i_mean_rank', 0)
        metrics['t2i_median_rank'] = results.get('t2i_median_rank', 0)
        
        # Extract average metrics
        metrics['avg_recall_at_1'] = results.get('avg_recall@1', 0)
        metrics['avg_recall_at_5'] = results.get('avg_recall@5', 0)
        metrics['avg_recall_at_10'] = results.get('avg_recall@10', 0)
        metrics['avg_mrr'] = results.get('avg_mrr', 0)
        metrics['avg_mean_rank'] = results.get('avg_mean_rank', 0)
        metrics['avg_median_rank'] = results.get('avg_median_rank', 0)
                
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return None
    
    return metrics

def generate_paper_results(all_results, output_file):
    """Generate paper-ready results summary."""
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate statistics for Image-to-Text
    i2t_stats = {
        'Mean Recall@1': df['i2t_recall_at_1'].mean(),
        'Std Recall@1': df['i2t_recall_at_1'].std(),
        'Mean Recall@5': df['i2t_recall_at_5'].mean(),
        'Std Recall@5': df['i2t_recall_at_5'].std(),
        'Mean Recall@10': df['i2t_recall_at_10'].mean(),
        'Std Recall@10': df['i2t_recall_at_10'].std(),
        'Mean MRR': df['i2t_mrr'].mean(),
        'Std MRR': df['i2t_mrr'].std(),
        'Mean Rank': df['i2t_mean_rank'].mean(),
        'Std Mean Rank': df['i2t_mean_rank'].std(),
        'Median Rank': df['i2t_median_rank'].median(),
        'Std Median Rank': df['i2t_median_rank'].std(),
    }
    
    # Calculate statistics for Text-to-Image
    t2i_stats = {
        'Mean Recall@1': df['t2i_recall_at_1'].mean(),
        'Std Recall@1': df['t2i_recall_at_1'].std(),
        'Mean Recall@5': df['t2i_recall_at_5'].mean(),
        'Std Recall@5': df['t2i_recall_at_5'].std(),
        'Mean Recall@10': df['t2i_recall_at_10'].mean(),
        'Std Recall@10': df['t2i_recall_at_10'].std(),
        'Mean MRR': df['t2i_mrr'].mean(),
        'Std MRR': df['t2i_mrr'].std(),
        'Mean Rank': df['t2i_mean_rank'].mean(),
        'Std Mean Rank': df['t2i_mean_rank'].std(),
        'Median Rank': df['t2i_median_rank'].median(),
        'Std Median Rank': df['t2i_median_rank'].std(),
    }
    
    # Calculate statistics for Average
    avg_stats = {
        'Mean Recall@1': df['avg_recall_at_1'].mean(),
        'Std Recall@1': df['avg_recall_at_1'].std(),
        'Mean Recall@5': df['avg_recall_at_5'].mean(),
        'Std Recall@5': df['avg_recall_at_5'].std(),
        'Mean Recall@10': df['avg_recall_at_10'].mean(),
        'Std Recall@10': df['avg_recall_at_10'].std(),
        'Mean MRR': df['avg_mrr'].mean(),
        'Std MRR': df['avg_mrr'].std(),
        'Mean Rank': df['avg_mean_rank'].mean(),
        'Std Mean Rank': df['avg_mean_rank'].std(),
        'Median Rank': df['avg_median_rank'].median(),
        'Std Median Rank': df['avg_median_rank'].std(),
    }
    
    # Generate report
    report = f"""
# Multi-Seed Model Evaluation Results
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Individual Seed Results
{df.to_string(index=False)}

## Statistical Summary (Mean ± Std)

### Image-to-Text Retrieval
- Recall@1: {i2t_stats['Mean Recall@1']:.4f} ± {i2t_stats['Std Recall@1']:.4f}
- Recall@5: {i2t_stats['Mean Recall@5']:.4f} ± {i2t_stats['Std Recall@5']:.4f}
- Recall@10: {i2t_stats['Mean Recall@10']:.4f} ± {i2t_stats['Std Recall@10']:.4f}
- MRR: {i2t_stats['Mean MRR']:.4f} ± {i2t_stats['Std MRR']:.4f}
- Mean Rank: {i2t_stats['Mean Rank']:.2f} ± {i2t_stats['Std Mean Rank']:.2f}
- Median Rank: {i2t_stats['Median Rank']:.2f} ± {i2t_stats['Std Median Rank']:.2f}

### Text-to-Image Retrieval
- Recall@1: {t2i_stats['Mean Recall@1']:.4f} ± {t2i_stats['Std Recall@1']:.4f}
- Recall@5: {t2i_stats['Mean Recall@5']:.4f} ± {t2i_stats['Std Recall@5']:.4f}
- Recall@10: {t2i_stats['Mean Recall@10']:.4f} ± {t2i_stats['Std Recall@10']:.4f}
- MRR: {t2i_stats['Mean MRR']:.4f} ± {t2i_stats['Std MRR']:.4f}
- Mean Rank: {t2i_stats['Mean Rank']:.2f} ± {t2i_stats['Std Mean Rank']:.2f}
- Median Rank: {t2i_stats['Median Rank']:.2f} ± {t2i_stats['Std Median Rank']:.2f}

### Average Performance (Both Directions)
- Recall@1: {avg_stats['Mean Recall@1']:.4f} ± {avg_stats['Std Recall@1']:.4f}
- Recall@5: {avg_stats['Mean Recall@5']:.4f} ± {avg_stats['Std Recall@5']:.4f}
- Recall@10: {avg_stats['Mean Recall@10']:.4f} ± {avg_stats['Std Recall@10']:.4f}
- MRR: {avg_stats['Mean MRR']:.4f} ± {avg_stats['Std MRR']:.4f}
- Mean Rank: {avg_stats['Mean Rank']:.2f} ± {avg_stats['Std Mean Rank']:.2f}
- Median Rank: {avg_stats['Median Rank']:.2f} ± {avg_stats['Std Median Rank']:.2f}

## Best Performance per Metric

### Image-to-Text
- Best Recall@1: {df['i2t_recall_at_1'].max():.4f} (Seed: {df.loc[df['i2t_recall_at_1'].idxmax(), 'seed']})
- Best Recall@5: {df['i2t_recall_at_5'].max():.4f} (Seed: {df.loc[df['i2t_recall_at_5'].idxmax(), 'seed']})
- Best Recall@10: {df['i2t_recall_at_10'].max():.4f} (Seed: {df.loc[df['i2t_recall_at_10'].idxmax(), 'seed']})
- Best MRR: {df['i2t_mrr'].max():.4f} (Seed: {df.loc[df['i2t_mrr'].idxmax(), 'seed']})

### Text-to-Image
- Best Recall@1: {df['t2i_recall_at_1'].max():.4f} (Seed: {df.loc[df['t2i_recall_at_1'].idxmax(), 'seed']})
- Best Recall@5: {df['t2i_recall_at_5'].max():.4f} (Seed: {df.loc[df['t2i_recall_at_5'].idxmax(), 'seed']})
- Best Recall@10: {df['t2i_recall_at_10'].max():.4f} (Seed: {df.loc[df['t2i_recall_at_10'].idxmax(), 'seed']})
- Best MRR: {df['t2i_mrr'].max():.4f} (Seed: {df.loc[df['t2i_mrr'].idxmax(), 'seed']})

## Reproducibility Analysis
- Standard Deviation of Average Recall@1: {avg_stats['Std Recall@1']:.4f}
- Coefficient of Variation (Average Recall@1): {(avg_stats['Std Recall@1']/avg_stats['Mean Recall@1'])*100:.2f}%
- Model Stability: {'High' if avg_stats['Std Recall@1'] < 0.01 else 'Medium' if avg_stats['Std Recall@1'] < 0.02 else 'Low'}

## Paper-Ready Results
For your research paper, you can report:

**Primary Results (Mean ± Standard Deviation):**

### Image-to-Text Retrieval
- Recall@1: {i2t_stats['Mean Recall@1']:.4f} ± {i2t_stats['Std Recall@1']:.4f}
- Recall@5: {i2t_stats['Mean Recall@5']:.4f} ± {i2t_stats['Std Recall@5']:.4f}
- Recall@10: {i2t_stats['Mean Recall@10']:.4f} ± {i2t_stats['Std Recall@10']:.4f}
- Mean Reciprocal Rank (MRR): {i2t_stats['Mean MRR']:.4f} ± {i2t_stats['Std MRR']:.4f}

### Text-to-Image Retrieval
- Recall@1: {t2i_stats['Mean Recall@1']:.4f} ± {t2i_stats['Std Recall@1']:.4f}
- Recall@5: {t2i_stats['Mean Recall@5']:.4f} ± {t2i_stats['Std Recall@5']:.4f}
- Recall@10: {t2i_stats['Mean Recall@10']:.4f} ± {t2i_stats['Std Recall@10']:.4f}
- Mean Reciprocal Rank (MRR): {t2i_stats['Mean MRR']:.4f} ± {t2i_stats['Std MRR']:.4f}

### Overall Performance
- Average Recall@1: {avg_stats['Mean Recall@1']:.4f} ± {avg_stats['Std Recall@1']:.4f}
- Average Recall@5: {avg_stats['Mean Recall@5']:.4f} ± {avg_stats['Std Recall@5']:.4f}
- Average Recall@10: {avg_stats['Mean Recall@10']:.4f} ± {avg_stats['Std Recall@10']:.4f}
- Average MRR: {avg_stats['Mean MRR']:.4f} ± {avg_stats['Std MRR']:.4f}

**Reproducibility Statement:**
"We evaluated our model across {len(df)} different random seeds to ensure reproducibility. The standard deviation of our primary metric (Average Recall@1) is {avg_stats['Std Recall@1']:.4f}, indicating {'high' if avg_stats['Std Recall@1'] < 0.01 else 'moderate' if avg_stats['Std Recall@1'] < 0.02 else 'low'} reproducibility."
"""
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Paper-ready results saved to: {output_file}")
    print(f"📈 Statistical summary generated with {len(df)} seed evaluations")

def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained models with different seeds')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='List of model paths to evaluate')
    parser.add_argument('--seeds', type=str, default='3407,2021,123,42,17',
                       help='Comma-separated list of seed names corresponding to model paths')
    parser.add_argument('--config_name', type=str, required=True,
                       help='Configuration name for the models')
    parser.add_argument('--output_base_dir', type=str, default='multi_seed_evaluation',
                       help='Base directory for evaluation results')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of test samples to use')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [s.strip() for s in args.seeds.split(',')]
    
    # Validate that we have the same number of model paths and seeds
    if len(args.model_paths) != len(seeds):
        print(f"❌ Error: Number of model paths ({len(args.model_paths)}) doesn't match number of seeds ({len(seeds)})")
        return
    
    # Create base output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    all_results = []
    
    # Evaluate each model
    for i, (model_path, seed) in enumerate(zip(args.model_paths, seeds)):
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        # Create output directory for this seed
        output_dir = os.path.join(args.output_base_dir, f"seed_{seed}")
        
        # Evaluate model
        results = evaluate_model(
            model_path=model_path,
            config_name=args.config_name,
            output_dir=output_dir,
            seed=seed,
            num_samples=args.num_samples
        )
        
        if results:
            # Extract metrics
            metrics = extract_metrics_from_results(results)
            if metrics:
                metrics['seed'] = seed
                metrics['model_path'] = model_path
                all_results.append(metrics)
    
    # Generate paper-ready results
    if all_results:
        output_file = os.path.join(args.output_base_dir, 'paper_results_summary.txt')
        generate_paper_results(all_results, output_file)
        
        # Also save as JSON for further analysis
        json_file = os.path.join(args.output_base_dir, 'all_results.json')
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n🎉 Evaluation completed for {len(all_results)} models!")
        print(f"📄 Paper results: {output_file}")
        print(f"📊 JSON results: {json_file}")
    else:
        print("❌ No successful evaluations completed!")

if __name__ == "__main__":
    main() 