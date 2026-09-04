#!/usr/bin/env python3
"""
Separate Modality Correlation Analysis
=====================================

This script addresses the supervisor's TODO to separate text and image features
since they are different modalities. It creates 4 separate correlation analyses:

1. Image-Image: synergy_img vs diff_img correlations
2. Text-Text: synergy_txt vs diff_txt correlations  
3. Cross-Modal Image-Text: synergy_img vs diff_txt correlations
4. Cross-Modal Text-Image: synergy_txt vs diff_img correlations

This provides more detailed insights into how each modality behaves
in the dual-branch architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import your model and data loading functions
from base_models_refactored_v1 import MultimodalFusion
# from data_loader_v1 import load_test_data
from config import get_vocab_size, get_embed_dim, get_current_config

def load_tokenizer_from_metadata(shard_subfolder="mimic_shards_hybrid_full_ori"):
    """Load tokenizer from metadata file"""
    import pickle
    metadata_path = f"/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/data/{shard_subfolder}/metadata.pkl"
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata['tokenizer']

def create_separate_modality_correlation_plots(synergy_img, diff_img, synergy_txt, diff_txt, timestamp):
    """Create 4 separate correlation analyses for different modality combinations"""
    
    # Set Scientific Reports style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Create output directory
    os.makedirs('modality_correlation_analysis', exist_ok=True)
    
    # Define the 4 correlation analyses
    analyses = [
        {
            'name': 'Image-Image',
            'synergy': synergy_img,
            'difference': diff_img,
            'description': 'Synergy vs Difference Image Embeddings',
            'filename': 'image_image_correlation'
        },
        {
            'name': 'Text-Text', 
            'synergy': synergy_txt,
            'difference': diff_txt,
            'description': 'Synergy vs Difference Text Embeddings',
            'filename': 'text_text_correlation'
        },
        {
            'name': 'Cross-Modal Image-Text',
            'synergy': synergy_img,
            'difference': diff_txt,
            'description': 'Synergy Image vs Difference Text Embeddings',
            'filename': 'image_text_correlation'
        },
        {
            'name': 'Cross-Modal Text-Image',
            'synergy': synergy_txt,
            'difference': diff_img,
            'description': 'Synergy Text vs Difference Image Embeddings',
            'filename': 'text_image_correlation'
        }
    ]
    
    results_summary = {}
    
    for analysis in analyses:
        print(f"\n📊 Analyzing {analysis['name']} correlations...")
        
        # Calculate correlation matrix
        synergy_data = analysis['synergy']
        diff_data = analysis['difference']
        
        # Ensure same number of samples
        min_samples = min(synergy_data.shape[0], diff_data.shape[0])
        synergy_data = synergy_data[:min_samples]
        diff_data = diff_data[:min_samples]
        
        # Calculate cross-branch correlation matrix
        cross_correlation_matrix = np.corrcoef(synergy_data.T, diff_data.T)
        
        # Extract cross-branch section (top-right quadrant)
        n_features = synergy_data.shape[1]
        cross_section = cross_correlation_matrix[:n_features, n_features:]
        
        # Flatten correlation values
        corr_values = cross_section.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        corr_values = corr_values[corr_values != 1.0]  # Remove self-correlations
        
        # Calculate statistics
        mean_corr = np.mean(corr_values)
        std_corr = np.std(corr_values)
        
        # Categorize correlations
        very_low = np.sum(np.abs(corr_values) <= 0.1)
        low = np.sum((np.abs(corr_values) > 0.1) & (np.abs(corr_values) <= 0.2))
        medium = np.sum((np.abs(corr_values) > 0.2) & (np.abs(corr_values) <= 0.5))
        high = np.sum(np.abs(corr_values) > 0.5)
        
        total_pairs = len(corr_values)
        independent_pairs = very_low + low
        success_rate = (independent_pairs / total_pairs) * 100
        
        # Store results
        results_summary[analysis['name']] = {
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'success_rate': success_rate,
            'very_low': very_low,
            'low': low,
            'medium': medium,
            'high': high,
            'total_pairs': total_pairs
        }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Panel 1: Histogram
        ax1.hist(corr_values, bins=30, color='#4A90E2', edgecolor='#2E5BBA', alpha=0.8, linewidth=1.2)
        
        # Format mean correlation with appropriate precision
        if abs(mean_corr) < 0.001:
            mean_corr_str = f"{mean_corr:.6f}"
        else:
            mean_corr_str = f"{mean_corr:.3f}"
        
        ax1.axvline(mean_corr, color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean: {mean_corr_str}')
        ax1.axvline(-mean_corr, color='#E74C3C', linestyle='--', linewidth=2)
        ax1.axvline(0.2, color='#F39C12', linestyle=':', linewidth=2, label='Target: ±0.2')
        ax1.axvline(-0.2, color='#F39C12', linestyle=':', linewidth=2)
        
        ax1.set_xlabel('Correlation Coefficient', fontsize=8)
        ax1.set_ylabel('Frequency', fontsize=8)
        ax1.set_title(f'{analysis["name"]} Correlation Distribution', fontsize=9, fontweight='bold')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Breakdown
        categories = ['Very Low', 'Low', 'Medium', 'High']
        counts = [very_low, low, medium, high]
        percentages = [count/total_pairs*100 for count in counts]
        
        colors = ['#27AE60', '#58D68D', '#F39C12', '#E74C3C']
        bars = ax2.bar(categories, percentages, color=colors, alpha=0.9, edgecolor='#2C3E50', linewidth=1.2)
        ax2.set_ylabel('Percentage (%)', fontsize=8)
        ax2.set_title(f'{analysis["name"]} Correlation Breakdown', fontsize=9, fontweight='bold')
        
        # Add percentage values inside bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{percentage:.1f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Add statistics to panels
        legend_text = f"Mean Corr: {mean_corr_str}\nIndependent Pairs: {success_rate:.1f}%\nTarget: <0.2"
        ax1.text(0.02, 0.98, legend_text, transform=ax1.transAxes, fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'),
                 verticalalignment='top', fontfamily='monospace')
        
        success_text = f"Success Rate: {success_rate:.1f}%\n(≤0.2 correlation)\n\nVery Low: {very_low} pairs\nLow: {low} pairs\nMedium: {medium} pairs\nHigh: {high} pairs"
        ax2.text(0.02, 0.98, success_text, transform=ax2.transAxes, fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='#cccccc'),
                 verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'modality_correlation_analysis/{analysis["filename"]}_{timestamp}.png',
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"   ✅ {analysis['name']} correlation analysis saved")
        print(f"   📊 Mean correlation: {mean_corr_str}")
        print(f"   📊 Success rate: {success_rate:.1f}%")
    
    # Create summary comparison plot
    create_modality_comparison_plot(results_summary, timestamp)
    
    return results_summary

def create_modality_comparison_plot(results_summary, timestamp):
    """Create a comparison plot showing all modality analyses"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Mean correlations comparison
    modalities = list(results_summary.keys())
    mean_corrs = [results_summary[mod]['mean_correlation'] for mod in modalities]
    success_rates = [results_summary[mod]['success_rate'] for mod in modalities]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars1 = ax1.bar(modalities, mean_corrs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Mean Correlation', fontsize=8)
    ax1.set_title('Mean Correlation by Modality', fontsize=9, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.2, color='red', linestyle='--', linewidth=1, label='Target: ±0.2')
    ax1.axhline(y=-0.2, color='red', linestyle='--', linewidth=1)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars1, mean_corrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=7)
    
    # Panel 2: Success rates comparison
    bars2 = ax2.bar(modalities, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Success Rate (%)', fontsize=8)
    ax2.set_title('Independence Success Rate by Modality', fontsize=9, fontweight='bold')
    ax2.axhline(y=50, color='green', linestyle='--', linewidth=1, label='Target: >50%')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars2, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'modality_correlation_analysis/modality_comparison_{timestamp}.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"   ✅ Modality comparison plot saved")

def print_detailed_results(results_summary):
    """Print detailed results for paper writing"""
    
    print(f"\n{'='*80}")
    print("📊 SEPARATE MODALITY CORRELATION ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    for modality, results in results_summary.items():
        print(f"\n🔍 {modality.upper()} ANALYSIS:")
        print(f"   Mean Correlation: {results['mean_correlation']:.6f}")
        print(f"   Std Correlation:  {results['std_correlation']:.6f}")
        print(f"   Success Rate:     {results['success_rate']:.1f}%")
        print(f"   Total Pairs:      {results['total_pairs']:,}")
        print(f"   Very Low (≤0.1):  {results['very_low']:,} ({results['very_low']/results['total_pairs']*100:.1f}%)")
        print(f"   Low (0.1-0.2):    {results['low']:,} ({results['low']/results['total_pairs']*100:.1f}%)")
        print(f"   Medium (0.2-0.5): {results['medium']:,} ({results['medium']/results['total_pairs']*100:.1f}%)")
        print(f"   High (>0.5):      {results['high']:,} ({results['high']/results['total_pairs']*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("📝 PAPER WRITING INSIGHTS:")
    print(f"{'='*80}")
    
    # Find best and worst performing modalities
    best_success = max(results_summary.items(), key=lambda x: x[1]['success_rate'])
    worst_success = min(results_summary.items(), key=lambda x: x[1]['success_rate'])
    
    print(f"\n✅ BEST PERFORMING: {best_success[0]}")
    print(f"   Success Rate: {best_success[1]['success_rate']:.1f}%")
    print(f"   Mean Correlation: {best_success[1]['mean_correlation']:.6f}")
    
    print(f"\n⚠️  CHALLENGING: {worst_success[0]}")
    print(f"   Success Rate: {worst_success[1]['success_rate']:.1f}%")
    print(f"   Mean Correlation: {worst_success[1]['mean_correlation']:.6f}")
    
    # Calculate overall statistics
    all_means = [r['mean_correlation'] for r in results_summary.values()]
    all_success_rates = [r['success_rate'] for r in results_summary.values()]
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Average Mean Correlation: {np.mean(all_means):.6f}")
    print(f"   Average Success Rate: {np.mean(all_success_rates):.1f}%")
    print(f"   Correlation Range: {np.min(all_means):.6f} to {np.max(all_means):.6f}")
    print(f"   Success Rate Range: {np.min(all_success_rates):.1f}% to {np.max(all_success_rates):.1f}%")

def main():
    """Main function to run separate modality correlation analysis"""
    
    print("🔬 SEPARATE MODALITY CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Load model
    print("\n📁 Loading trained model...")
    model_path = input("Enter the path to your trained model (.pth file): ").strip()
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load model
    vocab_size = get_vocab_size()
    embed_dim = get_embed_dim()
    config = get_current_config()
    
    model = MultimodalFusion(vocab_size, embed_dim, config["num_heads"], config["num_layers"])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Load test data
    print("\n📊 Loading test data...")
    # Use the same data loading as main script
    from data_loader_v1 import IndianaDataLoader
    data_loader = IndianaDataLoader("mimic_shards_hybrid_full_ori")
    data_loader.load_data(max_samples=640)
    test_dataset = data_loader.get_test_data(num_samples=640)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print("✅ Test data loaded")
    
    # Extract embeddings
    print("\n🔍 Extracting branch embeddings...")
    synergy_img_embeddings = []
    diff_img_embeddings = []
    synergy_txt_embeddings = []
    diff_txt_embeddings = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 20:  # Limit to 640 samples (20 batches × 32)
                break
                
            # Handle batch format
            if isinstance(batch, dict):
                images = batch['images']
                texts = batch['captions']
            else:
                images, texts = batch
            
            # Convert images to BCHW format if needed
            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)
            
            # Get branch embeddings
            _, _, synergy_img, synergy_txt, diff_img, diff_txt = model(
                (images, texts), training=False, return_branch_embeddings=True
            )
            
            synergy_img_embeddings.append(synergy_img.cpu().numpy())
            diff_img_embeddings.append(diff_img.cpu().numpy())
            synergy_txt_embeddings.append(synergy_txt.cpu().numpy())
            diff_txt_embeddings.append(diff_txt.cpu().numpy())
    
    # Concatenate all embeddings
    synergy_img = np.vstack(synergy_img_embeddings)
    diff_img = np.vstack(diff_img_embeddings)
    synergy_txt = np.vstack(synergy_txt_embeddings)
    diff_txt = np.vstack(diff_txt_embeddings)
    
    print(f"✅ Extracted embeddings:")
    print(f"   Synergy Image: {synergy_img.shape}")
    print(f"   Difference Image: {diff_img.shape}")
    print(f"   Synergy Text: {synergy_txt.shape}")
    print(f"   Difference Text: {diff_txt.shape}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run separate modality correlation analysis
    print(f"\n📊 Running separate modality correlation analysis...")
    results = create_separate_modality_correlation_plots(
        synergy_img, diff_img, synergy_txt, diff_txt, timestamp
    )
    
    # Print detailed results
    print_detailed_results(results)
    
    print(f"\n✅ Analysis complete! Results saved in 'modality_correlation_analysis/' directory")
    print(f"📁 Timestamp: {timestamp}")

if __name__ == "__main__":
    main()
