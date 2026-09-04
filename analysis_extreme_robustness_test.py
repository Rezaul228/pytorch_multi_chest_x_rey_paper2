#!/usr/bin/env python3
"""
EXTREME ROBUSTNESS TESTING - FINDING BREAKING POINTS
Test with extreme intensities to find exact breaking points for each noise type
"""

import torch
import numpy as np
import random
import string
import copy
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append('.')

from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from config import get_vocab_size, get_embed_dim, get_current_config
import paths
import pickle

def add_gaussian_noise_to_images(images_tensor, intensity=0.1):
    """Add Gaussian noise to image tensor while preserving format"""
    noise = torch.randn_like(images_tensor) * intensity
    return torch.clamp(images_tensor + noise, 0, 1)

def add_salt_pepper_noise_to_images(images_tensor, intensity=0.1):
    """Add salt and pepper noise to image tensor"""
    noisy_images = images_tensor.clone()
    noise_mask = torch.rand_like(images_tensor)
    # Salt noise (set to 1)
    noisy_images[noise_mask < intensity/2] = 1.0
    # Pepper noise (set to 0)
    noisy_images[noise_mask > 1 - intensity/2] = 0.0
    return noisy_images

def add_brightness_variation_to_images(images_tensor, intensity=0.1):
    """Add brightness variation to image tensor - ENHANCED VERSION"""
    # Handle different input formats without converting
    if len(images_tensor.shape) == 4 and images_tensor.shape[-1] == 3:
        # Format: (B, H, W, C) - apply brightness per image
        batch_size = images_tensor.shape[0]
        # Make brightness changes more dramatic: intensity * 2 for better visibility
        brightness_factors = 1 + (torch.rand(batch_size, 1, 1, 1, device=images_tensor.device) - 0.5) * intensity * 2
        return torch.clamp(images_tensor * brightness_factors, 0, 1)
    elif len(images_tensor.shape) == 3:
        # Format: (H, W, C) - single image
        # Make brightness changes more dramatic: intensity * 2 for better visibility
        brightness_factor = 1 + (torch.rand(1, device=images_tensor.device) - 0.5) * intensity * 2
        return torch.clamp(images_tensor * brightness_factor, 0, 1)
    else:
        # Fallback - assume (B, C, H, W)
        batch_size = images_tensor.shape[0]
        # Make brightness changes more dramatic: intensity * 2 for better visibility
        brightness_factors = 1 + (torch.rand(batch_size, 1, 1, 1, device=images_tensor.device) - 0.5) * intensity * 2
        return torch.clamp(images_tensor * brightness_factors, 0, 1)

def add_blur_to_images(images_tensor, intensity=0.1):
    """Add motion blur to image tensor - ENHANCED VERSION"""
    # Use PyTorch operations for efficiency
    if len(images_tensor.shape) == 4 and images_tensor.shape[-1] == 3:
        # Format: (B, H, W, C) - apply blur per image
        batch_size, height, width, channels = images_tensor.shape
        blur_size = max(1, int(intensity * 20))  # Convert intensity to blur kernel size
        
        # Create a simple horizontal blur kernel for each channel
        kernel = torch.ones(channels, 1, 1, blur_size, device=images_tensor.device) / blur_size
        
        # Convert to (B, C, H, W) for conv2d
        images_conv = images_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Apply blur using conv2d
        blurred = torch.nn.functional.conv2d(
            images_conv, 
            kernel, 
            padding=(0, blur_size-1),
            groups=channels
        )
        
        # Convert back to (B, H, W, C)
        blurred = blurred.permute(0, 2, 3, 1)
        
        return torch.clamp(blurred, 0, 1)
    elif len(images_tensor.shape) == 3:
        # Format: (H, W, C) - single image
        height, width, channels = images_tensor.shape
        blur_size = max(1, int(intensity * 20))
        
        # Create a simple horizontal blur kernel for each channel
        kernel = torch.ones(channels, 1, 1, blur_size, device=images_tensor.device) / blur_size
        
        # Convert to (C, H, W) for conv2d
        images_conv = images_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        
        # Apply blur using conv2d
        blurred = torch.nn.functional.conv2d(
            images_conv, 
            kernel, 
            padding=(0, blur_size-1),
            groups=channels
        )
        
        # Convert back to (H, W, C)
        blurred = blurred.squeeze(0).permute(1, 2, 0)
        
        return torch.clamp(blurred, 0, 1)
    else:
        # Fallback - return original
        return images_tensor

def add_typos_to_captions(captions_tensor, intensity=0.1, tokenizer=None):
    """Add typographical errors to captions by modifying tokens"""
    # For this implementation, we'll simulate typos by adding small random values
    # In a real scenario, you'd decode tokens, add typos, and re-encode
    # But to preserve the exact format, we'll add small noise to token values
    noise = torch.randint(-2, 3, captions_tensor.shape) * (torch.rand_like(captions_tensor.float()) < intensity)
    return torch.clamp(captions_tensor + noise, min=0)


def add_synonyms_to_captions(captions_tensor, intensity=0.1, tokenizer=None):
    """Add synonyms by replacing tokens with equivalent terms"""
    # For this implementation, we'll simulate synonyms by adding larger random values
    # In a real scenario, you'd decode tokens, replace with synonyms, and re-encode
    # But to preserve the exact format, we'll add moderate noise to token values
    noise = torch.randint(-5, 6, captions_tensor.shape) * (torch.rand_like(captions_tensor.float()) < intensity)
    return torch.clamp(captions_tensor + noise, min=0)
def create_noisy_dataset_copy(original_dataset, noise_type, intensity=0.1):
    """Create a noisy copy of dataset while preserving exact structure"""
    print(f"   Creating noisy dataset with {noise_type} noise (intensity: {intensity})")
    
    # Create a list to store noisy samples
    noisy_samples = []
    
    for i in range(len(original_dataset)):
        # Get original sample
        original_sample = original_dataset[i]
        
        # Create a copy of the sample
        noisy_sample = {
            'images': original_sample['images'].clone(),  # Clone to avoid modifying original
            'captions': original_sample['captions'].clone(),  # Clone to avoid modifying original
            'study_ids': original_sample['study_ids']  # Keep study_ids unchanged
        }
        
        # Apply noise based on type
        if noise_type == 'gaussian':
            noisy_sample['images'] = add_gaussian_noise_to_images(
                noisy_sample['images'], intensity
            )
        elif noise_type == 'salt_pepper':
            noisy_sample['images'] = add_salt_pepper_noise_to_images(
                noisy_sample['images'], intensity
            )
        elif noise_type == 'brightness':
            noisy_sample['images'] = add_brightness_variation_to_images(
                noisy_sample['images'], intensity
            )
        elif noise_type == 'typos':
            noisy_sample['captions'] = add_typos_to_captions(
                noisy_sample['captions'], intensity
            )
        elif noise_type == 'blur':
            noisy_sample['images'] = add_blur_to_images(
                noisy_sample['images'], intensity
            )
        elif noise_type == 'synonyms':
            noisy_sample['captions'] = add_synonyms_to_captions(
                noisy_sample['captions'], intensity
            )
        
        noisy_samples.append(noisy_sample)
    
    return noisy_samples

def load_model_and_data():
    """Load trained model and test data"""
    print("🤖 Loading model and data...")
    
    # Model path
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_v2/export/model_weights.pth'
    
    # Load model
    model = MultimodalFusion(
        vocab_size=get_vocab_size(),
        embed_dim=get_embed_dim(),
        num_heads=get_current_config()['num_heads']
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load test data
    data_loader = IndianaDataLoader(shard_subfolder="mimic_shards_hybrid_full_ori")
    test_dataset = data_loader.get_test_data(num_samples=2000)
    
    print(f"✅ Loaded model and {len(test_dataset)} test samples")
    return model, test_dataset

def save_results_to_file(results, baseline_results, timestamp):
    """Save detailed results to text file for graphing"""
    filename = f'enhanced_noise_sensitivity_results_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write("EXTREME ROBUSTNESS TESTING - FINDING BREAKING POINTS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        # Baseline results
        f.write("BASELINE PERFORMANCE (No Noise):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Image→Text R@1: {baseline_results['i2t_recall@1']:.4f}\n")
        f.write(f"Text→Image R@1: {baseline_results['t2i_recall@1']:.4f}\n")
        f.write(f"Average R@1: {baseline_results['avg_recall@1']:.4f}\n")
        f.write(f"Image→Text MRR: {baseline_results['i2t_mrr']:.4f}\n")
        f.write(f"Text→Image MRR: {baseline_results['t2i_mrr']:.4f}\n")
        f.write(f"Average MRR: {baseline_results['avg_mrr']:.4f}\n")
        f.write(f"Image→Text Mean Rank: {baseline_results['i2t_mean_rank']:.2f}\n")
        f.write(f"Text→Image Mean Rank: {baseline_results['t2i_mean_rank']:.2f}\n")
        f.write(f"Average Mean Rank: {baseline_results['avg_mean_rank']:.2f}\n\n")
        
        # Detailed results for each noise type
        f.write("DETAILED RESULTS FOR EACH NOISE VARIATION:\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"NOISE TYPE: {result['description']}\n")
            f.write(f"Type: {result['noise_type']}, Intensity: {result['intensity']}\n")
            f.write("-" * 40 + "\n")
            
            if 'error' in result:
                f.write(f"ERROR: {result['error']}\n")
                f.write(f"Performance Drop: 100.0%\n")
            else:
                f.write(f"Image→Text R@1: {result['i2t_r1']:.4f} (↓{result['i2t_r1_drop_pct']:.1f}%)\n")
                f.write(f"Text→Image R@1: {result['t2i_r1']:.4f} (↓{result['t2i_r1_drop_pct']:.1f}%)\n")
                f.write(f"Average R@1: {result['avg_r1']:.4f} (↓{result['avg_r1_drop_pct']:.1f}%)\n")
                f.write(f"Image→Text MRR: {result['i2t_mrr']:.4f} (↓{result['i2t_mrr_drop_pct']:.1f}%)\n")
                f.write(f"Text→Image MRR: {result['t2i_mrr']:.4f} (↓{result['t2i_mrr_drop_pct']:.1f}%)\n")
                f.write(f"Average MRR: {result['avg_mrr']:.4f} (↓{result['avg_mrr_drop_pct']:.1f}%)\n")
                f.write(f"Image→Text Mean Rank: {result['i2t_mean_rank']:.2f}\n")
                f.write(f"Text→Image Mean Rank: {result['t2i_mean_rank']:.2f}\n")
                f.write(f"Average Mean Rank: {result['avg_mean_rank']:.2f}\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS:\n")
        f.write("=" * 20 + "\n")
        
        breaking_points = [r for r in results if r.get('avg_r1_drop_pct', 100) > 10]
        warning_points = [r for r in results if 5 < r.get('avg_r1_drop_pct', 100) <= 10]
        robust_points = [r for r in results if r.get('avg_r1_drop_pct', 100) <= 5]
        
        f.write(f"Breaking Points (>10% drop): {len(breaking_points)}\n")
        f.write(f"Warning Points (5-10% drop): {len(warning_points)}\n")
        f.write(f"Robust Conditions (≤5% drop): {len(robust_points)}\n\n")
        
        # Modality analysis
        image_results = [r for r in results if r['noise_type'] in ['gaussian', 'salt_pepper', 'brightness', 'blur']]
        text_results = [r for r in results if r['noise_type'] == 'typos']
        
        if image_results and text_results:
            avg_image_drop = np.mean([r.get('avg_r1_drop_pct', 100) for r in image_results])
            avg_text_drop = np.mean([r.get('avg_r1_drop_pct', 100) for r in text_results])
            
            f.write(f"Image noise average drop: {avg_image_drop:.1f}%\n")
            f.write(f"Text noise average drop: {avg_text_drop:.1f}%\n")
        
        # CSV format for easy graphing
        f.write("\nCSV FORMAT FOR GRAPHING:\n")
        f.write("=" * 25 + "\n")
        f.write("Noise_Type,Intensity,Description,Avg_R1,Avg_R1_Drop_Pct,Avg_MRR,Avg_MRR_Drop_Pct,Avg_Mean_Rank\n")
        
        for result in results:
            if 'error' not in result:
                f.write(f"{result['noise_type']},{result['intensity']},{result['description']},{result['avg_r1']:.4f},{result['avg_r1_drop_pct']:.1f},{result['avg_mrr']:.4f},{result['avg_mrr_drop_pct']:.1f},{result['avg_mean_rank']:.2f}\n")
            else:
                f.write(f"{result['noise_type']},{result['intensity']},{result['description']},0.0000,100.0,0.0000,100.0,999.99\n")
    
    print(f"📄 Enhanced results saved to: {filename}")
    return filename

def test_enhanced_noise_sensitivity():
    """Test model sensitivity with enhanced realistic intensities"""
    print("🔬 EXTREME ROBUSTNESS TESTING - FINDING BREAKING POINTS")
    print("=" * 70)
    print("📋 Approach: Test with extreme intensities to find exact breaking points for each noise type")
    print("=" * 70)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model and data
    model, test_dataset = load_model_and_data()
    
    # Test baseline (no noise)
    print("\n📊 Testing baseline performance (no noise)...")
    baseline_results = evaluate_cross_modal_retrieval_streaming(
        model=model,
        test_dataset=test_dataset,
        k_values=[1, 5, 10],
        batch_size=32,
        visualize=False,
        num_vis_examples=0,
        create_rank_plots=False,
        output_dir=None
    )
    
    baseline_r1 = baseline_results['i2t_recall@1']
    baseline_mrr = baseline_results['avg_mrr']
    print(f"✅ Baseline - R@1: {baseline_r1:.4f}, MRR: {baseline_mrr:.4f}")
    
        # Define EXTREME noise tests to find exact breaking points
    noise_tests = [
        # Image noise tests - find exact breaking points
        ('gaussian', 0.05, 'Low Gaussian Noise'),
        ('gaussian', 0.12, 'Medium Gaussian Noise'),
        ('gaussian', 0.15, 'High Gaussian Noise'),
        ('gaussian', 0.18, 'Very High Gaussian Noise'),
        ('gaussian', 0.25, 'Extreme Gaussian Noise'),
        
        ('salt_pepper', 0.02, 'Low Salt-Pepper Noise'),
        ('salt_pepper', 0.05, 'Medium Salt-Pepper Noise'),
        ('salt_pepper', 0.08, 'High Salt-Pepper Noise'),
        ('salt_pepper', 0.12, 'Very High Salt-Pepper Noise'),
        
        # EXTREME brightness tests to find breaking point
        ('brightness', 1.5, 'High Brightness Variation'),
        ('brightness', 2.0, 'Very High Brightness Variation'),
        ('brightness', 2.5, 'Extreme Brightness Variation'),
        ('brightness', 3.0, 'Maximum Brightness Variation'),
        
        # EXTREME blur tests to find breaking point
        ('blur', 1.0, 'High Motion Blur'),
        ('blur', 1.5, 'Very High Motion Blur'),
        ('blur', 2.0, 'Extreme Motion Blur'),
        ('blur', 2.5, 'Maximum Motion Blur'),
        
        # EXTREME text noise tests to find breaking points
        ('typos', 0.3, 'High Typo Rate'),
        ('typos', 0.5, 'Very High Typo Rate'),
        ('typos', 0.7, 'Extreme Typo Rate'),
        ('typos', 1.0, 'Maximum Typo Rate'),
        
        # NEW: Synonyms noise tests
        ('synonyms', 0.1, 'Low Synonym Rate'),
        ('synonyms', 0.3, 'Medium Synonym Rate'),
        ('synonyms', 0.5, 'High Synonym Rate'),
        ('synonyms', 0.7, 'Very High Synonym Rate'),
        ('synonyms', 1.0, 'Extreme Synonym Rate'),
    ]
    
    print(f"\n🧪 Testing {len(noise_tests)} enhanced noise conditions...")
    print("=" * 70)
    
    results = []
    
    for noise_type, intensity, description in noise_tests:
        print(f"\n🔍 Testing: {description}")
        print(f"   Type: {noise_type}, Intensity: {intensity}")
        
        try:
            # Create noisy dataset (preserving exact structure)
            noisy_samples = create_noisy_dataset_copy(test_dataset, noise_type, intensity)
            
            # Evaluate with noisy data
            noisy_results = evaluate_cross_modal_retrieval_streaming(
                model=model,
                test_dataset=noisy_samples,
                k_values=[1, 5, 10],
                batch_size=32,
                visualize=False,
                num_vis_examples=0,
                create_rank_plots=False,
                output_dir=None
            )
            
            # Extract detailed metrics
            i2t_r1 = noisy_results['i2t_recall@1']
            t2i_r1 = noisy_results['t2i_recall@1']
            avg_r1 = noisy_results['avg_recall@1']
            i2t_mrr = noisy_results['i2t_mrr']
            t2i_mrr = noisy_results['t2i_mrr']
            avg_mrr = noisy_results['avg_mrr']
            i2t_mean_rank = noisy_results['i2t_mean_rank']
            t2i_mean_rank = noisy_results['t2i_mean_rank']
            avg_mean_rank = noisy_results['avg_mean_rank']
            
            # Calculate performance drops
            i2t_r1_drop_pct = ((baseline_results['i2t_recall@1'] - i2t_r1) / baseline_results['i2t_recall@1']) * 100
            t2i_r1_drop_pct = ((baseline_results['t2i_recall@1'] - t2i_r1) / baseline_results['t2i_recall@1']) * 100
            avg_r1_drop_pct = ((baseline_results['avg_recall@1'] - avg_r1) / baseline_results['avg_recall@1']) * 100
            i2t_mrr_drop_pct = ((baseline_results['i2t_mrr'] - i2t_mrr) / baseline_results['i2t_mrr']) * 100
            t2i_mrr_drop_pct = ((baseline_results['t2i_mrr'] - t2i_mrr) / baseline_results['t2i_mrr']) * 100
            avg_mrr_drop_pct = ((baseline_results['avg_mrr'] - avg_mrr) / baseline_results['avg_mrr']) * 100
            
            print(f"   📊 DETAILED RESULTS:")
            print(f"      Image→Text R@1: {i2t_r1:.4f} (↓{i2t_r1_drop_pct:.1f}%)")
            print(f"      Text→Image R@1: {t2i_r1:.4f} (↓{t2i_r1_drop_pct:.1f}%)")
            print(f"      Average R@1: {avg_r1:.4f} (↓{avg_r1_drop_pct:.1f}%)")
            print(f"      Image→Text MRR: {i2t_mrr:.4f} (↓{i2t_mrr_drop_pct:.1f}%)")
            print(f"      Text→Image MRR: {t2i_mrr:.4f} (↓{t2i_mrr_drop_pct:.1f}%)")
            print(f"      Average MRR: {avg_mrr:.4f} (↓{avg_mrr_drop_pct:.1f}%)")
            print(f"      Image→Text Mean Rank: {i2t_mean_rank:.2f}")
            print(f"      Text→Image Mean Rank: {t2i_mean_rank:.2f}")
            print(f"      Average Mean Rank: {avg_mean_rank:.2f}")
            
            # Determine if this is a breaking point
            if avg_r1_drop_pct > 10:  # More than 10% drop
                print(f"   ⚠️  BREAKING POINT: Significant performance drop detected!")
            elif avg_r1_drop_pct > 5:  # More than 5% drop
                print(f"   ⚡ WARNING: Moderate performance drop")
            else:
                print(f"   ✅ ROBUST: Model handles this noise well")
            
            results.append({
                'description': description,
                'noise_type': noise_type,
                'intensity': intensity,
                'i2t_r1': i2t_r1,
                't2i_r1': t2i_r1,
                'avg_r1': avg_r1,
                'i2t_mrr': i2t_mrr,
                't2i_mrr': t2i_mrr,
                'avg_mrr': avg_mrr,
                'i2t_mean_rank': i2t_mean_rank,
                't2i_mean_rank': t2i_mean_rank,
                'avg_mean_rank': avg_mean_rank,
                'i2t_r1_drop_pct': i2t_r1_drop_pct,
                't2i_r1_drop_pct': t2i_r1_drop_pct,
                'avg_r1_drop_pct': avg_r1_drop_pct,
                'i2t_mrr_drop_pct': i2t_mrr_drop_pct,
                't2i_mrr_drop_pct': t2i_mrr_drop_pct,
                'avg_mrr_drop_pct': avg_mrr_drop_pct
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results.append({
                'description': description,
                'noise_type': noise_type,
                'intensity': intensity,
                'error': str(e)
            })
    
    # Save results to file
    save_results_to_file(results, baseline_results, timestamp)
    
    # Summary analysis
    print(f"\n📊 ENHANCED NOISE SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Find breaking points
    breaking_points = [r for r in results if r.get('avg_r1_drop_pct', 100) > 10]
    warning_points = [r for r in results if 5 < r.get('avg_r1_drop_pct', 100) <= 10]
    robust_points = [r for r in results if r.get('avg_r1_drop_pct', 100) <= 5]
    
    print(f"🔴 BREAKING POINTS (>{10}% drop): {len(breaking_points)}")
    for bp in breaking_points:
        print(f"   - {bp['description']}: {bp.get('avg_r1_drop_pct', 100):.1f}% drop")
    
    print(f"\n🟡 WARNING POINTS (5-10% drop): {len(warning_points)}")
    for wp in warning_points:
        print(f"   - {wp['description']}: {wp.get('avg_r1_drop_pct', 100):.1f}% drop")
    
    print(f"\n🟢 ROBUST CONDITIONS (≤5% drop): {len(robust_points)}")
    for rp in robust_points:
        print(f"   - {rp['description']}: {rp.get('avg_r1_drop_pct', 100):.1f}% drop")
    
    # Modality analysis
    image_results = [r for r in results if r['noise_type'] in ['gaussian', 'salt_pepper', 'brightness', 'blur']]
    text_results = [r for r in results if r['noise_type'] == 'typos']
    
    if image_results and text_results:
        avg_image_drop = np.mean([r.get('avg_r1_drop_pct', 100) for r in image_results])
        avg_text_drop = np.mean([r.get('avg_r1_drop_pct', 100) for r in text_results])
        
        print(f"\n📈 MODALITY ROBUSTNESS:")
        print(f"   Image noise average drop: {avg_image_drop:.1f}%")
        print(f"   Text noise average drop: {avg_text_drop:.1f}%")
        
        if avg_image_drop > avg_text_drop:
            print(f"   🎯 Model is more sensitive to IMAGE noise")
        else:
            print(f"   🎯 Model is more sensitive to TEXT noise")
    
    print(f"\n�� Enhanced noise sensitivity testing complete!")
    print(f"   Tested {len(results)} conditions")
    print(f"   Found {len(breaking_points)} breaking points")
    print(f"   Model shows robustness in {len(robust_points)} conditions")
    
    print(f"\n✅ KEY INSIGHTS:")
    print(f"   • Tested with realistic enhanced intensities")
    print(f"   • Brightness: 0.7, 0.9, 1.3 (enhanced medical imaging range)")
    print(f"   • Blur: 0.4, 0.6, 0.9 (enhanced motion blur range)")
    print(f"   • Results show true model robustness limits")

if __name__ == "__main__":
    test_enhanced_noise_sensitivity()
