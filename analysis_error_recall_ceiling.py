#!/usr/bin/env python3
"""
Error Analysis for Recall@1 Ceiling Investigation
Analyzes failed retrievals to understand why R@1 plateaus at ~0.92
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
import config
import paths
from data_loader_v1 import IndianaDataLoader
from base_models_refactored_v1 import MultimodalFusion
from analysis_all_visualization_v1 import create_failure_rank_distribution_plot, create_failure_type_distribution_plot

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

def load_trained_model(model_path):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model

def analyze_retrieval_failures(model, val_loader, tokenizer, output_dir="error_analysis"):
    """Analyze failed retrievals to understand the 0.92 ceiling"""
    print("🔍 ANALYZING RETRIEVAL FAILURES")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all embeddings and metadata
    print("📊 Collecting embeddings and metadata...")
    all_image_emb = []
    all_text_emb = []
    all_captions = []
    all_study_ids = []
    all_images = []
    
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
            
            # Get embeddings
            img_emb, txt_emb = model((batch_images, batch_texts), training=False)
            
            # Store embeddings
            all_image_emb.append(img_emb.cpu())
            all_text_emb.append(txt_emb.cpu())
            
            # Store metadata
            all_captions.extend(batch_texts.cpu().numpy())
            all_images.extend(batch_images.cpu().numpy())
            
            # Handle study IDs
            if isinstance(batch_study_ids, torch.Tensor):
                study_ids_batch = [str(sid.item()) for sid in batch_study_ids]
            else:
                study_ids_batch = [str(sid) for sid in batch_study_ids]
            all_study_ids.extend(study_ids_batch)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Concatenate all embeddings
    all_image_emb = torch.cat(all_image_emb, dim=0)
    all_text_emb = torch.cat(all_text_emb, dim=0)
    
    print(f"✅ Collected {len(all_image_emb)} samples")
    
    # Compute similarity matrix
    print("🔢 Computing similarity matrix...")
    similarity_matrix = torch.matmul(all_image_emb, all_text_emb.transpose(0, 1))
    
    # Analyze failures for both directions
    print("📈 Analyzing failures...")
    
    # Image-to-Text analysis
    i2t_failures = analyze_direction_failures(
        similarity_matrix, 
        all_captions, 
        all_study_ids, 
        tokenizer, 
        direction="i2t"
    )
    
    # Text-to-Image analysis  
    t2i_failures = analyze_direction_failures(
        similarity_matrix.transpose(0, 1), 
        all_captions, 
        all_study_ids, 
        tokenizer, 
        direction="t2i"
    )
    
    # Combine and analyze
    print("🔍 Combining analysis results...")
    combined_failures = combine_failure_analysis(i2t_failures, t2i_failures)
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Rank distribution plots
    create_failure_rank_distribution_plot(
        combined_failures['failure_cases'], 
        combined_failures['success_cases'], 
        output_dir, 
        timestamp
    )
    
    # Failure type distribution
    if combined_failures['failure_types']:
        create_failure_type_distribution_plot(
            combined_failures['failure_types'], 
            output_dir, 
            timestamp
        )
    
    # Save detailed analysis
    save_detailed_analysis(combined_failures, output_dir, timestamp)
    
    # Print summary
    print_summary(combined_failures)
    
    return combined_failures

def analyze_direction_failures(similarity_matrix, captions, study_ids, tokenizer, direction):
    """Analyze failures for one direction (i2t or t2i)"""
    print(f"  Analyzing {direction} failures...")
    
    failures = []
    successes = []
    failure_types = []
    
    sim_np = similarity_matrix.cpu().numpy()
    correct_indices = np.arange(len(sim_np))
    
    for i in range(len(sim_np)):
        similarities = sim_np[i]
        correct_score = similarities[i]
        
        # Calculate rank
        rank = 1 + np.sum(similarities > correct_score)
        tie_count = np.sum(np.isclose(similarities, correct_score)) - 1
        if tie_count > 0:
            rank = rank + tie_count / 2
        
        # Get top matches for analysis
        top_indices = np.argsort(similarities)[::-1][:5]
        
        # Decode captions
        query_caption = decode_caption(captions[i], tokenizer)
        correct_caption = decode_caption(captions[i], tokenizer)  # Same for paired data
        
        top_captions = [decode_caption(captions[idx], tokenizer) for idx in top_indices]
        
        # Analyze failure type
        failure_type = classify_failure_type(
            query_caption, correct_caption, top_captions, rank, similarities
        )
        
        case_info = {
            'sample_idx': i,
            'study_id': study_ids[i],
            'rank': rank,
            'query_caption': query_caption,
            'correct_caption': correct_caption,
            'top_captions': top_captions,
            'top_similarities': similarities[top_indices].tolist(),
            'correct_similarity': correct_score,
            'failure_type': failure_type,
            'direction': direction
        }
        
        if rank > 1:  # Failure case
            failures.append(case_info)
            failure_types.append(failure_type)
        else:  # Success case
            successes.append(case_info)
    
    return {
        'failures': failures,
        'successes': successes,
        'failure_types': failure_types,
        'direction': direction
    }

def classify_failure_type(query_caption, correct_caption, top_captions, rank, similarities):
    """Classify the type of failure"""
    if rank <= 1:
        return "success"
    
    # Check for duplicate/near-duplicate captions
    if any(caption.lower() == correct_caption.lower() for caption in top_captions[1:]):
        return "duplicate_caption"
    
    # Check for very similar captions
    if any(calculate_caption_similarity(caption, correct_caption) > 0.8 for caption in top_captions[1:]):
        return "similar_caption"
    
    # Check for low similarity scores (model uncertainty)
    if similarities[0] < 0.5:
        return "low_confidence"
    
    # Check for ambiguous medical terms
    if contains_ambiguous_terms(query_caption, top_captions):
        return "ambiguous_medical_terms"
    
    # Check for generic vs specific descriptions
    if is_generic_vs_specific(query_caption, top_captions):
        return "generic_vs_specific"
    
    return "other"

def calculate_caption_similarity(caption1, caption2):
    """Calculate similarity between two captions"""
    words1 = set(caption1.lower().split())
    words2 = set(caption2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def contains_ambiguous_terms(caption, top_captions):
    """Check if captions contain ambiguous medical terms"""
    ambiguous_terms = [
        'normal', 'clear', 'unremarkable', 'no abnormality', 'no finding',
        'mild', 'moderate', 'severe', 'prominent', 'increased', 'decreased'
    ]
    
    caption_lower = caption.lower()
    for term in ambiguous_terms:
        if term in caption_lower:
            return True
    
    return False

def is_generic_vs_specific(query_caption, top_captions):
    """Check if there's a generic vs specific description mismatch"""
    query_words = set(query_caption.lower().split())
    
    for caption in top_captions:
        caption_words = set(caption.lower().split())
        
        # Check if one is much more specific than the other
        if len(query_words) < len(caption_words) * 0.5 or len(caption_words) < len(query_words) * 0.5:
            return True
    
    return False

def decode_caption(caption_seq, tokenizer):
    """Decode caption sequence to text"""
    if tokenizer is None:
        return str(caption_seq)
    
    try:
        if isinstance(caption_seq, np.ndarray):
            caption_seq = caption_seq.tolist()
        
        # Remove padding tokens (usually 0)
        caption_seq = [token for token in caption_seq if token != 0]
        
        # Decode using tokenizer
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(caption_seq)
        elif hasattr(tokenizer, 'convert_ids_to_tokens'):
            tokens = tokenizer.convert_ids_to_tokens(caption_seq)
            return ' '.join(tokens)
        else:
            return str(caption_seq)
    except Exception as e:
        return f"DECODE_ERROR: {str(e)}"

def combine_failure_analysis(i2t_failures, t2i_failures):
    """Combine failure analysis from both directions"""
    print("  Combining failure analysis...")
    
    # Combine all failures and successes
    all_failures = i2t_failures['failures'] + t2i_failures['failures']
    all_successes = i2t_failures['successes'] + t2i_failures['successes']
    all_failure_types = i2t_failures['failure_types'] + t2i_failures['failure_types']
    
    # Calculate statistics
    total_samples = len(all_failures) + len(all_successes)
    failure_rate = len(all_failures) / total_samples if total_samples > 0 else 0
    
    # Analyze failure patterns
    failure_patterns = analyze_failure_patterns(all_failures)
    
    return {
        'failure_cases': all_failures,
        'success_cases': all_successes,
        'failure_types': all_failure_types,
        'failure_rate': failure_rate,
        'total_samples': total_samples,
        'failure_patterns': failure_patterns,
        'i2t_failures': i2t_failures,
        't2i_failures': t2i_failures
    }

def analyze_failure_patterns(failures):
    """Analyze patterns in failures"""
    patterns = {
        'rank_distribution': Counter(),
        'failure_type_distribution': Counter(),
        'direction_distribution': Counter(),
        'similarity_score_ranges': Counter(),
        'caption_length_analysis': defaultdict(list)
    }
    
    for failure in failures:
        # Rank distribution
        rank_bin = f"rank_{min(failure['rank'], 10)}+" if failure['rank'] > 10 else f"rank_{failure['rank']}"
        patterns['rank_distribution'][rank_bin] += 1
        
        # Failure type distribution
        patterns['failure_type_distribution'][failure['failure_type']] += 1
        
        # Direction distribution
        patterns['direction_distribution'][failure['direction']] += 1
        
        # Similarity score analysis
        correct_sim = failure['correct_similarity']
        if correct_sim < 0.3:
            sim_bin = "very_low"
        elif correct_sim < 0.5:
            sim_bin = "low"
        elif correct_sim < 0.7:
            sim_bin = "medium"
        else:
            sim_bin = "high"
        patterns['similarity_score_ranges'][sim_bin] += 1
        
        # Caption length analysis
        query_len = len(failure['query_caption'].split())
        patterns['caption_length_analysis']['query_lengths'].append(query_len)
    
    return patterns

def save_detailed_analysis(analysis_results, output_dir, timestamp):
    """Save detailed analysis results"""
    print(f"  Saving detailed analysis...")
    
    # Save raw data
    analysis_file = os.path.join(output_dir, f'error_analysis_{timestamp}.pkl')
    with open(analysis_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    # Save text report
    report_file = os.path.join(output_dir, f'error_analysis_report_{timestamp}.txt')
    with open(report_file, 'w') as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Samples: {analysis_results['total_samples']}\n")
        f.write(f"Failure Rate: {analysis_results['failure_rate']:.4f} ({analysis_results['failure_rate']*100:.2f}%)\n")
        f.write(f"Success Rate: {1 - analysis_results['failure_rate']:.4f} ({(1 - analysis_results['failure_rate'])*100:.2f}%)\n\n")
        
        f.write("FAILURE TYPE DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        type_counts = Counter(analysis_results['failure_types'])
        for failure_type, count in type_counts.most_common():
            percentage = count / len(analysis_results['failure_types']) * 100
            f.write(f"{failure_type}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nRANK DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        rank_counts = Counter([f['rank'] for f in analysis_results['failure_cases']])
        for rank in sorted(rank_counts.keys()):
            f.write(f"Rank {rank}: {rank_counts[rank]} failures\n")
        
        f.write("\nTOP 10 FAILURE EXAMPLES:\n")
        f.write("-" * 25 + "\n")
        sorted_failures = sorted(analysis_results['failure_cases'], key=lambda x: x['rank'], reverse=True)
        for i, failure in enumerate(sorted_failures[:10]):
            f.write(f"\n{i+1}. Sample {failure['sample_idx']} (Rank {failure['rank']})\n")
            f.write(f"   Study ID: {failure['study_id']}\n")
            f.write(f"   Direction: {failure['direction']}\n")
            f.write(f"   Failure Type: {failure['failure_type']}\n")
            f.write(f"   Query Caption: {failure['query_caption']}\n")
            f.write(f"   Correct Caption: {failure['correct_caption']}\n")
            f.write(f"   Top Match: {failure['top_captions'][0]}\n")
            f.write(f"   Correct Similarity: {failure['correct_similarity']:.4f}\n")
            f.write(f"   Top Similarity: {failure['top_similarities'][0]:.4f}\n")
    
    print(f"✅ Analysis saved to {output_dir}")

def print_summary(analysis_results):
    """Print summary of error analysis"""
    print("\n" + "=" * 60)
    print("📊 ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_samples = analysis_results['total_samples']
    failure_rate = analysis_results['failure_rate']
    
    print(f"Total Samples Analyzed: {total_samples}")
    print(f"Failure Rate: {failure_rate:.4f} ({failure_rate*100:.2f}%)")
    print(f"Success Rate: {1 - failure_rate:.4f} ({(1 - failure_rate)*100:.2f}%)")
    
    print(f"\n🎯 This explains the 0.92 ceiling!")
    print(f"   Expected R@1 = {1 - failure_rate:.4f}")
    print(f"   Observed R@1 ≈ 0.92")
    print(f"   Difference: {abs(0.92 - (1 - failure_rate)):.4f}")
    
    print(f"\n📈 Failure Type Distribution:")
    type_counts = Counter(analysis_results['failure_types'])
    for failure_type, count in type_counts.most_common():
        percentage = count / len(analysis_results['failure_types']) * 100
        print(f"   {failure_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔍 Top Failure Causes:")
    sorted_failures = sorted(analysis_results['failure_cases'], key=lambda x: x['rank'], reverse=True)
    for i, failure in enumerate(sorted_failures[:3]):
        print(f"   {i+1}. Rank {failure['rank']} - {failure['failure_type']}")
        print(f"      Query: {failure['query_caption'][:80]}...")
        print(f"      Top Match: {failure['top_captions'][0][:80]}...")

def main():
    """Main function to run error analysis"""
    print("🔍 RECALL@1 CEILING ERROR ANALYSIS")
    print("=" * 60)
    
    # Configuration
    config.print_current_config()
    
    # Load model - using the correct model with vocab size 4446
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_al_vocab4446_to128_lr1e-4_b128_ep45_dualbr/export/model_weights.pth'
    model = load_trained_model(model_path)
    
    # Load validation data
    print("\n📁 Loading validation data...")
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder=config.DATASET_MODE
    )
    
    # Load tokenizer
    data_loader.tokenizer = load_tokenizer_from_metadata()
    
    data_loader.load_data(max_samples=None, skip_processing=True)
    val_dataset = data_loader.get_validation_data(num_samples=None)
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ Validation data loaded: {len(val_dataset)} samples")
    
    # Run error analysis
    results = analyze_retrieval_failures(
        model, 
        val_loader, 
        data_loader.tokenizer,
        output_dir="error_analysis_results"
    )
    
    print(f"\n✅ Error analysis complete!")
    print(f"📁 Results saved to: error_analysis_results/")

if __name__ == "__main__":
    main() 