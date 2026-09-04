#!/usr/bin/env python3
"""
VOCABULARY AND CAPTION LENGTH ANALYSIS
Detailed analysis of vocabulary diversity and caption length issues
"""

import pickle
import numpy as np
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def analyze_vocabulary_and_captions():
    """Analyze vocabulary diversity and caption length issues"""
    print("🔍 VOCABULARY AND CAPTION LENGTH ANALYSIS")
    print("=" * 60)
    print("Analyzing vocabulary diversity and caption length issues")
    print()
    
    base_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori"
    
    # Load tokenizer
    print("📊 Loading tokenizer...")
    metadata_path = os.path.join(base_path, "metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer', None)
    vocab_size = metadata.get('vocab_size', 0)
    
    print(f"📊 Vocabulary size from metadata: {vocab_size}")
    print(f"📊 Tokenizer type: {type(tokenizer)}")
    
    # Function to decode tokens to text
    def decode_caption(tokens):
        """Decode tokenized caption back to text"""
        try:
            if hasattr(tokenizer, 'decode'):
                return tokenizer.decode(tokens)
            elif hasattr(tokenizer, 'convert_ids_to_tokens'):
                tokens_list = tokenizer.convert_ids_to_tokens(tokens)
                return ' '.join([t for t in tokens_list if t not in ['[PAD]', '[CLS]', '[SEP]']])
            else:
                return str(tokens)
        except Exception as e:
            return f"Error decoding: {e}"
    
    # Collect captions and analyze
    print("📊 Collecting and analyzing captions...")
    
    all_captions = []
    all_tokens = []
    caption_lengths = []
    
    # Load sample data from each split
    for split_name, split_dir in [("train", "train"), ("val", "val"), ("test", "test")]:
        split_path = os.path.join(base_path, split_dir)
        print(f"📊 Loading {split_name} data...")
        
        for shard_file in os.listdir(split_path)[:5]:  # Load first 5 shards
            shard_path = os.path.join(split_path, shard_file)
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
                
                for caption in shard_data['captions']:
                    all_captions.append(caption)
                    
                    # Convert to tokens if possible
                    try:
                        if isinstance(caption, np.ndarray):
                            tokens = caption.tolist()
                        else:
                            tokens = caption
                        
                        # Remove padding tokens
                        tokens = [t for t in tokens if t != 0]
                        all_tokens.extend(tokens)
                        
                        # Count caption length
                        caption_lengths.append(len(tokens))
                        
                    except Exception as e:
                        print(f"Error processing caption: {e}")
    
    print(f"📊 Analyzed {len(all_captions)} captions")
    print(f"📊 Total tokens: {len(all_tokens)}")
    
    # Vocabulary analysis
    print("\n🔍 VOCABULARY ANALYSIS:")
    print("-" * 40)
    
    unique_tokens = set(all_tokens)
    token_counter = Counter(all_tokens)
    
    print(f"📊 Unique tokens found: {len(unique_tokens)}")
    print(f"📊 Vocabulary size from metadata: {vocab_size}")
    print(f"📊 Vocabulary utilization: {len(unique_tokens)/vocab_size*100:.2f}%")
    
    # Most common tokens
    most_common_tokens = token_counter.most_common(20)
    print(f"\n📊 Most common tokens:")
    for i, (token, count) in enumerate(most_common_tokens):
        print(f"   {i+1:2d}. Token {token}: {count} times")
    
    # Token distribution
    print(f"\n📊 Token distribution:")
    print(f"   Total unique tokens: {len(unique_tokens)}")
    print(f"   Tokens used only once: {sum(1 for count in token_counter.values() if count == 1)}")
    print(f"   Tokens used 2-5 times: {sum(1 for count in token_counter.values() if 2 <= count <= 5)}")
    print(f"   Tokens used 6-10 times: {sum(1 for count in token_counter.values() if 6 <= count <= 10)}")
    print(f"   Tokens used 11+ times: {sum(1 for count in token_counter.values() if count >= 11)}")
    
    # Caption length analysis
    print("\n🔍 CAPTION LENGTH ANALYSIS:")
    print("-" * 40)
    
    if caption_lengths:
        avg_length = np.mean(caption_lengths)
        median_length = np.median(caption_lengths)
        min_length = np.min(caption_lengths)
        max_length = np.max(caption_lengths)
        
        print(f"📊 Caption length statistics:")
        print(f"   Average length: {avg_length:.2f} tokens")
        print(f"   Median length: {median_length:.2f} tokens")
        print(f"   Minimum length: {min_length} tokens")
        print(f"   Maximum length: {max_length} tokens")
        print(f"   Standard deviation: {np.std(caption_lengths):.2f} tokens")
        
        # Length distribution
        length_counter = Counter(caption_lengths)
        print(f"\n📊 Length distribution:")
        print(f"   Very short (1-10 tokens): {sum(count for length, count in length_counter.items() if 1 <= length <= 10)}")
        print(f"   Short (11-20 tokens): {sum(count for length, count in length_counter.items() if 11 <= length <= 20)}")
        print(f"   Medium (21-50 tokens): {sum(count for length, count in length_counter.items() if 21 <= length <= 50)}")
        print(f"   Long (51-100 tokens): {sum(count for length, count in length_counter.items() if 51 <= length <= 100)}")
        print(f"   Very long (100+ tokens): {sum(count for length, count in length_counter.items() if length > 100)}")
        
        # Most common lengths
        most_common_lengths = length_counter.most_common(10)
        print(f"\n📊 Most common caption lengths:")
        for length, count in most_common_lengths:
            print(f"   {length} tokens: {count} captions")
    
    # Decode sample captions
    print("\n🔍 SAMPLE CAPTION ANALYSIS:")
    print("-" * 40)
    
    # Sample different length captions
    if caption_lengths:
        short_captions = [caption for caption, length in zip(all_captions, caption_lengths) if length <= 10]
        medium_captions = [caption for caption, length in zip(all_captions, caption_lengths) if 11 <= length <= 50]
        long_captions = [caption for caption, length in zip(all_captions, caption_lengths) if length > 50]
        
        print(f"📊 Sample short captions (≤10 tokens):")
        for i, caption in enumerate(short_captions[:3]):
            try:
                decoded = decode_caption(caption)
                print(f"   {i+1}. {decoded}")
            except:
                print(f"   {i+1}. {caption[:100]}...")
        
        print(f"\n📊 Sample medium captions (11-50 tokens):")
        for i, caption in enumerate(medium_captions[:3]):
            try:
                decoded = decode_caption(caption)
                print(f"   {i+1}. {decoded}")
            except:
                print(f"   {i+1}. {caption[:100]}...")
        
        print(f"\n📊 Sample long captions (>50 tokens):")
        for i, caption in enumerate(long_captions[:3]):
            try:
                decoded = decode_caption(caption)
                print(f"   {i+1}. {decoded}")
            except:
                print(f"   {i+1}. {caption[:100]}...")
    
    # Impact analysis
    print("\n🔍 IMPACT ANALYSIS:")
    print("-" * 40)
    
    print(f"📊 VOCABULARY DIVERSITY IMPACT:")
    print(f"   ✅ Current vocabulary: {len(unique_tokens)} unique tokens")
    print(f"   ✅ Vocabulary utilization: {len(unique_tokens)/vocab_size*100:.2f}%")
    
    if len(unique_tokens) < vocab_size * 0.5:
        print(f"   ⚠️  Low vocabulary utilization - model may not be learning diverse patterns")
    elif len(unique_tokens) < vocab_size * 0.8:
        print(f"   ⚠️  Moderate vocabulary utilization - some room for improvement")
    else:
        print(f"   ✅ Good vocabulary utilization")
    
    print(f"\n📊 CAPTION LENGTH IMPACT:")
    if avg_length < 10:
        print(f"   ⚠️  Very short captions - may not provide enough information")
    elif avg_length < 20:
        print(f"   ⚠️  Short captions - limited descriptive content")
    elif avg_length < 50:
        print(f"   ✅ Reasonable caption length")
    else:
        print(f"   ⚠️  Very long captions - may be difficult for model to process")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 30)
    
    print(f"1. VOCABULARY DIVERSITY:")
    if len(unique_tokens) < 2000:
        print(f"   - Consider expanding vocabulary through data augmentation")
        print(f"   - Add more diverse medical terminology")
        print(f"   - Include synonyms and related terms")
    else:
        print(f"   - Vocabulary diversity is adequate")
    
    print(f"\n2. CAPTION LENGTH:")
    if avg_length > 50:
        print(f"   - Consider truncating very long captions")
        print(f"   - Focus on key medical findings")
        print(f"   - Remove redundant information")
    elif avg_length < 10:
        print(f"   - Consider expanding short captions")
        print(f"   - Add more descriptive content")
        print(f"   - Include relevant medical details")
    else:
        print(f"   - Caption length is reasonable")
    
    print(f"\n3. DATA QUALITY:")
    print(f"   - Ensure captions are medically accurate")
    print(f"   - Standardize medical terminology")
    print(f"   - Remove non-medical content")
    
    print("\n" + "=" * 60)
    print("END OF VOCABULARY AND CAPTION ANALYSIS")
    print("=" * 60)

if __name__ == "__main__":
    analyze_vocabulary_and_captions() 