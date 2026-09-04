#!/usr/bin/env python3
"""
INDIANA DATASET ABLATION STUDY
Comprehensive ablation study with Indiana dataset using same vocab_size and tokenizer
"""

import torch
import config
import paths
import pickle
from datetime import datetime
import os
from base_models_refactored_v1 import MultimodalFusion
from data_loader_v1 import IndianaDataLoader
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval_streaming
from analysis_all_visualization_v1 import (
    create_ablation_performance_comparison,
    create_ablation_contribution_analysis
)

def load_tokenizer_from_metadata():
    """Load the tokenizer from metadata based on current dataset mode"""
    shard_subfolder = config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    if hasattr(tokenizer, 'word_index') and not hasattr(tokenizer, 'word2idx'):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word
    
    return tokenizer

class IndianaAblatedModel(MultimodalFusion):
    """Modified model class for Indiana dataset ablation studies"""
    
    def __init__(self, base_model, ablation_type="none"):
        super().__init__(
            vocab_size=config.get_vocab_size(),
            embed_dim=config.get_embed_dim(),
            num_heads=config.get_current_config()['num_heads'],
            num_layers=config.get_current_config()['num_layers']
        )
        self.load_state_dict(base_model.state_dict())
        self.ablation_type = ablation_type
        self.base_model = base_model
    
    def forward(self, inputs, training=True):
        images, texts = inputs
        image_tokens = self.image_encoder(images, training=training)
        text_tokens = self.text_encoder(texts, training=training)
        
        # Apply encoder-level ablations
        if self.ablation_type == "no_image_encoder":
            image_tokens = torch.zeros_like(image_tokens)
        elif self.ablation_type == "no_text_encoder":
            text_tokens = torch.zeros_like(text_tokens)
        
        # Default: use all branches
        synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
        difference_img_emb, difference_txt_emb = self.difference_branch(image_tokens, text_tokens)
        
        # Apply branch-level ablations
        if self.ablation_type == "no_synergy_branch":
            synergy_img_emb = torch.zeros_like(synergy_img_emb)
            synergy_txt_emb = torch.zeros_like(synergy_txt_emb)
        elif self.ablation_type == "no_difference_branch":
            difference_img_emb = torch.zeros_like(difference_img_emb)
            difference_txt_emb = torch.zeros_like(difference_txt_emb)
        elif self.ablation_type == "synergy_only":
            difference_img_emb = torch.zeros_like(difference_img_emb)
            difference_txt_emb = torch.zeros_like(difference_txt_emb)
        elif self.ablation_type == "difference_only":
            synergy_img_emb = torch.zeros_like(synergy_img_emb)
            synergy_txt_emb = torch.zeros_like(synergy_txt_emb)
        
        # Combine embeddings
        if self.ablation_type in ["synergy_only", "no_difference_branch"]:
            final_img_emb = synergy_img_emb
            final_txt_emb = synergy_txt_emb
        elif self.ablation_type in ["difference_only", "no_synergy_branch"]:
            final_img_emb = difference_img_emb
            final_txt_emb = difference_txt_emb
        else:
            # Fix: Use same combination as original model
            import torch.nn.functional as F
            final_img_emb = F.normalize((synergy_img_emb + difference_img_emb) / 2, p=2, dim=-1)
            final_txt_emb = F.normalize((synergy_txt_emb + difference_txt_emb) / 2, p=2, dim=-1)
        
        return final_img_emb, final_txt_emb

def run_indiana_ablation_study():
    """Run comprehensive ablation study with Indiana dataset"""
    print("🧪 INDIANA DATASET ABLATION STUDY")
    print("=" * 70)
    print("🎯 Testing model components with Indiana dataset")
    print("🎯 Using same vocab_size (10805) and max_token_length (128)")
    print("=" * 70)
    
    # Switch to Indiana dataset
    print("🔄 Switching to Indiana dataset...")
    config.switch_dataset("indiana_shards")
    config.print_current_config()
    
    # Load test data
    print("\n📁 Loading Indiana test data...")
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder=config.DATASET_MODE
    )
    data_loader.tokenizer = load_tokenizer_from_metadata()
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Use 2000 test samples for Indiana dataset
    test_dataset = data_loader.get_test_data(num_samples=2000)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ Indiana test data loaded: {len(test_dataset)} samples")
    
    # Load your trained model (using the same model trained on MIMIC)
    print("\n📁 Loading your trained model...")
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_rey_paper2/saved_models/mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy07_main_loss02_ortho01_branch_v3/export/model_weights.pth'
    
    base_model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()['num_heads'],
        num_layers=config.get_current_config()['num_layers']
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in base_model.parameters()):,} parameters")
    
    # Define Indiana ablation variants
    variants = [
        # Baseline
        ("full_model", "none"),
        
        # 🎯 Branch Ablations
        ("no_synergy_branch", "no_synergy_branch"),             # Primary branch
        ("no_difference_branch", "no_difference_branch"),       # Regularization branch
        ("synergy_only", "synergy_only"),                       # Only synergy
        ("difference_only", "difference_only"),                 # Only difference
        
        # 🎯 Encoder Ablations
        ("no_image_encoder", "no_image_encoder"),               # No image processing
        ("no_text_encoder", "no_text_encoder"),                 # No text processing
        
        # ⚙️ Fusion Ablation
        ("no_branch_fusion", "no_branch_fusion"),               # Final merge step
    ]
    
    # Store results
    ablation_results = {}
    
    # Evaluate each variant
    for variant_name, ablation_type in variants:
        print(f"\n{'='*60}")
        print(f"🎯 Evaluating: {variant_name}")
        print(f"{'='*60}")
        
        try:
            if variant_name == "full_model":
                model_to_test = base_model
            else:
                model_to_test = IndianaAblatedModel(base_model, ablation_type)
                model_to_test.eval()
            
            # Run evaluation
            results = evaluate_cross_modal_retrieval_streaming(
                model=model_to_test,
                test_dataset=test_loader,
                k_values=[1, 5, 10],
                batch_size=32,
                visualize=False,
                num_vis_examples=0
            )
            
            # Store results
            ablation_results[variant_name] = results
            
            print(f"📊 Results for {variant_name}:")
            print(f"   MRR: {results['avg_mrr']:.4f}")
            print(f"   Recall@1: {results['avg_recall@1']:.4f}")
            print(f"   Recall@5: {results['avg_recall@5']:.4f}")
            print(f"   Recall@10: {results['avg_recall@10']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {variant_name}: {e}")
            print(f"   Skipping this variant...")
            continue
    
    # Generate comprehensive visualizations
    print(f"\n📊 Generating Indiana ablation study visualizations...")
    
    save_dir = "indiana_ablation_results"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualizations
    create_ablation_performance_comparison(
        ablation_results, 
        save_dir, 
        timestamp
    )
    
    create_ablation_contribution_analysis(
        ablation_results, 
        save_dir, 
        timestamp
    )
    
    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("📊 INDIANA ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    
    baseline_mrr = ablation_results.get('full_model', {}).get('avg_mrr', 0)
    print(f"Your Trained Model (full_model): MRR = {baseline_mrr:.4f}")
    print(f"Dataset: {config.DATASET_MODE}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Variants evaluated: {len(ablation_results)}")
    
    print("\nComponent Contributions (Indiana dataset):")
    for variant, results in ablation_results.items():
        if variant != 'full_model':
            contribution = baseline_mrr - results['avg_mrr']
            percentage = (contribution / baseline_mrr) * 100 if baseline_mrr > 0 else 0
            print(f"  {variant}: {contribution:.4f} ({percentage:+.1f}%)")
    
    # Find most critical component
    contributions = {}
    for variant, results in ablation_results.items():
        if variant != 'full_model':
            contributions[variant] = baseline_mrr - results['avg_mrr']
    
    if contributions:
        most_critical = max(contributions.items(), key=lambda x: x[1])
        print(f"\n🎯 Most Critical Component in Indiana: {most_critical[0]} ({most_critical[1]:.4f} MRR drop)")
    
    # Compare with MIMIC results
    print(f"\n{'='*70}")
    print("🔍 INDIANA vs MIMIC COMPARISON")
    print(f"{'='*70}")
    
    print(f"📊 Key differences:")
    print(f"   • Indiana dataset: Different domain, same architecture")
    print(f"   • Same vocabulary size: 10805")
    print(f"   • Same max token length: 128")
    print(f"   • Same tokenizer: Ensures compatibility")
    
    print(f"\n✅ Indiana ablation study completed!")
    print(f"📁 Results saved to: {save_dir}/")
    print(f"📊 Visualizations: {len(ablation_results)} variants analyzed")
    print(f"🎯 Focused on Indiana-specific component analysis")
    
    return ablation_results

if __name__ == "__main__":
    run_indiana_ablation_study() 