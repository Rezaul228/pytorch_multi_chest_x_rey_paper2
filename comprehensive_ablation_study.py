#!/usr/bin/env python3
"""
Comprehensive Ablation Study with Your Trained Model
===================================================
This script runs a complete ablation study with ALL possible variants
based on your model architecture components.
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
from all_visualization_v1 import (
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

class ComprehensiveAblatedModel(MultimodalFusion):
    """Modified model class for comprehensive ablation studies"""
    
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
            final_img_emb = synergy_img_emb + difference_img_emb
            final_txt_emb = synergy_txt_emb + difference_txt_emb
        
        return final_img_emb, final_txt_emb

class DetailedAblatedModel(MultimodalFusion):
    """Advanced ablation model for detailed component analysis"""
    
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
        
        # Get original tokens
        image_tokens = self.image_encoder(images, training=training)
        text_tokens = self.text_encoder(texts, training=training)
        
        # Apply detailed ablations
        if self.ablation_type == "no_image_encoder":
            image_tokens = torch.zeros_like(image_tokens)
        elif self.ablation_type == "no_text_encoder":
            text_tokens = torch.zeros_like(text_tokens)
        elif self.ablation_type == "no_conv_blocks":
            # Skip conv blocks, use random features
            image_tokens = torch.randn_like(image_tokens)
        elif self.ablation_type == "no_lstm_layers":
            # Skip LSTM, use embedding only
            text_tokens = self.text_encoder.embedding(texts)
        
        # Process through branches with detailed ablation
        synergy_img_emb, synergy_txt_emb = self._process_branch_with_ablation(
            self.synergy_branch, image_tokens, text_tokens, "synergy"
        )
        difference_img_emb, difference_txt_emb = self._process_branch_with_ablation(
            self.difference_branch, image_tokens, text_tokens, "difference"
        )
        
        # Branch-level ablations
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
        
        # Final fusion
        if self.ablation_type == "no_branch_fusion":
            # Use only first branch
            final_img_emb = synergy_img_emb
            final_txt_emb = synergy_txt_emb
        else:
            final_img_emb = synergy_img_emb + difference_img_emb
            final_txt_emb = synergy_txt_emb + difference_txt_emb
        
        return final_img_emb, final_txt_emb
    
    def _process_branch_with_ablation(self, branch, image_tokens, text_tokens, branch_name):
        """Process through branch with detailed attention ablations"""
        
        # Skip co-attention layers for certain ablations
        if self.ablation_type in ["no_coattention_layers", "single_coattention_layer"]:
            # Use only projections
            image_emb = torch.mean(image_tokens, dim=1)
            text_emb = torch.mean(text_tokens, dim=1)
            image_emb = branch.image_proj(image_emb)
            text_emb = branch.text_proj(text_emb)
            return image_emb, text_emb
        
        # Process through co-attention layers
        for i, layer in enumerate(branch.co_attn_layers):
            # Skip layers for single layer ablation
            if self.ablation_type == "single_coattention_layer" and i > 0:
                continue
                
            image_tokens, text_tokens = self._apply_coattention_ablation(
                layer, image_tokens, text_tokens
            )
        
        # Global pooling and projections
        image_emb = torch.mean(image_tokens, dim=1)
        text_emb = torch.mean(text_tokens, dim=1)
        image_emb = branch.image_proj(image_emb)
        text_emb = branch.text_proj(text_emb)
        
        return image_emb, text_emb
    
    def _apply_coattention_ablation(self, layer, image_tokens, text_tokens):
        """Apply detailed co-attention ablations"""
        
        # Attention mechanism ablations
        if self.ablation_type == "no_local_attention":
            # Skip local attention, keep global only
            image_tokens = image_tokens
            text_tokens = text_tokens
        elif self.ablation_type == "no_global_attention":
            # Skip global attention, keep local only
            image_tokens, text_tokens = self._apply_local_attention_only(layer, image_tokens, text_tokens)
        elif self.ablation_type == "only_local_attention":
            # Use only local attention
            image_tokens, text_tokens = self._apply_local_attention_only(layer, image_tokens, text_tokens)
        elif self.ablation_type == "only_global_attention":
            # Use only global attention
            image_tokens, text_tokens = self._apply_global_attention_only(layer, image_tokens, text_tokens)
        elif self.ablation_type == "no_cross_attention":
            # Skip all attention, use projections only
            image_tokens = layer.norm1(image_tokens)
            text_tokens = layer.norm3(text_tokens)
        else:
            # Apply gating ablations
            image_tokens, text_tokens = self._apply_gating_ablations(layer, image_tokens, text_tokens)
        
        return image_tokens, text_tokens
    
    def _apply_local_attention_only(self, layer, image_tokens, text_tokens):
        """Apply only local cross-attention"""
        # Local cross-attention: Image -> Text
        attended_image, _ = layer.cross_attention1(
            query=image_tokens, key=text_tokens, value=text_tokens
        )
        image_tokens = layer.norm1(image_tokens + attended_image)
        image_tokens = layer.norm2(image_tokens + layer.ffn1(image_tokens))
        
        # Local cross-attention: Text -> Image
        attended_text, _ = layer.cross_attention2(
            query=text_tokens, key=image_tokens, value=image_tokens
        )
        text_tokens = layer.norm3(text_tokens + attended_text)
        text_tokens = layer.norm4(text_tokens + layer.ffn2(text_tokens))
        
        return image_tokens, text_tokens
    
    def _apply_global_attention_only(self, layer, image_tokens, text_tokens):
        """Apply only global cross-attention"""
        # Global tokens
        global_image_token = torch.mean(image_tokens, dim=1, keepdim=True)
        global_text_token = torch.mean(text_tokens, dim=1, keepdim=True)
        
        # Global cross-attention
        attended_global_image, _ = layer.global_cross_attention1(
            query=global_image_token, key=text_tokens, value=text_tokens
        )
        global_image_token = layer.global_norm1(global_image_token + attended_global_image)
        global_image_token = layer.global_norm2(global_image_token + layer.global_ffn1(global_image_token))
        
        attended_global_text, _ = layer.global_cross_attention2(
            query=global_text_token, key=image_tokens, value=image_tokens
        )
        global_text_token = layer.global_norm3(global_text_token + attended_global_text)
        global_text_token = layer.global_norm4(global_text_token + layer.global_ffn2(global_text_token))
        
        # Combine with local
        image_tokens = image_tokens + global_image_token.expand(-1, image_tokens.size(1), -1)
        text_tokens = text_tokens + global_text_token.expand(-1, text_tokens.size(1), -1)
        
        return image_tokens, text_tokens
    
    def _apply_gating_ablations(self, layer, image_tokens, text_tokens):
        """Apply gating mechanism ablations"""
        
        # Local cross-attention: Image -> Text
        attended_image, _ = layer.cross_attention1(
            query=image_tokens, key=text_tokens, value=text_tokens
        )
        
        # Apply gating ablations
        if self.ablation_type == "no_gating_mechanism":
            gated_image = attended_image
        elif self.ablation_type == "no_local_gating":
            gated_image = attended_image
        elif self.ablation_type == "no_image_gating":
            gated_image = attended_image
        else:
            local_image_gate = torch.sigmoid(layer.local_image_gate_weights).view(1, 1, layer.embed_dim)
            gated_image = local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
        
        image_tokens = layer.norm1(image_tokens + gated_image)
        image_tokens = layer.norm2(image_tokens + layer.ffn1(image_tokens))
        
        # Local cross-attention: Text -> Image
        attended_text, _ = layer.cross_attention2(
            query=text_tokens, key=image_tokens, value=image_tokens
        )
        
        # Apply text gating ablations
        if self.ablation_type == "no_gating_mechanism":
            gated_text = attended_text
        elif self.ablation_type == "no_local_gating":
            gated_text = attended_text
        elif self.ablation_type == "no_text_gating":
            gated_text = attended_text
        else:
            local_text_gate = torch.sigmoid(layer.local_text_gate_weights).view(1, 1, layer.embed_dim)
            gated_text = local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
        
        text_tokens = layer.norm3(text_tokens + gated_text)
        text_tokens = layer.norm4(text_tokens + layer.ffn2(text_tokens))
        
        # Global attention with gating ablations
        global_image_token = torch.mean(image_tokens, dim=1, keepdim=True)
        global_text_token = torch.mean(text_tokens, dim=1, keepdim=True)
        
        attended_global_image, _ = layer.global_cross_attention1(
            query=global_image_token, key=text_tokens, value=text_tokens
        )
        
        if self.ablation_type == "no_gating_mechanism":
            gated_global_image = attended_global_image
        elif self.ablation_type == "no_global_gating":
            gated_global_image = attended_global_image
        elif self.ablation_type == "no_image_gating":
            gated_global_image = attended_global_image
        else:
            global_image_gate = torch.sigmoid(layer.global_image_gate_weights).view(1, 1, layer.embed_dim)
            gated_global_image = global_image_gate * attended_global_image + (1 - global_image_gate) * global_image_token
        
        global_image_token = layer.global_norm1(global_image_token + gated_global_image)
        global_image_token = layer.global_norm2(global_image_token + layer.global_ffn1(global_image_token))
        
        attended_global_text, _ = layer.global_cross_attention2(
            query=global_text_token, key=image_tokens, value=image_tokens
        )
        
        if self.ablation_type == "no_gating_mechanism":
            gated_global_text = attended_global_text
        elif self.ablation_type == "no_global_gating":
            gated_global_text = attended_global_text
        elif self.ablation_type == "no_text_gating":
            gated_global_text = attended_global_text
        else:
            global_text_gate = torch.sigmoid(layer.global_text_gate_weights).view(1, 1, layer.embed_dim)
            gated_global_text = global_text_gate * attended_global_text + (1 - global_text_gate) * global_text_token
        
        global_text_token = layer.global_norm3(global_text_token + gated_global_text)
        global_text_token = layer.global_norm4(global_text_token + layer.global_ffn2(global_text_token))
        
        # Global→Local feedback ablation
        if self.ablation_type == "no_global_to_local_feedback":
            # Skip global→local broadcasting
            pass
        else:
            image_tokens = image_tokens + global_image_token.expand(-1, image_tokens.size(1), -1)
            text_tokens = text_tokens + global_text_token.expand(-1, text_tokens.size(1), -1)
        
        return image_tokens, text_tokens

def run_essential_ablation_study():
    """Run essential ablation study with your actual trained model - focused on most impactful components"""
    print("🧪 ESSENTIAL ABLATION STUDY WITH YOUR TRAINED MODEL")
    print("=" * 70)
    print("🎯 Focused on the most impactful components: attention, gating, feedback, branches, and depth")
    print("=" * 70)
    
    # Configuration
    config.print_current_config()
    
    # Load test data
    print("\n📁 Loading test data...")
    data_loader = IndianaDataLoader(
        batch_size=32, 
        use_shards=True, 
        shard_subfolder=config.DATASET_MODE
    )
    data_loader.tokenizer = load_tokenizer_from_metadata()
    data_loader.load_data(max_samples=None, skip_processing=True)
    
    # Use smaller test set for faster ablation study
    test_dataset = data_loader.get_test_data(num_samples=500)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    # Load the trained model
    model_path = '/home/abedin/Developments/pytorch_multi_chest_x_ray1/saved_models/train_mimic_128_1e-04_ep35_full/export/model_weights.pth'
    
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
    
    # Define ESSENTIAL ablation variants (minimal yet impactful set)
    variants = [
        # Baseline
        ("full_model", "none"),
        
        # 🎯 Attention Mechanism Ablations
        ("no_local_attention", "no_local_attention"),           # 🔥 Core attention mechanism
        ("no_global_attention", "no_global_attention"),         # ✔️ High-level attention
        
        # 🔒 Gating Mechanism Ablations  
        ("no_gating_mechanism", "no_gating_mechanism"),         # 🔥 All gates removed
        ("no_local_gating", "no_local_gating"),                 # ✔️ Fine-grained gating
        
        # 🔁 Feedback Mechanism Ablations
        ("no_global_to_local_feedback", "no_global_to_local_feedback"),  # 🔥 Your 43% drop result
        
        # 🌿 Branch Ablations
        ("no_synergy_branch", "no_synergy_branch"),             # ✔️ Primary branch
        ("no_difference_branch", "no_difference_branch"),       # ✔️ Regularization branch
        
        # ⚙️ Fusion Ablation
        ("no_branch_fusion", "no_branch_fusion"),               # ✔️ Final merge step
        
        # 📊 Depth Ablation
        ("single_coattention_layer", "single_coattention_layer"), # ✔️ Depth control
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
                # Use detailed ablation model for complex ablations
                if ablation_type in ["no_local_attention", "no_global_attention", 
                                   "no_gating_mechanism", "no_local_gating",
                                   "no_global_to_local_feedback", "single_coattention_layer"]:
                    model_to_test = DetailedAblatedModel(base_model, ablation_type)
                else:
                    model_to_test = ComprehensiveAblatedModel(base_model, ablation_type)
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
    print(f"\n📊 Generating comprehensive visualizations from your model...")
    
    save_dir = "comprehensive_ablation_results"
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
    print("📊 ESSENTIAL ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    
    baseline_mrr = ablation_results['full_model']['avg_mrr']
    print(f"Your Trained Model (full_model): MRR = {baseline_mrr:.4f}")
    print(f"Dataset: {config.DATASET_MODE}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Essential variants evaluated: {len(ablation_results)}")
    
    print("\nComponent Contributions (from your model):")
    for variant, results in ablation_results.items():
        if variant != 'full_model':
            contribution = baseline_mrr - results['avg_mrr']
            percentage = (contribution / baseline_mrr) * 100
            print(f"  {variant}: {contribution:.4f} ({percentage:+.1f}%)")
    
    # Find most critical component
    contributions = {}
    for variant, results in ablation_results.items():
        if variant != 'full_model':
            contributions[variant] = baseline_mrr - results['avg_mrr']
    
    if contributions:
        most_critical = max(contributions.items(), key=lambda x: x[1])
        print(f"\n🎯 Most Critical Component in Your Model: {most_critical[0]} ({most_critical[1]:.4f} MRR drop)")
    
    print(f"\n✅ Essential ablation study completed!")
    print(f"📁 Results saved to: {save_dir}/")
    print(f"📊 These visualizations now have {len(ablation_results)} bars from YOUR trained model!")
    print(f"🎯 Focused on the most impactful components: attention, gating, feedback, branches, and depth")
    
    return ablation_results

if __name__ == "__main__":
    results = run_essential_ablation_study() 