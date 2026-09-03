#!/usr/bin/env python3
"""
Complete Training Script for MIMIC-CXR Multimodal Retrieval - PyTorch Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
import psutil
import gc
import pickle
from tqdm import tqdm
from base_models_refactored_v1 import MultimodalFusion, ContrastiveLoss, SynergyLoss, DifferenceLoss
from all_visualization_v1 import TrainingVisualizer
from data_loader_v1 import IndianaDataLoader
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval
import paths
import glob
import config
import torch.nn.functional as F

# Import configuration values
VOCAB_SIZE = config.get_vocab_size()
MAX_TOKEN_LENGTH = config.get_max_token_length()
EMBED_DIM = config.get_embed_dim()
DEFAULT_BATCH_SIZE = config.get_default_batch_size()
DEFAULT_LEARNING_RATE = config.get_default_learning_rate()
DEFAULT_EPOCHS = config.get_default_epochs()

def setup_gpu_memory():
    """Configure GPU memory settings"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA available with {device_count} GPU(s)")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Clear cache
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
    else:
        print("CUDA not available, using CPU")

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

def print_memory_usage(stage=""):
    """Print current memory usage"""
    mem = get_memory_usage()
    system_mem = psutil.virtual_memory()
    print(f"Memory [{stage}]: RSS={mem['rss_mb']:.1f}MB, System={system_mem.percent:.1f}%")
    return mem

def force_garbage_collection():
    """Force garbage collection"""
    mem_before = get_memory_usage()['rss_mb']
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_after = get_memory_usage()['rss_mb']
    freed = mem_before - mem_after
    print(f"Garbage collection: Freed {freed:.1f}MB")

def check_memory_limit():
    """Check if memory usage is critical"""
    system_mem = psutil.virtual_memory()
    if system_mem.percent > 85:
        print(f"Critical: System memory at {system_mem.percent:.1f}%!")
        return True
    return False

def setup_directories(experiment_name):
    """Setup experiment directories"""
    train_viz_dir = os.path.join(paths.OUTPUTS_DIR, f"train_visualizations_{experiment_name}")
    models_dir = os.path.join(paths.SAVED_MODELS_DIR, experiment_name)
    results_dir = os.path.join(paths.OUTPUTS_DIR, f"results_{experiment_name}")
    
    for directory in [train_viz_dir, models_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return train_viz_dir, models_dir, results_dir

def clean_old_files(directory, pattern="*"):
    """Clean old files from directory"""
    if os.path.exists(directory):
        old_files = glob.glob(os.path.join(directory, pattern))
        for old_file in old_files:
            try:
                if os.path.isfile(old_file):
                    os.remove(old_file)
                elif os.path.isdir(old_file):
                    import shutil
                    shutil.rmtree(old_file)
            except Exception:
                pass

def compute_recall_k(similarity_matrix, k):
    """Compute recall@k for a similarity matrix"""
    batch_size = similarity_matrix.size(0)
    
    # Get top k indices
    _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
    
    # Create target indices
    target_indices = torch.arange(batch_size, device=similarity_matrix.device)
    target_indices = target_indices.unsqueeze(1).expand(-1, k)
    
    # Check if target indices are in top k
    correct = (top_k_indices == target_indices).any(dim=1)
    
    # Compute recall
    recall = correct.float().mean().item()
    
    return recall

class EnhancedRetrievalTrainer:
    """Enhanced trainer for dual branch architecture - PyTorch version"""

    def __init__(self, model, temperature=None, learning_rate=1e-5, 
                 viz_dir='visualizations', model_save_path=None, experiment_name='dual_branch_exp',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = model.to(device)
        
        # Use temperature from config if not specified
        if temperature is None:
            temperature = config.get_current_config()["temperature"]
        
        # Initialize task-specific loss functions
        self.main_loss_fn = ContrastiveLoss(temperature)
        self.synergy_loss_fn = SynergyLoss(temperature)
        self.difference_loss_fn = DifferenceLoss(temperature)
        
        # Get weight decay from config
        weight_decay = config.get_default_weight_decay()
        
        # Initialize optimizer with both learning rate and weight decay
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        print(f"Optimizer configured with:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Temperature: {temperature}")
        print(f"  Task-specific losses: Synergy + Difference")
        
        self.viz = TrainingVisualizer(save_dir=viz_dir)
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name
        
        self._ensure_model_built()
        
        self.training_history = {
            'epochs': [],
            'total_losses': [],
            'synergy_losses': [],
            'difference_losses': []
        }
    
    def _ensure_model_built(self):
        """Ensure model is fully built"""
        print("Building model with BranchEncoder architecture...")
        
        # PyTorch expects BCHW format (batch, channels, height, width)
        dummy_images = torch.zeros((1, 3, 224, 224), device=self.device)
        dummy_texts = torch.zeros((1, 115), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            _ = self.model((dummy_images, dummy_texts), training=False)
        
        has_synergy_branch = hasattr(self.model, 'synergy_branch')
        has_difference_branch = hasattr(self.model, 'difference_branch')
        
        if not (has_synergy_branch and has_difference_branch):
            print("BranchEncoder architecture not found!")
            raise ValueError("Model must have synergy_branch and difference_branch attributes")
    
    def train_step(self, batch):
        """Single training step with task-specific dual branch loss calculation"""
        self.model.train()
        
        # Unpack batch
        images, texts = batch
        
        # Convert images to BCHW format if needed
        if images.shape[1] != 3:  # If not already in BCHW format
            images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Forward pass - get all embeddings
        image_emb, text_emb, synergy_img_emb, synergy_txt_emb, diff_img_emb, diff_txt_emb = self.model(
            (images, texts),
            training=True,
            return_branch_embeddings=True
        )
        
        # Print embeddings shape (commented out to reduce log spam)
        # print(f"Image embeddings: {image_emb.shape}, Text embeddings: {text_emb.shape}")
        
        # Compute task-specific branch losses
        synergy_loss = self.synergy_loss_fn(None, (synergy_img_emb, synergy_txt_emb))
        difference_loss = self.difference_loss_fn(None, (diff_img_emb, diff_txt_emb))
        
        # Main loss using final combined embeddings (already normalized in model)
        main_loss = self.main_loss_fn(None, (image_emb, text_emb))
        
        # Balance losses to prevent one branch from dominating
        # Get loss magnitudes for balancing
        synergy_magnitude = synergy_loss.detach().item()
        diff_magnitude = difference_loss.detach().item()
        
        # Calculate weights to balance losses (avoid division by zero)
        total_magnitude = synergy_magnitude + diff_magnitude + 1e-8
        synergy_weight = total_magnitude / (synergy_magnitude + 1e-8)
        diff_weight = total_magnitude / (diff_magnitude + 1e-8)
        
        # Normalize weights to sum to 2.0
        weight_sum = synergy_weight + diff_weight
        synergy_weight = synergy_weight / weight_sum * 2.0
        diff_weight = diff_weight / weight_sum * 2.0
        
        # Combined loss: balanced task-specific losses
        total_loss = synergy_weight * synergy_loss + diff_weight * difference_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), synergy_loss.item(), difference_loss.item()
    
    def verify_branch_specialization(self, batch):
        """Verify that branches are learning different tasks"""
        self.model.eval()
        
        with torch.no_grad():
            # Unpack batch
            images, texts = batch
            
            # Convert images to BCHW format if needed
            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)
            
            # Get branch embeddings
            _, _, synergy_img_emb, synergy_txt_emb, diff_img_emb, diff_txt_emb = self.model(
                (images, texts),
                training=False,
                return_branch_embeddings=True
            )
            
            # Calculate similarities
            synergy_similarity = F.cosine_similarity(synergy_img_emb, synergy_txt_emb, dim=1)
            diff_similarity = F.cosine_similarity(diff_img_emb, diff_txt_emb, dim=1)
            
            # Synergy should have HIGH similarity, Difference should have LOW similarity
            synergy_mean = synergy_similarity.mean().item()
            diff_mean = diff_similarity.mean().item()
            
            print(f"\nBranch Specialization Check:")
            print(f"  Synergy Branch Similarity: {synergy_mean:.4f}")
            print(f"  Difference Branch Similarity: {diff_mean:.4f}")
            print(f"  Specialization Gap: {synergy_mean - diff_mean:.4f}")
            
            # Check if branches are properly specialized
            is_specialized = synergy_mean > diff_mean
            print(f"  Branches Specialized: {'YES' if is_specialized else 'NO'}")
            
            return is_specialized, synergy_mean, diff_mean
    
    def evaluate_recall_k_batched(self, data_loader, batch_size=100, k=[1, 5, 10], max_samples=None):
        """Evaluate recall@k on validation data in batches"""
        self.model.eval()
        
        # Initialize embeddings storage
        all_image_emb = []
        all_text_emb = []
        total_processed = 0
        
        # Process in batches
        for batch in data_loader:
            if max_samples and total_processed >= max_samples:
                break
            
            # Handle dictionary format from IndianaDataset
            if isinstance(batch, dict):
                batch_images = batch['images']
                batch_texts = batch['captions']
            else:
                batch_images, batch_texts = batch
            
            # Convert images to BCHW format
            if batch_images.shape[1] != 3:  # If not already in BCHW format
                batch_images = batch_images.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
            # Move to device
            batch_images = batch_images.to(self.device)
            batch_texts = batch_texts.to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                img_emb, txt_emb = self.model(
                    (batch_images, batch_texts),
                    training=False
                )
                
                # Store embeddings
                all_image_emb.append(img_emb.cpu())
                all_text_emb.append(txt_emb.cpu())
                total_processed += len(batch_images)
        
        print(f"Processed {total_processed} validation samples in batches")
        
        # Concatenate all embeddings
        all_image_emb = torch.cat(all_image_emb, dim=0)
        all_text_emb = torch.cat(all_text_emb, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_image_emb, all_text_emb.transpose(0, 1))
        
        # Calculate recalls
        recalls = {}
        for k_val in k:
            i2t_recall = compute_recall_k(similarity_matrix, k=k_val)
            t2i_recall = compute_recall_k(similarity_matrix.transpose(0, 1), k=k_val)
            recalls[f'recall@{k_val}'] = (i2t_recall + t2i_recall) / 2
        
        return recalls

    def evaluate_recall_k(self, val_data, k=[1, 5, 10]):
        """Evaluate recall@k on validation data"""
        self.model.eval()
        
        # Convert validation data to tensors
        val_images = torch.tensor(val_data['images'], dtype=torch.float32, device=self.device)
        val_texts = torch.tensor(val_data['captions'], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Get embeddings
            image_emb, text_emb = self.model(
                (val_images, val_texts),
                training=False
            )
            
            similarity_matrix = torch.matmul(image_emb, text_emb.transpose(0, 1))
        
        recalls = {}
        for k_val in k:
            _, top_k_indices = torch.topk(similarity_matrix, k=k_val, dim=1)
            correct_indices = torch.arange(similarity_matrix.size(0), device=self.device)
            correct_in_topk = torch.any(
                top_k_indices == correct_indices.unsqueeze(1),
                dim=1
            )
            recall = torch.mean(correct_in_topk.float())
            recalls[k_val] = float(recall.cpu())
        
        return recalls
    
    def train(self, train_loader, val_loader, num_epochs):
        """Training with dual branch monitoring"""
        
        steps_per_epoch = len(train_loader)
        
        print(f"\nStarting Dual Branch Training")
        print(f"Using BranchEncoder architecture")
        print(f"{num_epochs} epochs, {steps_per_epoch} steps per epoch")
        print(f"Device: {self.device}")
        print("="*60)
        
        # Initialize history
        self.history = {
            'epochs': [],
            'total_loss': [],
            'synergy_loss': [],
            'difference_loss': [],
            'loss_ratio': [],
            'recall@1': [],
            'recall@5': [],
            'recall@10': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Ensure model is in training mode for this epoch
            self.model.train()
            
            epoch_total_losses = []
            epoch_synergy_losses = []
            epoch_difference_losses = []
            
            pbar = tqdm(enumerate(train_loader), 
                       total=min(steps_per_epoch, len(train_loader)),
                       desc="Training")
            
            for step, batch in pbar:
                # Move data to device - handle dictionary format from IndianaDataset
                if isinstance(batch, dict):
                    images = batch['images']
                    texts = batch['captions']
                else:
                    images, texts = batch
                    
                images = images.to(self.device)
                texts = texts.to(self.device)
                
                # Forward pass
                total_loss, synergy_loss, difference_loss = self.train_step(
                    (images, texts)
                )
                
                # Store losses
                epoch_total_losses.append(total_loss)
                epoch_synergy_losses.append(synergy_loss)
                epoch_difference_losses.append(difference_loss)
                
                # Update progress bar
                pbar.set_description(
                    f"Total: {total_loss:.4f}, Syn: {synergy_loss:.4f}, Diff: {difference_loss:.4f}"
                )
            
            # Compute epoch metrics
            epoch_total_loss = np.mean(epoch_total_losses)
            epoch_synergy_loss = np.mean(epoch_synergy_losses)
            epoch_difference_loss = np.mean(epoch_difference_losses)
            loss_ratio = epoch_synergy_loss / epoch_difference_loss
            
            # Validation
            recalls = {}
            if val_loader is not None:
                recalls = self.evaluate_recall_k_batched(val_loader, k=[1, 5, 10])
                print(f"\nProcessed {len(val_loader.dataset)} validation samples in batches")
                print(f"Validation on {len(val_loader.dataset)} samples")
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"   Total Loss:      {epoch_total_loss:.6f}")
            print(f"   Synergy Loss:    {epoch_synergy_loss:.6f}")
            print(f"   Difference Loss: {epoch_difference_loss:.6f}")
            print(f"   Loss Ratio:      {loss_ratio:.3f}")
            
            if recalls:
                print("   Validation:")
                for k, recall in recalls.items():
                    print(f"     {k}: {recall:.4f}")
            
            # Verify branch specialization every 5 epochs
            if (epoch + 1) % 5 == 0 and val_loader is not None:
                # Get a sample batch for verification
                sample_batch = next(iter(val_loader))
                if isinstance(sample_batch, dict):
                    sample_images = sample_batch['images'].to(self.device)
                    sample_texts = sample_batch['captions'].to(self.device)
                else:
                    sample_images, sample_texts = sample_batch
                    sample_images = sample_images.to(self.device)
                    sample_texts = sample_texts.to(self.device)
                
                self.verify_branch_specialization((sample_images, sample_texts))
            
            # Update visualization
            viz_metrics = {
                'total_loss': epoch_total_loss,
                'synergy_loss': epoch_synergy_loss,
                'difference_loss': epoch_difference_loss,
                'loss_ratio': loss_ratio,
                **recalls
            }
            self.viz.update_history(viz_metrics)
            
            # Update history
            self.history['epochs'].append(epoch + 1)
            self.history['total_loss'].append(epoch_total_loss)
            self.history['synergy_loss'].append(epoch_synergy_loss)
            self.history['difference_loss'].append(epoch_difference_loss)
            self.history['loss_ratio'].append(loss_ratio)
            for k, v in recalls.items():
                self.history[k].append(v)
        
        if val_loader is not None:
            print("\nFINAL COMPREHENSIVE VALIDATION ON ALL DATA...")
            print("="*60)
            
            final_recalls = self.evaluate_recall_k_batched(
                val_loader, 
                k=[1, 5, 10]
            )
            
            print("\nFINAL VALIDATION RESULTS:")
            for k, recall in final_recalls.items():
                print(f"   Final {k}: {recall:.4f}")
            print("="*60)
            
            print("\nTRAINING SUMMARY")
            print("="*60)
            print("Final Losses:")
            print(f"   Total: {epoch_total_loss:.6f}")
            print(f"   Synergy: {epoch_synergy_loss:.6f}")
            print(f"   Difference: {epoch_difference_loss:.6f}")
            
            print("\nDual Branch Verification:")
            print(f"   Synergy learning: {'YES' if epoch_synergy_loss < 3.5 else 'NO'}")
            print(f"   Difference learning: {'YES' if epoch_difference_loss < 3.5 else 'NO'}")
            
            # Save model
            try:
                self.save_model()
                print("\nModel saved successfully!")
            except Exception as e:
                print(f"\nError saving model: {str(e)}")
                print("Model saving verification failed!")
        
        return epoch_total_loss, epoch_synergy_loss, epoch_difference_loss
    
    def save_model(self):
        """Save model"""
        if not self.model_save_path:
            print("No model save path specified!")
            return False
        
        try:
            # Get the base directory and experiment name from model_save_path
            base_dir = os.path.dirname(self.model_save_path)
            experiment_name = self.experiment_name
            
            # Create export directory
            export_dir = os.path.join(base_dir, 'export')
            os.makedirs(export_dir, exist_ok=True)
            
            # Save complete model state
            model_path = os.path.join(export_dir, 'model.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'experiment_name': self.experiment_name
            }, model_path)
            print(f"Model saved to: {model_path}")
            
            # Save just weights for compatibility
            weights_path = os.path.join(export_dir, 'model_weights.pth')
            torch.save(self.model.state_dict(), weights_path)
            print(f"Model weights saved to: {weights_path}")
            
            # Also save weights in originanl location for backward compatibility
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Model weights also saved to (backup): {self.model_save_path}")
            
            # Verify saved files
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            weights_size = os.path.getsize(weights_path) / (1024 * 1024)
            backup_weights_size = os.path.getsize(self.model_save_path) / (1024 * 1024)
            
            print(f"\nModel file sizes:")
            print(f"  Complete model: {model_size:.2f} MB")
            print(f"  Weights file (export): {weights_size:.2f} MB")
            print(f"  Weights file (backup): {backup_weights_size:.2f} MB")
            
            has_synergy = hasattr(self.model, 'synergy_branch')
            has_difference = hasattr(self.model, 'difference_branch')
    
            print(f"\nBranch verification:")
            print(f"   Synergy branch: {'YES' if has_synergy else 'NO'}")
            print(f"   Difference branch: {'YES' if has_difference else 'NO'}")
            
            # Verify model can be loaded
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                print("\n✅ Successfully verified saved model can be loaded")
                
                # Check if training history is saved
                if 'training_history' in checkpoint:
                    print("✅ Training history saved")
                
            except Exception as e:
                print(f"\n❌ Error verifying saved model: {e}")
                return False
            
            return has_synergy and has_difference and model_size > 1
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="PyTorch Multimodal Retrieval Training with Centralized Configuration")
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Experiment name (required)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size (default: from config.py)')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='Learning rate (default: from config.py)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs (default: from config.py)')
    parser.add_argument('--train_samples', type=int, default=None, 
                      help='Number of training samples to use (None = use all)')
    parser.add_argument('--val_samples', type=int, default=None,
                      help='Number of validation samples to use (None = use all)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                      help='Device to use for computation (default: cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Get parameters from config or command line
    batch_size = args.batch_size if args.batch_size is not None else config.get_default_batch_size()
    learning_rate = args.learning_rate if args.learning_rate is not None else config.get_default_learning_rate()
    epochs = args.epochs if args.epochs is not None else config.get_default_epochs()
    train_samples = args.train_samples if args.train_samples is not None else config.get_default_train_samples()
    val_samples = args.val_samples if args.val_samples is not None else config.get_default_val_samples()
    
    # Print current configuration
    config.print_current_config()
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size} (config: {config.get_default_batch_size()})")
    print(f"Learning Rate: {learning_rate} (config: {config.get_default_learning_rate()})")
    print(f"Epochs: {epochs} (config: {config.get_default_epochs()})")
    
    # Print data loading information
    if train_samples is None:
        print("📊 Training: Using ALL available samples")
    else:
        print(f"📊 Training: Using {train_samples} samples")
    
    if val_samples is None:
        print("📊 Validation: Using ALL available samples")
    else:
        print(f"📊 Validation: Using {val_samples} samples")
    
    # Memory tracking
    print(f"Memory [STARTUP]: {get_memory_usage()}")
    print(f"Memory [BEFORE_DATA_LOADING]: {get_memory_usage()}")
    
    # Create data loaders using centralized path configuration
    data_loader = IndianaDataLoader(
        batch_size=batch_size,
        use_shards=True,
        shard_size=config.get_current_config()["shard_size"],
        shard_subfolder=config.DATASET_MODE
    )
    
    # Load the data and get PyTorch datasets
    # Note: max_samples=None means use all samples
    data_loader.load_data(max_samples=train_samples, skip_processing=True)
    train_dataset = data_loader.get_data(max_samples=train_samples)
    val_dataset = data_loader.get_validation_data(num_samples=val_samples)
    
    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, vocab: {VOCAB_SIZE}")
    
    print(f"Memory [AFTER_DATA_LOADING]: {get_memory_usage()}")
    gc.collect()
    print(f"Garbage collection: Freed {gc.collect():.1f}MB")
    
    # Initialize model
    print("Building model with BranchEncoder architecture...")
    
    fusion_model = MultimodalFusion(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"]
    ).to(device)
    
    # Initialize trainer
    viz_dir = os.path.join('outputs', f'train_visualizations_{args.experiment_name}')
    trainer = EnhancedRetrievalTrainer(
        model=fusion_model,
        learning_rate=learning_rate,
        device=device,
        experiment_name=args.experiment_name,
        model_save_path=os.path.join('saved_models', args.experiment_name, f'model_{args.experiment_name}.pth'),
        viz_dir=viz_dir
    )
    
    print("\nStarting Dual Branch Training")
    print("Using BranchEncoder architecture")
    print(f"{epochs} epochs, {len(train_loader)} steps per epoch")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs
    )
    
    # Generate training visualizations
    print("\nGenerating training visualizations...")
    trainer.viz.plot_training_progress()
    
    # Generate dual branch loss visualizations
    print("\nGenerating dual branch loss visualizations...")
    trainer.viz.plot_dual_branch_losses()
    
    # If validation data is available, create comprehensive analysis
    if val_loader is not None:
        print("\nGenerating comprehensive analysis...")
        trainer.viz.create_comprehensive_analysis(fusion_model, val_loader, epoch=epochs)
    
    print("\nDUAL BRANCH TRAINING COMPLETED!")
    print("Training completed!")

if __name__ == "__main__":
    main() 