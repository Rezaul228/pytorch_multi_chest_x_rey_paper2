#!/usr/bin/env python3
"""
Centralized Configuration for MIMIC-CXR Multimodal Retrieval
All dataset-specific parameters can be changed from this single file.
"""

import os

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Dataset Selection
# Change this to switch between datasets
DATASET_MODE = "mimic_shards_hybrid_full_ori"  # Options: "mimic_shards", "indiana_shards", "augmented_data", "mimic_shards_hybrid_full_ori"

# Dataset-specific configurations
DATASET_CONFIGS = {
    "mimic_shards": {
        "vocab_size": 10805,           # Vocabulary size from tokenizer
        "max_token_length": 128,      # Maximum sequence length
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 128,             # Default batch size (changed from 128)
        "learning_rate": 1e-4,        # Default learning rate (changed from 5e-5)
        "epochs": 50,                 # Default number of epochs
        "shard_size": 100,            # Number of samples per shard file (for memory management)
        "train_samples": None,        # Number of training samples (None = use ALL available)
        "val_samples": None,          # Number of validation samples (None = use ALL available)
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    },
    
    "indiana_shards": {
        "vocab_size": 10805,#2559           # Vocabulary size from tokenizer (updated)
        "max_token_length": 128,#64      # Maximum sequence length (updated)
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 16,             # Default batch size
        "learning_rate": 5e-5, #1e-4       # Default learning rate
        "epochs": 15,                 # Default number of epochs
        "shard_size": 100,             # Number of samples per shard file (for memory management)
        "train_samples": None,        # Number of training samples (None = use ALL available)
        "val_samples": None,          # Number of validation samples (None = use ALL available)
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/indiana_shards"
    },
    
    "augmented_data": {
        "vocab_size": 2552,           # Vocabulary size from tokenizer (same as mimic_shards)
        "max_token_length": 128,      # Maximum sequence length (same as mimic_shards)
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 64,             # Default batch size
        "learning_rate": 1e-4,        # Default learning rate
        "epochs": 30,                 # Default number of epochs
        "shard_size": 100,            # Number of samples per shard file (for memory management)
        "train_samples": None,        # Number of training samples (None = use ALL available)
        "val_samples": None,          # Number of validation samples (None = use ALL available)
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/augmented_data"
    },
    
    "mimic_shards_hybrid_full_ori": {
        "vocab_size": 10805,           # Vocabulary size from tokenizer (same as indiana_shards)
        "max_token_length": 128,       # Maximum sequence length (same as indiana_shards)
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 256,             # Default batch size
        "learning_rate": 5e-5,#1e-4,        # Default learning rate
        "epochs": 5,                  # TEST: 5 epochs for quick test
        "shard_size": 100,             # Number of samples per shard file (for memory management)
        "train_samples": 1000,        # TEST: 1000 training samples for quick test
        "val_samples": 200,           # TEST: 200 validation samples for quick test
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori"
    },
    
    "mimic_shards_hufc4446-to128": {
        "vocab_size": 4446,            # Vocabulary size from tokenizer (previous working model)
        "max_token_length": 128,       # Maximum sequence length
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 128,             # Default batch size
        "learning_rate": 1e-4,        # Default learning rate
        "epochs": 60,                 # Default number of epochs
        "shard_size": 100,            # Number of samples per shard file (for memory management)
        "train_samples": None,        # Number of training samples (None = use ALL available)
        "val_samples": None,          # Number of validation samples (None = use ALL available)
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    },
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training parameters (can be overridden by command line arguments)
TRAINING_CONFIG = {
    "weight_decay": 0.0001,           # Weight decay for optimizer
    "gradient_clip": 1.0,             # Gradient clipping threshold
    "early_stopping_patience": 10,    # Early stopping patience
    "save_best_model": True,          # Save best model based on validation loss
    "save_frequency": 5,              # Save model every N epochs
    "validation_frequency": 1,        # Validate every N epochs
    "log_frequency": 10,              # Log training progress every N steps
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model architecture parameters
MODEL_CONFIG = {
    "use_pretrained": True,           # Use pretrained image encoder
    "freeze_image_encoder": False,    # Freeze image encoder weights
    "dropout_rate": 0.1,              # Dropout rate for model layers
    "activation": "relu",             # Activation function
    "normalization": "batch",         # Normalization type (batch, layer, none)
}

# =============================================================================
# LOSS CONFIGURATION
# =============================================================================

# Loss function parameters
LOSS_CONFIG = {
    "synergy_weight": 0.65,           # Weight for synergy loss
    "difference_weight": 0.35,        # Weight for difference loss
    "orthogonal_weight": 0.15,        # Weight for orthogonal loss
    "main_loss_weight": 0.20,         # Weight for main contrastive loss
    "temperature": 0.07,              # Temperature for contrastive loss
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def switch_dataset(dataset_name):
    """Switch the active dataset configuration."""
    global DATASET_MODE
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    DATASET_MODE = dataset_name
    return get_current_config()


def get_current_config():
    """Get the current dataset configuration"""
    return DATASET_CONFIGS.get(DATASET_MODE, DATASET_CONFIGS["mimic_shards_hybrid_full_ori"])

def get_vocab_size():
    """Get vocabulary size for current dataset"""
    return get_current_config()["vocab_size"]

def get_max_token_length():
    """Get maximum token length for current dataset"""
    return get_current_config()["max_token_length"]

def get_embed_dim():
    """Get embedding dimension for current dataset"""
    return get_current_config()["embed_dim"]

def get_batch_size():
    """Get default batch size for current dataset"""
    return get_current_config()["batch_size"]

def get_default_batch_size():
    """Get default batch size for current dataset (alias for compatibility)"""
    return get_current_config()["batch_size"]

def get_learning_rate():
    """Get default learning rate for current dataset"""
    return get_current_config()["learning_rate"]

def get_default_learning_rate():
    """Get default learning rate for current dataset (alias for compatibility)"""
    return get_current_config()["learning_rate"]

def get_epochs():
    """Get default number of epochs for current dataset"""
    return get_current_config()["epochs"]

def get_default_epochs():
    """Get default number of epochs for current dataset (alias for compatibility)"""
    return get_current_config()["epochs"]

def get_default_weight_decay():
    """Get default weight decay for training"""
    return TRAINING_CONFIG["weight_decay"]

def get_default_train_samples():
    """Get default number of training samples (alias for compatibility)"""
    return get_current_config()["train_samples"]

def get_default_val_samples():
    """Get default number of validation samples (alias for compatibility)"""
    return get_current_config()["val_samples"]

def get_data_path():
    """Get data path for current dataset"""
    return get_current_config()["data_path"]

def get_train_samples():
    """Get number of training samples for current dataset"""
    return get_current_config()["train_samples"]

def get_val_samples():
    """Get number of validation samples for current dataset"""
    return get_current_config()["val_samples"]

def print_current_config():
    """Print the current configuration"""
    config = get_current_config()
    print("\n📋 Current Dataset Configuration:")
    print(f"   Dataset: {DATASET_MODE}")
    print(f"   Vocab Size: {config['vocab_size']}")
    print(f"   Max Token Length: {config['max_token_length']}")
    print(f"   Embedding Dimension: {config['embed_dim']}")
    print(f"   Default Batch Size: {config['batch_size']}")
    print(f"   Default Learning Rate: {config['learning_rate']}")
    print(f"   Default Epochs: {config['epochs']}")
    print(f"   Shard Size: {config['shard_size']} (samples per shard file)")
    print(f"   Train Samples: {config['train_samples']}")
    print(f"   Val Samples: {config['val_samples']}")
    print(f"   Data Path: {config['data_path']}")
    
    print("\n🔧 Training Configuration:")
    print(f"   Weight Decay: {TRAINING_CONFIG['weight_decay']}")
    print(f"   Gradient Clip: {TRAINING_CONFIG['gradient_clip']}")
    print(f"   Early Stopping Patience: {TRAINING_CONFIG['early_stopping_patience']}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🔧 Configuration Module")
    print("=" * 50)
    print_current_config()
