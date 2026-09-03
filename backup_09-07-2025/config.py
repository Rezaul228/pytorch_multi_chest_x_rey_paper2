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
DATASET_MODE = "mimic_shards"  # Options: "mimic_shards", "indiana_shards"

# Dataset-specific configurations
DATASET_CONFIGS = {
    "mimic_shards": {
        "vocab_size": 2009,           # Vocabulary size from tokenizer
        "max_token_length": 115,      # Maximum sequence length
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 128,            # Default batch size
        "learning_rate": 5e-5,        # Default learning rate
        "epochs": 50,                 # Default number of epochs
        "shard_size": 100,            # Number of samples per shard file (for memory management)
        "train_samples": None,        # Number of training samples (None = use ALL available)
        "val_samples": None,          # Number of validation samples (None = use ALL available)
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    },
    
    "indiana_shards": {
        "vocab_size": 2007,           # Vocabulary size from tokenizer
        "max_token_length": 64,      # Maximum sequence length
        "embed_dim": 256,             # Embedding dimension
        "num_heads": 8,               # Number of attention heads
        "num_layers": 2,              # Number of co-attention layers
        "temperature": 0.07,          # Contrastive loss temperature
        "batch_size": 32,             # Default batch size
        "learning_rate": 1e-4,        # Default learning rate
        "epochs": 20,                 # Default number of epochs
        "shard_size": 50,             # Number of samples per shard file (for memory management)
        "train_samples": None,        # Number of training samples (None = use ALL available)
        "val_samples": None,          # Number of validation samples (None = use ALL available)
        "data_path": "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/indiana_shards"
    }
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Image Encoder Configuration
IMAGE_ENCODER_CONFIG = {
    "input_channels": 3,              # RGB channels
    "conv_channels": [32, 64, 128, 256],  # Channel progression
    "dropout_rate": 0.3,              # Dropout rate for conv layers
    "pool_size": 2,                   # Max pooling kernel size
}

# Text Encoder Configuration
TEXT_ENCODER_CONFIG = {
    "lstm1_hidden_size": 256,         # First LSTM hidden size
    "lstm2_hidden_size": 128,         # Second LSTM hidden size (embed_dim//2)
    "lstm_dropout": 0.5,              # LSTM dropout rate
    "dense_dropout": 0.3,             # Dense layer dropout rate
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Default training parameters (can be overridden by command line)
DEFAULT_TRAINING_CONFIG = {
    "device": "cuda",                 # Device to use (cuda/cpu)
    "num_workers": 0,                 # DataLoader workers
    "pin_memory": True,               # Pin memory for faster GPU transfer
    "gradient_clip": 1.0,             # Gradient clipping value
    "weight_decay": 1e-5,             # Weight decay for optimizer
    "scheduler_patience": 5,          # Learning rate scheduler patience
    "scheduler_factor": 0.5,          # Learning rate reduction factor
    "early_stopping_patience": 10,    # Early stopping patience
    "save_best_only": True,           # Save only best model
    "monitor_metric": "recall@1",     # Metric to monitor for best model
}

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Base paths
BASE_PATHS = {
    "outputs": "outputs",
    "saved_models": "saved_models", 
    "logs": "logs",
    "checkpoints": "checkpoints",
    "shards": "shards"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_current_config():
    """Get configuration for current dataset"""
    if DATASET_MODE not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset mode: {DATASET_MODE}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[DATASET_MODE]

def get_vocab_size():
    """Get vocabulary size for current dataset"""
    return get_current_config()["vocab_size"]

def get_max_token_length():
    """Get maximum token length for current dataset"""
    return get_current_config()["max_token_length"]

def get_embed_dim():
    """Get embedding dimension for current dataset"""
    return get_current_config()["embed_dim"]

def get_data_path():
    """Get data path for current dataset"""
    return get_current_config()["data_path"]

def get_default_batch_size():
    """Get default batch size for current dataset"""
    return get_current_config()["batch_size"]

def get_default_learning_rate():
    """Get default learning rate for current dataset"""
    return get_current_config()["learning_rate"]

def get_default_epochs():
    """Get default number of epochs for current dataset"""
    return get_current_config()["epochs"]

def get_default_weight_decay():
    """Get default weight decay for optimizer"""
    return DEFAULT_TRAINING_CONFIG["weight_decay"]

def get_default_train_samples():
    """Get default number of training samples for current dataset (None = use all)"""
    return get_current_config().get("train_samples", None)

def get_default_val_samples():
    """Get default number of validation samples for current dataset (None = use all)"""
    return get_current_config().get("val_samples", None)

def print_current_config():
    """Print current dataset configuration"""
    config = get_current_config()
    print(f"\n📋 Current Dataset Configuration:")
    print(f"   Dataset: {DATASET_MODE}")
    print(f"   Vocab Size: {config['vocab_size']}")
    print(f"   Max Token Length: {config['max_token_length']}")
    print(f"   Embedding Dimension: {config['embed_dim']}")
    print(f"   Default Batch Size: {config['batch_size']}")
    print(f"   Default Learning Rate: {config['learning_rate']}")
    print(f"   Default Epochs: {config['epochs']}")
    print(f"   Shard Size: {config['shard_size']} (samples per shard file)")
    print(f"   Train Samples: {config.get('train_samples', 'ALL')}")
    print(f"   Val Samples: {config.get('val_samples', 'ALL')}")
    print(f"   Data Path: {config['data_path']}")
    print(f"\n🔧 Training Configuration:")
    print(f"   Weight Decay: {DEFAULT_TRAINING_CONFIG['weight_decay']}")
    print(f"   Gradient Clip: {DEFAULT_TRAINING_CONFIG['gradient_clip']}")
    print(f"   Early Stopping Patience: {DEFAULT_TRAINING_CONFIG['early_stopping_patience']}")
    print()

def switch_dataset(dataset_name):
    """Switch to a different dataset configuration"""
    global DATASET_MODE
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    DATASET_MODE = dataset_name
    print(f"✅ Switched to dataset: {dataset_name}")
    print_current_config()

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate the current configuration"""
    config = get_current_config()
    
    # Check if data path exists
    if not os.path.exists(config["data_path"]):
        print(f"⚠️  Warning: Data path does not exist: {config['data_path']}")
    
    # Validate numeric values
    assert config["vocab_size"] > 0, "Vocab size must be positive"
    assert config["max_token_length"] > 0, "Max token length must be positive"
    assert config["embed_dim"] > 0, "Embedding dimension must be positive"
    assert config["batch_size"] > 0, "Batch size must be positive"
    assert config["learning_rate"] > 0, "Learning rate must be positive"
    assert config["epochs"] > 0, "Number of epochs must be positive"
    
    print("✅ Configuration validation passed")
    return True

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    print_current_config()
    validate_config() 