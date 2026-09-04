#!/usr/bin/env python3
"""
Simple Path Configuration for Chest X-ray Project

🎯 CENTRALIZED DATA LOADING: All preprocessed CXR shards under one directory!

FOR SERVER DEPLOYMENT:
---------------------
1. Copy project to server
2. Edit the CENTRALIZED_DATA_BASE_PATH below to match your server structure
3. Use shard_subfolder argument to switch between datasets
4. That's it!
"""

import os

# ==========================================
# 🎯 CENTRALIZED DATA CONFIGURATION
# ==========================================

# Centralized base path for all processed CXR data
CENTRALIZED_DATA_BASE_PATH = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/"

# Available shard subfolders
AVAILABLE_SHARD_SUBFOLDERS = ["mimic_shards_voc-2007","indiana_shards", "mimic_shards", "mimic_shards_hybrid_full_ori", "mimic_shards_hufc4446-to128", "indiana_shards2559-64"]

# Default subfolder selection
DEFAULT_SHARD_SUBFOLDER = "indiana_shards"

# ==========================================
# 🚀 CHANGE THESE PATHS FOR YOUR SERVER
# ==========================================

# Base directories (relative to project root)
DATA_DIR = "data"
SAVED_MODELS_DIR = "saved_models"  
OUTPUTS_DIR = "outputs"

# ==========================================
# 🔧 DYNAMIC PATH FUNCTIONS
# ==========================================

def get_shard_base_path(shard_subfolder=None):
    """Get the base path for shards based on subfolder selection"""
    if shard_subfolder is None:
        shard_subfolder = DEFAULT_SHARD_SUBFOLDER
    
    if shard_subfolder not in AVAILABLE_SHARD_SUBFOLDERS:
        raise ValueError(f"Invalid shard_subfolder: {shard_subfolder}. Available options: {AVAILABLE_SHARD_SUBFOLDERS}")
    
    return os.path.join(CENTRALIZED_DATA_BASE_PATH, shard_subfolder)

def get_train_shards_dir(shard_subfolder=None):
    """Get training shards directory"""
    return os.path.join(get_shard_base_path(shard_subfolder), "train")

def get_val_shards_dir(shard_subfolder=None):
    """Get validation shards directory"""
    return os.path.join(get_shard_base_path(shard_subfolder), "val")

def get_test_shards_dir(shard_subfolder=None):
    """Get test shards directory"""
    return os.path.join(get_shard_base_path(shard_subfolder), "test")

def get_metadata_path(shard_subfolder=None):
    """Get metadata file path"""
    return os.path.join(get_shard_base_path(shard_subfolder), "metadata.pkl")

# ==========================================
# 📁 LEGACY COMPATIBILITY (DEPRECATED)
# ==========================================

# Keep these for backward compatibility but mark as deprecated
# TODO: Remove these once all code is updated to use the new functions above

# Legacy paths - use get_shard_base_path() instead
SHARDS_DIR = get_shard_base_path(DEFAULT_SHARD_SUBFOLDER)
TRAIN_SHARDS_DIR = get_train_shards_dir(DEFAULT_SHARD_SUBFOLDER)
VAL_SHARDS_DIR = get_val_shards_dir(DEFAULT_SHARD_SUBFOLDER)
TEST_SHARDS_DIR = get_test_shards_dir(DEFAULT_SHARD_SUBFOLDER)

# External data paths (backup/alternative datasets) - deprecated
MIMIC_DATA_PATH = "/home/abedin/Developments/mimic_cxr_multimodal/data/hf_samples"
INDIANA_DATA_PATH = SHARDS_DIR

# ==========================================
# 📁 DERIVED PATHS (AUTO-GENERATED)
# ==========================================

# Data subdirectories
HF_SAMPLES_DIR = os.path.join(DATA_DIR, "hf_samples")
MINIMAL_SAMPLES_DIR = os.path.join(DATA_DIR, "minimal_samples")

# Model subdirectories  
INDIANA_TRAINED_DIR = os.path.join(SAVED_MODELS_DIR, "indiana_trained")

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def get_model_weights_path(model_name, epochs=5):
    """Get model weights path for a specific model."""
    return os.path.join(INDIANA_TRAINED_DIR, f"{model_name}_{epochs}epochs.weights.h5")

def get_train_viz_dir(model_name):
    """Get training visualization directory for a specific model."""
    return f"train_visualizations_{model_name}"

def get_output_file(filename):
    """Get path for output file."""
    return os.path.join(OUTPUTS_DIR, filename)

def ensure_dirs():
    """Create essential directories if they don't exist."""
    dirs_to_create = [
        DATA_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR,
        HF_SAMPLES_DIR, INDIANA_TRAINED_DIR
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
    
def print_current_config(shard_subfolder=None):
    """Print current path configuration"""
    if shard_subfolder is None:
        shard_subfolder = DEFAULT_SHARD_SUBFOLDER
    
    print(f"🔄 DATASET MODE: {shard_subfolder}")
    print(f"📁 Base path: {get_shard_base_path(shard_subfolder)}")
    print(f"🚂 Train: {get_train_shards_dir(shard_subfolder)}")
    print(f"✅ Val: {get_val_shards_dir(shard_subfolder)}")
    print(f"🧪 Test: {get_test_shards_dir(shard_subfolder)}")

# Auto-create directories when imported
ensure_dirs() 