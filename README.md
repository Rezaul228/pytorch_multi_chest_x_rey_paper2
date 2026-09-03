# PyTorch Multimodal Chest X-Ray Retrieval Model

This repository contains a PyTorch implementation of a multimodal model for chest X-ray image and text analysis. The model uses a dual-branch architecture with synergy and difference learning pathways, refactored to exactly match the original TensorFlow/Keras architecture.

## 🎯 Key Features

- **TensorFlow-Matching Architecture**: Refactored encoders that exactly match the original TensorFlow implementation
- **Dual-Branch Design**: Synergy and Difference branches for complementary learning
- **Centralized Configuration**: Easy dataset switching and parameter management
- **SLURM Integration**: Simplified job submission with automatic configuration loading
- **Comprehensive Evaluation**: Recall@1, @5, @10 metrics with training visualizations
- **GPU Acceleration**: CUDA support with memory optimization

## 🏗️ Model Architecture

The model consists of two main branches:
1. **Synergy Branch** (6.45M parameters) - Learns complementary features
2. **Difference Branch** (6.45M parameters) - Learns distinguishing features

**Total model size**: 12.9M parameters (49.24 MB)

### Key Components:
- **Image Encoder**: 4 conv blocks with batch norm, dropout, max pooling
- **Text Encoder**: 2 bidirectional LSTM layers with dense projections
- **Hierarchical Co-Attention**: Local and global attention with learnable gates
- **Cross-Modal Fusion**: Dual-branch processing with final combination
- **Contrastive Learning**: Temperature-scaled similarity learning

## 📁 Project Structure

```
.
├── config.py                              # Centralized configuration system
├── base_models_refactored_v1.py          # Refactored model architecture
├── train_retrieval_v1.py                 # Training script with dual branch monitoring
├── data_loader_v1.py                     # Data loading and preprocessing
├── all_visualization_v1.py               # Training visualization utilities
├── train_test_cross_modal_evaluation_v1.py # Cross-modal retrieval evaluation
├── submit_pytorch_training.sh            # SLURM job submission script
├── paths.py                              # Path configurations
├── saved_models/                         # Trained model weights
├── outputs/                              # Training visualizations and results
├── logs/                                 # SLURM job logs
└── shards/                               # Data shards
    ├── train/
    └── val/
```

## ⚙️ Configuration System

All parameters are centralized in `config.py`:

```python
# Dataset Selection
DATASET_MODE = "mimic_shards"  # Options: "mimic_shards", "indiana_shards"

# Dataset-specific configurations
DATASET_CONFIGS = {
    "mimic_shards": {
        "vocab_size": 2009,
        "max_token_length": 115,
        "embed_dim": 256,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 30,
        # ... more parameters
    },
    "indiana_shards": {
        "vocab_size": 2007,
        "max_token_length": 64,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 20,
        # ... more parameters
    }
}
```

### Switching Datasets

To switch between datasets, simply change `DATASET_MODE` in `config.py`:

```python
# For MIMIC-CXR dataset
DATASET_MODE = "mimic_shards"

# For Indiana dataset  
DATASET_MODE = "indiana_shards"
```

## 📈 Learning Rate Configuration

The model uses a **fixed learning rate** of 1e-04 for consistent training performance. This approach provides stable training without the complexity of learning rate scheduling.

### 🎯 Fixed Learning Rate Approach

The model maintains a constant learning rate throughout training:

- **Learning Rate**: 1e-04 (fixed)
- **No Warmup/Decay**: Consistent learning rate from start to finish
- **Stable Training**: Predictable training behavior

### ⚙️ Current Learning Rate Values

| Dataset | Learning Rate | Source |
|---------|---------------|---------|
| **MIMIC-CXR** | 1e-4 | `config.py` → `learning_rate: 1e-4` |
| **Indiana** | 1e-4 | `config.py` → `learning_rate: 1e-4` |

### 🔧 How to Modify Learning Rate

#### Option 1: Change in `config.py` (Recommended)

```python
# In config.py, modify the learning_rate for your dataset
DATASET_CONFIGS = {
    "mimic_shards": {
        # ... other parameters
        "learning_rate": 5e-5,  # ← Change this value
        # ... other parameters
    }
}
```

#### Option 2: Command Line Override

```bash
# Override learning rate via command line
python train_retrieval_v1.py \
    --experiment_name "lr_test" \
    --learning_rate 5e-5 \
    --epochs 10
```

### 📊 Training Impact

#### Benefits of Fixed Learning Rate:
- **🎯 Predictable Training**: Consistent learning rate throughout
- **🛡️ Stability**: No learning rate fluctuations
- **⚡ Simplicity**: No scheduler complexity
- **🔍 Easy Monitoring**: Learning rate remains constant

### 🔍 Monitoring Learning Rate

The learning rate remains constant throughout training. You can verify this during training:

```python
# Current learning rate (in training loop)
current_lr = trainer.optimizer.param_groups[0]['lr']
print(f"Current LR: {current_lr:.2e}")  # Will always show 1e-4
```

### 🚀 Quick Examples

#### Example 1: Higher Learning Rate
```python
# In config.py
"learning_rate": 2e-4,  # Double the learning rate
```

#### Example 2: Lower Learning Rate
```python
# In config.py
"learning_rate": 5e-5,  # Half the learning rate
```

### ⚠️ Important Notes

1. **Fixed Rate**: Learning rate remains constant throughout training
2. **No Scheduler**: No automatic learning rate adjustments
3. **Manual Changes**: Can be changed via config.py or command line
4. **Stable Training**: Predictable training behavior

### 🎯 Best Practices

- **Start with defaults**: Use the default 1e-4 learning rate
- **Adjust based on dataset size**: Larger datasets can use higher learning rates
- **Monitor training curves**: Watch for signs of overfitting or underfitting
- **Experiment carefully**: Try different learning rates if needed

## 🚀 Usage

### Quick Start

1. **Setup Environment**:
   ```bash
   conda create -n multi_pytorch python=3.8
   conda activate multi_pytorch
   pip install torch torchvision torchaudio
   pip install numpy tqdm matplotlib psutil
   ```

2. **Configure Dataset** (in `config.py`):
   ```python
   DATASET_MODE = "mimic_shards"  # or "indiana_shards"
   ```

3. **Run Training**:
   ```bash
   # Basic training (all parameters from config.py)
   ./submit_pytorch_training.sh --experiment_name my_experiment
   
   # High-resource training
   ./submit_pytorch_training.sh --experiment_name large_run --gpus 2 --mem 64G
   
   # Quick test
   ./submit_pytorch_training.sh --experiment_name quick_test --time_limit 02:00:00
   ```

### SLURM Job Submission (not working )

The simplified SLURM script only requires an experiment name:

```bash
# Basic usage
./submit_pytorch_training.sh --experiment_name pytorch_multimodal_test

# With SLURM options
./submit_pytorch_training.sh --experiment_name large_exp \
    --gpus 2 \
    --mem 64G \
    --time_limit 24:00:00 \
    --partition pascal
```

**Available SLURM Options**:
- `--experiment_name NAME` (required)
- `--partition PARTITION` (default: pascal)
- `--time_limit TIME` (default: 10:00:00)
- `--mem MEMORY` (default: 32G)
- `--cpus NUM` (default: 8)
- `--gpus NUM` (default: 1)

### Config-Based Training Scripts

For simplified training that uses `config.py` for all parameters except experiment name and device, use these scripts:

#### Option 1: Simple SLURM Script (`submit_training_simple.sh`) (working)

This script only requires an experiment name and optionally a device:

```bash (working)
# Basic usage (uses cuda by default)
./submit_training_simple.sh my_experiment

# Specify device explicitly
./submit_training_simple.sh my_experiment cuda
./submit_training_simple.sh my_experiment cpu

# Examples
./submit_training_simple.sh quick_test cuda
./submit_training_simple.sh full_training cuda
./submit_training_simple.sh cpu_test cpu
```

**What this script does:**
- ✅ Uses ALL parameters from `config.py`
- ✅ Only requires experiment name and device
- ✅ Automatically shows current configuration
- ✅ Easy to use and understand

#### Option 2: Direct SLURM Script (`submit_training_job.sh`)

This is a direct SLURM script that you can edit and submit:

```bash
# 1. Edit the experiment name in the script
nano submit_training_job.sh

# 2. Submit the job
sbatch submit_training_job.sh
```

**To modify the script:**
```bash
# Edit these lines in submit_training_job.sh:
python train_retrieval_v1.py \
    --experiment_name YOUR_EXPERIMENT_NAME \  # ← Change this
    --device cuda                            # ← Change to cpu if needed
```

#### Parameter Priority

When using these scripts, parameters are loaded in this order:

| Parameter | Source | Priority |
|-----------|--------|----------|
| **Experiment Name** | Script argument | 1st |
| **Device** | Script argument | 1st |
| **Dataset** | `config.py` | 2nd |
| **Batch Size** | `config.py` | 2nd |
| **Learning Rate** | `config.py` | 2nd |
| **Epochs** | `config.py` | 2nd |
| **All Others** | `config.py` | 2nd |

#### Quick Examples

```bash
# 1. Quick test with current config
./submit_training_simple.sh quick_test cuda

# 2. Full training with current config
./submit_training_simple.sh full_training cuda

# 3. CPU training for testing
./submit_training_simple.sh cpu_test cpu

# 4. Direct SLURM submission
sbatch submit_training_job.sh
```

#### Monitoring Config-Based Jobs

```bash
# Check all your jobs
squeue -u $USER

# Monitor specific job
squeue -j JOB_ID

# View output
tail -f logs/training_JOB_ID.out

# View errors
tail -f logs/training_JOB_ID.err

# Cancel job
scancel JOB_ID
```

#### Benefits of Config-Based Training

- **🎯 Simple**: Only specify experiment name and device
- **📋 Centralized**: All parameters in `config.py`
- **🔄 Flexible**: Easy to switch datasets by changing config
- **⚙️ Consistent**: Same parameters across all runs
- **🚀 Fast**: No need to remember all parameters

#### Switching Datasets

To switch datasets, simply edit `config.py`:

```python
# For MIMIC-CXR dataset
DATASET_MODE = "mimic_shards"

# For Indiana dataset
DATASET_MODE = "indiana_shards"
```

Then run your training script - it will automatically use the new dataset!

### Local Training (Without SLURM)

For local training without SLURM, you can override configuration parameters with command-line arguments:

```bash
# Basic local training with config defaults
python train_retrieval_v1.py --experiment_name local_test --device cuda

# Override configuration parameters
python train_retrieval_v1.py \
    --experiment_name="pytorch_viz_test" \
    --batch_size=16 \
    --learning_rate=1e-4 \
    --epochs=3 \
    --device=cpu

# Quick test with small dataset
python train_retrieval_v1.py \
    --experiment_name="quick_test" \
    --train_samples=1000 \
    --val_samples=200 \
    --epochs=2 \
    --device=cpu

# GPU training with custom parameters
python train_retrieval_v1.py \
    --experiment_name="gpu_test" \
    --batch_size=32 \
    --learning_rate=5e-5 \
    --epochs=10 \
    --device=cuda
```

**Available Local Training Options**:
- `--experiment_name NAME` (required)
- `--batch_size SIZE` (optional, default: from config.py)
- `--learning_rate RATE` (optional, default: from config.py)
- `--epochs NUM` (optional, default: from config.py)
- `--train_samples NUM` (optional, None = use all)
- `--val_samples NUM` (optional, None = use all)
- `--device DEVICE` (cpu/cuda, default: cpu)

**Note**: When running locally, you can override any parameter from `config.py` using command-line arguments. Parameters not specified will use the centralized configuration values.

### Model Testing and Evaluation

#### Testing with SLURM

Create a test script for SLURM submission:

```bash
# Create test script
cat > submit_test_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=pascal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

# Set working directory
cd /home/abedin/Developments/pytorch_multi_chest_x_ray

# Run evaluation on test data
python train_test_cross_modal_evaluation_v1.py

echo "Testing completed!"
EOF

# Submit test job
sbatch submit_test_job.sh
```

#### Testing Locally

The evaluation script provides functions for testing trained models. You can create a simple test script:

```python
# test_model.py
import torch
from base_models_refactored_v1 import MultimodalFusion
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval
import config

# Load trained model
model_path = "saved_models/EXPERIMENT_NAME/export/model_weights.pth"
model = MultimodalFusion(vocab_size=config.get_vocab_size())
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Load test data (you'll need to implement this based on your data format)
# test_data = load_test_data()

# Evaluate
results = evaluate_cross_modal_retrieval(
    model=model,
    test_data=test_data,
    k_values=[1, 5, 10],
    visualize=True,
    output_dir="outputs/test_results/"
)

print("Evaluation Results:", results)
```

#### Quick Model Verification

```bash
# Check if model can be loaded
python -c "
import torch
import config
from base_models_refactored_v1 import MultimodalFusion

# Create model
model = MultimodalFusion(vocab_size=config.get_vocab_size())
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

# Test forward pass
import torch
dummy_images = torch.randn(2, 3, 224, 224)
dummy_texts = torch.randint(0, config.get_vocab_size(), (2, config.get_max_token_length()))

with torch.no_grad():
    img_emb, txt_emb = model((dummy_images, dummy_texts))
    print(f'Forward pass successful: {img_emb.shape}, {txt_emb.shape}')
"

# Check model file sizes
ls -lh saved_models/EXPERIMENT_NAME/export/
```

#### Evaluation Functions

The evaluation script provides these main functions:

- `evaluate_cross_modal_retrieval()`: Evaluate on complete test data
- `evaluate_cross_modal_retrieval_streaming()`: Evaluate on streaming/batched data

**Available Metrics**:
- **MRR** (Mean Reciprocal Rank)
- **Mean/Median Rank**
- **Recall@K** (K=1, 5, 10)
- **Rank Percentages** (≤1, ≤5, ≤10, ≤50, ≤100)

### Complete Workflow Examples

#### 1. Quick Local Development

```bash
# 1. Train a small model locally
python train_retrieval_v1.py \
    --experiment_name="dev_test" \
    --train_samples=1000 \
    --val_samples=200 \
    --epochs=3 \
    --device=cpu

#Or activation with conda env
source /opt/conda/etc/profile.d/conda.sh && conda activate multi_pytorch && python train_retrieval_v1.py --experiment_name="final_simplified_test" --train_samples=150 --val_samples=30 --epochs=5 --device=cpu


# 2. Verify the model works
python -c "
import torch
model = torch.load('saved_models/dev_test/export/model_weights.pth', map_location='cpu')
print('Model loaded successfully')
"
```

#### 2. Full SLURM Training and Testing

```bash
# 1. Train model on SLURM
./submit_pytorch_training.sh --experiment_name full_training

# 2. Wait for training to complete, then verify
# Check job status
squeue -u $USER

# View results when complete
ls saved_models/full_training/export/
cat logs/pytorch_train_JOB_ID.out | tail -20
```

#### 3. Hyperparameter Tuning

```bash
# Test different configurations locally
for lr in 1e-4 5e-5 1e-5; do
    for bs in 16 32 64; do
        python train_retrieval_v1.py \
            --experiment_name="tune_lr${lr}_bs${bs}" \
            --train_samples=5000 \
            --val_samples=1000 \
            --epochs=5 \
            --device=cpu
    done
done
```

## 📊 Model Parameters

**Current Configuration** (MIMIC-CXR):
- **Vocab Size**: 2009
- **Max Token Length**: 115
- **Embedding Dimension**: 256
- **Number of Attention Heads**: 8
- **Number of Co-attention Layers**: 2
- **Batch Size**: 128
- **Learning Rate**: 1e-4
- **Number of Epochs**: 30
- **Temperature**: 0.07

## 📈 Performance Metrics

Latest training results (MIMIC-CXR, 30 epochs):
- **Recall@1**: 90.43%
- **Recall@5**: 99.36%
- **Recall@10**: 99.61%
- **Training Time**: ~58 minutes (GPU)

## 💾 Model Weights

Pre-trained weights are stored in:
```
saved_models/EXPERIMENT_NAME/
├── export/
│   ├── model.pth (Complete state)
│   └── model_weights.pth (Weights only)
└── model_EXPERIMENT_NAME.pth (Backup)
```

## 🔧 Data Requirements

The model expects data in the following format:
- **Images**: (batch_size, 3, 224, 224) - RGB format
- **Text**: (batch_size, sequence_length) - Tokenized text

Data should be organized in shards for efficient loading:
```
shards/
├── train/ (training shards)
└── val/ (validation shards)
```

## 📋 Monitoring and Visualization

### Job Monitoring
```bash
# Check job status
squeue -j JOB_ID

# View live output
tail -f logs/pytorch_train_JOB_ID.out

# View errors
tail -f logs/pytorch_train_JOB_ID.err

# Cancel job
scancel JOB_ID
```

### Training Visualizations
- **Training Progress**: Loss curves, recall metrics
- **Similarity Matrices**: Cross-modal similarity analysis
- **Comprehensive Analysis**: Detailed evaluation plots

Generated in: `outputs/train_visualizations_EXPERIMENT_NAME/`

## 🎯 Key Improvements

### Architecture Refactoring
- **4 distinct conv blocks** in ImageEncoder (matching TensorFlow)
- **Two sequential bidirectional LSTM layers** in TextEncoder
- **TensorFlow-style layer naming** for consistency
- **21% parameter reduction** vs old model
- **Exact output shape matching** with TensorFlow

### Configuration System
- **Centralized parameters** in `config.py`
- **Easy dataset switching** without code changes
- **Automatic parameter loading** in all scripts
- **No command-line parameter confusion**

### Training Enhancements
- **Proper dual-branch loss calculation** (fixed critical bug)
- **Individual branch monitoring** (synergy vs difference)
- **Comprehensive evaluation** with all samples
- **GPU memory optimization**

## 🔍 Troubleshooting

### Common Issues

1. **Job fails immediately**:
   ```bash
   # Check error log
   cat logs/pytorch_train_JOB_ID.err
   
   # Verify data path exists
   ls -la shards/
   ```

2. **Out of memory**:
   ```bash
   # Reduce batch size in config.py
   "batch_size": 64  # instead of 128
   ```

3. **Dataset not found**:
   ```bash
   # Check current dataset mode
   python -c "import config; print(config.DATASET_MODE)"
   
   # Switch dataset if needed
   python -c "import config; config.switch_dataset('indiana_shards')"
   ```

### Environment Issues
- Ensure `multi_pytorch` conda environment is available
- Check that PyTorch is installed with CUDA support
- Verify data shards are in `./shards/` directory

## 📚 References

- **Original TensorFlow Implementation**: MIMIC-CXR Multimodal Retrieval
- **Dual-Branch Architecture**: Synergy and Difference Learning
- **Hierarchical Co-Attention**: Local and Global Attention Mechanisms
- **Contrastive Learning**: Temperature-scaled Similarity Learning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the configuration system
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 