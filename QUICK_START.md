# Quick Start Guide - PyTorch Multimodal Chest X-Ray Project

## 🚀 Setup

1. **Clone or copy this project** to your desired location
2. **Run the setup script**:
   ```bash
   ./setup_new_project.sh
   ```
3. **Activate the conda environment**:
   ```bash
   conda activate multi_pytorch_new
   ```

## ⚙️ Configuration

### 1. Update Data Paths (`paths.py`)
Edit the centralized data paths in `paths.py`:
```python
# Base path for all processed data
CENTRALIZED_DATA_BASE = "/path/to/your/processed/data/"

# Dataset-specific paths
DATASET_PATHS = {
    "indiana_shards": "/path/to/your/indiana_shards/",
    "mimic_shards": "/path/to/your/mimic_shards/"
}
```

### 2. Configure Dataset (`config.py`)
Choose your dataset and configure parameters:
```python
# Dataset Selection
DATASET_MODE = "indiana_shards"  # or "mimic_shards"

# Dataset-specific configurations
DATASET_CONFIGS = {
    "indiana_shards": {
        "vocab_size": 2007,
        "max_token_length": 64,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 20,
        # ... other parameters
    }
}
```

## 🎯 Training

### Basic Training
```bash
python train_retrieval_v1.py --experiment_name my_experiment
```

### With Custom Parameters
```bash
python train_retrieval_v1.py \
    --experiment_name my_experiment \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --epochs 30 \
    --device cuda
```

### SLURM Submission
```bash
./submit_training_simple.sh my_experiment cuda
```

## 📊 Evaluation

### Test Trained Model
```bash
python test_trained_model.py
```

### Cross-Modal Evaluation
```python
from train_test_cross_modal_evaluation_v1 import evaluate_cross_modal_retrieval

results = evaluate_cross_modal_retrieval(
    model=your_model,
    test_data=test_data,
    k_values=[1, 5, 10],
    visualize=True
)
```

## 📁 Project Structure

```
.
├── config.py                              # Centralized configuration
├── base_models_refactored_v1.py          # Model architecture
├── train_retrieval_v1.py                 # Training script
├── data_loader_v1.py                     # Data loading
├── all_visualization_v1.py               # Visualization utilities
├── train_test_cross_modal_evaluation_v1.py # Evaluation functions
├── paths.py                              # Path configuration
├── test_trained_model.py                 # Model testing
├── submit_training_simple.sh             # SLURM submission
├── requirements.txt                      # Dependencies
├── setup_new_project.sh                  # Setup script
├── outputs/                              # Training outputs
├── logs/                                 # Training logs
├── saved_models/                         # Model weights
├── checkpoints/                          # Training checkpoints
├── shards/                               # Data shards
└── data/                                 # Raw data
```

## 🔧 Key Features

- **Dual-Branch Architecture**: Synergy and Difference branches
- **Centralized Configuration**: Easy dataset switching
- **SLURM Integration**: Automated job submission
- **Comprehensive Evaluation**: Recall@1, @5, @10 metrics
- **Visualization**: Training progress and retrieval examples
- **GPU Acceleration**: CUDA support with memory optimization

## 📈 Model Architecture

- **Image Encoder**: 4 conv blocks with batch norm, dropout, max pooling
- **Text Encoder**: 2 bidirectional LSTM layers with dense projections
- **Hierarchical Co-Attention**: Local and global attention with learnable gates
- **Cross-Modal Fusion**: Dual-branch processing with final combination
- **Contrastive Learning**: Temperature-scaled similarity learning

## 🎯 Performance Metrics

- **MRR (Mean Reciprocal Rank)**
- **Recall@K** (K=1, 5, 10)
- **Mean/Median Rank**
- **Rank Percentages**

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `config.py`
2. **Tokenizer Loading Error**: Check metadata.pkl exists in shards/
3. **Data Path Not Found**: Update paths in `paths.py`
4. **Import Errors**: Ensure all dependencies installed via `requirements.txt`

### Memory Optimization

- Use smaller batch sizes for large datasets
- Enable gradient checkpointing for large models
- Use mixed precision training (FP16)
- Monitor memory usage with `psutil`

## 📚 Documentation

- `README.md`: Comprehensive project documentation
- `SLURM_USAGE.md`: SLURM job submission guide
- `MULTIMODAL_CHEST_XRAY_RETRIEVAL_TECHNICAL_REPORT.txt`: Technical details
