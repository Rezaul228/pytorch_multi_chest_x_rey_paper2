# SLURM PyTorch Training Usage Guide

## Overview
The `submit_pytorch_training.sh` script allows you to submit PyTorch multimodal retrieval training jobs to the SLURM cluster with GPU acceleration.

## Quick Start

### Basic Usage
```bash
# Simple training job
./submit_pytorch_training.sh --experiment_name my_experiment

# Debug mode (CPU, smaller settings)
./submit_pytorch_training.sh --experiment_name debug_test --debug

# Custom configuration
./submit_pytorch_training.sh --experiment_name large_exp \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --gpus 2
```

### Training Options
- `--experiment_name NAME` - **Required**: Unique experiment identifier
- `--batch_size SIZE` - Batch size (default: 16)
- `--learning_rate RATE` - Learning rate (default: 1e-4)
- `--epochs NUM` - Number of training epochs (default: 10)
- `--embed_dim DIM` - Embedding dimension (default: 256)
- `--num_heads NUM` - Attention heads (default: 8)
- `--num_layers NUM` - Co-attention layers (default: 2)
- `--device DEVICE` - cuda or cpu (default: cuda)
- `--debug` - Debug mode: CPU, smaller settings

### SLURM Options
- `--partition PARTITION` - SLURM partition (default: pascal)
- `--time_limit TIME` - Time limit HH:MM:SS (default: 04:00:00)
- `--mem MEMORY` - Memory allocation (default: 32G)
- `--cpus NUM` - CPU cores (default: 8)
- `--gpus NUM` - Number of GPUs (default: 1)

## Monitoring Jobs

### Check Job Status
```bash
# View your jobs
squeue -u $USER

# Check specific job
squeue -j JOB_ID

# Detailed job info
scontrol show job JOB_ID
```

### View Output
```bash
# Live output
tail -f logs/pytorch_train_JOB_ID.out

# Error log
tail -f logs/pytorch_train_JOB_ID.err

# Job statistics
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,MaxRSS,ReqMem,AllocCPUS,ReqGRES,Elapsed
```

### Cancel Job
```bash
scancel JOB_ID
```

## Output Files

After successful training, you'll find:

### Model Files
- `saved_models/EXPERIMENT_NAME/export/model.pth` - Complete model
- `saved_models/EXPERIMENT_NAME/export/model_weights.pth` - Model weights
- `saved_models/EXPERIMENT_NAME/model_EXPERIMENT_NAME.pth` - Backup weights

### Visualizations
- `outputs/train_visualizations_EXPERIMENT_NAME/training_progress_*.png`
- `outputs/train_visualizations_EXPERIMENT_NAME/similarity_matrix_*.png`

### Logs
- `logs/pytorch_train_JOB_ID.out` - Training output and metrics
- `logs/pytorch_train_JOB_ID.err` - Error messages (should be empty)

## Example Workflows

### 1. Quick Test Run
```bash
./submit_pytorch_training.sh --experiment_name quick_test --debug
```
- Uses CPU, 2 epochs, batch size 8
- Good for testing setup

### 2. Standard Training
```bash
./submit_pytorch_training.sh --experiment_name standard_run \
    --epochs 15 \
    --batch_size 16
```
- Uses 1 GPU, standard settings
- Good for most experiments

### 3. Large Scale Training
```bash
./submit_pytorch_training.sh --experiment_name large_scale \
    --epochs 30 \
    --batch_size 32 \
    --gpus 2 \
    --mem 64G \
    --time_limit 08:00:00
```
- Uses 2 GPUs, larger batch size
- Extended time and memory

### 4. Hyperparameter Exploration
```bash
./submit_pytorch_training.sh --experiment_name hp_search_lr5e5 \
    --learning_rate 5e-5 \
    --num_heads 16 \
    --num_layers 3
```
- Custom hyperparameters
- Different architecture settings

## Architecture Features

The script uses the **refactored TensorFlow-matching architecture**:

### ✅ Key Improvements
- **4 distinct conv blocks** in ImageEncoder
- **Two sequential bidirectional LSTM layers** in TextEncoder
- **TensorFlow-style layer naming** for consistency
- **21% parameter reduction** vs old model
- **Exact output shape matching** with TensorFlow

### 🔄 Dual-Branch Design
- **Synergy branch**: Learns complementary features
- **Difference branch**: Learns distinguishing features
- **Hierarchical co-attention**: Local and global attention mechanisms

### 📊 Performance Metrics
- **Recall@1, @5, @10**: Standard retrieval metrics
- **Training visualizations**: Progress plots and similarity matrices
- **Comprehensive evaluation**: Validation during training

## Troubleshooting

### Common Issues

1. **Job fails immediately**
   ```bash
   # Check error log
   cat logs/pytorch_train_JOB_ID.err
   
   # Try debug mode
   ./submit_pytorch_training.sh --experiment_name debug --debug
   ```

2. **Out of memory**
   ```bash
   # Reduce batch size
   ./submit_pytorch_training.sh --experiment_name small_batch \
       --batch_size 8 --mem 16G
   ```

3. **GPU not available**
   ```bash
   # Use CPU mode
   ./submit_pytorch_training.sh --experiment_name cpu_run \
       --device cpu --gpus 0
   ```

4. **Data not found**
   ```bash
   # Ensure shards directory exists
   ls -la shards/
   
   # Should see train/ and val/ subdirectories
   ```

### Environment Issues
- Ensure `multi_pytorch` conda environment is available
- Check that PyTorch is installed with CUDA support
- Verify data shards are in `./shards/` directory

## Best Practices

1. **Use descriptive experiment names** with dates/versions
2. **Start with debug mode** for new configurations
3. **Monitor resource usage** with `sacct`
4. **Save important experiments** with different names
5. **Check logs regularly** during long training runs

## Performance Tips

- **GPU training**: Use `--gpus 1` or `--gpus 2` for faster training
- **Batch size**: Increase if you have enough GPU memory
- **CPU cores**: Match `--cpus` to your workload
- **Memory**: Start with 32G, adjust based on usage
- **Time limit**: Set generously to avoid job termination 