#!/bin/bash

# Setup script for new PyTorch Multimodal Chest X-Ray project
echo "🚀 Setting up new PyTorch Multimodal Chest X-Ray project"
echo "========================================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found"
    
    # Create new conda environment
    echo "📦 Creating conda environment 'multi_pytorch_new'..."
    conda create -n multi_pytorch_new python=3.8 -y
    
    # Activate environment
    echo "🔄 Activating conda environment..."
    source /opt/conda/etc/profile.d/conda.sh
    conda activate multi_pytorch_new
    
    # Install PyTorch with CUDA support
    echo "🔥 Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install other requirements
    echo "📚 Installing other dependencies..."
    pip install -r requirements.txt
    
    echo "✅ Environment setup complete!"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Activate environment: conda activate multi_pytorch_new"
    echo "2. Update config.py with your dataset configuration"
    echo "3. Update paths.py with your data paths"
    echo "4. Run training: python train_retrieval_v1.py --experiment_name your_exp"
    
else
    echo "⚠️  Conda not found. Please install dependencies manually:"
    echo "   pip install -r requirements.txt"
fi

echo ""
echo "📁 Project structure created:"
echo "   ├── config.py                              # Configuration"
echo "   ├── base_models_refactored_v1.py          # Model architecture"
echo "   ├── train_retrieval_v1.py                 # Training script"
echo "   ├── data_loader_v1.py                     # Data loading"
echo "   ├── all_visualization_v1.py               # Visualization"
echo "   ├── train_test_cross_modal_evaluation_v1.py # Evaluation"
echo "   ├── paths.py                              # Path configuration"
echo "   ├── test_trained_model.py                 # Model testing"
echo "   ├── submit_training_simple.sh             # SLURM submission"
echo "   ├── requirements.txt                      # Dependencies"
echo "   ├── outputs/                              # Training outputs"
echo "   ├── logs/                                 # Training logs"
echo "   ├── saved_models/                         # Model weights"
echo "   ├── checkpoints/                          # Training checkpoints"
echo "   ├── shards/                               # Data shards"
echo "   └── data/                                 # Raw data"
