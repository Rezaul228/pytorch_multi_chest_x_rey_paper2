#!/bin/bash

# Simple PyTorch Training Submission Script
# Uses config.py for all parameters except experiment_name and device

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide an experiment name"
    echo "Usage: $0 <experiment_name> [device]"
    echo "Example: $0 my_experiment cuda"
    echo "Example: $0 my_experiment cpu"
    exit 1
fi

EXPERIMENT_NAME="$1"
DEVICE="${2:-cuda}"  # Default to cuda if not specified

echo "🚀 Submitting PyTorch Training Job"
echo "=================================="
echo "🧪 Experiment: $EXPERIMENT_NAME"
echo "🖥️  Device: $DEVICE"
echo "📋 All other parameters from config.py"
echo ""

# Create logs directory
mkdir -p logs

# Create the SLURM job script
cat > /tmp/simple_training_job_$$.sh << EOF
#!/bin/bash
#SBATCH --job-name=pytorch_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=volta

# Load conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate multi_pytorch

# Set working directory
cd /home/abedin/Developments/pytorch_multi_chest_x_ray1

# Print job information
echo "=========================================="
echo "SLURM_JOB_ID = \$SLURM_JOB_ID"
echo "SLURM_NODELIST = \$SLURM_NODELIST"
echo "=========================================="

# Show current configuration
echo "📋 Current Configuration:"
python -c "import config; config.print_current_config()"

# Run the training script with only experiment_name and device
# All other parameters come from config.py
python train_retrieval_v1.py \\
    --experiment_name $EXPERIMENT_NAME \\
    --device $DEVICE

echo "Training completed!"
EOF

# Submit the job
echo "📤 Submitting job to SLURM..."
JOB_ID=$(sbatch /tmp/simple_training_job_$$.sh | awk '{print $4}')

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Job submitted successfully!"
    echo "   🆔 Job ID: $JOB_ID"
    echo "   📁 Output: logs/training_${JOB_ID}.out"
    echo "   📁 Error: logs/training_${JOB_ID}.err"
    echo ""
    echo "📊 Monitor job status:"
    echo "   squeue -j $JOB_ID"
    echo ""
    echo "📋 View output:"
    echo "   tail -f logs/training_${JOB_ID}.out"
    echo ""
    echo "❌ Cancel job:"
    echo "   scancel $JOB_ID"
else
    echo "❌ Failed to submit job"
    exit 1
fi

# Clean up temporary file
rm -f /tmp/simple_training_job_$$.sh

echo ""
echo "🎯 Training job submitted!"
echo "   All parameters loaded from config.py"
echo "   Dataset: $(python -c "import config; print(config.DATASET_MODE)")"
echo "   Batch Size: $(python -c "import config; print(config.get_default_batch_size())")"
echo "   Learning Rate: $(python -c "import config; print(config.get_default_learning_rate())")"
echo "   Epochs: $(python -c "import config; print(config.get_default_epochs())")" 