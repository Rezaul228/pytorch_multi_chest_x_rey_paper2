#!/usr/bin/env python3
"""
Test script for 2-epoch training to check visualization functionality
"""

import torch
import argparse
import os
import sys
from train_retrieval_v1 import main

def test_2epoch_training():
    """Run a small 2-epoch training to test visualizations"""
    
    # Set up arguments for 2-epoch training
    sys.argv = [
        'test_2epoch_training.py',
        '--experiment_name', 'test_2epoch_viz',
        '--device', 'cpu',  # Use CPU instead of CUDA
        '--epochs', '2',
        '--train_samples', '1000',  # Use only 1000 samples for quick test
        '--val_samples', '200'      # Use only 200 validation samples
    ]
    
    print("🧪 Running 2-epoch training test...")
    print("📊 Training samples: 1000")
    print("📊 Validation samples: 200")
    print("🔄 Epochs: 2")
    print("🖥️  Device: CPU")
    print("🎯 Purpose: Test visualization functionality")
    
    try:
        # Run the main training function
        main()
        print("✅ 2-epoch training test completed successfully!")
        
        # Check if visualization files were created
        viz_dir = os.path.join('outputs', 'train_visualizations_test_2epoch_viz')
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
            print(f"📊 Visualization files created: {len(viz_files)}")
            for file in viz_files:
                print(f"   📁 {file}")
        else:
            print("❌ No visualization directory found")
            
    except Exception as e:
        print(f"❌ Error during 2-epoch training test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_2epoch_training() 