#!/bin/bash
#SBATCH --job-name=eeg_debug      # Job name
#SBATCH --output=eeg_debug_%j.log # Standard output and error log
#SBATCH --nodes=1                 # Run on a single node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=16GB                # Total memory limit
#SBATCH --time=24:00:00           # Time limit hrs:min:sec
#SBATCH --partition=short       # Request compute partition

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules if needed (modify as per your cluster setup)
module purge
module load anaconda3/2023.07

# Create and activate a conda environment
echo "Creating conda environment..."
conda create -n eeg_env python=3.9 -y
source activate eeg_env

# Make sure pip3 is available and install dependencies
echo "Checking for pip3..."
which pip3
echo "Installing dependencies..."
conda install -y pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy matplotlib seaborn scikit-learn tqdm

# Create a modified version of the script with debugging
echo "Creating debug version of the script..."
cat > debug_comparison_classification.py << 'EOL'
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from datetime import datetime
import traceback

# Print Python version and environment
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Custom DottedDict implementation with better error handling
class DottedDict(dict):
    """Dict that can access nested keys with dot notation"""
    def __init__(self, *args, **kwargs):
        print(f"Initializing DottedDict with args: {args}, kwargs: {kwargs}")
        super(DottedDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DottedDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            print(f"ERROR: Missing key '{key}' in {self.__class__.__name__}")
            print(f"Available keys: {list(self.keys())}")
            print(f"Dict content: {dict(self)}")
            traceback.print_stack()
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value

# Path handling utilities
def sanitize_filename(name):
    """Sanitize a filename to remove problematic characters"""
    name = name.replace(' ', '_')
    name = name.replace('/', '_')
    name = name.replace('\\', '_')
    return name

def ensure_dir_exists(path):
    """Ensure directory exists, creating it if necessary"""
    os.makedirs(path, exist_ok=True)
    return path

# Model Arguments class with print debugging
class ModelArgs:
    """Class to hold model arguments"""
    def __init__(self, **kwargs):
        print(f"Initializing ModelArgs with kwargs: {kwargs}")
        # Set default values
        self.dropout = 0.1
        self.cl_decay_steps = 1000
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            print(f"  Setting {key}={value}")
            setattr(self, key, value)
        
        # Print all attributes after initialization
        print(f"  ModelArgs initialized with attributes: {vars(self)}")

# Create model factory with debugging
def create_model(model_name, data_shape, num_classes, device):
    """Create a new model for classification"""
    try:
        print(f"Creating model: {model_name}")
        print(f"  data_shape: {data_shape}")
        print(f"  num_classes: {num_classes}")
        print(f"  device: {device}")
        
        channels, windows, freq_bins = data_shape
        print(f"  Unpacked data_shape: channels={channels}, windows={windows}, freq_bins={freq_bins}")
        
        # Determine the actual input size for LSTM models
        input_size = freq_bins if freq_bins > 0 else 1
        print(f"  input_size determined as: {input_size}")
        
        if model_name == "cnn":
            # Create CNN model
            print("  Creating CNN model")
            model = SimpleCNN(input_shape=data_shape, num_classes=num_classes)
            
        elif model_name == "lstm":
            # Arguments for LSTM
            print("  Creating LSTM model")
            print(f"  Setting up ModelArgs with: rnn_layers=2, rnn_units=64, num_nodes={channels}, etc.")
            args = ModelArgs(
                num_rnn_layers=2,
                rnn_units=64,
                num_nodes=channels,
                bidirectional=True,
                input_dim=input_size,  # Use detected input size
                dropout=0.1
            )
            print(f"  ModelArgs created with keys: {vars(args)}")
            
            # Create model
            print("  Initializing LSTMModel")
            model = LSTMModel(args, num_classes, device)
            
        elif model_name == "cnnlstm":
            # CNN-LSTM with input shape
            print("  Creating CNN-LSTM model")
            model = CNNLSTM(input_shape=data_shape, num_classes=num_classes)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"  Model created successfully: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"ERROR in create_model for {model_name}: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        raise

# Rest of your script remains the same, except for the main function
# where we add additional debug handling:

def main():
    parser = argparse.ArgumentParser(description="Train and compare models for EEG classification")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    
    # Model selection
    parser.add_argument("--models", type=str, default="all", 
                      help="Comma-separated list of models to train: cnn,lstm,cnnlstm,all")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of evaluation runs")
    
    args = parser.parse_args()
    
    # Print and validate arguments
    print("Command line arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Check if data directory exists
    print(f"Checking data directory: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return
    else:
        print(f"Data directory exists: {args.data_dir}")
        # Print some files from the directory
        files = os.listdir(args.data_dir)
        print(f"First 5 files in directory: {files[:5] if len(files) >= 5 else files}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, f"eeg_comparison_{timestamp}")
    ensure_dir_exists(result_dir)
    print(f"Created result directory: {result_dir}")
    
    # Save arguments
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading training dataset...")
    try:
        train_dataset = EEGClassificationDataset(args.data_dir, is_train=True, verbose=True)
        print(f"Training dataset loaded successfully with {len(train_dataset)} items")
        
        print("\nLoading validation dataset...")
        val_dataset = EEGClassificationDataset(args.data_dir, is_train=False, verbose=True)
        print(f"Validation dataset loaded successfully with {len(val_dataset)} items")
    except Exception as e:
        print(f"ERROR loading datasets: {e}")
        traceback.print_exc()
        return
    
    # Check if datasets are valid
    if len(train_dataset) == 0 or train_dataset.data_shape is None:
        print("ERROR: Invalid training dataset. Please check the data directory and file format.")
        return
    
    print(f"Data shape: {train_dataset.data_shape}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    # Truncated script to focus on the error point - typically it's in dataset loading or model creation
    # Try with just one model for simplicity
    print("\nTrying to create the CNN model which is the simplest...")
    try:
        model = create_model(
            model_name="cnn",
            data_shape=train_dataset.data_shape,
            num_classes=train_dataset.num_classes,
            device=device
        )
        print("CNN model created successfully!")
    except Exception as e:
        print(f"Failed to create CNN model: {e}")
        traceback.print_exc()

    print("\nScript completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        traceback.print_exc()
EOL

# Set variables
DATA_DIR="$1"                   # Get data directory from command line argument
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: No data directory provided. Using current directory."
    DATA_DIR="."
fi

echo "Using data directory: $DATA_DIR"

# Run the debug script 
echo "Running debug script..."
python debug_comparison_classification.py --data_dir $DATA_DIR --models cnn

# Print job summary
echo "End Time: $(date)"
echo "Job $SLURM_JOB_ID completed"