#!/bin/bash
#SBATCH --job-name=eeg_classification
#SBATCH --output=eeg_classification_%j.out
#SBATCH --error=eeg_classification_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short  # Adjust based on your cluster's CPU partition name

# Load modules (adjust based on your cluster's available modules)
module load python/3.8  # Using Python 3.6+ for f-string compatibility


# Create and activate a virtual environment
python -m venv eeg_env
source eeg_env/bin/activate

pip install --upgrade pip
pip install \
    torch \
    torchvision \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    tqdm \
    torchaudio \
    seaborn



# Run the script - make sure to use CPU-only mode
python eeg-ssl/comparison_classification.py --data_dir preproc_data/classification/clipLen12_timeStepSize1 --result_dir ./results --models all --batch_size 32 --epochs 50

# Deactivate virtual environment
deactivate