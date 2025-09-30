#!/bin/bash

# Script to run DCRNN model comparison with and without pre-training

# Create directories
DATA_DIR="./eeg_data"
RESULT_DIR="./results/dcrnn_comparison"

mkdir -p $DATA_DIR
mkdir -p $RESULT_DIR

# Run correlation-based DCRNN comparison
echo "Running correlation-based DCRNN comparison"
python pretraining_dcrnn_implementation.py \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --epochs 30 \
    --pretraining_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --connectivity correlation

# Run distance-based DCRNN comparison
echo "Running distance-based DCRNN comparison"
python pretraining_dcrnn_implementation.py \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --epochs 30 \
    --pretraining_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --connectivity distance

# Run both DCRNN comparison
echo "Running both DCRNN comparison"
python pretraining_dcrnn_implementation.py \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
    --epochs 30 \
    --pretraining_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --connectivity both

echo "All comparisons completed"