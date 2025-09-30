# Deep Learning for EEG Signal Analysis

This repository contains a collection of scripts for analyzing EEG (electroencephalogram) signals using various deep learning architectures. The implementation includes support for seizure detection and classification tasks with multiple model architectures and a novel self-supervised pretraining approach.

Please find the full source code with data files from this link as dataset was too big to upload:https://drive.google.com/drive/folders/1WnzR1gMXjJYcB96GdnuMGPmaF-fMwbeK?usp=sharing

You can directly run the models as preprocessed data is already formulated, still for new data-resampling and preprocessing scripts are available in the repo.

## Features

- **Multiple model architectures**:
  - DCRNN (Dynamic Graph Convolutional Recurrent Neural Network)
  - LSTM-based models
  - CNN-based models
  - Hybrid CNN-LSTM models
  - Attention-enhanced models

- **Two primary tasks**:
  - Binary seizure detection (seizure vs. non-seizure)
  - Seizure type classification (support for multiple classes)

- **Advanced techniques**:
  - Self-supervised pretraining with contrastive learning
  - Dynamic graph construction (correlation and distance-based approaches)
  - Multi-head attention mechanism
  - Extensive evaluation metrics (Accuracy, F1-score, AUROC)

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- tqdm
- h5py
- pyedflib

### Data Preparation

1. Preprocessed data is there in directory in preproc_data.



### Model Training

#### DCRNN with Pretraining for Seizure Detection

Train DCRNN models with both correlation and distance-based approaches:

```bash
python dcrnn_with.py --data_dir "preproc_data\detection_clip12"  --result_dir results  --batch_size 32 --lr 0.001 --connectivity both --epoch 5
```

For attention-enhanced DCRNN:

```bash
python attention.py --data_dir "preproc_data\detection_clip12"  --result_dir results  --batch_size 32 --lr 0.001 --connectivity both --num_attention_heads 4 --epoch 10  
```



#### Model Comparison

Compare multiple model architectures:

1. For classification:
   ```bash
   python comparison_classification.py --data_dir preproc_data/classification/clipLen12_timeStepSize1  --models all      
   ```

2. For detection:
   ```bash
   python comparison_detection.py --data_dir preproc_data/detection_clip12  --result_dir results --epochs 30 --batch_size 4 --lr 0.001 --model all   
   ```

## File Descriptions

- **dcrnn_with.py**: DCRNN implementation with contrastive pretraining for seizure detection
- **attention.py**: Enhanced DCRNN with multi-head attention mechanism
=- **comparison_classification.py**: Script to compare different models for seizure classification
- **comparison_detection.py**: Script to compare different models for seizure detection

## Model Architectures

### DCRNN

The DCRNN model combines graph neural networks with recurrent neural networks:

1. **Dynamic Graph Construction**: Builds channel connectivity graphs using either correlation or distance metrics
2. **Graph Convolution**: Applies graph convolution to capture spatial dependencies
3. **Recurrent Processing**: Uses LSTM layers to capture temporal dependencies
4. **Classification**: Fully connected layers for final prediction

### Attention-Enhanced DCRNN

Extends the DCRNN model with multi-head attention mechanism:

1. **Dynamic Graph Construction**: Same as DCRNN
2. **Graph Convolution**: Same as DCRNN
3. **Multi-Head Attention**: Applies attention mechanism to focus on important features
4. **Recurrent Processing**: Uses LSTM layers to capture temporal dependencies
5. **Classification**: Fully connected layers for final prediction

### Other Models

- **Simple CNN**: Convolutional architecture for feature extraction
- **LSTM**: Recurrent architecture for temporal sequence processing
- **CNN-LSTM**: Hybrid architecture combining CNN feature extraction with LSTM temporal processing

## Pretraining Strategy

The self-supervised pretraining uses a contrastive learning approach:

1. Pairs of EEG segments are created (same subject vs different subjects)
2. The model learns to determine if two segments are from the same subject
3. This pretraining helps the model learn meaningful representations before fine-tuning for the target task

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- F1-score
- AUROC (Area Under the Receiver Operating Characteristic curve)
- Confusion matrices
- ROC curves

## Results Visualization

Training results include:
- Loss curves
- Accuracy/F1-score/AUROC evolution
- Confusion matrices
- ROC curves
- Model comparison charts

## License

[Specify your license here]

## Acknowledgments

[Add any acknowledgments here]
