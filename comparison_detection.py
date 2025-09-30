import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import glob
import argparse
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

# Path handling utilities for safe file operations
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

# Simple EEG Dataset
class SimpleEEGDataset(Dataset):
    def __init__(self, preproc_dir, is_train=True, verbose=True):
        self.preproc_dir = preproc_dir
        self.verbose = verbose
        
        # Check if directory exists
        if not os.path.exists(preproc_dir):
            if verbose:
                print(f"ERROR: Directory '{preproc_dir}' does not exist!")
            self.file_list = []
            self.data_shape = None
            return
        
        # Find npz files
        self.file_list = glob.glob(os.path.join(preproc_dir, "*.npz"))
        if verbose:
            print(f"Found {len(self.file_list)} files in {preproc_dir}")
        
        if len(self.file_list) == 0:
            if verbose:
                print("ERROR: No .npz files found in the directory!")
            self.data_shape = None
            return
        
        # Split into train/test (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.file_list))
        split = int(len(indices) * 0.8)
        
        if is_train:
            self.file_list = [self.file_list[i] for i in indices[:split]]
        else:
            self.file_list = [self.file_list[i] for i in indices[split:]]
            
        if verbose:
            print(f"Using {len(self.file_list)} files for {'training' if is_train else 'testing'}")
        
        # Load one file to get data shape
        if len(self.file_list) > 0:
            try:
                sample_data = np.load(self.file_list[0])
                
                # Check for required keys
                if 'data' not in sample_data:
                    if verbose:
                        print("ERROR: 'data' key not found in the npz file!")
                    self.data_shape = None
                    return
                
                self.data_shape = sample_data['data'].shape
                if verbose:
                    print(f"Data shape: {self.data_shape}")
                    
                if 'label' not in sample_data:
                    if verbose:
                        print("ERROR: 'label' key not found in the npz file!")
                
            except Exception as e:
                if verbose:
                    print(f"ERROR loading npz file: {e}")
                self.data_shape = None
        else:
            self.data_shape = None
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            data = np.load(file_path)
            x = data['data']
            y = int(data['label'])  # Make sure label is an integer
            
            # Convert to torch tensors
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            
            return x, y
        except Exception as e:
            if self.verbose:
                print(f"Error loading {file_path}: {e}")
            # Return dummy data with same shape as first sample
            if hasattr(self, 'data_shape') and self.data_shape is not None:
                x = torch.zeros(self.data_shape, dtype=torch.float32)
            else:
                x = torch.zeros((19, 10, 100), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

# 1. Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Save input shape
        self.input_shape = input_shape
        channels, windows, freq_bins = input_shape
        
        # Very simple architecture
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate output size
        output_windows = windows // 4  # After 2 pooling layers
        output_freq = freq_bins // 4
        
        # Make sure dimensions don't become zero
        output_windows = max(1, output_windows)
        output_freq = max(1, output_freq)
        
        # Final layer size
        flat_size = 32 * output_windows * output_freq
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        print(f"Model will flatten to size: {flat_size}")
        
    def forward(self, x):
        # Handle different input formats
        if len(x.shape) == 3:
            # If input is (batch, channels, features), reshape
            x = x.unsqueeze(3)
        
        # Ensure 4D tensor of shape (batch, channels, height, width)
        if len(x.shape) == 4 and x.shape[1] != self.input_shape[0]:
            # If channels dimension is not first, permute
            x = x.permute(0, 1, 2, 3)
            
        # Apply CNN layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 2. LSTM model (similar to the one in the paper)
class LSTMModel(nn.Module):
    def __init__(self, input_shape, num_classes=2, hidden_size=64, num_layers=2, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.input_shape = input_shape
        channels, windows, freq_bins = input_shape
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output dimensions from LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_output_size * channels, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Handle different input formats
        if len(x.shape) == 3:
            # If input is (batch, channels, features), reshape to (batch, channels, windows, freq_bins)
            channels, features = x.shape[1], x.shape[2]
            freq_bins = self.input_shape[2]
            windows = features // freq_bins
            x = x.reshape(x.shape[0], channels, windows, freq_bins)
        
        # Ensure 4D tensor of shape (batch, channels, windows, freq_bins)
        if len(x.shape) == 4 and x.shape[1] != self.input_shape[0]:
            # If channels dimension is not first, permute
            x = x.permute(0, 1, 2, 3)
            
        batch_size, channels, windows, freq_bins = x.shape
        
        # Process each channel with LSTM
        channel_outputs = []
        for i in range(channels):
            # Extract this channel's data: [batch, windows, freq_bins]
            channel_data = x[:, i, :, :]
            
            # Process with LSTM
            lstm_out, _ = self.lstm(channel_data)
            
            # Use the output from the last time step
            channel_outputs.append(lstm_out[:, -1, :])
        
        # Concatenate outputs from all channels
        combined = torch.cat(channel_outputs, dim=1)
        
        # Apply dropout
        combined = self.dropout(combined)
        
        # Final classification
        output = self.fc(combined)
        
        return output

# 3. CNN-LSTM hybrid model
class CNNLSTM(nn.Module):
    def __init__(self, input_shape, num_classes=2, hidden_size=64, num_layers=1):
        super(CNNLSTM, self).__init__()
        self.input_shape = input_shape
        channels, windows, freq_bins = input_shape
        
        # CNN feature extraction
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate CNN output dimensions
        output_windows = windows // 4  # After 2 pooling layers
        output_freq = freq_bins // 4
        
        # Make sure dimensions don't become zero
        output_windows = max(1, output_windows)
        output_freq = max(1, output_freq)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=32 * output_freq,  # Features per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Handle different input formats
        if len(x.shape) == 3:
            # If input is (batch, channels, features), reshape
            x = x.unsqueeze(3)
        
        # Ensure 4D tensor of shape (batch, channels, windows, freq_bins)
        if len(x.shape) == 4 and x.shape[1] != self.input_shape[0]:
            # If channels dimension is not first, permute
            x = x.permute(0, 1, 2, 3)
            
        batch_size = x.shape[0]
        
        # CNN feature extraction
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        
        # Reshape for LSTM input: [batch, seq_len, features]
        # We treat the spatial dimension as sequence length
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, x.shape[1], -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Final classification
        output = self.fc(lstm_out)
        
        return output

# Enhanced model training function with AUROC
def train_model(model, model_name, train_loader, val_loader, device, num_epochs=30, learning_rate=0.001, patience=5, result_dir='.'):
    # Ensure result directory exists
    ensure_dir_exists(result_dir)
    
    # Sanitize model name for file paths
    safe_model_name = sanitize_filename(model_name)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=patience)
    
    # Training history with added AUROC tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auroc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auroc': []
    }
    
    # Best model tracking
    best_val_acc = 0
    best_val_auroc = 0
    best_model_weights = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_probs = []
        all_train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get class probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Backward pass
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and true labels for AUROC calculation
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_probs.append(probs.detach().cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        epoch_acc = 100 * correct / total if total > 0 else 0
        
        # Calculate AUROC for training
        all_train_probs = np.vstack(all_train_probs)
        all_train_labels = np.array(all_train_labels)
        
        # For binary classification
        if all_train_probs.shape[1] == 2:  # Binary classification
            try:
                train_auroc = roc_auc_score(all_train_labels, all_train_probs[:, 1])
            except ValueError:  # If only one class is present in labels
                train_auroc = 0.5  # Default value
        else:  # Multi-class
            try:
                train_auroc = roc_auc_score(
                    np.eye(all_train_probs.shape[1])[all_train_labels], 
                    all_train_probs, 
                    multi_class='ovr'
                )
            except ValueError:  # If issue with labels
                train_auroc = 0.5  # Default value
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_auroc'].append(float(train_auroc))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_probs = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Get class probabilities
                probs = torch.softmax(outputs, dim=1)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store predictions and true labels for AUROC calculation
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Calculate AUROC for validation
        all_val_probs = np.vstack(all_val_probs)
        all_val_labels = np.array(all_val_labels)
        
        # For binary classification
        if all_val_probs.shape[1] == 2:  # Binary classification
            try:
                val_auroc = roc_auc_score(all_val_labels, all_val_probs[:, 1])
            except ValueError:  # If only one class is present in labels
                val_auroc = 0.5  # Default value
        else:  # Multi-class
            try:
                val_auroc = roc_auc_score(
                    np.eye(all_val_probs.shape[1])[all_val_labels], 
                    all_val_probs, 
                    multi_class='ovr'
                )
            except ValueError:  # If issue with labels
                val_auroc = 0.5  # Default value
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auroc'].append(float(val_auroc))
        
        # Print metrics including AUROC
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Train AUROC: {train_auroc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, Val AUROC: {val_auroc:.4f}")
        
        # Update learning rate scheduler using AUROC
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auroc)  # Using AUROC for scheduler instead of accuracy
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change if it happened
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model based on both AUROC and accuracy
        if val_auroc > best_val_auroc or (val_auroc == best_val_auroc and epoch_val_acc > best_val_acc):
            best_val_acc = epoch_val_acc
            best_val_auroc = val_auroc
            best_model_weights = model.state_dict().copy()
            print(f"New best model saved with validation AUROC: {val_auroc:.4f}, Acc: {epoch_val_acc:.2f}%")
            
            # Save model checkpoint with safe path handling
            model_path = os.path.join(result_dir, f'best_{safe_model_name}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_auroc': val_auroc,
            }, model_path)
            
            # Generate and save confusion matrix
            cm = confusion_matrix(all_val_labels, all_val_preds)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.colorbar()
            
            # Add labels to confusion matrix
            num_classes = len(np.unique(all_val_labels))
            class_names = [str(i) for i in range(num_classes)]
            tick_marks = np.arange(num_classes)
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            
            # Add text values in cells
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
            
            # Save confusion matrix
            cm_path = os.path.join(result_dir, f'{safe_model_name}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            
            # Generate ROC curve for binary classification
            if all_val_probs.shape[1] == 2:
                plt.figure(figsize=(8, 6))
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs[:, 1])
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Receiver Operating Characteristic - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Save ROC curve
                roc_path = os.path.join(result_dir, f'{safe_model_name}_roc_curve.png')
                plt.savefig(roc_path)
                plt.close()
        
        # Check for early stopping
        if epoch > 10 and optimizer.param_groups[0]['lr'] <= learning_rate / 8:
            print("Learning rate reduced significantly with no improvement. Stopping early.")
            break
    
    # Load best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    else:
        # Save final model if no best model was saved
        model_path = os.path.join(result_dir, f'final_{safe_model_name}_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_val_acc,
            'val_auroc': val_auroc,
        }, model_path)
    
    # Plot and save training curves (now including AUROC)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add AUROC plot
    plt.subplot(1, 3, 3)
    plt.plot(history['train_auroc'], label='Training')
    plt.plot(history['val_auroc'], label='Validation')
    plt.title(f'{model_name} - AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    curves_path = os.path.join(result_dir, f'{safe_model_name}_training_curves.png')
    plt.savefig(curves_path)
    plt.close()
    
    return model, history, best_val_auroc

# Enhanced evaluation function with AUROC
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get class probabilities
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    
    # Calculate AUROC
    if all_probs.shape[1] == 2:  # Binary classification
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auroc = 0.5
    else:  # Multi-class
        try:
            auroc = roc_auc_score(
                np.eye(all_probs.shape[1])[all_labels], 
                all_probs, 
                multi_class='ovr'
            )
        except ValueError:
            auroc = 0.5
    
    # Return metrics
    return {
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist()
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="EEG model training with multiple architectures and AUROC evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, choices=['cnn', 'lstm', 'cnn-lstm', 'all'], default='all',
                    help="Model architecture to train")
    
    args = parser.parse_args()
    
    # Create result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, f"{args.model}_{timestamp}")
    ensure_dir_exists(result_dir)
    
    # Save arguments
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets with verbose output
    print("Loading training dataset...")
    train_dataset = SimpleEEGDataset(args.data_dir, is_train=True, verbose=True)
    
    print("\nLoading validation dataset...")
    val_dataset = SimpleEEGDataset(args.data_dir, is_train=False, verbose=True)
    
    # Check if datasets are valid
    if len(train_dataset) == 0 or train_dataset.data_shape is None:
        print("ERROR: Invalid training dataset. Please check the data directory and file format.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=0
    )
    
    # Define models to train
    models_to_train = []
    if args.model == 'all' or args.model == 'cnn':
        models_to_train.append(('CNN', SimpleCNN(input_shape=train_dataset.data_shape, num_classes=2)))
    if args.model == 'all' or args.model == 'lstm':
        models_to_train.append(('LSTM', LSTMModel(input_shape=train_dataset.data_shape, num_classes=2)))
    if args.model == 'all' or args.model == 'cnn-lstm':
        models_to_train.append(('CNN-LSTM', CNNLSTM(input_shape=train_dataset.data_shape, num_classes=2)))
    
    # Train and evaluate each model
    results = {}
    
    for model_name, model in models_to_train:
        print(f"\n{'-'*20} Training {model_name} {'-'*20}")
        
        model.to(device)
        
        # Train model
        trained_model, history, best_val_auroc = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=5,
            result_dir=result_dir
        )
        
        # Evaluate model
        eval_results = evaluate_model(
            model=trained_model, 
            test_loader=val_loader, 
            device=device
        )
        
        # Store results
        results[model_name] = {
            'best_val_auroc': best_val_auroc,
            'final_metrics': eval_results,
            'history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'train_auroc': [float(x) for x in history['train_auroc']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_acc': [float(x) for x in history['val_acc']],
                'val_auroc': [float(x) for x in history['val_auroc']]
            }
        }
        
        print(f"Completed training {model_name}")
        print(f"Best validation AUROC: {best_val_auroc:.4f}")
        print(f"Final evaluation - Accuracy: {eval_results['accuracy']:.4f}, AUROC: {eval_results['auroc']:.4f}")
    
    # Save overall results
    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison plots
    if len(models_to_train) > 1:
        # Validation AUROC comparison
        plt.figure(figsize=(10, 6))
        for model_name in results.keys():
            plt.plot(results[model_name]['history']['val_auroc'], label=model_name)
        plt.title('Validation AUROC Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('AUROC')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_dir, 'model_comparison_auroc.png'))
        
        # Validation accuracy comparison
        plt.figure(figsize=(10, 6))
        for model_name in results.keys():
            plt.plot(results[model_name]['history']['val_acc'], label=model_name)
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_dir, 'model_comparison_accuracy.png'))
        
        # Best validation metrics bar plot
        plt.figure(figsize=(10, 6))
        model_names = list(results.keys())
        
        # Create data for bar chart
        metrics = {
            'AUROC': [results[model]['best_val_auroc'] for model in model_names],
            'Accuracy': [results[model]['final_metrics']['accuracy'] for model in model_names]
        }
        
        # Set up bar positions
        x = np.arange(len(model_names))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, metrics['AUROC'], width, label='AUROC')
        plt.bar(x + width/2, metrics['Accuracy'], width, label='Accuracy')
        
        # Add labels and formatting
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'model_metrics_comparison.png'))
    
    print(f"\nAll training completed. Results saved to {result_dir}")

if __name__ == "__main__":
    main()