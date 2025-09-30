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

# Custom DottedDict implementation to avoid dependencies
class DottedDict(dict):
    """Dict that can access nested keys with dot notation"""
    def __init__(self, *args, **kwargs):
        super(DottedDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DottedDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
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

# EEG Dataset class for classification
class EEGClassificationDataset(Dataset):
    def __init__(self, preproc_dir, is_train=True, split_ratio=0.8, verbose=True):
        self.preproc_dir = preproc_dir
        self.verbose = verbose
        
        # Find npz files recursively
        self.file_list = glob.glob(os.path.join(preproc_dir, "*.npz"))
        
        # Remove non-data files
        self.file_list = [f for f in self.file_list if "class_distribution" not in f]
        
        if verbose:
            print(f"Found {len(self.file_list)} preprocessed files")
        
        if len(self.file_list) == 0:
            if verbose:
                print("ERROR: No .npz files found in the directory!")
            self.data_shape = None
            self.num_classes = 0
            return
        
        # Split into train/val sets
        np.random.seed(42)
        indices = np.random.permutation(len(self.file_list))
        split_idx = int(len(indices) * split_ratio)
        
        if is_train:
            self.file_list = [self.file_list[i] for i in indices[:split_idx]]
        else:
            self.file_list = [self.file_list[i] for i in indices[split_idx:]]
            
        if verbose:
            print(f"Using {len(self.file_list)} files for {'training' if is_train else 'validation'}")
        
        # Check class distribution
        self._check_class_balance()
        
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
                        print("WARNING: 'label' key not found in the npz file!")
                
            except Exception as e:
                if verbose:
                    print(f"ERROR loading npz file: {e}")
                self.data_shape = None
        else:
            self.data_shape = None

    def _check_class_balance(self):
        """Check class balance in dataset"""
        label_counts = {}
        for file_path in tqdm(self.file_list, desc="Checking class balance"):
            try:
                data = np.load(file_path)
                if 'label' not in data:
                    continue
                    
                label = int(data['label'])
                label_counts[label] = label_counts.get(label, 0) + 1
            except Exception:
                continue
        
        total = sum(label_counts.values())
        if total > 0:
            if self.verbose:
                print("Class distribution:")
                for label, count in sorted(label_counts.items()):
                    print(f"  Class {label}: {count} samples ({count/total*100:.1f}%)")
            
            # Store class distribution
            self.class_counts = label_counts
            
            # Get unique classes
            self.num_classes = len(label_counts)
            if self.verbose:
                print(f"Total number of classes: {self.num_classes}")
        else:
            self.class_counts = {}
            self.num_classes = 0
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            data = np.load(file_path)
            x = data['data']
            y = int(data['label'])
            
            # Convert to torch tensors
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            
            return x, y
        except Exception as e:
            if self.verbose:
                print(f"Error loading {file_path}: {e}")
            # Return dummy data
            if hasattr(self, 'data_shape') and self.data_shape is not None:
                x = torch.zeros(self.data_shape, dtype=torch.float32)
            else:
                x = torch.zeros((19, 10, 100), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

# Custom collate function for batching
def collate_fn(batch):
    """Custom collate function to format data for models"""
    # Filter out None items
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Separate data and labels
    data, labels = zip(*batch)
    
    # Stack tensors
    data = torch.stack(data)
    labels = torch.stack(labels)
    
    # Create dummy sequence lengths (all same length)
    seq_lengths = torch.ones(data.shape[0], dtype=torch.long) * data.shape[2]
    
    return data, labels, seq_lengths

# Model Arguments class
class ModelArgs:
    """Class to hold model arguments"""
    def __init__(self, **kwargs):
        # Set default values
        self.dropout = 0.1
        self.cl_decay_steps = 1000
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

# 1. CNN Model - Fixed implementation
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Store input shape
        self.input_shape = input_shape
        
        # Use 1D convolutions instead of 2D to avoid dimension issues
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(8)  # Force output to fixed size
        )
        
        # Fixed-size fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8, 128),  # 64 channels * 8 features per channel
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, seq_lengths=None):
        # Handle input format
        batch_size = x.shape[0]
        
        # Convert to the format expected by 1D convolutions: [batch, channels, features]
        if len(x.shape) == 4:  # [batch, channels, windows, freq]
            # Flatten windows and freq dimensions
            x = x.reshape(batch_size, x.shape[1], -1)
        
        # Apply feature extraction
        x = self.features(x)
        
        # Flatten for classifier
        x = x.view(batch_size, -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x

# 2. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, args, num_classes, device):
        super(LSTMModel, self).__init__()
        self.num_nodes = args.num_nodes
        self.rnn_units = args.rnn_units
        self.num_rnn_layers = args.num_rnn_layers
        self.device = device
        self.bidirectional = args.bidirectional
        self.input_dim = args.input_dim
        
        # Process each channel separately
        self.channel_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.rnn_units,
                num_layers=self.num_rnn_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=args.dropout if self.num_rnn_layers > 1 else 0
            ) for _ in range(self.num_nodes)
        ])
        
        # Output dimension will be doubled if bidirectional
        self.lstm_output_dim = self.rnn_units * 2 if self.bidirectional else self.rnn_units
        
        # Feature dimension for classifier
        self.feature_dim = self.lstm_output_dim * self.num_nodes
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, inputs, seq_lengths=None):
        # Expected shape: [batch_size, num_nodes, seq_len, input_dim]
        batch_size = inputs.shape[0]
        
        # Process each channel separately
        channel_outputs = []
        
        for i in range(self.num_nodes):
            # Extract this channel's data
            if len(inputs.shape) == 4:
                # If input has feature dimension: [batch_size, num_nodes, seq_len, features]
                channel_data = inputs[:, i, :, :]  # Already [batch, seq_len, features]
            else:
                # If input is already [batch_size, num_nodes, seq_len]
                channel_data = inputs[:, i, :].unsqueeze(-1)  # Add feature dim
            
            # Process with the channel's LSTM
            lstm_out, _ = self.channel_lstms[i](channel_data)
            
            # Use the last output
            channel_outputs.append(lstm_out[:, -1, :])  # [batch, lstm_output_dim]
        
        # Concatenate outputs from all channels: [batch, num_nodes * lstm_output_dim]
        combined = torch.cat(channel_outputs, dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output

# 3. CNN-LSTM Model
class CNNLSTM(nn.Module):
    def __init__(self, input_shape, num_classes=2, hidden_size=64, num_layers=2):
        super(CNNLSTM, self).__init__()
        self.input_shape = input_shape
        channels, windows, freq_bins = input_shape
        
        # Use 1D CNN for feature extraction (more stable)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Use adaptive pooling to get fixed temporal dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=64,  # Features per time step from CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, seq_lengths=None):
        batch_size = x.shape[0]
        
        # Format inputs for 1D CNN
        if len(x.shape) == 4:  # [batch, channels, windows, freq_bins]
            # Flatten the time and frequency dimensions
            x = x.reshape(batch_size, x.shape[1], -1)
        
        # Apply CNN feature extraction
        x = self.features(x)
        
        # Apply adaptive pooling to get fixed temporal dimension
        x = self.adaptive_pool(x)
        
        # Prepare for LSTM [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, time, channels]
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get final time step output
        x = lstm_out[:, -1, :]
        
        # Apply dropout and classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

# Create model factory
def create_model(model_name, data_shape, num_classes, device):
    """Create a new model for classification"""
    channels, windows, freq_bins = data_shape
    
    # Determine the actual input size for LSTM models
    input_size = freq_bins if freq_bins > 0 else 1
    
    if model_name == "cnn":
        # Create CNN model
        model = SimpleCNN(input_shape=data_shape, num_classes=num_classes)
        
    elif model_name == "lstm":
        # Arguments for LSTM
        args = ModelArgs(
            num_rnn_layers=2,
            rnn_units=64,
            num_nodes=channels,
            bidirectional=True,
            input_dim=input_size,  # Use detected input size
            dropout=0.1
        )
        
        # Create model
        model = LSTMModel(args, num_classes, device)
        
    elif model_name == "cnnlstm":
        # CNN-LSTM with input shape
        model = CNNLSTM(input_shape=data_shape, num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# Training function
def train_model(model, model_name, train_loader, val_loader, device, save_dir, num_epochs, learning_rate=0.001):
    """Train the model with the data"""
    # Ensure directory exists
    ensure_dir_exists(save_dir)
    
    # Sanitize model name for file paths
    safe_model_name = sanitize_filename(model_name)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Set up LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    
    # Set up metrics
    best_val_f1 = 0.0
    patience_counter = 0
    max_patience = 15  # Early stopping patience
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_auroc': [],  # Add AUROC tracking
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auroc': []     # Add AUROC tracking
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_probs = []  # For AUROC calculation
        all_train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            # Skip empty batches
            if batch is None:
                continue
                
            # Unpack batch
            inputs, labels, seq_lengths = batch
            
            # Move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            seq_lengths = seq_lengths.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, seq_lengths)
            loss = criterion(outputs, labels)
            
            # Get probabilities for AUROC
            probs = torch.softmax(outputs, dim=1)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Save predictions and probabilities
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_probs.append(probs.detach().cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # Calculate F1 score if we have predictions from multiple classes
        unique_labels = np.unique(all_train_labels)
        if len(unique_labels) > 1:
            train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
            
            # Calculate AUROC for training
            all_train_probs = np.vstack(all_train_probs)
            if len(unique_labels) == 2:  # Binary classification
                try:
                    train_auroc = roc_auc_score(all_train_labels, all_train_probs[:, 1])
                except:
                    train_auroc = 0.5  # Default for error cases
            else:  # Multi-class
                try:
                    train_auroc = roc_auc_score(
                        np.eye(all_train_probs.shape[1])[all_train_labels], 
                        all_train_probs, 
                        multi_class='ovr'
                    )
                except:
                    train_auroc = 0.5  # Default for error cases
        else:
            train_f1 = 0.0
            train_auroc = 0.5
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(float(train_f1))
        history['train_auroc'].append(float(train_auroc))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_probs = []  # For AUROC calculation
        all_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                # Skip empty batches
                if batch is None:
                    continue
                    
                # Unpack batch
                inputs, labels, seq_lengths = batch
                
                # Move to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                seq_lengths = seq_lengths.to(device)
                
                # Forward pass
                outputs = model(inputs, seq_lengths)
                loss = criterion(outputs, labels)
                
                # Get probabilities for AUROC
                probs = torch.softmax(outputs, dim=1)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Save predictions and probabilities
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Calculate F1 score if we have predictions from multiple classes
        unique_labels = np.unique(all_val_labels)
        if len(unique_labels) > 1:
            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
            
            # Calculate AUROC for validation
            all_val_probs = np.vstack(all_val_probs)
            if len(unique_labels) == 2:  # Binary classification
                try:
                    val_auroc = roc_auc_score(all_val_labels, all_val_probs[:, 1])
                except:
                    val_auroc = 0.5  # Default for error cases
            else:  # Multi-class
                try:
                    val_auroc = roc_auc_score(
                        np.eye(all_val_probs.shape[1])[all_val_labels], 
                        all_val_probs, 
                        multi_class='ovr'
                    )
                except:
                    val_auroc = 0.5  # Default for error cases
        else:
            val_f1 = 0.0
            val_auroc = 0.5
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(float(val_f1))
        history['val_auroc'].append(float(val_auroc))
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}, Train AUROC: {train_auroc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1 or (val_f1 == 0 and val_acc > best_val_f1):  # Use accuracy if F1 is 0
            best_val_f1 = max(val_f1, best_val_f1)
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(save_dir, f"best_{safe_model_name}_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auroc': val_auroc,
            }, save_path)
            
            # Save confusion matrix
            if len(unique_labels) > 1:
                cm = confusion_matrix(all_val_labels, all_val_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig(os.path.join(save_dir, f'{safe_model_name}_confusion_matrix_epoch_{epoch+1}.png'))
                plt.close()
            
            # Save classification report
            report = classification_report(all_val_labels, all_val_preds, output_dict=True)
            with open(os.path.join(save_dir, f'{safe_model_name}_classification_report_epoch_{epoch+1}.json'), 'w') as f:
                json.dump(report, f, indent=4)
                
            print(f"  New best model saved with F1 score: {val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs (best F1: {best_val_f1:.4f})")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Save final model
    save_path = os.path.join(save_dir, f"final_{safe_model_name}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_auroc': val_auroc,
    }, save_path)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 4, 2)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 4, 3)
    plt.plot(history['train_f1'], label='Training')
    plt.plot(history['val_f1'], label='Validation')
    plt.title(f'{model_name} - F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add AUROC plot
    plt.subplot(1, 4, 4)
    plt.plot(history['train_auroc'], label='Training')
    plt.plot(history['val_auroc'], label='Validation')
    plt.title(f'{model_name} - AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{safe_model_name}_training_curves.png'))
    plt.close()
    
    return model, history, best_val_f1

# Model evaluation function
def evaluate_model(model, model_name, test_loader, device, num_runs=5):
    """Evaluate a trained model with multiple runs for robust results"""
    f1_values = []
    acc_values = []
    auroc_values = []  # Add AUROC tracking
    
    for run in range(num_runs):
        print(f"Evaluation run {run+1}/{num_runs}")
        
        model.eval()
        all_preds = []
        all_probs = []  # For AUROC calculation
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Skip empty batches
                if batch is None:
                    continue
                
                # Unpack batch
                inputs, labels, seq_lengths = batch
                
                # Move to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                seq_lengths = seq_lengths.to(device)
                
                # Forward pass
                outputs = model(inputs, seq_lengths)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        acc_values.append(accuracy)
        
        # Calculate F1 score
        unique_labels = np.unique(all_labels)
        if len(unique_labels) > 1:
            f1 = f1_score(all_labels, all_preds, average='weighted')
            f1_values.append(f1)
            
            # Calculate AUROC
            if len(unique_labels) == 2:  # Binary classification
                try:
                    auroc = roc_auc_score(all_labels, all_probs[:, 1])
                except:
                    auroc = 0.5  # Default for error cases
            else:  # Multi-class
                try:
                    auroc = roc_auc_score(
                        np.eye(all_probs.shape[1])[all_labels], 
                        all_probs, 
                        multi_class='ovr'
                    )
                except:
                    auroc = 0.5  # Default for error cases
            
            auroc_values.append(auroc)
        else:
            f1_values.append(0.0)
            auroc_values.append(0.5)
    
    # Calculate mean and std
    mean_f1 = np.mean(f1_values)
    std_f1 = np.std(f1_values)
    
    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values)
    
    mean_auroc = np.mean(auroc_values)
    std_auroc = np.std(auroc_values)
    
    # Return metrics
    return {
        'model_name': model_name,
        'accuracy': float(mean_acc),
        'accuracy_std': float(std_acc),
        'f1': float(mean_f1),
        'f1_std': float(std_f1),
        'auroc': float(mean_auroc),
        'auroc_std': float(std_auroc)
    }

# Main function
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
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, f"eeg_comparison_{timestamp}")
    ensure_dir_exists(result_dir)
    
    # Save arguments
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading training dataset...")
    train_dataset = EEGClassificationDataset(args.data_dir, is_train=True, verbose=True)
    
    print("\nLoading validation dataset...")
    val_dataset = EEGClassificationDataset(args.data_dir, is_train=False, verbose=True)
    
    # Check if datasets are valid
    if len(train_dataset) == 0 or train_dataset.data_shape is None:
        print("ERROR: Invalid training dataset. Please check the data directory and file format.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Determine which models to train
    if args.models.lower() == 'all':
        model_names = ['cnn', 'lstm', 'cnnlstm']
    else:
        model_names = [model.strip().lower() for model in args.models.split(',')]
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for model_name in model_names:
        print(f"\n{'-'*20} Training {model_name.upper()} {'-'*20}")
        
        # Create model directory
        model_dir = os.path.join(result_dir, model_name)
        ensure_dir_exists(model_dir)
        
        # Create model
        model = create_model(
            model_name=model_name,
            data_shape=train_dataset.data_shape,
            num_classes=train_dataset.num_classes,
            device=device
        )
        model.to(device)
        
        # Print model information
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters")
        
        # Train model
        model, history, best_val_f1 = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=model_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )
        
        # Evaluate model
        eval_results = evaluate_model(
            model=model,
            model_name=model_name,
            test_loader=val_loader,
            device=device,
            num_runs=args.num_runs
        )
        
        # Store results
        results[model_name] = {
            'best_val_f1': float(best_val_f1),
            'eval_results': eval_results,
            'history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_acc': [float(x) for x in history['train_acc']],
                'train_f1': [float(x) for x in history['train_f1']],
                'train_auroc': [float(x) for x in history['train_auroc']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_acc': [float(x) for x in history['val_acc']],
                'val_f1': [float(x) for x in history['val_f1']],
                'val_auroc': [float(x) for x in history['val_auroc']]
            }
        }
        
        print(f"Completed training and evaluation of {model_name.upper()}")
        print(f"F1 Score: {eval_results['f1']:.4f} ± {eval_results['f1_std']:.4f}")
        print(f"Accuracy: {eval_results['accuracy']:.4f} ± {eval_results['accuracy_std']:.4f}")
        print(f"AUROC: {eval_results['auroc']:.4f} ± {eval_results['auroc_std']:.4f}")
    
    # Save all results
    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison plots
    if len(results) > 1:
        # F1 Score comparison
        plt.figure(figsize=(15, 5))
        
        # Create data for bar charts
        models = list(results.keys())
        f1_scores = [results[model]['eval_results']['f1'] for model in models]
        f1_stds = [results[model]['eval_results']['f1_std'] for model in models]
        accuracies = [results[model]['eval_results']['accuracy'] for model in models]
        acc_stds = [results[model]['eval_results']['accuracy_std'] for model in models]
        aurocs = [results[model]['eval_results']['auroc'] for model in models]
        auroc_stds = [results[model]['eval_results']['auroc_std'] for model in models]
        
        # Plot F1 scores
        plt.subplot(1, 3, 1)
        bars = plt.bar(models, f1_scores, yerr=f1_stds, capsize=10, alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{f1_scores[i]:.4f}", ha='center', va='bottom'
            )
        
        plt.title('F1 Score Comparison')
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Accuracy comparison
        plt.subplot(1, 3, 2)
        bars = plt.bar(models, accuracies, yerr=acc_stds, capsize=10, alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{accuracies[i]:.4f}", ha='center', va='bottom'
            )
        
        plt.title('Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # AUROC comparison
        plt.subplot(1, 3, 3)
        bars = plt.bar(models, aurocs, yerr=auroc_stds, capsize=10, alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{aurocs[i]:.4f}", ha='center', va='bottom'
            )
        
        plt.title('AUROC Comparison')
        plt.xlabel('Model')
        plt.ylabel('AUROC')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'model_comparison.png'))
        plt.close()
        
        # Create metric evolution plots
        plt.figure(figsize=(15, 5))
        
        # F1 score evolution
        plt.subplot(1, 3, 1)
        for model_name in results:
            epochs = range(1, len(results[model_name]['history']['val_f1'])+1)
            plt.plot(epochs, results[model_name]['history']['val_f1'], marker='o', linestyle='-', label=model_name)
        
        plt.title('Validation F1 Score Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Accuracy evolution
        plt.subplot(1, 3, 2)
        for model_name in results:
            epochs = range(1, len(results[model_name]['history']['val_acc'])+1)
            plt.plot(epochs, results[model_name]['history']['val_acc'], marker='o', linestyle='-', label=model_name)
        
        plt.title('Validation Accuracy Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # AUROC evolution
        plt.subplot(1, 3, 3)
        for model_name in results:
            epochs = range(1, len(results[model_name]['history']['val_auroc'])+1)
            plt.plot(epochs, results[model_name]['history']['val_auroc'], marker='o', linestyle='-', label=model_name)
        
        plt.title('Validation AUROC Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('AUROC')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'metrics_evolution.png'))
        plt.close()
    
    print(f"\nTraining and evaluation complete. Results saved to: {result_dir}")
    
    # Print final comparison table
    print("\nModel Comparison Results:")
    print("-" * 100)
    print(f"{'Model':<10} {'F1 Score':<20} {'Accuracy':<20} {'AUROC':<20}")
    print("-" * 100)
    
    for model_name in results:
        f1 = results[model_name]['eval_results']['f1']
        f1_std = results[model_name]['eval_results']['f1_std']
        acc = results[model_name]['eval_results']['accuracy']
        acc_std = results[model_name]['eval_results']['accuracy_std']
        auroc = results[model_name]['eval_results']['auroc']
        auroc_std = results[model_name]['eval_results']['auroc_std']
        
        print(f"{model_name:<10} {f1:.4f} ± {f1_std:.4f} {acc:.4f} ± {acc_std:.4f} {auroc:.4f} ± {auroc_std:.4f}")
    
    print("-" * 100)

if __name__ == "__main__":
    main()