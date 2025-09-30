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
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Path handling utilities for safety
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

# Dynamic Graph Constructor
class DynamicGraphConstructor(nn.Module):
    """
    Creates dynamic adjacency matrices based on input data
    """
    def __init__(self, connectivity_type='correlation', threshold=0.5):
        super(DynamicGraphConstructor, self).__init__()
        self.connectivity_type = connectivity_type
        self.threshold = threshold
        
    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, channels, time_steps, features]
        Returns: adjacency matrix of shape [batch_size, channels, channels]
        """
        batch_size, channels, time_steps, features = x.shape
        
        # Reshape for computing connectivity
        x_reshaped = x.permute(0, 1, 3, 2).reshape(batch_size, channels, -1)
        
        if self.connectivity_type == 'correlation':
            # Compute correlation coefficient between channels
            # First normalize the data
            x_mean = x_reshaped.mean(dim=2, keepdim=True)
            x_std = x_reshaped.std(dim=2, keepdim=True) + 1e-8
            x_norm = (x_reshaped - x_mean) / x_std
            
            # Compute correlation matrix
            adj_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) / time_steps
            
            # Apply threshold to create binary adjacency matrix
            adj_matrix = (adj_matrix > self.threshold).float()
            
        elif self.connectivity_type == 'distance':
            # Compute Euclidean distance between channel features
            adj_matrix = torch.zeros(batch_size, channels, channels, device=x.device)
            
            for i in range(channels):
                for j in range(channels):
                    if i != j:
                        # Calculate Euclidean distance
                        dist = torch.sqrt(torch.sum((x_reshaped[:, i, :] - x_reshaped[:, j, :])**2, dim=1))
                        # Normalize distances to [0, 1] range
                        dist = dist / (dist.max() + 1e-8)
                        # Convert distance to similarity (closer = higher value)
                        sim = 1 - dist
                        adj_matrix[:, i, j] = sim
            
            # Apply threshold to create binary adjacency matrix
            adj_matrix = (adj_matrix > self.threshold).float()
        
        # Add self-connections
        adj_matrix = adj_matrix + torch.eye(channels, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Normalize adjacency matrix (D^-1/2 * A * D^-1/2)
        degree = torch.sum(adj_matrix, dim=2, keepdim=True)
        degree_sqrt_inv = torch.pow(degree + 1e-8, -0.5)
        adj_matrix = adj_matrix * degree_sqrt_inv * degree_sqrt_inv.transpose(1, 2)
        
        return adj_matrix

# Graph Convolution Layer
class GraphConvolution(nn.Module):
    """
    Graph convolution layer
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x, adj):
        """
        x: Input features [batch_size, channels, features]
        adj: Adjacency matrix [batch_size, channels, channels]
        """
        # Graph convolution: adj @ x @ weight
        support = torch.bmm(adj, x)  # [batch_size, channels, features]
        output = torch.matmul(support, self.weight)  # [batch_size, channels, out_features]
        return output + self.bias

# DCRNN base model
class DCRNN(nn.Module):
    """
    Dynamic Connectivity Recurrent Neural Network
    """
    def __init__(self, input_shape, num_classes=2, connectivity_type='correlation',
                 hidden_size=64, num_layers=2, threshold=0.5, bidirectional=True, dropout=0.5):
        super(DCRNN, self).__init__()
        
        self.input_shape = input_shape
        channels, windows, freq_bins = input_shape
        
        # Dynamic graph constructor
        self.graph_constructor = DynamicGraphConstructor(
            connectivity_type=connectivity_type,
            threshold=threshold
        )
        
        # Graph convolution layer
        self.gcn = GraphConvolution(freq_bins, hidden_size)
        
        # LSTM layer to process temporal sequences after graph convolution
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output dimensions from LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_output_size * channels, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
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
        
        # Construct dynamic adjacency matrix
        adj_matrix = self.graph_constructor(x)  # [batch_size, channels, channels]
        
        # Process each time window with graph convolution
        gcn_outputs = []
        for t in range(windows):
            # Get features for current time window [batch_size, channels, freq_bins]
            features_t = x[:, :, t, :]
            
            # Apply graph convolution
            gcn_out = self.gcn(features_t, adj_matrix)  # [batch_size, channels, hidden_size]
            gcn_outputs.append(gcn_out.unsqueeze(2))  # Add time dimension
            
        # Concatenate outputs along time dimension
        gcn_outputs = torch.cat(gcn_outputs, dim=2)  # [batch_size, channels, windows, hidden_size]
        
        # Process each channel with LSTM
        channel_outputs = []
        for i in range(channels):
            # Extract this channel's data: [batch, windows, hidden_size]
            channel_data = gcn_outputs[:, i, :, :]
            
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

# Function to train DCRNN model
def train_model(model, model_name, train_loader, val_loader, device, 
                num_epochs=30, learning_rate=0.001, patience=5, result_dir='.'):
    """
    Train a DCRNN model for classification
    """
    # Ensure directory exists
    result_dir = ensure_dir_exists(result_dir)
    
    # Sanitize model name for filesystem safety
    safe_model_name = sanitize_filename(model_name)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=patience)
    
    # Training history
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
    
    print(f"Starting training of {model_name}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_probs = []
        all_train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Collect probabilities for AUROC
            probs = torch.softmax(outputs, dim=1)
            all_train_probs.append(probs.detach().cpu().numpy())
            all_train_labels.append(labels.cpu().numpy())
            
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
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        epoch_acc = 100 * correct / total if total > 0 else 0
        
        # Calculate training AUROC
        all_train_probs = np.vstack(all_train_probs)
        all_train_labels = np.concatenate(all_train_labels)
        
        # For binary classification
        if all_train_probs.shape[1] == 2:  # Binary classification
            train_auroc = roc_auc_score(all_train_labels, all_train_probs[:, 1])
        else:  # Multi-class
            # One-vs-Rest approach for multiclass
            train_auroc = roc_auc_score(
                np.eye(all_train_probs.shape[1])[all_train_labels], 
                all_train_probs, 
                multi_class='ovr'
            )
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_auroc'].append(float(train_auroc))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_probs = []
        all_val_labels = []
        all_val_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Collect probabilities for AUROC
                probs = torch.softmax(outputs, dim=1)
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_val_preds.extend(predicted.cpu().numpy())
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Calculate validation AUROC
        all_val_probs = np.vstack(all_val_probs)
        all_val_labels = np.concatenate(all_val_labels)
        
        # For binary classification
        if all_val_probs.shape[1] == 2:  # Binary classification
            val_auroc = roc_auc_score(all_val_labels, all_val_probs[:, 1])
        else:  # Multi-class
            # One-vs-Rest approach for multiclass
            val_auroc = roc_auc_score(
                np.eye(all_val_probs.shape[1])[all_val_labels], 
                all_val_probs, 
                multi_class='ovr'
            )
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auroc'].append(float(val_auroc))
        
        # Print metrics
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Train AUROC: {train_auroc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, Val AUROC: {val_auroc:.4f}")
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_auroc)  # Using AUROC for learning rate scheduling
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change if it happened
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model based on validation AUROC
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_acc = epoch_val_acc  # Update best accuracy too
            best_model_weights = model.state_dict().copy()
            
            print(f"New best model saved with validation AUROC: {val_auroc:.4f}")
            
            # Save model checkpoint
            model_path = os.path.join(result_dir, f'best_{safe_model_name}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_auroc': val_auroc,
            }, model_path)
            
            # Create confusion matrix
            all_val_preds_array = np.array(all_val_preds)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_val_labels, all_val_preds_array)
            
            # Plot and save confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.colorbar()
            
            # Add labels
            classes = np.unique(all_val_labels)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
            
            # Save confusion matrix
            cm_path = os.path.join(result_dir, f'{safe_model_name}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            
            # Plot and save ROC curve for binary classification
            if all_val_probs.shape[1] == 2:
                plt.figure(figsize=(8, 8))
                fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Save ROC curve
                roc_path = os.path.join(result_dir, f'{safe_model_name}_roc_curve.png')
                plt.savefig(roc_path)
                plt.close()
    
    # Load best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    # Plot and save training curves
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
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_auroc'], label='Training')
    plt.plot(history['val_auroc'], label='Validation')
    plt.title(f'{model_name} - AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{safe_model_name}_training_curves.png'))
    plt.close()
    
    return model, history, best_val_acc, best_val_auroc

# Function to evaluate and compare models
def evaluate_models(models, model_names, test_loader, device, num_runs=5, result_dir='.'):
    """
    Evaluate and compare multiple models
    """
    # Ensure directory exists
    result_dir = ensure_dir_exists(result_dir)
    
    results = {}
    
    for model, model_name in zip(models, model_names):
        print(f"\n{'-'*20} Evaluating {model_name} {'-'*20}")
        
        # Safe model name for file paths
        safe_model_name = sanitize_filename(model_name)
        
        # Evaluation metrics
        auroc_values = []
        acc_values = []
        
        for run in range(num_runs):
            print(f"Evaluation run {run+1}/{num_runs}")
            
            model.eval()
            all_preds = []
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
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
            
            # Calculate AUROC
            if all_probs.shape[1] == 2:  # Binary classification
                auroc = roc_auc_score(all_labels, all_probs[:, 1])
            else:  # Multi-class
                auroc = roc_auc_score(
                    np.eye(all_probs.shape[1])[all_labels], 
                    all_probs, 
                    multi_class='ovr'
                )
            auroc_values.append(auroc)
            
            # Plot ROC curve for this run
            if run == 0:  # Only plot for first run
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_name} - Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(result_dir, f'{safe_model_name}_roc_curve.png'))
                plt.close()
        
        # Calculate mean and std
        mean_auroc = np.mean(auroc_values)
        std_auroc = np.std(auroc_values)
        
        mean_acc = np.mean(acc_values)
        std_acc = np.std(acc_values)
        
        # Store results
        results[model_name] = {
            'accuracy': float(mean_acc),
            'accuracy_std': float(std_acc),
            'auroc': float(mean_auroc),
            'auroc_std': float(std_auroc),
            'all_auroc_values': [float(x) for x in auroc_values]
        }
        
        print(f"Results for {model_name}:")
        print(f"  - Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  - AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    
    # Create comparison plot
    if len(models) > 1:
        # AUROC comparison bar plot with error bars
        plt.figure(figsize=(10, 6))
        model_names_list = list(results.keys())
        auroc_values = [results[model]['auroc'] for model in model_names_list]
        auroc_stds = [results[model]['auroc_std'] for model in model_names_list]
        
        plt.bar(model_names_list, auroc_values, yerr=auroc_stds, alpha=0.7, capsize=10)
        plt.title('AUROC Comparison (with standard deviation)')
        plt.xlabel('Model')
        plt.ylabel('AUROC')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(auroc_values):
            plt.text(i, v + 0.02, f"{v:.3f}±{auroc_stds[i]:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'model_auroc_comparison.png'))
        plt.close()
        
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        acc_values = [results[model]['accuracy'] for model in model_names_list]
        acc_stds = [results[model]['accuracy_std'] for model in model_names_list]
        
        plt.bar(model_names_list, acc_values, yerr=acc_stds, alpha=0.7, capsize=10)
        plt.title('Accuracy Comparison (with standard deviation)')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(acc_values):
            plt.text(i, v + 0.02, f"{v:.3f}±{acc_stds[i]:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'model_accuracy_comparison.png'))
        plt.close()
    
    # Save results to file
    with open(os.path.join(result_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and compare DCRNN models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--connectivity", type=str, choices=['correlation', 'distance', 'both'], default='both',
                       help="Connectivity type for DCRNN")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, f"dcrnn_comparison_{timestamp}")
    ensure_dir_exists(result_dir)
    
    # Save arguments
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets for training and evaluation
    print("Loading training dataset...")
    train_dataset = SimpleEEGDataset(args.data_dir, is_train=True, verbose=True)
    
    print("\nLoading validation dataset...")
    val_dataset = SimpleEEGDataset(args.data_dir, is_train=False, verbose=True)
    
    # Check if datasets are valid
    if len(train_dataset) == 0 or train_dataset.data_shape is None:
        print("ERROR: Invalid training dataset. Please check the data directory and file format.")
        return
    
    # Create data loaders for training and evaluation
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Define models to train
    models_to_evaluate = []
    model_names = []
    
    if args.connectivity == 'correlation' or args.connectivity == 'both':
        # Correlation-based DCRNN
        corr_dcrnn = DCRNN(
            input_shape=train_dataset.data_shape,
            num_classes=2,
            connectivity_type='correlation',
            hidden_size=64,
            num_layers=2,
            threshold=0.5,
            bidirectional=True,
            dropout=0.5
        )
        
        # Apply weight initialization
        for name, param in corr_dcrnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        corr_dcrnn.to(device)
        
        # Train model
        print(f"\n{'-'*20} Training Correlation-DCRNN {'-'*20}")
        
        corr_dcrnn_trained, corr_history, corr_val_acc, corr_val_auroc = train_model(
            model=corr_dcrnn,
            model_name="Correlation-DCRNN",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=5,
            result_dir=result_dir
        )
        
        models_to_evaluate.append(corr_dcrnn_trained)
        model_names.append("Correlation-DCRNN")
        
    if args.connectivity == 'distance' or args.connectivity == 'both':
        # Distance-based DCRNN
        dist_dcrnn = DCRNN(
            input_shape=train_dataset.data_shape,
            num_classes=2,
            connectivity_type='distance',
            hidden_size=64,
            num_layers=2,
            threshold=0.5,
            bidirectional=True,
            dropout=0.5
        )
        
        # Apply weight initialization
        for name, param in dist_dcrnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        dist_dcrnn.to(device)
        
        # Train model
        print(f"\n{'-'*20} Training Distance-DCRNN {'-'*20}")
        
        dist_dcrnn_trained, dist_history, dist_val_acc, dist_val_auroc = train_model(
            model=dist_dcrnn,
            model_name="Distance-DCRNN",
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=5,
            result_dir=result_dir
        )
        
        models_to_evaluate.append(dist_dcrnn_trained)
        model_names.append("Distance-DCRNN")
    
    # Add a simple CNN model for comparison
    cnn_model = nn.Sequential(
        nn.Conv2d(train_dataset.data_shape[0], 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * (train_dataset.data_shape[1]//4) * (train_dataset.data_shape[2]//4), 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    ).to(device)
    
    # Train the CNN model
    print(f"\n{'-'*20} Training CNN Baseline {'-'*20}")
    
    cnn_trained, cnn_history, cnn_val_acc, cnn_val_auroc = train_model(
        model=cnn_model,
        model_name="CNN-Baseline",
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=5,
        result_dir=result_dir
    )
    
    models_to_evaluate.append(cnn_trained)
    model_names.append("CNN-Baseline")
    
    # Evaluate all models
    print("\nEvaluating all models...")
    evaluation_results = evaluate_models(
        models=models_to_evaluate,
        model_names=model_names,
        test_loader=val_loader,
        device=device,
        num_runs=5,
        result_dir=result_dir
    )
    
    # Print final results
    print("\nFinal Results (Mean ± Std):")
    print("-" * 50)
    print(f"{'Model':<25} {'AUROC':<15} {'Accuracy':<15}")
    print("-" * 50)
    
    for model_name in model_names:
        auroc = evaluation_results[model_name]['auroc']
        auroc_std = evaluation_results[model_name]['auroc_std']
        acc = evaluation_results[model_name]['accuracy']
        acc_std = evaluation_results[model_name]['accuracy_std']
        
        print(f"{model_name:<25} {auroc:.4f} ± {auroc_std:.4f} {acc:.4f} ± {acc_std:.4f}")
    
    print("-" * 50)
    print(f"\nAll training and evaluation completed. Results saved to {result_dir}")

if __name__ == "__main__":
    main()