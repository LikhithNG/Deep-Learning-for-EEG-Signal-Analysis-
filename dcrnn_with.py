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

# Dynamic Graph Constructor
class DynamicGraphConstructor(nn.Module):
    def __init__(self, connectivity_type='correlation', threshold=0.5):
        super(DynamicGraphConstructor, self).__init__()
        self.connectivity_type = connectivity_type
        self.threshold = threshold
        
    def forward(self, x):
        """
        x: [batch_size, channels, time_steps, features]
        """
        batch_size, channels, time_steps, features = x.shape
        
        # Reshape for computing connectivity
        x_reshaped = x.permute(0, 1, 3, 2).reshape(batch_size, channels, -1)
        
        if self.connectivity_type == 'correlation':
            # Compute correlation coefficient between channels
            x_mean = x_reshaped.mean(dim=2, keepdim=True)
            x_std = x_reshaped.std(dim=2, keepdim=True) + 1e-8
            x_norm = (x_reshaped - x_mean) / x_std
            
            # Compute correlation matrix
            adj_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) / time_steps
            
            # Apply threshold
            adj_matrix = (adj_matrix > self.threshold).float()
            
        elif self.connectivity_type == 'distance':
            # Compute Euclidean distance between channel features
            adj_matrix = torch.zeros(batch_size, channels, channels, device=x.device)
            
            for i in range(channels):
                for j in range(channels):
                    if i != j:
                        dist = torch.sqrt(torch.sum((x_reshaped[:, i, :] - x_reshaped[:, j, :])**2, dim=1))
                        dist = dist / (dist.max() + 1e-8)
                        sim = 1 - dist
                        adj_matrix[:, i, j] = sim
            
            # Apply threshold
            adj_matrix = (adj_matrix > self.threshold).float()
        
        # Add self-connections
        adj_matrix = adj_matrix + torch.eye(channels, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Normalize adjacency matrix
        degree = torch.sum(adj_matrix, dim=2, keepdim=True)
        degree_sqrt_inv = torch.pow(degree + 1e-8, -0.5)
        adj_matrix = adj_matrix * degree_sqrt_inv * degree_sqrt_inv.transpose(1, 2)
        
        return adj_matrix

# Graph Convolution Layer
class GraphConvolution(nn.Module):
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
        x: [batch_size, channels, features]
        adj: [batch_size, channels, channels]
        """
        support = torch.bmm(adj, x)  # [batch_size, channels, features]
        output = torch.matmul(support, self.weight)  # [batch_size, channels, out_features]
        return output + self.bias

# DCRNN model
class DCRNN(nn.Module):
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
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output dimensions 
        self.lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer for classification
        self.fc = nn.Linear(self.lstm_output_size * channels, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Handle input formats
        if len(x.shape) == 3:
            channels, features = x.shape[1], x.shape[2]
            freq_bins = self.input_shape[2]
            windows = features // freq_bins
            x = x.reshape(x.shape[0], channels, windows, freq_bins)
        
        batch_size, channels, windows, freq_bins = x.shape
        
        # Graph construction
        adj_matrix = self.graph_constructor(x)  # [batch_size, channels, channels]
        
        # Process each time window with GCN
        gcn_outputs = []
        for t in range(windows):
            features_t = x[:, :, t, :]
            gcn_out = self.gcn(features_t, adj_matrix)  # [batch_size, channels, hidden_size]
            gcn_outputs.append(gcn_out.unsqueeze(2))  # Add time dimension
            
        # Concatenate along time dimension
        gcn_outputs = torch.cat(gcn_outputs, dim=2)  # [batch_size, channels, windows, hidden_size]
        
        # Process each channel with LSTM
        channel_outputs = []
        for i in range(channels):
            channel_data = gcn_outputs[:, i, :, :]  # [batch, windows, hidden_size]
            lstm_out, _ = self.lstm(channel_data)
            channel_outputs.append(lstm_out[:, -1, :])  # Use last time step output
        
        # Concatenate channel outputs
        combined = torch.cat(channel_outputs, dim=1)  # [batch_size, channels*lstm_output_size]
        
        # Apply dropout
        combined = self.dropout(combined)
        
        # Classification
        output = self.fc(combined)
        
        return output
    
    def extract_features(self, x):
        """Used for extracting features for pretraining"""
        # Handle input formats
        if len(x.shape) == 3:
            channels, features = x.shape[1], x.shape[2]
            freq_bins = self.input_shape[2]
            windows = features // freq_bins
            x = x.reshape(x.shape[0], channels, windows, freq_bins)
        
        batch_size, channels, windows, freq_bins = x.shape
        
        # Graph construction
        adj_matrix = self.graph_constructor(x)
        
        # Process each time window with GCN
        gcn_outputs = []
        for t in range(windows):
            features_t = x[:, :, t, :]
            gcn_out = self.gcn(features_t, adj_matrix)
            gcn_outputs.append(gcn_out.unsqueeze(2))
            
        gcn_outputs = torch.cat(gcn_outputs, dim=2)  # [batch_size, channels, windows, hidden_size]
        
        # Process each channel with LSTM
        lstm_features = []
        for i in range(channels):
            channel_data = gcn_outputs[:, i, :, :]
            lstm_out, _ = self.lstm(channel_data)
            lstm_features.append(lstm_out[:, -1, :])  # Just take the final output
            
        # Concatenate features
        combined = torch.cat(lstm_features, dim=1)  # [batch_size, channels*lstm_output_size]
        
        return combined, gcn_outputs

# Pretraining model wrapper for contrastive learning
class PretrainingModel(nn.Module):
    def __init__(self, base_model):
        super(PretrainingModel, self).__init__()
        self.base_model = base_model
        
        # Get LSTM output size
        lstm_size = base_model.lstm_output_size
        channels = base_model.input_shape[0]
        
        # Create a contrastive learning task for pretraining
        self.fc = nn.Sequential(
            nn.Linear(lstm_size * channels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary task: 0=different subject, 1=same subject
        )
        
    def forward(self, x1, x2=None):
        # Extract features from first input
        features1, _ = self.base_model.extract_features(x1)
        
        if x2 is not None:
            # Extract features from second input 
            features2, _ = self.base_model.extract_features(x2)
            
            # Compute similarity score
            output = self.fc(torch.abs(features1 - features2))
            return output
        else:
            # Just return features for single input
            return features1

# EEG Dataset
class EEGDataset(Dataset):
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
        
        # Find npz files recursively
        import fnmatch
        self.file_list = []
        for root, dirs, files in os.walk(preproc_dir):
            for filename in fnmatch.filter(files, "*.npz"):
                self.file_list.append(os.path.join(root, filename))
        
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
        
        if is_train is not None:  # Allow None for using all data (pretraining)
            if is_train:
                self.file_list = [self.file_list[i] for i in indices[:split]]
            else:
                self.file_list = [self.file_list[i] for i in indices[split:]]
                
            if verbose:
                print(f"Using {len(self.file_list)} files for {'training' if is_train else 'testing'}")
        
        # Load one file to get data shape
        if len(self.file_list) > 0:
            try:
                sample_data = np.load(self.file_list[0], allow_pickle=True)
                
                # Check for required keys
                if 'data' not in sample_data:
                    if verbose:
                        print(f"ERROR: 'data' key not found in the npz file! Available keys: {list(sample_data.keys())}")
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
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            data = np.load(file_path, allow_pickle=True)
            x = data['data']
            
            # Convert to torch tensors
            x = torch.tensor(x, dtype=torch.float32)
            
            # Get label if it exists
            if 'label' in data:
                y = int(data['label'])
                y = torch.tensor(y, dtype=torch.long)
            else:
                # Default label
                y = torch.tensor(0, dtype=torch.long)
            
            return x, y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data
            if hasattr(self, 'data_shape') and self.data_shape is not None:
                x = torch.zeros(self.data_shape, dtype=torch.float32)
            else:
                x = torch.zeros((19, 10, 100), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

# Pretraining function
def pretrain_model(model, pretraining_model, train_loader, val_loader, device, 
                   num_epochs=30, learning_rate=0.001, patience=5, result_dir='.', batch_size=16):
    """
    Pretrain using a contrastive learning approach
    """
    # Ensure directory exists
    result_dir = ensure_dir_exists(result_dir)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretraining_model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    print(f"Starting pretraining with {num_epochs} epochs")
    print("Using a contrastive learning approach for pretraining")
    
    # Create data pairs for training
    train_data = [(i, j, 1 if i//4 == j//4 else 0) 
                  for i in range(len(train_loader.dataset)) 
                  for j in range(i+1, min(i+5, len(train_loader.dataset)))]
    
    val_data = [(i, j, 1 if i//4 == j//4 else 0) 
                for i in range(len(val_loader.dataset)) 
                for j in range(i+1, min(i+5, len(val_loader.dataset)))]
    
    # Calculate batch size for pairs - use smaller of provided or 16
    pair_batch_size = min(batch_size // 2, 16)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        pretraining_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle pairs
        np.random.shuffle(train_data)
        
        # Process in batches
        num_batches = len(train_data) // pair_batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Pretraining"):
            # Get batch of pairs
            batch_pairs = train_data[batch_idx * pair_batch_size:(batch_idx + 1) * pair_batch_size]
            
            # Prepare data
            data1 = []
            data2 = []
            labels = []
            
            for idx1, idx2, label in batch_pairs:
                # Get data samples
                try:
                    x1, _ = train_loader.dataset[idx1]
                    x2, _ = train_loader.dataset[idx2]
                    
                    data1.append(x1)
                    data2.append(x2)
                    labels.append(label)
                except Exception as e:
                    continue
            
            if not data1:
                continue
                
            # Convert to tensors
            data1 = torch.stack(data1).to(device)
            data2 = torch.stack(data2).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = pretraining_model(data1, data2)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretraining_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        epoch_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        epoch_acc = 100 * correct / total if total > 0 else 0
        
        history['train_loss'].append(epoch_loss)
        
        # Validation phase
        pretraining_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Process validation in batches
        val_num_batches = min(len(val_data) // pair_batch_size, 10)  # Limit eval time
        
        with torch.no_grad():
            for batch_idx in tqdm(range(val_num_batches), desc="Validating"):
                # Get batch of pairs
                batch_pairs = val_data[batch_idx * pair_batch_size:(batch_idx + 1) * pair_batch_size]
                
                # Prepare data
                data1 = []
                data2 = []
                labels = []
                
                for idx1, idx2, label in batch_pairs:
                    try:
                        x1, _ = val_loader.dataset[idx1]
                        x2, _ = val_loader.dataset[idx2]
                        
                        data1.append(x1)
                        data2.append(x2)
                        labels.append(label)
                    except Exception:
                        continue
                
                if not data1:
                    continue
                    
                # Convert to tensors
                data1 = torch.stack(data1).to(device)
                data2 = torch.stack(data2).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                
                # Forward pass
                outputs = pretraining_model(data1, data2)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / val_num_batches if val_num_batches > 0 else float('inf')
        epoch_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        history['val_loss'].append(epoch_val_loss)
        
        # Print metrics
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            
            print(f"New best model saved with validation loss: {epoch_val_loss:.4f}")
            
            # Save model checkpoint with safe path
            model_path = os.path.join(result_dir, 'best_pretrained_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': epoch_val_loss,
            }, model_path)
            
        # Early stopping check
        if epoch >= 10 and epoch_val_loss > history['val_loss'][-2]:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        else:
            patience_counter = 0
    
    # Load best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        print(f"Loaded best pretrained model weights")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(result_dir, 'pretraining_loss.png'))
    plt.close()
    
    return model, history, best_val_loss

# Fine-tuning function
def train_model(model, model_name, train_loader, val_loader, device, 
                num_epochs=30, learning_rate=0.001, patience=5, result_dir='.', pretrained=False):
    """
    Fine-tune a pretrained model or train from scratch
    """
    # Ensure directory exists
    result_dir = ensure_dir_exists(result_dir)
    
    # Sanitize model name for file system
    safe_model_name = sanitize_filename(model_name)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler - use AUROC for scheduling
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
    best_val_auroc = 0.0
    best_model_weights = None
    patience_counter = 0
    
    print(f"Starting training of {model_name}")
    if pretrained:
        print("Using pretrained weights")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_probs = []  # Store probabilities for AUROC
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
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Calculate training AUROC
        all_train_probs = np.vstack(all_train_probs)
        all_train_labels = np.concatenate(all_train_labels)
        
        # For binary classification (adaptation for multi-class if needed)
        train_auroc = roc_auc_score(all_train_labels, all_train_probs[:, 1])
        
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
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        # Calculate validation AUROC
        all_val_probs = np.vstack(all_val_probs)
        all_val_labels = np.concatenate(all_val_labels)
        
        # For binary classification
        val_auroc = roc_auc_score(all_val_labels, all_val_probs[:, 1])
        
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
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            
            print(f"New best model saved with validation AUROC: {val_auroc:.4f}")
            
            # Save model checkpoint
            model_path = os.path.join(result_dir, f'best_{safe_model_name}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
            }, model_path)
            
            # Create confusion matrix
            all_val_preds = np.array(all_val_preds)
            cm = confusion_matrix(all_val_labels, all_val_preds)
            
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
            
            # Plot and save ROC curve
            plt.figure(figsize=(8, 8))
            fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
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
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. (Best AUROC: {best_val_auroc:.4f})")
            
            # Early stopping
            if patience_counter >= patience * 2:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
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
    
    return model, history, best_val_auroc

# Evaluation function
def evaluate_models(models, model_names, test_loader, device, num_runs=5, result_dir='.'):
    """Evaluate and compare multiple models"""
    # Ensure directory exists
    result_dir = ensure_dir_exists(result_dir)
    
    results = {}
    
    for model, model_name in zip(models, model_names):
        print(f"\n{'-'*20} Evaluating {model_name} {'-'*20}")
        
        # Safe model name for file paths
        safe_model_name = sanitize_filename(model_name)
        
        # Run multiple evaluations for robust results
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
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            auroc_values.append(auroc)
            
            # Plot ROC curve for first run
            if run == 0:
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_name} - ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(result_dir, f'{safe_model_name}_eval_roc_curve.png'))
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
    
    # Create comparison plots if multiple models
    if len(models) > 1:
        # AUROC comparison
        plt.figure(figsize=(10, 6))
        model_names_list = list(results.keys())
        auroc_values = [results[model]['auroc'] for model in model_names_list]
        auroc_stds = [results[model]['auroc_std'] for model in model_names_list]
        
        # Sort models by AUROC
        indices = np.argsort(auroc_values)[::-1]  # Descending order
        sorted_models = [model_names_list[i] for i in indices]
        sorted_aurocs = [auroc_values[i] for i in indices]
        sorted_stds = [auroc_stds[i] for i in indices]
        
        # Plot bars
        bars = plt.bar(sorted_models, sorted_aurocs, yerr=sorted_stds, alpha=0.7, capsize=10)
        
        # Color bars by model type
        for i, model_name in enumerate(sorted_models):
            if "Pretrained" in model_name:
                bars[i].set_color('#2c7fb8')  # Blue for pretrained
            else:
                bars[i].set_color('#7fcdbb')  # Light blue for non-pretrained
        
        plt.title('AUROC Comparison (with standard deviation)')
        plt.xlabel('Model')
        plt.ylabel('AUROC')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(sorted_aurocs):
            plt.text(i, v + 0.02, f"{v:.3f}±{sorted_stds[i]:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'model_auroc_comparison.png'))
        plt.close()
        
        # Create specific comparison of pretraining effect
        plt.figure(figsize=(12, 5))
        
        # Find pairs of models with and without pretraining
        pairs = []
        for model in model_names_list:
            if "Pretrained" in model:
                base_model = model.replace("Pretrained", "NoPretrain")
                if base_model in model_names_list:
                    pairs.append((base_model, model))
        
        if pairs:
            # Side by side comparison
            plt.subplot(1, 2, 1)
            
            labels = []
            no_pretrain_values = []
            pretrain_values = []
            
            for no_pretrain, pretrain in pairs:
                labels.append(no_pretrain.split('-')[0])  # Just use connectivity type
                no_pretrain_values.append(results[no_pretrain]['auroc'])
                pretrain_values.append(results[pretrain]['auroc'])
            
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, no_pretrain_values, width, label='Without Pretraining')
            plt.bar(x + width/2, pretrain_values, width, label='With Pretraining')
            
            plt.xlabel('Connectivity Type')
            plt.ylabel('AUROC')
            plt.title('Effect of Pretraining on AUROC')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Improvement percentage
            plt.subplot(1, 2, 2)
            improvements = []
            
            for i in range(len(labels)):
                improvement = ((pretrain_values[i] - no_pretrain_values[i]) / no_pretrain_values[i]) * 100
                improvements.append(improvement)
            
            plt.bar(labels, improvements, alpha=0.7)
            plt.xlabel('Connectivity Type')
            plt.ylabel('Improvement (%)')
            plt.title('Percentage Improvement from Pretraining')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(improvements):
                plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, 'pretraining_effect.png'))
            plt.close()
    
    # Save results to file
    with open(os.path.join(result_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DCRNN models with pretraining")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--pretraining_epochs", type=int, default=20, help="Number of epochs for pretraining")
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
    result_dir = os.path.join(args.result_dir, f"dcrnn_pretraining_{timestamp}")
    ensure_dir_exists(result_dir)
    
    # Save arguments
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading training dataset...")
    train_dataset = EEGDataset(args.data_dir, is_train=True, verbose=True)
    
    print("\nLoading validation dataset...")
    val_dataset = EEGDataset(args.data_dir, is_train=False, verbose=True)
    
    # Check if datasets are valid
    if len(train_dataset) == 0 or train_dataset.data_shape is None:
        print("ERROR: Invalid training dataset. Please check the data directory and file format.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Models to evaluate
    models_to_evaluate = []
    model_names = []
    
    # Process each connectivity type
    connectivity_types = []
    if args.connectivity == 'correlation' or args.connectivity == 'both':
        connectivity_types.append('correlation')
    if args.connectivity == 'distance' or args.connectivity == 'both':
        connectivity_types.append('distance')
    
    for conn_type in connectivity_types:
        # 1. Train DCRNN without pretraining
        print(f"\n{'-'*20} Training {conn_type.capitalize()}-DCRNN without pretraining {'-'*20}")
        
        # Create model
        dcrnn = DCRNN(
            input_shape=train_dataset.data_shape,
            num_classes=2,  # Binary classification
            connectivity_type=conn_type,
            hidden_size=64,
            num_layers=2,
            threshold=0.5,
            bidirectional=True,
            dropout=0.5
        )
        
        # Initialize weights
        for name, param in dcrnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        dcrnn.to(device)
        
        # Train without pretraining
        model_name = f"{conn_type.capitalize()}-DCRNN-NoPretrain"
        no_pretrain_dir = os.path.join(result_dir, sanitize_filename(model_name))
        ensure_dir_exists(no_pretrain_dir)
        
        dcrnn_no_pretrain, history, best_auroc = train_model(
            model=dcrnn,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=5,
            result_dir=no_pretrain_dir
        )
        
        models_to_evaluate.append(dcrnn_no_pretrain)
        model_names.append(model_name)
        
        # 2. Train DCRNN with pretraining
        print(f"\n{'-'*20} Training {conn_type.capitalize()}-DCRNN with pretraining {'-'*20}")
        
        # Create model for pretraining
        dcrnn_for_pretrain = DCRNN(
            input_shape=train_dataset.data_shape,
            num_classes=2,
            connectivity_type=conn_type,
            hidden_size=64,
            num_layers=2,
            threshold=0.5,
            bidirectional=True,
            dropout=0.5
        )
        
        # Initialize weights
        for name, param in dcrnn_for_pretrain.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        dcrnn_for_pretrain.to(device)
        
        # Create pretraining model
        pretraining_model = PretrainingModel(dcrnn_for_pretrain)
        pretraining_model.to(device)
        
        # Pretrain the model
        pretrain_dir = os.path.join(result_dir, f'pretrain_{conn_type}')
        ensure_dir_exists(pretrain_dir)
        
        pretrained_dcrnn, pretrain_history, _ = pretrain_model(
            model=dcrnn_for_pretrain,
            pretraining_model=pretraining_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.pretraining_epochs,
            learning_rate=args.lr,
            patience=5,
            result_dir=pretrain_dir,
            batch_size=args.batch_size
        )
        
        # Fine-tune the pretrained model
        model_name = f"{conn_type.capitalize()}-DCRNN-Pretrained"
        finetune_dir = os.path.join(result_dir, sanitize_filename(model_name))
        ensure_dir_exists(finetune_dir)
        
        finetuned_dcrnn, finetune_history, best_auroc_pretrained = train_model(
            model=pretrained_dcrnn,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            learning_rate=args.lr / 2,  # Lower learning rate for fine-tuning
            patience=5,
            result_dir=finetune_dir,
            pretrained=True
        )
        
        models_to_evaluate.append(finetuned_dcrnn)
        model_names.append(model_name)
    
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
    print("-" * 60)
    print(f"{'Model':<30} {'AUROC':<15} {'Accuracy':<15}")
    print("-" * 60)
    
    for model_name in model_names:
        auroc = evaluation_results[model_name]['auroc']
        auroc_std = evaluation_results[model_name]['auroc_std']
        acc = evaluation_results[model_name]['accuracy']
        acc_std = evaluation_results[model_name]['accuracy_std']
        
        print(f"{model_name:<30} {auroc:.4f} ± {auroc_std:.4f} {acc:.4f} ± {acc_std:.4f}")
    
    print("-" * 60)
    print(f"\nAll training and evaluation completed. Results saved to {result_dir}")

if __name__ == "__main__":
    main()