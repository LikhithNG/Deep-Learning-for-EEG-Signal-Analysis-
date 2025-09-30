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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import copy

# Add the repository root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# Basic model implementations without external dependencies
class SimpleDCRNNModel(nn.Module):
    def __init__(self, args, num_classes, device):
        super(SimpleDCRNNModel, self).__init__()
        self.num_nodes = args.num_nodes
        self.rnn_units = args.rnn_units
        self.num_rnn_layers = args.num_rnn_layers
        self.device = device
        
        # Process each channel separately using GRU
        self.channel_grus = nn.ModuleList([
            nn.GRU(
                input_size=args.input_dim,  # Use the input_dim from args
                hidden_size=self.rnn_units,
                num_layers=self.num_rnn_layers,
                batch_first=True,
                dropout=args.dropout if self.num_rnn_layers > 1 else 0
            ) for _ in range(self.num_nodes)
        ])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_units * self.num_nodes, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, inputs, seq_lengths, supports=None):
        # Expected shape: [batch_size, num_nodes, seq_len, input_dim]
        batch_size = inputs.shape[0]
        
        # Process each channel separately
        channel_outputs = []
        
        for i in range(self.num_nodes):
            # Extract this channel's data
            if len(inputs.shape) == 4:
                # If input has feature dimension: [batch_size, num_nodes, seq_len, features]
                channel_data = inputs[:, i, :, :]  # [batch, seq_len, features]
            else:
                # If input is already [batch_size, num_nodes, seq_len]
                channel_data = inputs[:, i, :].unsqueeze(-1)  # Add feature dim
            
            # Process with the channel's GRU
            gru_out, _ = self.channel_grus[i](channel_data)
            
            # Use the last output
            channel_outputs.append(gru_out[:, -1, :])  # [batch, rnn_units]
        
        # Concatenate outputs from all channels: [batch, num_nodes * rnn_units]
        combined = torch.cat(channel_outputs, dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output

class DenseCNN(nn.Module):
    def __init__(self, params, data_shape, num_classes):
        super(DenseCNN, self).__init__()
        self.seq_len, self.num_nodes = data_shape
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(
            in_channels=self.num_nodes, 
            out_channels=params.initialFilters, 
            kernel_size=5, 
            padding=2
        )
        
        # Dense layers
        self.dense1 = nn.Sequential(
            nn.BatchNorm1d(params.initialFilters),
            nn.ReLU(),
            nn.Conv1d(params.initialFilters, params.initialFilters * 2, kernel_size=3, padding=1),
            nn.Dropout(params.spatialDropout)
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(params.initialFilters * 2, 128),
            nn.ReLU(),
            nn.Dropout(params.depthDropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Input shape: [batch, seq_len, num_nodes]
        x = x.transpose(1, 2)  # -> [batch, num_nodes, seq_len]
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Dense block
        x = self.dense1(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

class LSTMModel(nn.Module):
    def __init__(self, args, num_classes, device):
        super(LSTMModel, self).__init__()
        self.num_nodes = args.num_nodes
        self.rnn_units = args.rnn_units
        self.num_rnn_layers = args.num_rnn_layers
        self.device = device
        self.bidirectional = args.bidirectional
        
        # Process each channel separately
        self.channel_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=args.input_dim,  # Use the input_dim from args
                hidden_size=self.rnn_units,
                num_layers=self.num_rnn_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=args.dropout if self.num_rnn_layers > 1 else 0
            ) for _ in range(self.num_nodes)
        ])
        
        # Output dimension will be doubled if bidirectional
        self.lstm_output_dim = self.rnn_units * 2 if self.bidirectional else self.rnn_units
        
        # Calculate the feature dimension for the classifier
        self.feature_dim = self.lstm_output_dim * self.num_nodes
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, inputs, seq_lengths, supports=None):
        # Expected shape: [batch_size, num_nodes, seq_len, input_dim=1]
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

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, input_dim=1):
        super(CNN_LSTM, self).__init__()
        
        # Store input dimension
        self.input_dim = input_dim
        
        # CNN for feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=32,  # Output from CNN
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 128),  # Bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, inputs, seq_lengths, supports=None):
        # Expected shape: [batch_size, channels, seq_len, features]
        batch_size = inputs.shape[0]
        
        # Add channel dimension for 2D convolution
        if len(inputs.shape) == 3:  # [batch, channels, seq_len]
            x = inputs.unsqueeze(1)  # [batch, 1, channels, seq_len]
        else:  # [batch, channels, seq_len, features]
            x = inputs.permute(0, 3, 1, 2).contiguous()  # [batch, features, channels, seq_len]
            if x.shape[1] == 0:  # If no feature dimension
                x = x.squeeze(1).unsqueeze(1)  # [batch, 1, channels, seq_len]
        
        # Apply CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape for LSTM: [batch, seq, features]
        # Flatten the spatial dimensions
        x = x.permute(0, 2, 1, 3).contiguous()
        seq_length = x.shape[1]
        x = x.view(batch_size, seq_length, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Classifier
        output = self.classifier(last_output)
        
        return output

# Dataset class for preprocessed npz files
class ClassificationDataset(Dataset):
    def __init__(self, preproc_dir, is_train=True, split_ratio=0.8):
        self.preproc_dir = preproc_dir
        self.file_list = glob.glob(os.path.join(preproc_dir, "*.npz"))
        
        # Remove non-data files
        self.file_list = [f for f in self.file_list if "class_distribution" not in f]
        
        print(f"Found {len(self.file_list)} preprocessed files")
        
        # Split into train/val sets
        np.random.seed(42)
        indices = np.random.permutation(len(self.file_list))
        split_idx = int(len(indices) * split_ratio)
        
        if is_train:
            self.file_list = [self.file_list[i] for i in indices[:split_idx]]
        else:
            self.file_list = [self.file_list[i] for i in indices[split_idx:]]
            
        print(f"Using {len(self.file_list)} files for {'training' if is_train else 'validation'}")
        
        # Check class distribution
        self._check_class_balance()
        
        # Load one file to get data shape
        if len(self.file_list) > 0:
            try:
                sample_data = np.load(self.file_list[0])
                self.data_shape = sample_data['data'].shape
                print(f"Data shape: {self.data_shape}")
            except Exception as e:
                print(f"Error loading sample file: {e}")
                self.data_shape = None
        else:
            self.data_shape = None
    
    def _check_class_balance(self):
        """Check class balance in dataset"""
        label_counts = {}
        for file_path in tqdm(self.file_list, desc="Checking class balance"):
            try:
                data = np.load(file_path)
                label = int(data['label'])
                label_counts[label] = label_counts.get(label, 0) + 1
            except Exception:
                continue
        
        total = sum(label_counts.values())
        if total > 0:
            print("Class distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"  Class {label}: {count} samples ({count/total*100:.1f}%)")
            
            # Store class distribution
            self.class_counts = label_counts
            
            # Get unique classes
            self.num_classes = len(label_counts)
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
            print(f"Error loading {file_path}: {e}")
            # Return dummy data
            if self.data_shape:
                x = torch.zeros(self.data_shape, dtype=torch.float32)
            else:
                x = torch.zeros((19, 10, 100), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

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
    
    # Create dummy supports (identity matrices)
    num_nodes = data.shape[1]
    supports = [torch.eye(num_nodes)]
    
    return data, labels, seq_lengths, supports

class ModelArgs:
    """Class to hold model arguments"""
    def __init__(self, **kwargs):
        # Set default values for potential missing attributes
        self.dropout = 0.1  # Default dropout rate
        self.cl_decay_steps = 1000  # Default curriculum learning decay steps
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_model(model_name, data_shape, num_classes, device):
    """Create a new model for classification"""
    channels, windows, freq_bins = data_shape
    
    # Determine the actual input size for LSTM/GRU models
    input_size = freq_bins if freq_bins > 0 else 1
    
    if model_name == "dcrnn":
        # Arguments for classification model
        class_args = ModelArgs(
            max_diffusion_step=2,
            num_rnn_layers=2,
            rnn_units=64,
            filter_type="dual_random_walk",
            num_nodes=channels,
            use_curriculum_learning=False,
            task="classification",
            input_dim=input_size,  # Use detected input size
            seq_len=windows,
            output_dim=1,
            dcgru_activation='tanh',
            use_gc_for_ru=True,
            temporal_activation='linear',
            dropout=0.1
        )
        
        # Create classification model
        model = SimpleDCRNNModel(args=class_args, num_classes=num_classes, device=device)
        
    elif model_name == "densecnn":
        # Load DenseCNN parameters
        params = {
            "initialFilters": 16,
            "filterMultiplier": 1.5,
            "spatialDropout": 0.1,
            "depthDropout": 0.1,
            "bottleneckFilters": 32,
            "bottleneckKernel": 14,
            "growthRate": 4,
            "reduction": 0.5,
            "numTransitions": 3,
            "widths": [3, 5, 7, 9],
            "init": "glorot_uniform"
        }
        params = DottedDict(params)
        
        # For DenseCNN, data shape is different
        data_shape_densecnn = (windows*freq_bins, channels)
        
        # Create model
        model = DenseCNN(params, data_shape=data_shape_densecnn, num_classes=num_classes)
        
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
        # CNN-LSTM with input dimension
        model = CNN_LSTM(num_classes, input_dim=input_size)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def train_model(model, model_name, train_loader, val_loader, device, save_dir, num_epochs, learning_rate=0.001):
    """Train the model with the new data"""
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Set up LR scheduler - removed verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    
    # Set up metrics
    best_val_f1 = 0.0
    patience_counter = 0
    max_patience = 15  # Early stopping patience
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            # Skip empty batches
            if batch is None:
                continue
                
            # Unpack batch
            inputs, labels, seq_lengths, supports = batch
            
            # Move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            seq_lengths = seq_lengths.to(device)
            supports = [s.to(device) for s in supports]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - different for each model
            if model_name == "dcrnn":
                outputs = model(inputs, seq_lengths, supports)
            elif model_name == "densecnn":
                # DenseCNN expects different input format
                batch_size = inputs.shape[0]
                # Reshape to (batch_size, seq_len*feature, num_nodes)
                inputs_reshaped = inputs.transpose(-1, -2).reshape(batch_size, -1, inputs.shape[1])
                outputs = model(inputs_reshaped)
            elif model_name == "lstm" or model_name == "cnnlstm":
                outputs = model(inputs, seq_lengths)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Save predictions for F1 score
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # Calculate F1 score if we have predictions from multiple classes
        unique_labels = np.unique(all_train_labels)
        if len(unique_labels) > 1:
            train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        else:
            train_f1 = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                # Skip empty batches
                if batch is None:
                    continue
                    
                # Unpack batch
                inputs, labels, seq_lengths, supports = batch
                
                # Move to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                seq_lengths = seq_lengths.to(device)
                supports = [s.to(device) for s in supports]
                
                # Forward pass - different for each model
                if model_name == "dcrnn":
                    outputs = model(inputs, seq_lengths, supports)
                elif model_name == "densecnn":
                    # DenseCNN expects different input format
                    batch_size = inputs.shape[0]
                    # Reshape to (batch_size, seq_len*feature, num_nodes)
                    inputs_reshaped = inputs.transpose(-1, -2).reshape(batch_size, -1, inputs.shape[1])
                    outputs = model(inputs_reshaped)
                elif model_name == "lstm" or model_name == "cnnlstm":
                    outputs = model(inputs, seq_lengths)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Save predictions for F1 score
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Calculate F1 score if we have predictions from multiple classes
        unique_labels = np.unique(all_val_labels)
        if len(unique_labels) > 1:
            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        else:
            val_f1 = 0.0
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Update learning rate - manually check and print changes
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if val_f1 > best_val_f1 or (val_f1 == 0 and val_acc > best_val_f1):  # Use accuracy if F1 is 0
            best_val_f1 = max(val_f1, best_val_f1)
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(save_dir, f"best_{model_name}_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, save_path)
            
            # Save confusion matrix
            if len(unique_labels) > 1:
                cm = confusion_matrix(all_val_labels, all_val_preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Validation Confusion Matrix - Epoch {epoch+1}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
                plt.close()
            
            # Save classification report
            report = classification_report(all_val_labels, all_val_preds, output_dict=True)
            with open(os.path.join(save_dir, f'classification_report_epoch_{epoch+1}.json'), 'w') as f:
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
    save_path = os.path.join(save_dir, f"final_{model_name}_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_f1': val_f1,
    }, save_path)
    
    return model, best_val_f1

def main():
    parser = argparse.ArgumentParser(description="Train models for seizure classification")
    
    # Data arguments
    parser.add_argument("--preproc_dir", type=str, required=True,
                       help="Directory containing preprocessed classification data")
    parser.add_argument("--save_dir", type=str, default="results/classification",
                       help="Directory to save results")
    
    # Model selection
    parser.add_argument("--model", type=str, default="dcrnn",
                       choices=["dcrnn", "densecnn", "lstm", "cnnlstm"],
                       help="Model architecture to use")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ClassificationDataset(args.preproc_dir, is_train=True)
    val_dataset = ClassificationDataset(args.preproc_dir, is_train=False)
    
    # Check if datasets are valid
    if len(train_dataset) == 0 or train_dataset.data_shape is None:
        print("No training data found or could not determine data shape. Exiting.")
        return
    
    # Determine number of classes
    num_classes = train_dataset.num_classes
    print(f"Training model for {num_classes} classes")
    
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
    
    # Create new model
    print(f"Creating new {args.model} model")
    model = create_model(
        model_name=args.model,
        data_shape=train_dataset.data_shape,
        num_classes=num_classes,
        device=device
    )
    model.to(device)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    
    # Train model
    print(f"Training {args.model} model for classification...")
    model, best_val_f1 = train_model(
        model=model,
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print(f"\nTraining complete. Best validation F1 score: {best_val_f1:.4f}")
    print(f"Models and results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()