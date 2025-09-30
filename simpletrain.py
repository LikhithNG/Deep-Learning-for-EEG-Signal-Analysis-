import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

# Simplified dataset that only uses preprocessed npz files
class SimpleEEGDataset(Dataset):
    def __init__(self, preproc_dir, is_train=True):
        self.preproc_dir = preproc_dir
        self.file_list = glob.glob(os.path.join(preproc_dir, "*.npz"))
        print(f"Found {len(self.file_list)} files in {preproc_dir}")
        
        # Split into train/test (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.file_list))
        split = int(len(indices) * 0.8)
        
        if is_train:
            self.file_list = [self.file_list[i] for i in indices[:split]]
        else:
            self.file_list = [self.file_list[i] for i in indices[split:]]
            
        print(f"Using {len(self.file_list)} files for {'training' if is_train else 'testing'}")
        
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
            print(f"Error loading {file_path}: {e}")
            # Return dummy data with same shape as first sample
            if hasattr(self, 'data_shape') and self.data_shape is not None:
                x = torch.zeros(self.data_shape, dtype=torch.float32)
            else:
                x = torch.zeros((19, 10, 100), dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
            return x, y

# Flexible CNN model that adapts to input shape
class FlexibleCNN(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(FlexibleCNN, self).__init__()
        
        # Save input shape
        self.input_shape = input_shape
        channels, windows, freq_bins = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate output size after conv+pool layers
        # After 2 pooling layers, dimensions are reduced by factor of 4
        output_windows = windows // 4
        output_freq = freq_bins // 4
        
        # Make sure dimensions don't become zero
        output_windows = max(1, output_windows)
        output_freq = max(1, output_freq)
        
        # Final layer size depends on output dimensions
        flat_size = 64 * output_windows * output_freq
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        print(f"Model will flatten to size: {flat_size}")
        
    def forward(self, x):
        # Input shape: (batch, channels, windows, freq_bins)
        
        # Handle different input formats
        if len(x.shape) == 3:
            # If input is (batch, channels, features), reshape to (batch, channels, windows, freq_bins)
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

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total if total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Print progress
        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader) if len(val_loader) > 0 else 0:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    return model

def main():
    # Parameters
    preproc_dir = "preproc_data/detection_clip12"
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001
    
    # Create datasets
    train_dataset = SimpleEEGDataset(preproc_dir, is_train=True)
    val_dataset = SimpleEEGDataset(preproc_dir, is_train=False)
    
    # Exit if no data was found
    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return
    
    # Get data shape from the dataset
    if not hasattr(train_dataset, 'data_shape') or train_dataset.data_shape is None:
        print("Could not determine data shape. Exiting.")
        return
    
    # Create data loaders with appropriate batch size
    # Use smaller batch size if limited data
    actual_batch_size = min(batch_size, len(train_dataset))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size,
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=actual_batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Create model that adapts to the data shape
    model = FlexibleCNN(input_shape=train_dataset.data_shape, num_classes=2)
    
    # Train model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), "final_model.pth")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main()