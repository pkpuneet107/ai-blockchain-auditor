import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import numpy as np
from models.gcn_model import GCN
from tqdm import tqdm

def load_data_from_json(json_path):
    """
    Load and convert the JSON data to PyTorch Geometric Data objects
    """
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    
    dataset = []
    for item in data_list:
        # Extract graph structure
        edge_index = torch.tensor(item['graph'], dtype=torch.long)
        # Extract node features
        x = torch.tensor(item['node_features'], dtype=torch.float)
        # Extract target/label
        y = torch.tensor(int(item['targets']), dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    
    return dataset

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)
    
    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
    
    acc = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_path = "train_data/combined/train.json"
    val_path = "train_data/combined/valid.json"
    
    train_dataset = load_data_from_json(train_path)
    val_dataset = load_data_from_json(val_path)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Get input dimension from first data point
    input_dim = train_dataset[0].x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Initialize model
    model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=4)  # 4 classes (safe, reentrancy, timestamp, integer overflow)
    model = model.to(device)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "pretrained/trained_gcn_model.pt")
            print("Saved best model!")
        
        print("-----------------------------")
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()