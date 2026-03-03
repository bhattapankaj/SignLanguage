#!/usr/bin/env python3
"""
Quick demo setup: Generate synthetic data and train minimal models for demonstration.
This allows you to see the app working without needing to download the full Kaggle dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data, IMAGE_SIZE, NUM_CLASSES, get_torch_dataloaders
from src.model import MLP, ConvNet

def generate_synthetic_data():
    """Generate synthetic Sign Language MNIST data for demo."""
    os.makedirs("data", exist_ok=True)
    
    print("Generating synthetic training data...")
    # Generate 1000 training samples
    X_train = np.random.randint(0, 256, (1000, 28 * 28), dtype=np.uint8)
    # Valid labels: 0-8 (A-I) and 10-24 (K-Y), excluding 9 (J) and 25 (Z)
    valid_labels = [i for i in range(26) if i not in (9, 25)]
    y_train = np.random.choice(valid_labels, 1000).astype(np.int64)
    
    train_df = pd.DataFrame(X_train)
    train_df.insert(0, 'label', y_train)
    train_df.to_csv('data/sign_mnist_train.csv', index=False)
    print(f"✓ Created data/sign_mnist_train.csv with {len(train_df)} samples")
    
    print("Generating synthetic test data...")
    # Generate 200 test samples
    X_test = np.random.randint(0, 256, (200, 28 * 28), dtype=np.uint8)
    y_test = np.random.choice(valid_labels, 200).astype(np.int64)
    
    test_df = pd.DataFrame(X_test)
    test_df.insert(0, 'label', y_test)
    test_df.to_csv('data/sign_mnist_test.csv', index=False)
    print(f"✓ Created data/sign_mnist_test.csv with {len(test_df)} samples")

def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3, device='cpu'):
    """Train a model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")
    
    return model

def main():
    print("=" * 60)
    print("SIGN LANGUAGE DEMO SETUP")
    print("=" * 60)
    
    # Generate synthetic data
    generate_synthetic_data()
    
    # Load data
    print("\nLoading data...")
    data = load_data(data_dir='data')
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train MLP
    print("\n" + "=" * 60)
    print("Training MLP model...")
    print("=" * 60)
    mlp = MLP(input_size=784, hidden_units=[128, 64], num_classes=NUM_CLASSES, dropout=0.2)
    train_loader, test_loader = get_torch_dataloaders(data, batch_size=32)
    mlp = train_model(mlp, train_loader, test_loader, epochs=5, device=device)
    torch.save({
        "model_state_dict": mlp.state_dict(),
        "model_type": "mlp",
        "hidden_units": [128, 64],
        "dropout": 0.2,
        "num_classes": NUM_CLASSES,
    }, "models/mlp_model.pth")
    print("✓ Saved to models/mlp_model.pth")
    
    # Train CNN
    print("\n" + "=" * 60)
    print("Training CNN model...")
    print("=" * 60)
    cnn = ConvNet(num_classes=NUM_CLASSES, dropout=0.25)
    train_loader_cnn = DataLoader(
        TensorDataset(data.X_train_tensor, data.y_train_tensor),
        batch_size=32, shuffle=True
    )
    test_loader_cnn = DataLoader(
        TensorDataset(data.X_test_tensor, data.y_test_tensor),
        batch_size=32, shuffle=False
    )
    cnn = train_model(cnn, train_loader_cnn, test_loader_cnn, epochs=5, device=device)
    torch.save({
        "model_state_dict": cnn.state_dict(),
        "model_type": "cnn",
        "dropout": 0.25,
        "num_classes": NUM_CLASSES,
    }, "models/cnn_model.pth")
    print("✓ Saved to models/cnn_model.pth")
    
    print("\n" + "=" * 60)
    print("✓ Setup complete! Run with:")
    print("  streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
