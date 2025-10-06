#!/usr/bin/env python3

"""
ffnn.py - Feed Forward Neural Network 
Author: Suchi Jain
NetID: SXJ210111
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

### Load Data
class ReviewDataset(Dataset):
    def __init__(self, file_path, vocab=None, build_vocab=False, max_vocab=20000):
        # Read data from JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        self.texts = [item["text"] for item in data]
        self.labels = [item["label"] for item in data]

        # Build vocabulary
        if build_vocab:
            counter = Counter()
            for text in self.texts:
                counter.update(text.lower().split())
            most_common = counter.most_common(max_vocab)
            self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        else:
            self.vocab = vocab or {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Convert text into bag-of-words vector
        bow = torch.zeros(len(self.vocab))
        for word in self.texts[idx].lower().split():
            if word in self.vocab:
                bow[self.vocab[word]] += 1
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bow, label


### Define Simple FFNN Model
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.output(x)
        return x

### Evaluate Model
def evaluate(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


### Training Logic
def train_ffnn(train_file, val_file, epochs=5, batch_size=32, lr=0.001, hidden_dim=128):
    print("### 🚀 Starting FFNN Training ###")

    # Load training and validation datasets
    train_data = ReviewDataset(train_file, build_vocab=True)
    val_data = ReviewDataset(val_file, vocab=train_data.vocab)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Initialize model, optimizer, and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNN(len(train_data.vocab), hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train for given epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        # Evaluate after each epoch
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_data):.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    print("✅ Training Complete! Model saved as ffnn_model.pt")
    torch.save(model.state_dict(), "ffnn_model.pt")


### Main Run Block
if __name__ == "__main__":
    train_ffnn("training.json", "validation.json", epochs=5)