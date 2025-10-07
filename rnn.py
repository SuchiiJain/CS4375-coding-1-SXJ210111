#!/usr/bin/env python3
"""
rnn.py - Simple RNN (GRU) 
Author: Suchi Jain
NetID: SXJ210111
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

### Load Data as Sequences
class SeqDataset(Dataset):
    def __init__(self, file_path, vocab=None, build_vocab=False, max_vocab=20000, max_len=200):
        with open(file_path, "r") as f:
            data = json.load(f)

        self.texts = [d["text"].lower().split() for d in data]
        self.labels = [d["label"] for d in data]
        self.max_len = max_len

        # Build vocabulary if needed
        if build_vocab:
            counter = Counter()
            for t in self.texts:
                counter.update(t)
            common = counter.most_common(max_vocab)
            self.vocab = {w: i + 1 for i, (w, _) in enumerate(common)}  # reserve 0 for PAD
            self.vocab["<PAD>"] = 0
        else:
            self.vocab = vocab or {"<PAD>": 0}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx][:self.max_len]
        ids = [self.vocab.get(t, 0) for t in tokens]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx])


### Define GRU Model
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len]
        emb = self.embedding(x)
        output, _ = self.gru(emb)
        logits = self.fc(output)
        logits = logits.sum(dim=1)  # sum across time
        return logits


### Evaluate
def evaluate(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


### Train RNN
def train_rnn(train_file, val_file, epochs=5, batch_size=32, lr=0.001, emb_dim=100, hidden_dim=128):
    print("### Start RNN Training ###")

    train_data = SeqDataset(train_file, build_vocab=True)
    val_data = SeqDataset(val_file, vocab=train_data.vocab)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRNN(len(train_data.vocab), emb_dim, hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_data):.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    print("Training Complete! Model saved as rnn_model.pt")
    torch.save(model.state_dict(), "rnn_model.pt")


### Main
if __name__ == "__main__":
    train_rnn("training.json", "validation.json", epochs=5)
