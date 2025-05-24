import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_accuracy

def train(model, train_dataset, val_dataset, epochs, batch_size, lr, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = compute_accuracy(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")
