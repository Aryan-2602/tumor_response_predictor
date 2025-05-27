import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn, optim
from src.evaluation.metrics import (
    compute_accuracy,
    evaluate_model,
    print_classification_metrics,
    plot_roc_curve
)
from src.config import MODEL_DIR

def train(model, train_dataset, val_dataset, epochs, batch_size, lr, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

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

        # Optional quick accuracy per epoch (you can comment this if not needed)
        val_acc = compute_accuracy(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")

    # === Evaluate after training ===
    print("\nFinal evaluation on validation set:")
    y_true, y_pred, y_probs = evaluate_model(model, val_loader, device)
    print_classification_metrics(y_true, y_pred)

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === Save ROC curve ===
    roc_path = os.path.join(MODEL_DIR, f"roc_curve_{timestamp}.png")
    plot_roc_curve(y_true, y_probs, save_path=roc_path)

    # === Save model ===  m
    model_path = os.path.join(MODEL_DIR, f"model_final_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

