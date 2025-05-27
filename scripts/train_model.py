import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE
from src.dataloader.histo_dataset import load_dataset, HistoDataset
from src.preprocessing.transforms import get_transforms
from src.models.classifier import build_model
from src.training.train_classifier import train
from src.evaluation.metrics import evaluate_model, print_classification_metrics, plot_roc_curve

def main():
    # Use Apple M1 GPU (MPS) if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # Load ALL image paths and labels
    full_dataset = load_dataset(RAW_DATA_PATH, transform=None)
    image_paths = full_dataset.image_paths
    labels = full_dataset.labels
    print(f"Total samples loaded: {len(image_paths)}")
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    # Apply transforms separately to train and val
    train_dataset = HistoDataset(train_paths, train_labels, transform=get_transforms(train=True))
    val_dataset = HistoDataset(val_paths, val_labels, transform=get_transforms(train=False))
    # Build and train model
    model = build_model(pretrained=True)
    train(model, train_dataset, val_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, device=device)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    y_true, y_pred, y_probs = evaluate_model(model, val_loader, device)
    print_classification_metrics(y_true, y_pred)
    plot_roc_curve(y_true, y_probs, save_path="roc_curve.png")
    
if __name__ == "__main__":
    main()
