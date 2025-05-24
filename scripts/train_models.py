import os
import torch
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMAGE_SIZE
from src.dataloader.histo_dataset import load_dataset
from src.preprocessing.transforms import get_transforms
from src.models.classifier import build_model
from src.training.train_classifier import train

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load full dataset (initially, just get a few hundred for testing)
    transform = get_transforms(train=True)
    dataset = load_dataset(RAW_DATA_PATH, transform)

    # Subset to avoid overloading during dev
    total_samples = 1000  
    image_paths = [dataset.image_paths[i] for i in range(total_samples)]
    labels = [dataset.labels[i] for i in range(total_samples)]

    # Recreate subset dataset
    from src.dataloader.histo_dataset import HistoDataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = HistoDataset(train_paths, train_labels, transform=get_transforms(train=True))
    val_dataset = HistoDataset(val_paths, val_labels, transform=get_transforms(train=False))

    model = build_model(pretrained=True)
    train(model, train_dataset, val_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, device=device)

if __name__ == "__main__":
    main()
