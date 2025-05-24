import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HistoDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def load_dataset(data_dir, transform):
    class_0_paths = glob(os.path.join(data_dir, "**", "0", "*.png"), recursive=True)
    class_1_paths = glob(os.path.join(data_dir, "**", "1", "*.png"), recursive=True)

    all_paths = class_0_paths + class_1_paths
    all_labels = [0] * len(class_0_paths) + [1] * len(class_1_paths)

    return HistoDataset(all_paths, all_labels, transform)
