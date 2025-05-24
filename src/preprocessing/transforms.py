import torchvision.transforms as T
from src.config import IMAGE_SIZE

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
        ])
