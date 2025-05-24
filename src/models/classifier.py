import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def build_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model
