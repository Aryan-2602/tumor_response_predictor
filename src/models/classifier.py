import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from src.config import NUM_CLASSES

def build_model(pretrained=True):
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model
