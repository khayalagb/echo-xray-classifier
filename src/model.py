"""ResNet18 factory. Kept from v1 as-is — correct choice for this dataset size."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 3) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
