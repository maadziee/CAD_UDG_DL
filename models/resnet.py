"""
This is implementation of ResNet with Low-Rank Adaptation (LoRA) 
applied to convolutional layers in each block.
"""

import torch
import torch.nn as nn
from torchvision import models

class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA)"""
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALinear, self).__init__()
        self.base_layer = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) * 0.01)
        self.rank = rank

    def forward(self, x):
        # Base layer output
        base_output = self.base_layer(x)

        # LoRA update
        lora_update = x @ self.lora_A @ self.lora_B

        # Add the LoRA update to the base layer's output
        return base_output + lora_update


class ResNetLoRA(nn.Module):
    """ResNet with Low-Rank Adaptation (LoRA)"""
    __name__ = "ResNetLoRA"
    def __init__(self, weights, base_model_name="resnet50", num_classes=2, rank=4):
        """
        Args:
            weights (str): Pre-training weights to load.
            base_model_name (str): Name of the base ResNet model to load.
            num_classes (int): Number of output classes.
            rank (int): Rank of the LoRA approximation.
        """

        super(ResNetLoRA, self).__init__()
        # Load a pre-trained ResNet
        self.base_model = getattr(models, base_model_name)(weights=weights)

        # Replace the last fully connected layer with LoRALinear
        in_features = self.base_model.fc.in_features
        self.base_model.fc = LoRALinear(in_features, num_classes, rank=rank)

    def forward(self, x):
        return self.base_model(x)