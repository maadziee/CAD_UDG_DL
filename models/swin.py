import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.swin_transformer import (
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)


class SwinTransformer(nn.Module):
    """Swin Transformer model"""

    def __init__(self, num_classes, name, pretrained=True):
        """Initialize the model

        Args:
            num_classes (int): Number of classes
            name (string): Name of the model (e.g., "t", "s", "b", "v2-t", "v2-s", "v2-b")
            pretrained (bool, optional): Use pretrained weights. Defaults to True.
        """
        super(SwinTransformer, self).__init__()
        if name == "t":
            self.model = models.swin_t(
                weights=Swin_T_Weights.DEFAULT if pretrained else None
            )
        elif name == "s":
            self.model = models.swin_s(
                weights=Swin_S_Weights.DEFAULT if pretrained else None
            )
        elif name == "b":
            self.model = models.swin_b(
                weights=Swin_B_Weights.DEFAULT if pretrained else None
            )
        elif name == "v2-t":
            self.model = models.swin_v2_t(
                weights=Swin_V2_T_Weights.DEFAULT if pretrained else None
            )
        elif name == "v2-s":
            self.model = models.swin_v2_s(
                weights=Swin_V2_S_Weights.DEFAULT if pretrained else None
            )
        elif name == "v2-b":
            self.model = models.swin_v2_b(
                weights=Swin_V2_B_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError("Invalid model name")

        # Modify the classification head for the given number of classes
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        """Forward pass

        Args:
            x (tensor): Input tensor

        Returns:
            output: Output tensor
        """
        return self.model(x)
