import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)


class EfficientNet(nn.Module):
    """EfficientNet model"""

    def __init__(self, num_classes, name, pretrained=True):
        """Initialize the model

        Args:
            num_classes (int): Number of classes
            name (string): Name of the model from b0 to b7, v2-s, v2-m, v2-l
            pretrained (bool, optional): Use pretrained weights. Defaults to True.
        """
        super(EfficientNet, self).__init__()
        if name == "b0":
            self.model = models.efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
        elif name == "b1":
            self.model = models.efficientnet_b1(
                weights=EfficientNet_B1_Weights.DEFAULT if pretrained else None
            )
        elif name == "b2":
            self.model = models.efficientnet_b2(
                weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None
            )
        elif name == "b3":
            self.model = models.efficientnet_b3(
                weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None
            )
        elif name == "b4":
            self.model = models.efficientnet_b4(
                weights=EfficientNet_B4_Weights.DEFAULT if pretrained else None
            )
        elif name == "b5":
            self.model = models.efficientnet_b5(
                weights=EfficientNet_B5_Weights.DEFAULT if pretrained else None
            )
        elif name == "b6":
            self.model = models.efficientnet_b6(
                weights=EfficientNet_B6_Weights.DEFAULT if pretrained else None
            )
        elif name == "b7":
            self.model = models.efficientnet_b7(
                weights=EfficientNet_B7_Weights.DEFAULT if pretrained else None
            )
        elif name == "v2-s":
            self.model = models.efficientnet_v2_s(
                weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            )
        elif name == "v2-m":
            self.model = models.efficientnet_v2_m(
                weights=EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            )
        elif name == "v2-l":
            self.model = models.efficientnet_v2_l(
                weights=EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError("Invalid model name")

        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        """Forward pass

        Args:
            x (tensor): Input tensor

        Returns:
            output: Output tensor
        """
        return self.model(x)
