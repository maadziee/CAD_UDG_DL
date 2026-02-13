import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Focal Loss as described in:
        "Focal Loss for Dense Object Detection" by Lin et al.

        Args:
            alpha (float): Balancing factor for the positive class.
            gamma (float): Focusing parameter to reduce the impact of easy examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Logits (before softmax) of shape (N, C), where C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: Loss value.
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Get the probabilities corresponding to the ground truth labels
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute the focal loss scaling factor
        focal_factor = (1 - pt) ** self.gamma

        # Apply class balancing factor alpha
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute the focal loss
        loss = (
            -alpha_factor * focal_factor * torch.log(pt + 1e-12)
        )  # Add epsilon to avoid log(0)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
