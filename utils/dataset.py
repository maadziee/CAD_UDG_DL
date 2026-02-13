"""
This module contains a custom PyTorch dataset class for the skin lesion dataset.
It is used to load the images and their corresponding labels (if available) from the disk.
"""

import os
from collections import Counter
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class SkinDataset(Dataset):
    """Custom PyTorch dataset for the skin lesion dataset."""

    def __init__(
        self, root_path, names, labels=None, transform=None, inference=False
    ):
        """
        Args:
            root_path (str): Root directory containing the images.
            names (list): List of filenames of the images (relative to root_path).
            labels (list, optional): List of corresponding class IDs for the images.
                                    Required if inference=False.
            transform (callable, optional): Optional transform to apply to the images.
            inference (bool): If True, the dataset will not expect labels (default=False).
        """
        self.root_path = root_path
        self.names = names
        self.labels = labels
        self.transform = transform
        self.inference = inference

        if not self.inference and self.labels is None:
            raise ValueError("Labels must be provided if inference=False.")
        if not self.inference and len(names) != len(labels):
            raise ValueError("Names and labels must have the same length.")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.names)

    def __str__(self):
        """Return a string representation of the dataset."""
        str_repr = f"Dataset: {len(self)} samples\n"
        if not self.inference:
            str_repr += f"Class distribution: {self._get_class_distribution()}\n"
        return str_repr

    def _get_class_distribution(self):
        """Calculate the class distribution."""
        if self.labels is not None:
            class_counts = Counter(self.labels)
            return dict(class_counts)
        return {}

    def __getitem__(self, idx):
        """Fetch the image (and label if available) at the given index."""
        # Construct the full path to the image
        img_path = os.path.join(self.root_path, self.names[idx])

        # Load the image
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if needed

        # Convert the PIL image to a NumPy array for Albumentations
        image = np.array(image)

        # Apply transformations, if provided
        if self.transform:
            augmented = self.transform(image=image)  # Pass as named argument
            image = augmented["image"]  # Extract the transformed image

        # Return only the image for inference
        if self.inference:
            return image, self.names[idx]

        # Fetch the label
        label = self.labels[idx]
        return image, label
