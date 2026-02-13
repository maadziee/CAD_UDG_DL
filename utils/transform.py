"""
This module contains custom transformations for data augmentation.
"""

import os
import random
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform

import torch
from torchvision.transforms import functional as F


class ResizeKeepRatio:
    """Resize an image while maintaining the aspect ratio by adding padding."""

    def __init__(self, size, fill_color=(0, 0, 0)):
        """
        Resize an image while maintaining the aspect ratio by adding padding.

        Args:
            size (int): Desired size of the shorter side.
            fill_color (tuple): RGB color for padding (default: black).
        """
        self.size = size
        self.fill_color = fill_color

    def __call__(self, image):
        # Get original dimensions
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        # Resize while keeping the aspect ratio
        if original_width > original_height:
            new_width = self.size
            new_height = int(self.size / aspect_ratio)
        else:
            new_height = self.size
            new_width = int(self.size * aspect_ratio)

        image = F.resize(image, (new_height, new_width))

        # Add padding to make the image square
        delta_width = self.size - new_width
        delta_height = self.size - new_height
        padding = (
            delta_width // 2,
            delta_height // 2,
            delta_width - delta_width // 2,
            delta_height - delta_height // 2,
        )
        image = F.pad(image, padding, fill=self.fill_color)

        return image


class GaussianNoiseInjection:
    """Add Gaussian noise to an image."""

    def __init__(self, mean=0, std=0.1):
        """
        Add Gaussian noise to an image.

        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Convert image to tensor
        image = F.to_tensor(image)

        # Add noise
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = image + noise

        # Clip the image to [0, 1]
        noisy_image = torch.clamp(noisy_image, 0, 1)

        # Convert tensor back to image
        noisy_image = F.to_pil_image(noisy_image)

        return noisy_image


class ObjectCentricCropping:
    def __init__(self):
        """
        Object-centric cropping for skin lesion images.
        This crops the image to the bounding box containing the object (lesion) and returns the cropped image.
        """
        pass

    def __call__(self, img):
        """
        Crop the image around the object (lesion).

        Args:
            img (PIL.Image or numpy.ndarray): Input image.

        Returns:
            PIL.Image: Cropped image centered on the object.
        """
        # Convert PIL Image to OpenCV format (BGR)
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours of the object in the binary mask
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # If no contours are found, return the original image as is
            print("No object detected. Returning the original image.")
            crop = img
        else:
            # Get the largest contour (assuming it's the lesion)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Ensure the bounding box is within image dimensions
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            # Crop the image around the bounding box
            crop = img[y : y + h, x : x + w]

        return crop


class ObjectAwareRandomCropping(ImageOnlyTransform):
    """Object-aware random cropping for skin lesion images."""

    def __init__(self, width=224, height=224, p=1.0):
        """
        Initialize the ObjectAwareRandomCropping class.

        Args:
            width (int): Width of the cropped image.
            height (int): Height of the cropped image.
        """
        super(ObjectAwareRandomCropping, self).__init__()
        self.width = width
        self.height = height
        self.crop_size = (width, height)
        self.object_centric_cropper = ObjectCentricCropping()
        self.p = p

    def apply(self, img, **params):
        """
        Perform object-aware random cropping on the image.

        Args:
            img (PIL.Image or numpy.ndarray): Input image.

        Returns:
            PIL.Image: Randomly cropped image centered on the object.
        """
        if np.random.uniform(0, 1) > self.p:
            return img
        # Convert PIL Image to OpenCV format (BGR)
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Perform object-centric cropping
        cropped_img = self.object_centric_cropper(img)

        # Get the bounding box coordinates
        h, w = cropped_img.shape[:2]

        if w < self.crop_size[1] and h < self.crop_size[0]:
            # If the bounding box is smaller than the desired crop size, return the original image
            print("Object is too small. Returning the original image.")
            return img

        # Generate random offsets within the image bounds
        max_x_offset = max(0, w - self.crop_size[1])
        max_y_offset = max(0, h - self.crop_size[0])
        x_offset = random.randint(0, max_x_offset)
        y_offset = random.randint(0, max_y_offset)

        # Calculate the crop coordinates
        x_crop = max(0, min(x_offset, cropped_img.shape[1] - self.crop_size[1]))
        y_crop = max(0, min(y_offset, cropped_img.shape[0] - self.crop_size[0]))

        # Perform the random crop
        random_crop = cropped_img[
            y_crop : y_crop + self.crop_size[0], x_crop : x_crop + self.crop_size[1]
        ]

        return random_crop

    def get_transform_init_args_names(self):
        """
        Return the arguments required to initialize the transform.

        Returns:
            tuple: Tuple of argument names.
        """
        return ("width", "height", "p")


class AdvancedHairAugmentation:
    """Add hairs to an image."""

    def __init__(self, hairs: int = 4, hairs_folder: str = ""):
        """
        Hair augmentation for images.

        Args:
            hairs (int): Maximum number of hairs to add.
            hairs_folder (str): Path to the folder containing the hair images.
        """
        self.hairs = hairs
        self.hairs_folder = hairs_folder
        self.hair_images = [im for im in os.listdir(self.hairs_folder) if "png" in im]

    def __call__(self, img):
        """
        Add hairs to an image.

        Args:
            img (PIL Image or ndarray): Image to add hairs to.

        Returns:
            PIL Image: Image with hairs added.
        """
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        # Convert PIL Image to OpenCV format (BGR)
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        height, width, _ = img.shape  # target image width and height

        for _ in range(n_hairs):
            hair = cv2.imread(
                os.path.join(self.hairs_folder, random.choice(self.hair_images))
            )
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img


if __name__ == "__main__":
    # Example of how to use the transformations
    from torchvision.transforms import Compose

    # Load an image
    image = Image.open("/root/madz/datasets/Binary/train/others/ack00364.jpg")

    # Define the transformations
    transform = Compose(
        [
            AdvancedHairAugmentation(
                hairs=4,
                hairs_folder="/root/madz/Deep-Skin-Lesion-Classification/datasets/hairs",
            ),
        ]
    )

    # Apply the transformations
    transformed_image = transform(image)
    transformed_image.save("transformed_image.jpg")
