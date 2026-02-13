"""Utility functions for training and testing the model."""

import json
import mlflow
import numpy as np

import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score

from utils.transform import (
    GaussianNoiseInjection,
    AdvancedHairAugmentation,
    ObjectCentricCropping,
    ObjectAwareRandomCropping,
)

CUSTOM_TRANSFORMS = {
    "GaussianNoiseInjection": GaussianNoiseInjection,
    "AdvancedHairAugmentation": AdvancedHairAugmentation,
    "ObjectCentricCropping": ObjectCentricCropping,
    "ObjectAwareRandomCropping": ObjectAwareRandomCropping,
}


def freeze_layers(model, layers):
    """
    Freeze the specified layers of the model.

    Args:
        model: The PyTorch model.
        layers (list): List of layer names to freeze.
    """
    # Parse the freeze layers argument
    # Freeze specified layers
    if layers:
        layers = layers.split(",")
        print(f"Freezing specified layers: {layers}")
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
            else:
                param.requires_grad = True


def load_data_file(input_file, inference=False):
    """
    Load image names and labels from a text file.

    Args:
        input_file (str): Path to the input file.
        inference (bool): If True, assume no labels are present in the file.

    Returns:
        list: List of image names.
        list or None: List of labels, or None if inference=True.
    """
    names, labels = [], []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.isspace():
                continue
            if inference:
                names.append(line)
            else:
                try:
                    path, class_id = line.rsplit(" ", 1)
                    names.append(path)
                    labels.append(int(class_id))
                except ValueError:
                    raise ValueError(f"Malformed line in dataset file: {line}")

    return names, None if inference else labels


def load_config(config_path):
    """Load the configuration file.
    Args: config_path (str): Path to the configuration file.
    Returns: dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def build_transforms(transform_config):
    """Build a transform pipeline dynamically from the configuration."""

    def parse_transform(transform):
        """Parse a single transformation from JSON."""
        transform_name, params = list(transform.items())[0]
        if transform_name == "OneOf":
            # Special handling for OneOf transformations
            sub_transforms = [parse_transform(sub) for sub in params["transforms"]]
            return A.OneOf(sub_transforms, p=params.get("p", 1.0))
        elif hasattr(A, transform_name):
            # Regular Albumentations transformations
            transform_class = getattr(A, transform_name)
            return transform_class(**params)
        elif transform_name == "ToTensorV2":
            return ToTensorV2()  # Handle ToTensorV2 separately
        elif transform_name in CUSTOM_TRANSFORMS:
            # Custom transformations from utils.transform
            return CUSTOM_TRANSFORMS[transform_name](**params)
        else:
            raise ValueError(f"Unknown transformation: {transform_name}")

    return A.Compose([parse_transform(t) for t in transform_config])


def log_metrics(monitor, epoch, mode="train"):
    """Log the computed metrics to MLflow.

    Args:
        monitor (MetricsMonitor): Instance of MetricsMonitor to track metrics.
        epoch (int): Current epoch.
        mode (str): Mode of operation (train/val/test).
    """
    # Log Metrics
    loss = monitor.compute_average("loss")
    acc = monitor.compute_average("accuracy")
    kappa = monitor.compute_average("kappa")

    mlflow.log_metric(f"{mode}_loss", loss, step=epoch)
    mlflow.log_metric(f"t{mode}_accuracy", acc, step=epoch)
    mlflow.log_metric(f"{mode}_kappa", kappa, step=epoch)


def compute_class_weights_from_dataset(dataset, num_classes):
    """
    Compute class weights based on the class distribution in the dataset.

    Args:
        dataset: Dataset object that has a `_get_class_distribution` method.
        num_classes: Total number of classes.

    Returns:
        torch.Tensor: Computed class weights.
    """
    if not hasattr(dataset, "_get_class_distribution"):
        raise AttributeError("Dataset must have a '_get_class_distribution' method.")

    # Get class distribution from the dataset
    class_distribution = dataset._get_class_distribution()

    # Total samples in the dataset
    total_samples = sum(class_distribution.values())

    # Compute weights inversely proportional to class frequencies
    class_weights = [
        total_samples / (num_classes * class_distribution.get(i, 1))
        for i in range(num_classes)
    ]
    return torch.tensor(class_weights, dtype=torch.float32)


def log_first_batch_images(
    dataloader, save_path="first_batch_image.png", artifact_path="images"
):
    """
    Logs the first batch of images to MLflow as an artifact.

    Args:
        dataloader (DataLoader): DataLoader for the training dataset.
        save_path (str): Path to save the image locally before logging.
        artifact_path (str): Artifact path in MLflow to store the image.
    """
    first_batch = next(iter(dataloader))
    inputs, _ = first_batch  # Ignore labels here
    grid_image = torchvision.utils.make_grid(inputs, nrow=8, normalize=True)
    torchvision.transforms.ToPILImage()(grid_image).save(save_path)
    mlflow.log_artifact(save_path, artifact_path="images")
    print(f"First batch image logged to MLflow: {save_path}")


def export_predictions(
    predictions,
    export_path,
):
    """
    Export the predictions to a npy file.

    Args:
        predictions (list): List of predicted class labels.
        export_path (str): Path to the npy file to save the predictions.
    """
    # Save the predictions to a npy file
    print(f"=============== Exporting predictions to: {export_path} ===============")
    with open(export_path, "wb") as f:
        np.save(f, predictions)


def train(model, dataloader, criterion, optimizer, device, monitor, log_kappa=False):
    """
    Train the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer for the model.
        device: Device to run the computations on (CPU or GPU).
        monitor: Instance of MetricsMonitor to track metrics.
        log_kappa: Whether to compute and log Cohen's Kappa Score.
    """
    model.train()
    monitor.reset()
    total_iterations = len(dataloader)
    all_labels = []
    all_predictions = []

    for iteration, (inputs, labels) in enumerate(dataloader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)

        # Collect predictions and labels for overall metrics
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        monitor.update("loss", loss.item(), count=labels.size(0))
        monitor.update("accuracy", accuracy, count=labels.size(0))
        monitor.print_iteration(iteration, total_iterations, phase="Train")

    # Compute Kappa Score after collecting all predictions
    if log_kappa and len(set(all_labels)) > 1 and len(set(all_predictions)) > 1:
        kappa = cohen_kappa_score(all_labels, all_predictions)
        monitor.update("kappa", kappa)
    monitor.print_final(phase="Train")


def validate(model, dataloader, criterion, device, monitor, log_kappa=False):
    """
    Validate the model on the validation dataset.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the validation data.
        criterion: Loss function.
        device: Device to run the computations on (CPU or GPU).
        monitor: Instance of MetricsMonitor to track metrics.
        log_kappa: Whether to compute and log Cohen's Kappa Score.
    """
    model.eval()
    monitor.reset()
    total_iterations = len(dataloader)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for iteration, (inputs, labels) in enumerate(dataloader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Metrics
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Collect predictions and labels for overall metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            monitor.update("loss", loss.item(), count=labels.size(0))
            monitor.update("accuracy", accuracy, count=labels.size(0))
            monitor.print_iteration(iteration, total_iterations, phase="Validation")

    # Compute Kappa Score after collecting all predictions
    if log_kappa and len(set(all_labels)) > 1 and len(set(all_predictions)) > 1:
        kappa = cohen_kappa_score(all_labels, all_predictions)
        monitor.update("kappa", kappa)
    monitor.print_final(phase="Validation")


def test(model, dataloader, criterion, device, monitor, log_kappa=False):
    """
    Test the model on the test dataset.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the test data.
        criterion: Loss function.
        device: Device to run the computations on (CPU or GPU).
        monitor: Instance of MetricsMonitor to track metrics.
        log_kappa: Whether to compute and log Cohen's Kappa Score.
    """
    model.eval()
    monitor.reset()
    total_iterations = len(dataloader)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for iteration, (inputs, labels) in enumerate(dataloader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)

            # Metrics
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Collect predictions and labels for overall metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            monitor.update("accuracy", accuracy, count=labels.size(0))
            if criterion is not None:
                monitor.update("loss", loss.item(), count=labels.size(0))
            monitor.print_iteration(iteration, total_iterations, phase="Test")

    # Compute Kappa Score after collecting all predictions
    if log_kappa and len(set(all_labels)) > 1 and len(set(all_predictions)) > 1:
        kappa = cohen_kappa_score(all_labels, all_predictions)
        monitor.update("kappa", kappa)
    monitor.print_final(phase="Test")
