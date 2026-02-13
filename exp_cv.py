"""
This script trains and validates a deep learning model for skin lesion classification.
It uses MLflow for tracking experiments and PyTorch for model training.

Run the script using the following command:
    python exp.py --batch_size 64 --epochs 100 --lr 0.001 --workers 4 --warmup_epochs 10 --warmup_lr 0.00005
"""

import os
import argparse
import dagshub
import mlflow
import mlflow.pytorch
import mlflow
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from models.efficientnet import EfficientNet
from models.swin import SwinTransformer
from models.resnet import ResNetLoRA
from models.loss import FocalLoss
from test import test
from utils.dataset import SkinDataset
from utils.metric import MetricsMonitor
from utils.utils import (
    train,
    validate,
    load_data_file,
    load_config,
    build_transforms,
    freeze_layers,
    compute_class_weights_from_dataset,
    export_predictions,
    log_first_batch_images,
    log_metrics,
)


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: command line
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Experiment"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument(
        "--freeze_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer names to freeze (e.g., 'layer1,layer2,fc')",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0, help="Number of warm-up epochs"
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=0.00005, help="Learning rate during warm-up"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of workers"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/huy/datasets/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--num_tta",
        type=int,
        default=10,
        help="Number of TTA iterations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary",
        help="Dataset type (binary/multiclass)",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    os.makedirs("weights", exist_ok=True)
    # Constants
    CONFIG_PATH = "config.json"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WORKERS = args.workers
    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Warm-up settings
    WARMUP_EPOCHS = args.warmup_epochs  # Number of warm-up epochs
    WARMUP_LR = args.warmup_lr  # Learning rate during warm-up
    # Dataset type
    DATASET = "Binary" if args.dataset.lower() == "binary" else "Multiclass"
    # Classes
    CLASSES = ["bcc", "mel", "scc"] if DATASET == "Multiclass" else ["nevous", "others"]

    dagshub.init(
        repo_owner="madzie", repo_name="CAD_UDG_DL", mlflow=True
    )
    # Start MLflow tracking
    mlflow.start_run(run_name=f"Skin Lesion Classification - {DATASET}")

    # Data Transformations
    # Load the configuration file
    config = load_config(CONFIG_PATH)
    # Build train and test transforms from the configuration
    train_transform = build_transforms(config["transformations"]["train"])
    test_transform = build_transforms(config["transformations"]["test"])
    print("Train Transformations:", train_transform)
    print("Test Transformations:", test_transform)

    # Load data paths and labels
    train_names, train_labels = load_data_file(f"datasets/{DATASET}/train.txt")

    # Cross-Validation Training
    train_names = np.array(train_names)
    train_labels = np.array(train_labels)
    kf = StratifiedKFold(n_splits=args.nfolds, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(
        kf.split(train_names, train_labels)
    ):
        print(f"Fold {fold + 1}/{args.nfolds}")
        train_dataset = SkinDataset(
            os.path.join(args.data_root, DATASET),
            train_names[train_index],
            train_labels[train_index],
            train_transform,
        )
        val_dataset = SkinDataset(
            os.path.join(args.data_root, DATASET),
            train_names[val_index],
            train_labels[val_index],
            test_transform,
        )

        print(
            "================== Train dataset Info: ==================\n", train_dataset
        )
        print(
            "================== Validation dataset Info: ==================\n",
            val_dataset,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
        )

        # Model
        model = EfficientNet(num_classes=len(CLASSES), name="b6", pretrained=True)
        # model = SwinTransformer(num_classes=len(CLASSES), name="v2-b", pretrained=True)
        model = model.to(DEVICE)

        # Loss
        class_weights = compute_class_weights_from_dataset(train_dataset, len(CLASSES))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

        # Monitors
        train_monitor = MetricsMonitor(metrics=["loss", "accuracy", "kappa"])
        val_monitor = MetricsMonitor(
            metrics=["loss", "accuracy", "kappa"],
            patience=args.patience,
            mode="max",
            export_path=f"weights/{DATASET}_best_fold{fold}.pth",
        )

        # Optimizer
        main_optimizer = getattr(torch.optim, args.optimizer)(
            [param for param in model.parameters() if param.requires_grad], lr=LR
        )

        # Scheduler: Cosine Annealing
        scheduler = CosineAnnealingLR(main_optimizer, T_max=EPOCHS, eta_min=0.00001)

        # Training Loop
        # Log Artifacts
        mlflow.log_artifact(CONFIG_PATH, artifact_path="config")
        # Log Parameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("warmup_lr", WARMUP_LR)
        mlflow.log_param("warmup_epochs", WARMUP_EPOCHS)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("device", DEVICE)
        mlflow.log_param("classes", CLASSES)
        mlflow.log_param("model", model.__class__.__name__)
        mlflow.log_param("optimizer", main_optimizer.__class__.__name__)
        mlflow.log_param("patience", args.patience)
        mlflow.log_param("freeze_layers", args.freeze_layers)
        mlflow.log_param("num_workers", WORKERS)
        mlflow.log_param("scheduler", scheduler.__class__.__name__)

        # Training Phase
        print("====================== Training phase ======================")
        # Freeze layers

        freeze_layers(model, args.freeze_layers)

        for epoch in range(EPOCHS):
            if epoch == 0:
                log_first_batch_images(
                    train_loader, save_path="images/train_images.png"
                )
                log_first_batch_images(val_loader, save_path="images/val_images.png")
            print(f"Fold {fold + 1} | Epoch {epoch + 1}/{EPOCHS}")

            # Training phase
            train(
                model,
                train_loader,
                criterion,
                main_optimizer,
                DEVICE,
                train_monitor,
                log_kappa=True,
            )

            # Validation phase
            validate(
                model,
                val_loader,
                criterion,
                DEVICE,
                val_monitor,
                log_kappa=True,
            )

            # Adjust learning rate with cosine scheduler
            scheduler.step()

            # Log Metrics
            log_metrics(train_monitor, epoch, f"train_fold{fold}")
            log_metrics(val_monitor, epoch, f"val_fold{fold}")

            # Early Stopping
            if val_monitor.early_stopping_check(
                val_monitor.compute_average("kappa"), model
            ):
                print("Early stopping triggered.")
                break

        # Log the Best Model
        print(
            f"Logging the best model with accuracy: {abs(val_monitor.best_score):.4f}"
        )
        best_model_state = torch.load(val_monitor.export_path, weights_only=True)
        model.load_state_dict(best_model_state)  # Load the best model state_dict
        mlflow.pytorch.log_model(model, artifact_path=f"skin_lesion_model_fold_{fold}")

    #################### Test the model with k-folds ####################
    fold_test_predictions = []
    fold_test_predictions_tta = []
    # Load k-folds models
    for fold in range(args.nfolds):
        print(f"Fold {fold + 1}/{args.nfolds}")
        model = EfficientNet(num_classes=len(CLASSES), name="b6", pretrained=True)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(f"weights/{DATASET}_best_fold{fold}.pth"))

        #################### Test the model with k-folds on validation set ####################
        test_acc, kappa_score, prediction_probs, test_labels = test(
            model=model,
            config=config,
            data_file=f"datasets/{DATASET}/val.txt",
            data_root=os.path.join(args.data_root, DATASET),
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            device=DEVICE,
            mode="val",
            tta=False,
            log_kappa=True,
            inference=False,
        )
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Kappa Score: {kappa_score:.4f}")

        # Test the model
        test_acc_tta, kappa_score_tta, tta_prediction_probs, test_labels_tta = test(
            model=model,
            config=config,
            data_file=f"datasets/{DATASET}/val.txt",
            data_root=os.path.join(args.data_root, DATASET),
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            mode="val",
            device=DEVICE,
            tta=True,
            num_tta=args.num_tta,
            log_kappa=True,
        )
        print(f"Test with TTA Accuracy: {test_acc_tta:.4f}")
        print(f"Kappa Score with TTA: {kappa_score_tta:.4f}")

        fold_test_predictions.append(prediction_probs)
        fold_test_predictions_tta.append(tta_prediction_probs)

        ############ Generate predictions for the test set ############
        test_predictions = test(
            model=model,
            config=config,
            data_file=f"datasets/{DATASET}/test.txt",
            data_root=os.path.join(args.data_root, DATASET),
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            device=DEVICE,
            mode="test",
            tta=False,
            num_tta=args.num_tta,
            log_kappa=True,
            inference=True,
        )
        export_path = f"results/{DATASET}/test_predictions.npy"
        export_predictions(test_predictions, export_path)
        mlflow.log_artifact(export_path, artifact_path="results")
        # TTA
        test_predictions = test(
            model=model,
            config=config,
            data_file=f"datasets/{DATASET}/test.txt",
            data_root=os.path.join(args.data_root, DATASET),
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            device=DEVICE,
            mode="test",
            tta=True,
            num_tta=args.num_tta,
            log_kappa=True,
            inference=True,
        )
        export_path = f"results/{DATASET}/test_tta_predictions.npy"
        export_predictions(test_predictions, export_path)
        mlflow.log_artifact(export_path, artifact_path="results")

    # Average the predictions
    prediction_probs = np.mean(fold_test_predictions, axis=0)
    tta_prediction_probs = np.mean(fold_test_predictions_tta, axis=0)
    # Class predictions
    predictions = np.argmax(prediction_probs, axis=1)
    predictions_tta = np.argmax(tta_prediction_probs, axis=1)
    # Test accuracy
    test_acc = np.mean(predictions == test_labels)
    test_acc_tta = np.mean(predictions_tta == test_labels_tta)
    # Cohen's Kappa
    kappa_score = cohen_kappa_score(predictions, test_labels)
    kappa_score_tta = cohen_kappa_score(predictions_tta, test_labels_tta)
    # Log metrics
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_accuracy_tta", test_acc_tta)
    mlflow.log_metric("kappa_score", kappa_score)
    mlflow.log_metric("kappa_score_tta", kappa_score_tta)
    # Export predictions
    export_predictions(prediction_probs, f"results/{DATASET}/prediction_probs.npy")
    export_predictions(
        tta_prediction_probs, f"results/{DATASET}/tta_prediction_probs.npy"
    )
    ## Log predictions to artifacts
    mlflow.log_artifact(
        f"results/{DATASET}/prediction_probs.npy", artifact_path="results"
    )
    mlflow.log_artifact(
        f"results/{DATASET}/tta_prediction_probs.npy", artifact_path="results"
    )
