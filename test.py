import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import dagshub
import mlflow.pytorch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import SkinDataset
from utils.utils import (
    load_data_file,
    load_config,
    build_transforms,
    export_predictions,
)
from sklearn.metrics import cohen_kappa_score


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: Command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Testing Script"
    )
    parser.add_argument(
        "--tta", default=False, help="Use Test Time Augmentation", action="store_true"
    )
    parser.add_argument(
        "--num_tta",
        type=int,
        default=10,
        help="Number of TTA iterations",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/madz/datasets/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--log_kappa",
        default=False,
        help="Log Cohen's Kappa Score",
        action="store_true",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="aca333832cbf492981651b12b6f27c84",
        help="MLflow run ID",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary",
        help="Dataset type (binary/multiclass)",
    )
    return parser.parse_args()


def tta_step_batch(model, images, transform, num_tta=5, device="cpu"):
    """Perform Test Time Augmentation (TTA) for a batch of images."""
    model.eval()
    tta_outputs = []

    with torch.no_grad():
        for _ in range(num_tta):
            augmented_images = []
            import pdb

            pdb.set_trace()
            for img in images:
                augmented_images.append(transform(image=img.numpy()))
            augmented_images = torch.stack(augmented_images).to(device)

            # Perform inference
            output = model(augmented_images)
            tta_outputs.append(softmax(output, dim=1))

    # Average predictions across TTA iterations
    avg_output = torch.stack(tta_outputs).mean(dim=0)
    return avg_output


def test(
    model,
    config,
    data_file,
    data_root,
    batch_size,
    num_workers,
    device="cuda",
    mode="val",
    tta=False,
    num_tta=10,
    log_kappa=False,
    inference=False,
):
    """Test a trained model on a dataset."""
    # Data Transformations
    tta_transform = build_transforms(config["transformations"]["tta"])
    test_transform = build_transforms(config["transformations"]["test"])

    # Load test data
    test_names, test_labels = load_data_file(data_file, inference=inference)

    # Create test dataset and dataloader
    test_dataset = SkinDataset(
        data_root,
        test_names,
        test_labels,
        tta_transform if tta else test_transform,
        inference=inference,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("================== Test dataset Info: ==================\n", test_dataset)

    # Testing Phase
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        all_probs = []
        all_labels = []
        all_image_names = []

        for i in range(num_tta if tta else 1):
            current_probs = []
            current_labels = []

            for batch in tqdm(
                test_loader,
                desc=f"TTA Iteration {i + 1}/{num_tta}" if tta else "Testing",
            ):
                if inference:
                    batch_images, image_names = batch
                    batch_images = batch_images.to(device)
                    all_image_names.extend(image_names)
                else:
                    batch_images, batch_labels = batch
                    batch_images = batch_images.to(device)

                # Model inference
                outputs = model(batch_images)
                batch_probs = softmax(outputs, dim=1)

                current_probs.append(batch_probs.detach().cpu())
                # Collect labels only in the first iteration
                if not inference and i == 0:
                    current_labels.append(batch_labels)

            # Store current iteration probabilities
            all_probs.append(torch.cat(current_probs))

            # Store labels only once
            if not inference and i == 0:
                all_labels = torch.cat(current_labels)

    # Ensemble predictions across TTA iterations if TTA is enabled
    if tta:
        all_probs = torch.stack(all_probs).mean(dim=0).numpy()
    else:
        all_probs = all_probs[0].numpy()

    if inference:
        return all_probs, all_image_names

    # Calculate discrete predictions
    all_preds = all_probs.argmax(axis=1)

    # Calculate accuracy
    test_acc = (all_preds == all_labels).float().mean().item()

    # Calculate and log Kappa Score if enabled
    kappa_score = None
    if log_kappa:
        kappa_score = cohen_kappa_score(all_labels.numpy(), all_preds)

    return test_acc, kappa_score, all_probs, all_labels.numpy()


def load_model_and_config(run_id, artifact_path="config.json", model_path="skin_lesion_model", device="cuda"):
    """Load the trained model and configuration."""
    model_uri = f"runs:/{run_id}/{model_path}"
    config_path = "config.json"

    # Download the configuration file from the run
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    if Path(local_path).is_file():
        print(f"Config file downloaded to: {local_path}")
        config_path = local_path

    # Load model
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model = model.to(device)

    # Load config
    config = load_config(config_path)
    return model, config


def main(args):
    """Main function to test the model."""
    # Constants
    RUN_ID = args.run_id
    ARTIFACT_PATH = "config/config.json"
    # Dataset type
    DATASET = "Binary" if args.dataset.lower() == "binary" else "Multiclass"
    dagshub.init(
        repo_owner="madzie", repo_name="CAD_UDG_DL", mlflow=True
    )

    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {DEVICE}")
    # Load model and configuration
    model, config = load_model_and_config(RUN_ID, ARTIFACT_PATH, DEVICE)

    with mlflow.start_run(run_id=RUN_ID):
        run_name = mlflow.active_run().info.run_name
        print(f"MLflow run: {run_name}")
        # Test the model
        test_acc, kappa_score, prediction_probs, labels = test(
            model=model,
            config=config,
            data_file=f"datasets/{DATASET}/val.txt",
            data_root=os.path.join(args.data_root, DATASET),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
            mode="val",
            tta=args.tta,
            num_tta=args.num_tta,
            log_kappa=args.log_kappa,
            inference=False,
        )
        print(f"Test Accuracy: {test_acc:.4f}")
        if args.log_kappa:
            print(f"Cohen's Kappa Score: {kappa_score:.4f}")

        ## Save predictions
        export_name = (
            "prediction_probs.npy" if not args.tta else "tta_prediction_probs.npy"
        )
        ## Log metrics
        mlflow.log_metric(
            "test_accuracy" if not args.tta else "test_accuracy_tta", test_acc
        )
        if args.log_kappa:
            mlflow.log_metric(
                "test_kappa_score" if args.tta else "test_kappa_score_tta", kappa_score
            )

        export_path = f"results/{DATASET}/{export_name}"
        export_predictions(prediction_probs, export_path)
        mlflow.log_artifact(export_path, artifact_path="results")

        ############ Generate predictions for the test set ############
        test_predictions = test(
            model=model,
            config=config,
            data_file=f"datasets/{DATASET}/test.txt",
            data_root=os.path.join(args.data_root, DATASET),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=DEVICE,
            mode="test",
            tta=args.tta,
            num_tta=args.num_tta,
            log_kappa=args.log_kappa,
            inference=True,
        )
        export_path = f"results/{DATASET}/test_predictions.npy"
        export_predictions(test_predictions, export_path)
        mlflow.log_artifact(export_path, artifact_path="results")


if __name__ == "__main__":
    args = arg_parser()
    os.environ["MLFLOW_TRACKING_URI"] = (
        "https://dagshub.com/madzie/CAD_UDG_DL.mlflow"
    )
    main(args)
