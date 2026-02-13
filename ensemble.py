import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
)

from utils.ensemble import Ensemble
from utils.utils import load_data_file

# DATASET = "Multiclass"
DATASET = "Binary"
TTA = False

if __name__ == "__main__":
    # Load the model
    ## Binary run_ids
    if DATASET == "Binary":
        run_ids = [
            "16c2e4453bde440db1b5d22058f95fab",
            "aca333832cbf492981651b12b6f27c84",
            "696928eecfe4429f8de4d13e07cad64c",
            "aca333832cbf492981651b12b6f27c84",
            "8c153b477bfe4cb9b4b08c741b6ccbea",
            "3002b0c7c0584728b7198d26d2b73e6a",
            "fe00b64a5b3c4ceb9d46fe5c73db8e1c", # Cross-Validation
            "923a7b14fcbb4bbba68036011ce957dc", # Cross-Validation 2
            "e343a3f6a8f8401390d2bdd7c8ae87e7", # Cross-Validation 736
            # "eee52d223bb9492ab7d4d992e758fd10" # Fine-tuning from Multiclass
        ]
    else:
        ## Multiclass run_ids
        run_ids = [
            # "776b2e8f8853416a9c959b312a5a4611",
            # "f62f6e145791420caf0346263e4b14fa",
            # "c539872187ee434dac603e2a148fcb33", # Cross-Validation 640
            'bfd44c64e1474d04bffc650b29329594', # Cross-Validation 736
            # "ae0c32cb6a98498d96bf00cd317586dc", # Cross-Validation 640
            # "0c91e8dd73824d09a8975a3ca2660402", # Fine-tuning from binary
        ]
    classes = np.loadtxt(f"./datasets/{DATASET}/classes.txt", dtype=str)
    names, labels = load_data_file(f"./datasets/{DATASET}/val.txt")

    ensemble = Ensemble(
        run_ids=run_ids,
        tta=False,
        weights=None,
    )
    for mode in ["majority", "average", "dempster_shafer", "geometric_mean"]:
        print(f"============================= Mode: {mode} =============================")
        predicts = ensemble.predict(mode=mode)
        # Calculate the accuracy
        acc = accuracy_score(labels, predicts)
        print(f"Ensemble Accuracy: {acc:.4f}")

        # Classification report
        print(classification_report(labels, predicts, target_names=classes))

        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, predicts)
        print(f"Cohen's Kappa: {kappa:.4f}")
