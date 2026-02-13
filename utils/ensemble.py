import numpy as np

import dagshub
import mlflow.pytorch


class Ensemble:
    def __init__(self, run_ids, tta=False, weights=None):
        """
        Initialize the ensemble.

        Args:
            run_ids (list): List of MLflow run IDs for the models.
            tta (bool): Whether to use Test Time Augmentation (TTA).
            weights (list or None): Weights for models in weighted voting modes.
        """
        self.run_ids = run_ids
        self.tta = tta
        self.weights = weights
        self.artifact_path = (
            "results/prediction_probs.npy"
            if not tta
            else "results/tta_prediction_probs.npy"
        )
        self.predictions = []
        dagshub.init(
            repo_owner="huytrnq",
            repo_name="Deep-Skin-Lesion-Classification",
            mlflow=True,
        )
        
        # Load predictions from MLflow artifacts
        self.load_predictions_from_mlflow()

    def load_predictions_from_mlflow(self):
        """
        Load predictions from MLflow artifacts.
        """
        for run_id in self.run_ids:
            with mlflow.start_run(run_id=run_id):
                predictions_npy_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=self.artifact_path
                )
                self.predictions.append(np.load(predictions_npy_path))

    def majority_voting(self):
        """
        Perform majority voting on the predictions.

        Returns:
            list: Final predictions.
        """
        final_predictions = []
        for i in range(len(self.predictions[0])):
            votes = [np.argmax(pred[i]) for pred in self.predictions]
            final_predictions.append(max(set(votes), key=votes.count))
        return final_predictions

    def average_voting(self):
        """
        Perform average voting on the predictions.

        Returns:
            list: Final predictions after averaging probabilities.
        """
        final_predictions = []
        for i in range(len(self.predictions[0])):
            votes = [pred[i] for pred in self.predictions]
            averaged_probs = np.mean(votes, axis=0)
            final_predictions.append(np.argmax(averaged_probs))
        return final_predictions

    def weighted_average_voting(self):
        """
        Perform weighted average voting on the predictions.

        Returns:
            list: Final predictions after weighting probabilities.
        """
        if not self.weights or len(self.weights) != len(self.predictions):
            raise ValueError("Weights must be provided and match the number of models.")

        final_predictions = []
        for i in range(len(self.predictions[0])):
            votes = [pred[i] * w for pred, w in zip(self.predictions, self.weights)]
            weighted_avg_probs = np.sum(votes, axis=0)
            final_predictions.append(np.argmax(weighted_avg_probs))
        return final_predictions

    def geometric_mean_voting(self):
        """
        Perform geometric mean voting on the predictions.

        Returns:
            list: Final predictions after geometric mean calculation.
        """
        final_predictions = []
        for i in range(len(self.predictions[0])):
            votes = [pred[i] for pred in self.predictions]
            geometric_mean_probs = np.prod(votes, axis=0) ** (1 / len(votes))
            final_predictions.append(np.argmax(geometric_mean_probs))
        return final_predictions

    def weighted_majority_voting(self):
        """
        Perform weighted majority voting on the predictions.

        Returns:
            list: Final predictions after weighting votes.
        """
        if not self.weights or len(self.weights) != len(self.predictions):
            raise ValueError("Weights must be provided and match the number of models.")

        final_predictions = []
        for i in range(len(self.predictions[0])):
            votes = [np.argmax(pred[i]) for pred in self.predictions]
            weighted_votes = {}
            for vote, weight in zip(votes, self.weights):
                weighted_votes[vote] = weighted_votes.get(vote, 0) + weight
            final_predictions.append(max(weighted_votes, key=weighted_votes.get))
        return final_predictions

    def resolve_prediction(self, prediction):
        """
        Resolve prediction by handling conflicts and mapping class names to numeric labels.

        Args:
            prediction (str or int): The prediction to resolve.

        Returns:
            int: Resolved numeric label.
        """
        if prediction == "conflict":
            # Handle conflicts (default to a random class or specific logic)
            return np.random.choice(
                len(self.predictions[0][0])
            )  # Assuming 0-indexed classes
        elif isinstance(prediction, str) and prediction.lower().startswith("class_"):
            # Map class names like 'class_0', 'Class_0' to numeric labels
            return int(prediction.split("_")[1])
        elif isinstance(prediction, int):
            # Return the numeric label directly
            return prediction
        else:
            raise ValueError(f"Unexpected prediction format: {prediction}")

    def dempster_shafer_combination(self):
        """
        Perform ensemble using Dempster-Shafer Combination.

        Returns:
            list: Final predictions after combining beliefs.
        """

        def get_evidence_from_probabilities(probs):
            evidence = {f"Class_{i}": prob for i, prob in enumerate(probs)}
            evidence["uncertainty"] = 1 - sum(probs)
            return evidence

        def combine_evidence(evidence_list):
            combined_evidence = evidence_list[0]
            for evidence in evidence_list[1:]:
                new_combined_evidence = {}
                for hypo_i, belief_i in combined_evidence.items():
                    for hypo_j, belief_j in evidence.items():
                        if hypo_i == hypo_j:
                            new_combined_evidence[hypo_i] = (
                                new_combined_evidence.get(hypo_i, 0)
                                + belief_i * belief_j
                            )
                        elif hypo_i != "uncertainty" and hypo_j != "uncertainty":
                            new_combined_evidence["conflict"] = (
                                new_combined_evidence.get("conflict", 0)
                                + belief_i * belief_j
                            )
                # Normalize
                conflict = new_combined_evidence.get("conflict", 0)
                if conflict < 1:
                    for hypo in combined_evidence:
                        if hypo != "conflict":
                            new_combined_evidence[hypo] = new_combined_evidence.get(
                                hypo, 0
                            ) / (1 - conflict)
                    new_combined_evidence.pop("conflict", None)
                combined_evidence = new_combined_evidence
            return combined_evidence

        final_predictions = []
        for i in range(len(self.predictions[0])):
            evidence_list = [
                get_evidence_from_probabilities(pred[i]) for pred in self.predictions
            ]
            combined_evidence = combine_evidence(evidence_list)
            final_predictions.append(max(combined_evidence, key=combined_evidence.get))
        return final_predictions

    def predict(self, mode):
        """
        Perform the ensemble prediction using the selected strategy.
        
        Args:
            mode (str): Ensemble mode ("majority", "average", "weighted_average",
                "dempster_shafer", "geometric_mean", "weighted_majority").

        Returns:
            list: Final predictions.
        """

        if mode == "average":
            final_predictions = self.average_voting()
        elif mode == "majority":
            final_predictions = self.majority_voting()
        elif mode == "weighted_average":
            final_predictions = self.weighted_average_voting()
        elif mode == "geometric_mean":
            final_predictions = self.geometric_mean_voting()
        elif mode == "weighted_majority":
            final_predictions = self.weighted_majority_voting()
        elif mode == "dempster_shafer":
            final_predictions = self.dempster_shafer_combination()

            # Handle conflicts and class names
            final_predictions = [
                self.resolve_prediction(pred) for pred in final_predictions
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return final_predictions
