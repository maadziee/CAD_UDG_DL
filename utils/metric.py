"""
This module contains a class for tracking metrics and implementing early stopping.
It is used to monitor the performance of the model during training and validation.
"""

import torch


class MetricsMonitor:
    """Monitor for tracking metrics and implementing early stopping."""

    def __init__(
        self,
        metrics=None,
        patience=5,
        delta=0.0001,
        mode="min",
        export_path="best_model.pth",
    ):
        """
        Combines metric tracking and early stopping with real-time updates.

        Args:
            metrics (list): List of metric names to track (e.g., ['loss', 'accuracy']).
            patience (int): Patience for early stopping.
            delta (float): Minimum change to qualify as an improvement for early stopping.
            mode (str): 'min' for loss or 'max' for accuracy.
        """
        self.metrics = metrics if metrics else ["loss", "accuracy"]
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.export_path = export_path
        self.reset()

    def reset(self):
        """
        Resets metrics and early stopping variables for a new epoch or phase.
        """
        self.metric_totals = {metric: 0.0 for metric in self.metrics}
        self.metric_counts = {metric: 0 for metric in self.metrics}

    def update(self, metric_name, value, count=1):
        """
        Updates a specific metric with a new value.

        Args:
            metric_name (str): Name of the metric to update.
            value (float): Value to add to the metric.
            count (int): Number of samples contributing to the metric (default is 1).
        """
        if metric_name in self.metric_totals:
            self.metric_totals[metric_name] += value * count
            self.metric_counts[metric_name] += count
        else:
            raise ValueError(f"Metric '{metric_name}' is not being tracked.")

    def compute_average(self, metric_name):
        """
        Computes the average value for a specific metric.

        Args:
            metric_name (str): Name of the metric to compute.

        Returns:
            float: The average value of the metric.
        """
        if self.metric_counts[metric_name] > 0:
            return self.metric_totals[metric_name] / self.metric_counts[metric_name]
        return 0.0

    def print_iteration(self, iteration, total_iterations, phase="Train"):
        """
        Prints real-time metrics for each iteration.

        Args:
            iteration (int): Current iteration index.
            total_iterations (int): Total number of iterations in the epoch.
            phase (str): Name of the current phase (e.g., 'Train', 'Validation').
        """
        metrics_str = ", ".join(
            f"{metric}: {self.compute_average(metric):.4f}" for metric in self.metrics
        )
        print(
            f"\r[{phase}] Iteration {iteration}/{total_iterations} - {metrics_str}",
            end="",
            flush=True,
        )

    def print_final(self, phase="Train"):
        """
        Prints the final average values of all tracked metrics.

        Args:
            phase (str): Name of the current phase (e.g., 'Train', 'Validation').
        """
        metrics_str = ", ".join(
            f"{metric}: {self.compute_average(metric):.4f}" for metric in self.metrics
        )
        print(f"\n{phase} Metrics - {metrics_str}")

    def early_stopping_check(self, metric, model):
        """
        Implements early stopping based on a monitored metric.

        Args:
            metric (float): The validation metric to monitor.
            model: The model to save if performance improves.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = -metric if self.mode == "max" else metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Save the best model."""
        torch.save(model.state_dict(), self.export_path)
        print("Model improved and saved!\n")
