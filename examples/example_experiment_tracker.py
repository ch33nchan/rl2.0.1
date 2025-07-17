#!/usr/bin/env python3
"""
Example: Experiment Tracker
"""
from RL2.experiment_tracking import ExperimentTracker

if __name__ == "__main__":
    tracker = ExperimentTracker(
        experiment_name="Example_Experiment",
        enable_mlflow=False,
        enable_wandb=False
    )
    tracker.log_metrics({"reward": 1.0, "kl": 0.01}, step=1)
    tracker.log_system_metrics()
    tracker.finish()
    print("Experiment tracking example complete.")
