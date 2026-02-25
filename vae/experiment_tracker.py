"""
experiment_tracker.py — Lightweight wrapper around Vertex AI Experiments.

Falls back to a no-op tracker when Vertex AI is unavailable or disabled,
so the training script works identically on local GPUs.

Usage:
    from experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(config)
    tracker.log_params({"lr": 0.0001, "batch_size": 32})
    tracker.log_metrics({"recon_loss": 1.23, "kl_loss": 0.5}, step=1000)
    tracker.end_run()
"""

import time
from datetime import datetime

try:
    from google.cloud import aiplatform
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False


class _NoOpTracker:
    """Silent tracker used when Vertex AI is disabled or unavailable."""
    def log_params(self, params: dict): pass
    def log_metrics(self, metrics: dict, step: int = 0): pass
    def log_time_series_metrics(self, metrics: dict, step: int = 0): pass
    def end_run(self): pass


class VertexTracker:
    """Thin wrapper around Vertex AI Experiments."""

    def __init__(self, project: str, region: str, experiment_name: str):
        aiplatform.init(
            project=project,
            location=region,
        )
        # Create or get the experiment
        self.experiment = aiplatform.Experiment.get_or_create(
            experiment_name=experiment_name,
        )
        # Start a new run with a timestamp ID
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.run = aiplatform.ExperimentRun.create(
            run_id,
            experiment=experiment_name,
        )
        print(f"[VertexTracker] Experiment: {experiment_name} | Run: {run_id}")

    def log_params(self, params: dict):
        """Log hyperparameters (called once at the start)."""
        self.run.log_params(params)

    def log_metrics(self, metrics: dict, step: int = 0):
        """Log summary metrics (e.g. final val loss)."""
        self.run.log_metrics(metrics)

    def log_time_series_metrics(self, metrics: dict, step: int = 0):
        """Log time-series metrics (loss curves visible in Vertex UI)."""
        self.run.log_time_series_metrics(metrics, step=step)

    def end_run(self):
        """Mark the run as complete."""
        self.run.end_run()
        print("[VertexTracker] Run ended.")


class ExperimentTracker:
    """
    Factory that returns a VertexTracker or a no-op tracker depending on config.
    
    Required config keys (when enabled):
        - enable_vertex_tracking: bool
        - gcp_project: str
        - vertex_experiment_region: str
        - vertex_experiment_name: str
    """

    def __new__(cls, config: dict):
        enabled = config.get("enable_vertex_tracking", False)
        is_gcp = config.get("platform", "local") == "gcp"

        if enabled and is_gcp:
            if not HAS_VERTEX:
                print(
                    "[ExperimentTracker] WARNING: google-cloud-aiplatform not installed. "
                    "Install with: pip install google-cloud-aiplatform. "
                    "Falling back to no-op tracker."
                )
                return _NoOpTracker()
            try:
                return VertexTracker(
                    project=config["gcp_project"],
                    region=config["vertex_experiment_region"],
                    experiment_name=config["vertex_experiment_name"],
                )
            except Exception as e:
                print(f"[ExperimentTracker] WARNING: Could not init Vertex AI: {e}")
                print("[ExperimentTracker] Falling back to no-op tracker.")
                return _NoOpTracker()
        else:
            if not enabled:
                print("[ExperimentTracker] Tracking disabled in config.")
            else:
                print("[ExperimentTracker] Not on GCP — using no-op tracker.")
            return _NoOpTracker()