"""
run_manager.py — Run lifecycle management for CLUTTR VAE training.

Handles:
  • Generating unique run IDs (timestamp + optional tag)
  • Saving the config that launched each run alongside its checkpoints
  • Maintaining a run log (YAML registry of all runs)
  • Resuming from a previous run by ID (auto-finds config + latest checkpoint)

Directory structure (relative paths, works with both IOManager local & GCS):

    runs/
      20250224_143052_lr5e-05_lat64/
        config.yaml              ← frozen config for this run
        checkpoints/
          checkpoint_5000.pkl    ← includes run_id inside
          checkpoint_10000.pkl
          checkpoint_final.pkl
        plot.png
      20250225_091011_lr1e-04_lat128/
        ...
    run_log.yaml                 ← append-only registry of all runs

Usage:
    from run_manager import RunManager

    rm = RunManager(config, io)
    run_id, config = rm.setup_run()       # new run or resume
    rm.save_checkpoint(state, step)
    rm.save_plot(fig)
    rm.finalize_run(final_metrics)
"""

import os
import yaml
from datetime import datetime


class RunManager:
    """Manages run directories, config snapshots, checkpoint paths, and the run log."""

    def __init__(self, config: dict, io):
        """
        Args:
            config: the raw CONFIG dict loaded from vae_train_config.yml.
            io: an IOManager instance (handles local vs GCS transparently).
        """
        self.io = io
        self.original_config = config
        self.run_id = None
        self.run_dir = None  # relative path like "runs/20250224_143052_lr5e-05_lat64"
        self.config = None   # the effective config for this run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_run(self) -> tuple:
        """
        Either resume an existing run or start a new one.

        Returns:
            (run_id, effective_config, start_step, params_or_None)
        """
        resume_run_id = self.original_config.get("resume_run_id")

        if resume_run_id:
            return self._resume_run(resume_run_id)
        else:
            return self._new_run()

    def save_checkpoint(self, state, step: int):
        """Save a checkpoint into this run's directory."""
        ckpt_rel = f"{self.run_dir}/checkpoints/checkpoint_{step}.pkl"
        payload = {
            "params": state.params,
            "step": step,
            "run_id": self.run_id,
        }
        self.io.save_pickle(payload, ckpt_rel)
        print(f"  [Checkpoint] {self.run_id} → step {step}")

    def save_plot(self, fig):
        """Save the training plot into this run's directory."""
        plot_name = self.config.get("KL_recon_plot_name", "plot.png")
        self.io.save_figure(fig, f"{self.run_dir}/{plot_name}")

    def finalize_run(self, final_metrics: dict = None):
        """Mark the run as complete in the run log."""
        self._update_run_log(status="completed", extra=final_metrics or {})
        print(f"[RunManager] Run {self.run_id} finalized.")

    def get_checkpoint_dir(self) -> str:
        """Return the relative path to this run's checkpoint directory."""
        return f"{self.run_dir}/checkpoints"

    # ------------------------------------------------------------------
    # New run
    # ------------------------------------------------------------------

    def _new_run(self) -> tuple:
        """Create a new run with a unique ID."""
        self.run_id = self._generate_run_id(self.original_config)
        self.run_dir = f"runs/{self.run_id}"
        self.config = dict(self.original_config)  # shallow copy
        self.config["run_id"] = self.run_id

        # Save frozen config
        self.io.save_yaml(self.config, f"{self.run_dir}/config.yaml")

        # Register in run log
        self._update_run_log(status="started")

        print(f"[RunManager] NEW run: {self.run_id}")
        print(f"[RunManager] Config saved to {self.run_dir}/config.yaml")

        return self.run_id, self.config, 0, None

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resume_run(self, resume_run_id: str) -> tuple:
        """
        Resume a previous run by its ID.
        Loads the config that was used for that run (not the current config file)
        and finds the latest checkpoint.
        """
        self.run_id = resume_run_id
        self.run_dir = f"runs/{self.run_id}"

        # 1. Load the ORIGINAL config from that run
        config_path = f"{self.run_dir}/config.yaml"
        if not self.io.exists(config_path):
            raise FileNotFoundError(
                f"Cannot resume run '{resume_run_id}': "
                f"config not found at {config_path}"
            )

        self.config = self.io.load_yaml(config_path)
        print(f"[RunManager] RESUMING run: {self.run_id}")
        print(f"[RunManager] Loaded config from {config_path}")

        # 2. Find the latest checkpoint
        ckpt_dir = f"{self.run_dir}/checkpoints"
        latest_path, latest_step = self._find_latest_checkpoint(ckpt_dir)

        if latest_path is None:
            print(f"[RunManager] WARNING: No checkpoints found in {ckpt_dir}. Starting from step 0.")
            return self.run_id, self.config, 0, None

        # 3. Load checkpoint
        ckpt = self.io.load_pickle(latest_path)
        params = ckpt["params"]
        start_step = ckpt["step"]
        print(f"[RunManager] Loaded checkpoint: step {start_step} from {latest_path}")

        # Update run log
        self._update_run_log(status="resumed", extra={"resumed_from_step": start_step})

        return self.run_id, self.config, start_step, params

    def _find_latest_checkpoint(self, ckpt_dir: str):
        """
        Scan checkpoint directory for the highest-step checkpoint.
        Returns (rel_path, step) or (None, 0) if none found.
        """
        # List files in the checkpoint directory
        try:
            files = self.io.list_dir(ckpt_dir)
        except Exception:
            return None, 0

        best_step = -1
        best_path = None

        for fname in files:
            if not fname.startswith("checkpoint_") or not fname.endswith(".pkl"):
                continue
            # Parse step from "checkpoint_5000.pkl" or "checkpoint_final.pkl"
            stem = fname.replace("checkpoint_", "").replace(".pkl", "")
            if stem == "final":
                step = float("inf")
            else:
                try:
                    step = int(stem)
                except ValueError:
                    continue

            if step > best_step:
                best_step = step
                best_path = f"{ckpt_dir}/{fname}"

        if best_path is None:
            return None, 0

        # For "final", we need to load it to get the actual step number
        if best_step == float("inf"):
            ckpt = self.io.load_pickle(best_path)
            best_step = ckpt.get("step", 0)

        return best_path, best_step

    # ------------------------------------------------------------------
    # Run log (append-only YAML registry)
    # ------------------------------------------------------------------

    def _update_run_log(self, status: str, extra: dict = None):
        """Append an entry to the global run_log.yaml."""
        log_path = "run_log.yaml"

        # Load existing log
        try:
            existing = self.io.load_yaml(log_path)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "platform": self.config.get("platform", "unknown"),
            "learning_rate": self.config.get("learning_rate"),
            "recon_weight": self.config.get("recon_weight"),
            "latent_dim": self.config.get("latent_dim"),
            "batch_size": self.config.get("batch_size"),
            "num_steps": self.config.get("num_steps"),
            "train_data": self.config.get("train_data_path"),
        }
        if extra:
            entry.update(extra)

        existing.append(entry)
        self.io.save_yaml(existing, log_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_id(config: dict) -> str:
        """
        Generate a human-readable, unique run ID.
        Format: YYYYMMDD_HHMMSS_lr{lr}_lat{latent_dim}[_tag]
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr = config.get("learning_rate", 0)
        lat = config.get("latent_dim", 0)
        tag = config.get("run_tag", "")  # optional user-provided tag

        run_id = f"{ts}_lr{lr}_lat{lat}"
        if tag:
            # Sanitise: replace spaces/slashes
            tag = tag.replace(" ", "_").replace("/", "_")
            run_id = f"{run_id}_{tag}"
        return run_id