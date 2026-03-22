"""
gcp_io.py — Unified I/O layer that works on both local filesystems and GCS.

Usage:
    from gcp_io import IOManager

    io = IOManager(config)       # auto-selects local vs GCS based on config["platform"]
    data = io.load_npy("datasets/train.npy")
    io.save_pickle(obj, "checkpoints/step_1000.pkl")
    io.save_figure(fig, "plots/loss.png")
"""

import os
import io as _io
import pickle
import tempfile
import numpy as np
import matplotlib.figure

# GCS imports are optional — only needed on GCP
try:
    from google.cloud import storage as gcs_storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


class IOManager:
    """Transparent I/O that routes to local disk or GCS depending on config."""

    def __init__(self, config: dict):
        self.platform = config.get("platform", "local")
        self.config = config

        if self.platform == "gcp":
            if not HAS_GCS:
                raise ImportError(
                    "google-cloud-storage is required for GCP mode. "
                    "Install with: pip install google-cloud-storage"
                )
            self.bucket_name = config["gcp_bucket"]
            self.prefix = config.get("gcp_bucket_prefix", "vae")
            self._client = gcs_storage.Client(project=config.get("gcp_project"))
            self._bucket = self._client.bucket(self.bucket_name)
            print(f"[IOManager] GCS mode  →  gs://{self.bucket_name}/{self.prefix}/")
        else:
            self.base_dir = os.path.join(config["working_path"], config["vae_folder"])
            print(f"[IOManager] Local mode → {self.base_dir}")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _local_path(self, rel_path: str) -> str:
        return os.path.join(self.base_dir, rel_path)

    def _gcs_key(self, rel_path: str) -> str:
        return f"{self.prefix}/{rel_path}"

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------
    def exists(self, rel_path: str) -> bool:
        if self.platform == "gcp":
            blob = self._bucket.blob(self._gcs_key(rel_path))
            return blob.exists()
        else:
            return os.path.exists(self._local_path(rel_path))

    # ------------------------------------------------------------------
    # NumPy arrays (.npy)
    # ------------------------------------------------------------------
    def load_npy(self, rel_path: str) -> np.ndarray:
        if self.platform == "gcp":
            blob = self._bucket.blob(self._gcs_key(rel_path))
            data = blob.download_as_bytes()
            return np.load(_io.BytesIO(data))
        else:
            return np.load(self._local_path(rel_path))

    def save_npy(self, arr: np.ndarray, rel_path: str):
        if self.platform == "gcp":
            buf = _io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            blob = self._bucket.blob(self._gcs_key(rel_path))
            blob.upload_from_file(buf, content_type="application/octet-stream")
        else:
            path = self._local_path(rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, arr)

    # ------------------------------------------------------------------
    # Pickle (checkpoints)
    # ------------------------------------------------------------------
    def load_pickle(self, rel_path: str):
        if self.platform == "gcp":
            blob = self._bucket.blob(self._gcs_key(rel_path))
            data = blob.download_as_bytes()
            return pickle.loads(data)
        else:
            with open(self._local_path(rel_path), "rb") as f:
                return pickle.load(f)

    def save_pickle(self, obj, rel_path: str):
        if self.platform == "gcp":
            data = pickle.dumps(obj)
            blob = self._bucket.blob(self._gcs_key(rel_path))
            blob.upload_from_string(data, content_type="application/octet-stream")
        else:
            path = self._local_path(rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(obj, f)

    # ------------------------------------------------------------------
    # Matplotlib figures
    # ------------------------------------------------------------------
    def save_figure(self, fig: matplotlib.figure.Figure, rel_path: str, dpi: int = 150):
        if self.platform == "gcp":
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            blob = self._bucket.blob(self._gcs_key(rel_path))
            blob.upload_from_file(buf, content_type="image/png")
        else:
            path = self._local_path(rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")

    # ------------------------------------------------------------------
    # YAML config
    # ------------------------------------------------------------------
    def save_yaml(self, config_dict: dict, rel_path: str):
        import yaml
        if self.platform == "gcp":
            data = yaml.safe_dump(config_dict).encode("utf-8")
            blob = self._bucket.blob(self._gcs_key(rel_path))
            blob.upload_from_string(data, content_type="text/yaml")
        else:
            path = self._local_path(rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                yaml.safe_dump(config_dict, f)

    def load_yaml(self, rel_path: str):
        import yaml
        if self.platform == "gcp":
            blob = self._bucket.blob(self._gcs_key(rel_path))
            data = blob.download_as_text()
            return yaml.safe_load(data)
        else:
            with open(self._local_path(rel_path), "r") as f:
                return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Directory listing
    # ------------------------------------------------------------------
    def list_dir(self, rel_path: str) -> list:
        """
        List file names (not full paths) in a directory.
        For GCS, lists blobs with the given prefix and returns basenames.
        """
        if self.platform == "gcp":
            prefix = self._gcs_key(rel_path)
            if not prefix.endswith("/"):
                prefix += "/"
            blobs = self._client.list_blobs(
                self.bucket_name, prefix=prefix, delimiter="/"
            )
            names = []
            for blob in blobs:
                name = blob.name[len(prefix):]
                if name:  # skip the prefix itself
                    names.append(name)
            return names
        else:
            local = self._local_path(rel_path)
            if not os.path.isdir(local):
                return []
            return os.listdir(local)

    # ------------------------------------------------------------------
    # Generic text / bytes
    # ------------------------------------------------------------------
    def save_text(self, text: str, rel_path: str):
        if self.platform == "gcp":
            blob = self._bucket.blob(self._gcs_key(rel_path))
            blob.upload_from_string(text, content_type="text/plain")
        else:
            path = self._local_path(rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)