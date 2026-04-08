#!/usr/bin/env python3
"""Download the public COMP0258 notebook assets (zip on Google Drive) into notebooks/assets.

Zip file (shared link):
https://drive.google.com/file/d/1_gqw6v4cNDxLm1BYmtfz2dLyt3_jG3n9/view?usp=drive_link

Usage:
  python notebooks/download_notebook_assets.py
  python notebooks/download_notebook_assets.py --force
  NOTEBOOK_ASSETS_FORCE=1 python notebooks/download_notebook_assets.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Public assets.zip on Google Drive
DRIVE_FILE_ID = "1_gqw6v4cNDxLm1BYmtfz2dLyt3_jG3n9"
DRIVE_ZIP_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"


def _run_stream(cmd: list[str]) -> None:
    """Run a command and stream combined stdout/stderr (works well in Colab notebooks)."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _ensure_gdown() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        check=True,
    )


def _extract_zip_to_assets(zip_path: Path, asset_dir: Path) -> None:
    """Extract zip layout into asset_dir.

    Supports:
    - zip with top-level ``assets/`` folder -> merge that folder's contents
    - zip with files/dirs at root -> merge into asset_dir
    """
    asset_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        top = list(tmp_path.iterdir())
        if not top:
            raise RuntimeError("Zip archive is empty.")

        if len(top) == 1 and top[0].is_dir():
            # e.g. a single top-level ``assets/`` folder or any one wrapper directory
            src_root = top[0]
        else:
            src_root = tmp_path

        for item in src_root.iterdir():
            dest = asset_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)


def download_assets(asset_dir: Path, *, force: bool = False) -> None:
    env_force = os.environ.get("NOTEBOOK_ASSETS_FORCE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    force = force or env_force

    marker = asset_dir / "cluttr_vae_aligned_checkpoint_2060000.pkl"
    if marker.exists() and not force:
        print(f"Assets already present ({marker.name} exists). Skip download.")
        print("Use --force or NOTEBOOK_ASSETS_FORCE=1 to re-download.")
        return

    _ensure_gdown()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "assets.zip"
        _run_stream(
            [
                sys.executable,
                "-m",
                "gdown",
                "--fuzzy",
                DRIVE_ZIP_URL,
                "-O",
                str(zip_path),
            ]
        )
        if not zip_path.is_file():
            raise FileNotFoundError(f"Expected downloaded zip at {zip_path}")
        _extract_zip_to_assets(zip_path, asset_dir)

    print("Done. Assets directory:", asset_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the default marker checkpoint already exists.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    asset_dir = script_dir / "assets"
    download_assets(asset_dir, force=args.force)


if __name__ == "__main__":
    main()
