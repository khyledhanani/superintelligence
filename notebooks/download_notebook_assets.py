#!/usr/bin/env python3
"""Download the public COMP0258 notebook assets from Google Drive into notebooks/assets.

Uses gdown (--folder) for the shared folder:
https://drive.google.com/drive/folders/1e1T4aFZ7lMPcNxA-i_f8345DANUZANba

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
from pathlib import Path

DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1e1T4aFZ7lMPcNxA-i_f8345DANUZANba"
)


def _ensure_gdown() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        check=True,
    )


def _merge_into_assets(staging: Path, asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    top = list(staging.iterdir())
    if not top:
        raise RuntimeError("gdown produced an empty download directory.")
    # gdown --folder usually creates a single subdirectory named after the Drive folder.
    if len(top) == 1 and top[0].is_dir():
        root_dir = top[0]
    else:
        root_dir = staging

    for item in root_dir.iterdir():
        dest = asset_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copytree(item, dest)
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
        cmd = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            "--remaining-ok",
            "-O",
            str(tmp_path),
            DRIVE_FOLDER_URL,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        _merge_into_assets(tmp_path, asset_dir)
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
