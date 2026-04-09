#!/usr/bin/env python3
"""Download the LLM demo assets (zip on Google Drive) into notebooks/llm_assets.

Zip file (shared link):
https://drive.google.com/file/d/1T0R-MOT9z-B01-7nH_qPdPtkqlIDElHi/view?usp=sharing

Override the zip source:
  export LLM_ASSETS_ZIP_URL='https://.../llm_assets.zip'
  python notebooks/download_llm_assets.py

Usage:
  python notebooks/download_llm_assets.py
  python notebooks/download_llm_assets.py --force
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

# ── UPDATE THIS after uploading to Google Drive ──
DRIVE_FILE_ID = "1T0R-MOT9z-B01-7nH_qPdPtkqlIDElHi"


def _ensure_gdown() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"],
        check=True,
    )


def _download_zip(zip_path: Path) -> None:
    override = os.environ.get("LLM_ASSETS_ZIP_URL", "").strip()
    if override:
        print(f"Using LLM_ASSETS_ZIP_URL override: {override}", flush=True)
        import gdown
        gdown.download(override, str(zip_path), quiet=False, fuzzy=True)
    else:
        _ensure_gdown()
        import gdown
        url = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing"
        print(f"Downloading LLM assets from Drive ...", flush=True)
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)

    if not zip_path.is_file() or zip_path.stat().st_size < 1024:
        raise RuntimeError("Download failed: file is missing or too small.")
    with open(zip_path, "rb") as f:
        if f.read(4) != b"PK\x03\x04":
            raise RuntimeError("Downloaded file is not a zip (PK header missing).")


def download_llm_assets(asset_dir: Path, *, force: bool = False) -> None:
    env_force = os.environ.get("NOTEBOOK_ASSETS_FORCE", "").lower() in (
        "1", "true", "yes",
    )
    force = force or env_force

    marker = asset_dir / "buffer_dump_1250.npz"
    if marker.exists() and not force:
        print(f"LLM assets already present ({marker.name} exists). Skip download.")
        print("Use --force or NOTEBOOK_ASSETS_FORCE=1 to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "llm_assets.zip"
        _download_zip(zip_path)

        asset_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(asset_dir)

        # Handle nested directory: if zip contains a single top-level folder,
        # move its contents up to asset_dir.
        children = [
            c for c in asset_dir.iterdir()
            if c.name not in ("__MACOSX", ".DS_Store")
        ]
        if len(children) == 1 and children[0].is_dir():
            nested = children[0]
            for item in nested.iterdir():
                dest = asset_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            shutil.rmtree(nested)

    print(f"Done. LLM assets directory: {asset_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the marker file already exists.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    asset_dir = script_dir / "llm_assets"
    download_llm_assets(asset_dir, force=args.force)


if __name__ == "__main__":
    main()
