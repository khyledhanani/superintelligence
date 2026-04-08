#!/usr/bin/env python3
"""Download the public COMP0258 notebook assets (zip on Google Drive) into notebooks/assets.

Zip file (shared link):
https://drive.google.com/file/d/1_gqw6v4cNDxLm1BYmtfz2dLyt3_jG3n9/view?usp=drive_link

``gdown`` often fails from Colab on Drive (quota, virus-scan page, permissions). This script
tries several methods and falls back to a small ``requests``-based download with the usual
``confirm`` cookie flow.

Override the zip source (e.g. Hugging Face, GitHub raw, your own mirror):

  export NOTEBOOK_ASSETS_ZIP_URL='https://.../assets.zip'
  python notebooks/download_notebook_assets.py

Usage:
  python notebooks/download_notebook_assets.py
  python notebooks/download_notebook_assets.py --force
  NOTEBOOK_ASSETS_FORCE=1 python notebooks/download_notebook_assets.py
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Public assets.zip on Google Drive (file id from share link)
DRIVE_FILE_ID = "1_gqw6v4cNDxLm1BYmtfz2dLyt3_jG3n9"
DRIVE_VIEW_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing"
DRIVE_UC_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
DRIVE_EXPORT_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"


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


def _ensure_deps() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "gdown",
            "requests",
        ],
        check=True,
    )


def _gdown_try(url: str, zip_path: Path) -> bool:
    """Return True if gdown wrote a non-trivial zip file."""
    try:
        import gdown

        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
    except Exception as exc:
        print(f"gdown failed for {url!r}: {exc}", flush=True)
        return False
    if not zip_path.is_file() or zip_path.stat().st_size < 1024:
        return False
    # Drive sometimes saves an HTML error page with .zip name
    with open(zip_path, "rb") as f:
        head = f.read(4)
    return head == b"PK\x03\x04"


def _collect_confirm_tokens(page_html: str) -> list[str]:
    """Extract possible ``confirm`` tokens from a Drive virus-scan / interstitial HTML page."""
    t = html.unescape(page_html)
    found: list[str] = []

    for m in re.finditer(r"confirm=([0-9A-Za-z_-]+)", t):
        found.append(m.group(1))
    for m in re.finditer(r'"confirm"\s*:\s*"([^"]+)"', t):
        found.append(m.group(1))
    for m in re.finditer(r"name=\"confirm\"\s+value=\"([^\"]+)\"", t):
        found.append(m.group(1))

    # Prefer longer tokens (real confirms are usually long); keep stable order for ties.
    seen: set[str] = set()
    ordered: list[str] = []
    for c in sorted(found, key=len, reverse=True):
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    return ordered


def _download_drive_via_requests(file_id: str, zip_path: Path) -> None:
    """Download a Drive file by id; handles virus-scan HTML and several ``confirm`` shapes."""
    import requests

    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    timeout = (60, 600)  # (connect, read) — large zips are slow

    def _write_stream(resp: requests.Response, first_chunk: bytes) -> None:
        total = len(first_chunk)
        with open(zip_path, "wb") as f:
            f.write(first_chunk)
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        if total < 1024:
            raise RuntimeError("Downloaded file is too small; likely not the real zip.")
        with open(zip_path, "rb") as f:
            if f.read(4) != b"PK\x03\x04":
                raise RuntimeError("Downloaded file is not a zip (PK header missing).")

    def _finish_if_zip(resp: requests.Response) -> bool:
        first = next(resp.iter_content(chunk_size=65536), b"") or b""
        if first.startswith(b"PK\x03\x04"):
            _write_stream(resp, first)
            resp.close()
            return True
        for _ in resp.iter_content(chunk_size=65536):
            pass
        resp.close()
        return False

    def _read_html(resp: requests.Response, first: bytes) -> str:
        parts = [first.decode("utf-8", errors="replace")]
        for chunk in resp.iter_content(chunk_size=65536, decode_unicode=False):
            if not chunk:
                break
            parts.append(chunk.decode("utf-8", errors="replace"))
            if sum(len(p) for p in parts) > 3_000_000:
                break
        resp.close()
        return "".join(parts)

    base_hosts = (
        "https://drive.google.com",
        "https://docs.google.com",
    )
    params0 = {"export": "download", "id": file_id}

    last_html: str | None = None

    for host in base_hosts:
        url = f"{host}/uc"
        try:
            r = session.get(url, params=params0, headers=headers, stream=True, timeout=timeout)
            r.raise_for_status()

            token = None
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    token = v
                    break

            if token:
                r.close()
                r = session.get(
                    url,
                    params={**params0, "confirm": token},
                    headers=headers,
                    stream=True,
                    timeout=timeout,
                )
                r.raise_for_status()

            first = next(r.iter_content(chunk_size=65536), b"") or b""
            if first.startswith(b"PK\x03\x04"):
                _write_stream(r, first)
                r.close()
                return

            page = _read_html(r, first)
            last_html = page
            candidates = _collect_confirm_tokens(page)
            # Short fallbacks Google has used on interstitial pages
            for extra in ("t", "yes", "1"):
                if extra not in candidates:
                    candidates.append(extra)

            for confirm in candidates:
                r2 = session.get(
                    url,
                    params={**params0, "confirm": confirm},
                    headers=headers,
                    stream=True,
                    timeout=timeout,
                )
                r2.raise_for_status()
                if _finish_if_zip(r2):
                    print(
                        f"OK: Drive download passed virus-scan interstitial (host={host.split('//')[1]}).",
                        flush=True,
                    )
                    return
        except requests.RequestException as exc:
            print(f"Drive download attempt skipped for {host}: {exc}", flush=True)
            continue

    # usercontent host (sometimes used for large files)
    uc_url = "https://drive.usercontent.google.com/download"
    for confirm in _collect_confirm_tokens(last_html or "") + ["t", "yes", "1"]:
        r3 = session.get(
            uc_url,
            params={"id": file_id, "export": "download", "confirm": confirm},
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        if r3.status_code != 200:
            r3.close()
            continue
        if _finish_if_zip(r3):
            print("OK: Drive usercontent download.", flush=True)
            return

    raise RuntimeError(
        "Google Drive still returned HTML (virus-scan / confirm page) after all retries. "
        "Try: (1) open the zip link in a browser and confirm you can download without signing in; "
        "(2) wait and retry if download quota is exceeded; "
        "(3) host assets.zip elsewhere and set NOTEBOOK_ASSETS_ZIP_URL to a direct URL."
    )


def _download_zip(zip_path: Path) -> None:
    override = os.environ.get("NOTEBOOK_ASSETS_ZIP_URL", "").strip()
    if override:
        print("Using NOTEBOOK_ASSETS_ZIP_URL override.", flush=True)
        if _gdown_try(override, zip_path):
            return
        # Try as direct URL with requests
        import requests

        r = requests.get(override, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        with open(zip_path, "rb") as f:
            if f.read(4) != b"PK\x03\x04":
                raise RuntimeError("Override URL did not yield a zip file.")
        return

    _ensure_deps()

    for label, url in (
        ("gdown view URL", DRIVE_VIEW_URL),
        ("gdown uc export", DRIVE_EXPORT_URL),
        ("gdown uc?id", DRIVE_UC_URL),
    ):
        print(f"Trying {label} …", flush=True)
        if zip_path.exists():
            zip_path.unlink()
        if _gdown_try(url, zip_path):
            print(f"OK via gdown ({label}).", flush=True)
            return

    print("gdown failed; trying requests + Drive confirm flow …", flush=True)
    if zip_path.exists():
        zip_path.unlink()
    _download_drive_via_requests(DRIVE_FILE_ID, zip_path)
    print("OK via requests.", flush=True)


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

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "assets.zip"
        _download_zip(zip_path)
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
