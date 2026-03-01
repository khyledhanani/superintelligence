"""
pull_from_gcp.py — Download trained VAE runs from GCS to local disk.

Usage:
    python pull_from_gcp.py                    # list all runs
    python pull_from_gcp.py --run <run_id>     # download a specific run
    python pull_from_gcp.py --all              # download all runs
    python pull_from_gcp.py --latest           # download the most recent run
    python pull_from_gcp.py --plots            # download only plot.png from every run
"""
import argparse
import os
import yaml
from google.cloud import storage as gcs_storage

# Load config for bucket info
with open("vae_train_config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

BUCKET   = CONFIG["gcp_bucket"]
PREFIX   = CONFIG.get("gcp_bucket_prefix", "vae")
PROJECT  = CONFIG.get("gcp_project")
LOCAL_BASE = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"])


def get_client():
    return gcs_storage.Client(project=PROJECT)


def list_runs(client):
    """List all run IDs in the bucket."""
    bucket = client.bucket(BUCKET)
    prefix = f"{PREFIX}/runs/"

    # Use delimiter to get "subdirectories"
    iterator = client.list_blobs(BUCKET, prefix=prefix, delimiter="/")

    # We need to consume the iterator to populate prefixes
    _ = list(iterator)

    run_ids = []
    for p in iterator.prefixes:
        # p looks like "vae/runs/20250224_143052_lr5e-05_lat64/"
        run_id = p.rstrip("/").split("/")[-1]
        run_ids.append(run_id)

    return sorted(run_ids)


def list_run_files(client, run_id):
    """List all files in a specific run."""
    prefix = f"{PREFIX}/runs/{run_id}/"
    blobs = client.list_blobs(BUCKET, prefix=prefix)
    return [(b.name, b.size) for b in blobs]


def download_run(client, run_id, files_filter=None):
    """Download all files for a run to the local runs directory."""
    bucket = client.bucket(BUCKET)
    prefix = f"{PREFIX}/runs/{run_id}/"

    blobs = list(client.list_blobs(BUCKET, prefix=prefix))
    if not blobs:
        print(f"  No files found for run {run_id}")
        return

    downloaded = 0
    for blob in blobs:
        # Relative path under runs/<run_id>/
        rel = blob.name[len(f"{PREFIX}/"):]  # e.g. "runs/<run_id>/checkpoints/checkpoint_10000.pkl"

        if files_filter and not any(f in rel for f in files_filter):
            continue

        local_path = os.path.join(LOCAL_BASE, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and os.path.getsize(local_path) == blob.size:
            print(f"  [skip] {rel} (already exists, same size)")
            continue

        size_mb = blob.size / (1024 * 1024)
        print(f"  [pull] {rel} ({size_mb:.1f} MB)")
        blob.download_to_filename(local_path)
        downloaded += 1

    print(f"  Done — {downloaded} file(s) downloaded for {run_id}")


def main():
    parser = argparse.ArgumentParser(description="Pull VAE runs from GCS")
    parser.add_argument("--run", type=str, help="Download a specific run by ID")
    parser.add_argument("--all", action="store_true", help="Download all runs")
    parser.add_argument("--latest", action="store_true", help="Download the most recent run")
    parser.add_argument("--plots", action="store_true", help="Download only plots from all runs")
    parser.add_argument("--configs", action="store_true", help="Download only configs from all runs")
    parser.add_argument("--list", action="store_true", dest="list_only", help="Just list runs (default if no flags)")
    args = parser.parse_args()

    client = get_client()
    runs = list_runs(client)

    if not runs:
        print(f"No runs found in gs://{BUCKET}/{PREFIX}/runs/")
        return

    # Default: list runs
    if not (args.run or args.all or args.latest or args.plots or args.configs):
        args.list_only = True

    if args.list_only:
        print(f"\nRuns in gs://{BUCKET}/{PREFIX}/runs/  ({len(runs)} total):\n")
        for r in runs:
            files = list_run_files(client, r)
            ckpts = [f for f, _ in files if "checkpoint_" in f]
            total_mb = sum(s for _, s in files) / (1024 * 1024)
            print(f"  {r}  ({len(ckpts)} checkpoints, {total_mb:.0f} MB)")
        print(f"\nTo download: python pull_from_gcp.py --run <run_id>")
        return

    if args.plots:
        print(f"\nDownloading plots from {len(runs)} runs...\n")
        for r in runs:
            download_run(client, r, files_filter=["plot.png"])
        return

    if args.configs:
        print(f"\nDownloading configs from {len(runs)} runs...\n")
        for r in runs:
            download_run(client, r, files_filter=["config.yaml"])
        return

    if args.latest:
        run_id = runs[-1]
        print(f"\nDownloading latest run: {run_id}\n")
        download_run(client, run_id)
        return

    if args.all:
        print(f"\nDownloading all {len(runs)} runs...\n")
        for r in runs:
            print(f"\n--- {r} ---")
            download_run(client, r)
        return

    if args.run:
        if args.run not in runs:
            print(f"Run '{args.run}' not found. Available runs:")
            for r in runs:
                print(f"  {r}")
            return
        print(f"\nDownloading run: {args.run}\n")
        download_run(client, args.run)


if __name__ == "__main__":
    main()
