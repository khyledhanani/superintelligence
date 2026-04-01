"""Download data from GCS needed to reproduce the LLM injection plots.

Downloads:
  1. Merged buffers (pre-training) for seeds 1,2 (seed 0 not on GCS injection/)
  2. Training buffer dumps (with embeddings + origins) for all 3 seeds / 5 pcts
  3. Eligible pools

Local layout matches what plot_embedding_training.py and plot_env_space.py expect:
  {output_dir}/llm_inject_seed{s}/training_{pct}/buffer_dumps/buffer_dump_{N}.npz
  {output_dir}/llm_inject_seed{s}/merged_buffer_{pct}.npz

Usage:
    python scripts/download_plot_data.py
    python scripts/download_plot_data.py --seeds 0,1,2 --updates 1000,5000,10000
    python scripts/download_plot_data.py --training_only
"""
import argparse
import os

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)
from google.cloud import storage

BUCKET = "ucl-ued-project-bucket"
PROJECT = "ucl-ued-project"

INJECTION_PREFIX = "llm-exp/injection"
TRAINING_PREFIX = "llm-exp/training"


def download_blob(bucket, blob_name, local_path):
    """Download one blob, skipping if already exists locally."""
    if os.path.exists(local_path):
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None  # not found on GCS
    blob.download_to_filename(local_path)
    return True


def download_merged_buffers(bucket, seeds, pcts, local_root):
    """Download pre-training merged buffers + eligible pools."""
    print("=== Downloading merged buffers ===")
    downloaded, cached, missing = 0, 0, 0
    for seed in seeds:
        gcs_dir = f"{INJECTION_PREFIX}/llm_inject_seed{seed}"
        local_dir = os.path.join(local_root, f"llm_inject_seed{seed}")

        for pct in pcts:
            blob_name = f"{gcs_dir}/merged_buffer_{pct}.npz"
            local_path = os.path.join(local_dir, f"merged_buffer_{pct}.npz")
            result = download_blob(bucket, blob_name, local_path)
            if result is True:
                downloaded += 1
                print(f"  Downloaded s{seed}/merged_buffer_{pct}.npz")
            elif result is False:
                cached += 1
            else:
                missing += 1

        # Eligible pool
        for extra in ["eligible_pool.npz", "experiment_log.json"]:
            blob_name = f"{gcs_dir}/{extra}"
            local_path = os.path.join(local_dir, extra)
            download_blob(bucket, blob_name, local_path)

    print(f"  Merged buffers: {downloaded} new, {cached} cached, {missing} not on GCS")


def download_training_dumps(bucket, seeds, pcts, updates, local_root):
    """Download training buffer dumps (contain embeddings + origins)."""
    print("\n=== Downloading training buffer dumps ===")
    downloaded, cached, missing_list = 0, 0, []
    total = len(seeds) * len(pcts) * len(updates)

    for seed in seeds:
        for pct in pcts:
            pct_num = pct.replace("pct", "")
            run_name = f"inject_llm_{pct_num}pct_seed{seed}"
            gcs_base = f"{TRAINING_PREFIX}/{run_name}/buffer_dumps/{run_name}/{seed}"
            local_dir = os.path.join(local_root, f"llm_inject_seed{seed}",
                                     f"training_{pct}", "buffer_dumps")

            for update in updates:
                blob_name = f"{gcs_base}/buffer_dump_{update}.npz"
                local_path = os.path.join(local_dir, f"buffer_dump_{update}.npz")
                result = download_blob(bucket, blob_name, local_path)
                if result is True:
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"  [{downloaded + cached}/{total}] s{seed}/{pct}/u{update}")
                elif result is False:
                    cached += 1
                else:
                    missing_list.append(f"s{seed}/{pct}/u{update}")

    print(f"  Training dumps: {downloaded} new, {cached} cached, {len(missing_list)} missing")
    if missing_list:
        print(f"  Missing: {missing_list[:10]}{'...' if len(missing_list) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--pcts", type=str, default="5pct,10pct,15pct,20pct,25pct")
    parser.add_argument("--updates", type=str,
                        default="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000")
    parser.add_argument("--output_dir", type=str,
                        default="gcs_artifacts/plot_data",
                        help="Local directory to store downloaded data")
    parser.add_argument("--training_only", action="store_true")
    parser.add_argument("--merged_only", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pcts = args.pcts.split(",")
    updates = [int(u) for u in args.updates.split(",")]

    print(f"Output: {args.output_dir}")
    print(f"Seeds: {seeds}, Pcts: {pcts}")
    print(f"Updates: {updates}")
    print(f"Expected files: {len(seeds) * len(pcts) * len(updates)} training "
          f"+ {len(seeds) * len(pcts)} merged\n")

    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.training_only:
        download_merged_buffers(bucket, seeds, pcts, args.output_dir)

    if not args.merged_only:
        download_training_dumps(bucket, seeds, pcts, updates, args.output_dir)

    print(f"\nDone. Data in: {args.output_dir}/")


if __name__ == "__main__":
    main()
