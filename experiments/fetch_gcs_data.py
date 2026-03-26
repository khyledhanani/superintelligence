"""Download experiment data (checkpoint, buffer, seeds) from GCS.

Usage:
    python experiments/fetch_gcs_data.py \
        --gcs_bucket ucl-ued-project-bucket \
        --gcs_project ucl-ued-project \
        --local_dir /tmp/injection_data \
        [--checkpoint_step 39]

Downloads:
    - checkpoint (config.json + models/<step>/) -> <local_dir>/checkpoint/
    - buffer_dump_10000.npz                     -> <local_dir>/buffer/
    - seeds_10k_gated/ (seeds_levels.pkl, seeds.npz) -> <local_dir>/seeds/

Prints the local paths at the end for use by downstream scripts.
"""
import argparse
import os
import sys


def download_prefix(bucket, gcs_prefix, local_dir):
    """Download all blobs under a GCS prefix to local_dir, preserving structure."""
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    n = 0
    for blob in blobs:
        # Strip the prefix to get relative path
        rel_path = blob.name[len(gcs_prefix):].lstrip("/")
        if not rel_path:
            continue
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Fetch experiment data from GCS")
    parser.add_argument("--gcs_bucket", type=str, default="ucl-ued-project-bucket")
    parser.add_argument("--gcs_project", type=str, default="ucl-ued-project")
    parser.add_argument("--local_dir", type=str, default="/tmp/injection_data")
    parser.add_argument("--checkpoint_step", type=int, default=39,
                        help="Orbax checkpoint step (39 = 10k updates with eval_freq=250)")
    parser.add_argument("--baseline_name", type=str, default="accel_sfl_baseline_13x13")
    parser.add_argument("--baseline_seed", type=int, default=0)
    parser.add_argument("--buffer_update", type=int, default=10000,
                        help="Buffer dump update number to download")
    args = parser.parse_args()

    from google.cloud import storage
    client = storage.Client(project=args.gcs_project)
    bucket = client.bucket(args.gcs_bucket)

    base = f"llm-exp"
    seed_str = str(args.baseline_seed)
    os.makedirs(args.local_dir, exist_ok=True)

    # --- 1. Checkpoint ---
    ckpt_local = os.path.join(args.local_dir, "checkpoint")
    if os.path.exists(os.path.join(ckpt_local, "config.json")):
        print(f"[SKIP] Checkpoint already at {ckpt_local}")
    else:
        os.makedirs(ckpt_local, exist_ok=True)
        # config.json
        config_blob = bucket.blob(f"{base}/checkpoints/{args.baseline_name}/{seed_str}/config.json")
        config_local = os.path.join(ckpt_local, "config.json")
        config_blob.download_to_filename(config_local)
        print(f"[OK] config.json")

        # models/<step>/ (only the specific step we need)
        step = args.checkpoint_step
        prefix = f"{base}/checkpoints/{args.baseline_name}/{seed_str}/models/{step}/"
        models_local = os.path.join(ckpt_local, "models")
        n = download_prefix(bucket, prefix.rstrip("/"), os.path.join(models_local, str(step)))
        print(f"[OK] Checkpoint step {step}: {n} files")

    # --- 2. Buffer ---
    buf_local_dir = os.path.join(args.local_dir, "buffer")
    buf_filename = f"buffer_dump_{args.buffer_update}.npz"
    buf_local = os.path.join(buf_local_dir, buf_filename)
    if os.path.exists(buf_local):
        print(f"[SKIP] Buffer already at {buf_local}")
    else:
        os.makedirs(buf_local_dir, exist_ok=True)
        blob = bucket.blob(f"{base}/buffer_dumps/{args.baseline_name}/{seed_str}/{buf_filename}")
        blob.download_to_filename(buf_local)
        print(f"[OK] {buf_filename}")

    # --- 3. Seeds (gated) ---
    seeds_local = os.path.join(args.local_dir, "seeds")
    if os.path.exists(os.path.join(seeds_local, "seeds_levels.pkl")) or \
       os.path.exists(os.path.join(seeds_local, "seeds.npz")):
        print(f"[SKIP] Seeds already at {seeds_local}")
    else:
        os.makedirs(seeds_local, exist_ok=True)
        prefix = f"{base}/seeds/{args.baseline_name}/{seed_str}/seeds_10k_gated"
        # Only download seeds_levels.pkl and seeds.npz (not renders)
        for fname in ["seeds_levels.pkl", "seeds.npz"]:
            blob = bucket.blob(f"{prefix}/{fname}")
            if blob.exists():
                blob.download_to_filename(os.path.join(seeds_local, fname))
                print(f"[OK] {fname}")
            else:
                print(f"[WARN] {fname} not found at {prefix}/{fname}")

    # --- Print paths for downstream scripts ---
    print(f"\n=== Local paths ===")
    print(f"AGENT_CKPT={ckpt_local}")
    print(f"BUFFER_NPZ={buf_local}")
    print(f"SEEDS_DIR={seeds_local}")


if __name__ == "__main__":
    main()
