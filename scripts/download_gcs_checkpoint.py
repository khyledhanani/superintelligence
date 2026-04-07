"""Download a single checkpoint from GCS.

Usage:
    python scripts/download_gcs_checkpoint.py \
        --gcs_dir accel/run/checkpoints/run/0 \
        --local_dir /tmp/ckpt \
        --step 119
"""
import argparse
import os
from google.cloud import storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_dir", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--bucket", default="ucl-ued-project-bucket")
    parser.add_argument("--project", default="open-endedness-ued-project")
    args = parser.parse_args()

    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)

    os.makedirs(os.path.join(args.local_dir, "models", str(args.step)), exist_ok=True)

    # config.json
    config_blob = bucket.blob(f"{args.gcs_dir}/config.json")
    config_local = os.path.join(args.local_dir, "config.json")
    config_blob.download_to_filename(config_local)
    print(f"  Downloaded config.json")

    # models/{step}/
    prefix = f"{args.gcs_dir}/models/{args.step}/"
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel:
            continue
        local_file = os.path.join(args.local_dir, "models", str(args.step), rel)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
        n += 1
    print(f"  Downloaded {n} model files for step {args.step}")


if __name__ == "__main__":
    main()
