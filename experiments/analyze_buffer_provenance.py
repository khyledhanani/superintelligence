"""Post-hoc analysis of injected level survival across buffer dumps.

Takes the eligible pool (with origin_ids) and a directory of buffer dumps,
and tracks which injected levels survived, their score evolution, and when
they were evicted.

Usage:
    python experiments/analyze_buffer_provenance.py \
        --eligible_pool /path/to/eligible_pool.npz \
        --buffer_dumps_dir /path/to/buffer_dumps/ \
        --output_dir /path/to/analysis/

Outputs:
    - survival_curve.png: fraction of injected levels surviving over updates
    - score_evolution.png: mean score of injected vs organic levels over time
    - eviction_timeline.png: when each injected level was evicted
    - provenance_stats.json: per-dump statistics
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def token_hash(tokens_row):
    return int(hashlib.md5(tokens_row.tobytes()).hexdigest()[:16], 16)


def analyze(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load eligible pool ---
    pool = np.load(args.eligible_pool, allow_pickle=True)
    pool_tokens = pool["tokens"]
    pool_scores = pool["scores"]
    pool_origin_ids = pool["origin_ids"]
    pool_seed_indices = pool["seed_indices"]
    n_pool = len(pool_tokens)
    print(f"[Pool] {n_pool} eligible levels")

    injected_id_set = set(pool_origin_ids.astype(np.uint64))

    # --- Find and sort buffer dumps ---
    dump_dir = args.buffer_dumps_dir
    dump_files = sorted([
        f for f in os.listdir(dump_dir)
        if f.startswith("buffer_dump_") and f.endswith(".npz") and "final" not in f
    ])
    # Also include final if present
    final = [f for f in os.listdir(dump_dir)
             if f.startswith("buffer_dump") and "final" in f and f.endswith(".npz")]
    dump_files += final

    if not dump_files:
        print(f"No buffer dumps found in {dump_dir}")
        return

    print(f"[Dumps] Found {len(dump_files)} buffer dumps")

    # --- Track per-dump stats ---
    stats = []
    for dump_file in dump_files:
        dump_path = os.path.join(dump_dir, dump_file)
        data = np.load(dump_path, allow_pickle=True)
        tokens = data["tokens"]
        scores = data["scores"]
        size = int(data["size"])
        update_num = int(data.get("update_num", 0))

        # Compute hashes for current buffer
        current_hashes = set()
        injected_alive = []
        organic_scores_list = []
        injected_scores_list = []

        for i in range(size):
            h = token_hash(tokens[i])
            current_hashes.add(h)
            if h in injected_id_set:
                injected_alive.append(h)
                injected_scores_list.append(float(scores[i]))
            else:
                organic_scores_list.append(float(scores[i]))

        n_survived = len(injected_alive)
        survival_rate = n_survived / max(n_pool, 1)

        entry = {
            "dump_file": dump_file,
            "update_num": update_num,
            "buffer_size": size,
            "n_injected_survived": n_survived,
            "n_injected_total": n_pool,
            "survival_rate": survival_rate,
            "injected_score_mean": float(np.mean(injected_scores_list)) if injected_scores_list else 0.0,
            "injected_score_std": float(np.std(injected_scores_list)) if injected_scores_list else 0.0,
            "organic_score_mean": float(np.mean(organic_scores_list)) if organic_scores_list else 0.0,
            "organic_score_std": float(np.std(organic_scores_list)) if organic_scores_list else 0.0,
            "buffer_score_mean": float(scores[:size].mean()),
        }
        stats.append(entry)
        print(f"  {dump_file}: {n_survived}/{n_pool} survived ({survival_rate:.0%}), "
              f"inj_score={entry['injected_score_mean']:.4f}, "
              f"org_score={entry['organic_score_mean']:.4f}")

    # --- Per-level survival timeline ---
    # Track first dump where each injected level disappears
    level_last_seen = {}  # origin_id -> last update_num where it was alive
    for entry_idx, dump_file in enumerate(dump_files):
        dump_path = os.path.join(dump_dir, dump_file)
        data = np.load(dump_path, allow_pickle=True)
        tokens = data["tokens"]
        size = int(data["size"])
        update_num = stats[entry_idx]["update_num"]

        for i in range(size):
            h = token_hash(tokens[i])
            if h in injected_id_set:
                level_last_seen[h] = update_num

    # --- Save stats JSON ---
    with open(os.path.join(args.output_dir, "provenance_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # --- Plot 1: Survival curve ---
    updates = [s["update_num"] for s in stats]
    survival = [s["survival_rate"] for s in stats]
    n_survived_list = [s["n_injected_survived"] for s in stats]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(updates, [s * 100 for s in survival], "b-o", markersize=3, label="Survival %")
    ax1.set_xlabel("Training Update")
    ax1.set_ylabel("Injected Levels Surviving (%)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(updates, n_survived_list, "r--", alpha=0.5, label="Count")
    ax2.set_ylabel("Count", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title("Injected Level Survival Through Training")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "survival_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved survival_curve.png")

    # --- Plot 2: Score evolution ---
    inj_means = [s["injected_score_mean"] for s in stats]
    org_means = [s["organic_score_mean"] for s in stats]
    buf_means = [s["buffer_score_mean"] for s in stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(updates, inj_means, "r-o", markersize=3, label="Injected (mean SFL)")
    ax.plot(updates, org_means, "b-o", markersize=3, label="Organic (mean SFL)")
    ax.plot(updates, buf_means, "k--", alpha=0.5, label="Buffer overall")
    ax.set_xlabel("Training Update")
    ax.set_ylabel("Mean SFL Score")
    ax.set_title("Score Evolution: Injected vs Organic Levels")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "score_evolution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved score_evolution.png")

    # --- Plot 3: Eviction timeline ---
    if level_last_seen:
        last_update = max(updates) if updates else 0
        eviction_times = []
        for oid in pool_origin_ids:
            oid_int = int(oid)
            if oid_int in level_last_seen:
                if level_last_seen[oid_int] < last_update:
                    eviction_times.append(level_last_seen[oid_int])
                else:
                    eviction_times.append(None)  # still alive at end
            else:
                eviction_times.append(0)  # never seen (wasn't in this injection %)

        evicted = [t for t in eviction_times if t is not None and t > 0]
        if evicted:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(evicted, bins=min(30, len(set(evicted))), color="coral", edgecolor="black", alpha=0.7)
            n_alive = sum(1 for t in eviction_times if t is None)
            ax.set_xlabel("Last Update Seen")
            ax.set_ylabel("Count")
            ax.set_title(f"Eviction Timeline of Injected Levels\n"
                         f"({len(evicted)} evicted, {n_alive} still alive at end)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, "eviction_timeline.png"), dpi=150)
            plt.close(fig)
            print(f"  Saved eviction_timeline.png")

    # --- Per-seed survival breakdown ---
    if pool_seed_indices is not None:
        unique_seeds = sorted(set(pool_seed_indices))
        print(f"\n  Per-seed survival at final dump:")
        final_hashes = set()
        final_data = np.load(os.path.join(dump_dir, dump_files[-1]), allow_pickle=True)
        for i in range(int(final_data["size"])):
            final_hashes.add(token_hash(final_data["tokens"][i]))

        for s in unique_seeds:
            mask = pool_seed_indices == s
            seed_ids = set(pool_origin_ids[mask].astype(np.uint64))
            alive = len(seed_ids & final_hashes)
            total = mask.sum()
            print(f"    Seed {s}: {alive}/{total} survived ({alive/max(total,1):.0%})")

    print(f"\n  Output: {args.output_dir}")

    # --- Upload to GCS ---
    if args.gcs_bucket:
        gcs_base = f"{args.gcs_prefix}/{os.path.basename(args.output_dir)}"
        print(f"\n[GCS] Uploading to gs://{args.gcs_bucket}/{gcs_base}/...")
        for fname in os.listdir(args.output_dir):
            fpath = os.path.join(args.output_dir, fname)
            if os.path.isfile(fpath):
                _upload_to_gcs(fpath, args.gcs_bucket, f"{gcs_base}/{fname}")


def _upload_to_gcs(local_path, gcs_bucket, gcs_path):
    """Upload a local file to GCS."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    except (ImportError, Exception) as e:
        import subprocess
        dest = f"gs://{gcs_bucket}/{gcs_path}"
        subprocess.run(["gcloud", "storage", "cp", local_path, dest], check=True)
    print(f"[GCS] {os.path.basename(local_path)} -> gs://{gcs_bucket}/{gcs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze injected level survival in buffer dumps")
    parser.add_argument("--eligible_pool", type=str, required=True,
                        help="Path to eligible_pool.npz from run_injection_experiment.py")
    parser.add_argument("--buffer_dumps_dir", type=str, required=True,
                        help="Directory containing buffer_dump_*.npz files from training")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None,
                        help="GCS bucket for uploading results")
    parser.add_argument("--gcs_prefix", type=str, default="llm-exp/analysis",
                        help="GCS prefix path within bucket")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.buffer_dumps_dir, "provenance_analysis")
    analyze(args)
