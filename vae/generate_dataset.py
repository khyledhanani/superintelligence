"""Generate a maze token dataset for VAE training at any grid size.

Supports variable wall counts to ensure the VAE sees a range of difficulties.
Optionally uploads to GCS bucket for TPU training.

Usage:
    # Fixed wall count (like original 13x13 training)
    python vae/generate_dataset.py --grid_size 21 --n_levels 1000000 --n_walls 130 \
        --output /tmp/train_21x21_1M.npy

    # Variable wall count (uniform over a range — recommended)
    python vae/generate_dataset.py --grid_size 21 --n_levels 1000000 \
        --n_walls_min 20 --n_walls_max 180 \
        --output /tmp/train_21x21_1M.npy

    # With GCS upload
    python vae/generate_dataset.py --grid_size 21 --n_levels 1000000 \
        --n_walls_min 20 --n_walls_max 180 \
        --output /tmp/train_21x21_1M.npy \
        --gcs_bucket ucl-ued-project-bucket --gcs_prefix vae/datasets
"""
import argparse
import os
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from jaxued.environments.maze.util import make_level_generator
from vae_level_utils import level_to_tokens, grid_constants


def main():
    parser = argparse.ArgumentParser(description="Generate maze token dataset for VAE training")
    parser.add_argument("--grid_size", type=int, default=21, help="Grid height and width")
    parser.add_argument("--n_levels", type=int, default=1_000_000, help="Number of levels to generate")

    # Wall count options (mutually exclusive modes)
    parser.add_argument("--n_walls", type=int, default=None,
                        help="Fixed number of walls per level")
    parser.add_argument("--n_walls_min", type=int, default=None,
                        help="Min walls for variable mode (default: ~5%% of cells)")
    parser.add_argument("--n_walls_max", type=int, default=None,
                        help="Max walls for variable mode (default: ~45%% of cells)")

    parser.add_argument("--output", type=str, required=True, help="Output .npy file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Batch size for generation (larger = faster but more memory)")

    # GCS upload
    parser.add_argument("--gcs_bucket", type=str, default=None,
                        help="GCS bucket name for upload (e.g. ucl-ued-project-bucket)")
    parser.add_argument("--gcs_prefix", type=str, default="vae/datasets",
                        help="GCS path prefix within bucket")
    args = parser.parse_args()

    gs = args.grid_size
    num_cells = gs * gs
    consts = grid_constants(gs)
    seq_len = consts["seq_len"]
    max_walls = consts["max_walls"]

    # Determine wall count mode
    if args.n_walls is not None:
        # Fixed mode
        wall_counts = [args.n_walls]
        mode_str = f"fixed n_walls={args.n_walls}"
    else:
        # Variable mode
        wmin = args.n_walls_min if args.n_walls_min is not None else max(1, int(num_cells * 0.05))
        wmax = args.n_walls_max if args.n_walls_max is not None else min(max_walls, int(num_cells * 0.45))
        wall_counts = list(range(wmin, wmax + 1))
        mode_str = f"variable n_walls in [{wmin}, {wmax}]"

    # Print comparison with 13x13 baseline
    baseline_cells = 13 * 13  # 169
    baseline_walls = 50
    baseline_ratio = baseline_walls / baseline_cells
    print(f"=== Dataset Generation ===")
    print(f"Grid: {gs}x{gs} ({num_cells} cells)")
    print(f"Wall mode: {mode_str}")
    print(f"Token format: seq_len={seq_len}, vocab_size={consts['vocab_size']}, max_walls={max_walls}")
    print(f"Levels: {args.n_levels:,}")
    print(f"\n--- Wall density comparison with 13x13 baseline ---")
    print(f"  13x13: {baseline_walls}/{baseline_cells} = {baseline_ratio:.1%} density")
    if args.n_walls is not None:
        ratio = args.n_walls / num_cells
        print(f"  {gs}x{gs}: {args.n_walls}/{num_cells} = {ratio:.1%} density")
        equivalent_13 = int(baseline_cells * ratio)
        print(f"  Equivalent to {equivalent_13} walls on 13x13")
    else:
        ratio_min = wall_counts[0] / num_cells
        ratio_max = wall_counts[-1] / num_cells
        print(f"  {gs}x{gs}: [{wall_counts[0]}, {wall_counts[-1]}]/{num_cells} "
              f"= [{ratio_min:.1%}, {ratio_max:.1%}] density")
    print()

    # Pre-compile generators for each wall count (or just one if fixed)
    # For variable mode, we create one generator per wall count and cycle through them
    if len(wall_counts) == 1:
        # Fixed mode: one generator, simple
        level_gen = make_level_generator(gs, gs, wall_counts[0])
        gen_batch = jax.jit(jax.vmap(level_gen))
        tok_batch = jax.jit(jax.vmap(lambda lvl: level_to_tokens(lvl, grid_size=gs)))

        @jax.jit
        def generate_and_tokenize(rng):
            rngs = jax.random.split(rng, args.batch_size)
            levels = gen_batch(rngs)
            return tok_batch(levels)

        all_tokens = []
        rng = jax.random.PRNGKey(args.seed)
        n_remaining = args.n_levels

        while n_remaining > 0:
            rng, rng_batch = jax.random.split(rng)
            batch_tokens = generate_and_tokenize(rng_batch)
            batch_np = np.array(batch_tokens)
            take = min(args.batch_size, n_remaining)
            all_tokens.append(batch_np[:take])
            n_remaining -= take
            done = args.n_levels - n_remaining
            if done % 100000 < args.batch_size or n_remaining == 0:
                print(f"  Generated {done:,}/{args.n_levels:,} levels")

    else:
        # Variable mode: cycle through wall counts
        # Pre-compile a generator for a few representative wall counts to avoid
        # recompiling for every single value. Use bins.
        n_bins = min(len(wall_counts), 20)  # max 20 different generators
        bin_edges = np.linspace(wall_counts[0], wall_counts[-1], n_bins + 1).astype(int)
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) // 2).tolist()
        # Remove duplicates
        bin_centers = sorted(set(bin_centers))

        print(f"Using {len(bin_centers)} wall-count bins: {bin_centers}")

        generators = {}
        for nw in bin_centers:
            gen = make_level_generator(gs, gs, nw)
            generators[nw] = jax.jit(jax.vmap(gen))

        tok_batch = jax.jit(jax.vmap(lambda lvl: level_to_tokens(lvl, grid_size=gs)))

        all_tokens = []
        rng = jax.random.PRNGKey(args.seed)
        n_remaining = args.n_levels
        bin_idx = 0

        while n_remaining > 0:
            rng, rng_batch = jax.random.split(rng)
            nw = bin_centers[bin_idx % len(bin_centers)]
            bin_idx += 1

            rngs = jax.random.split(rng_batch, args.batch_size)
            levels = generators[nw](rngs)
            batch_tokens = np.array(tok_batch(levels))

            take = min(args.batch_size, n_remaining)
            all_tokens.append(batch_tokens[:take])
            n_remaining -= take
            done = args.n_levels - n_remaining
            if done % 100000 < args.batch_size or n_remaining == 0:
                print(f"  Generated {done:,}/{args.n_levels:,} levels (last batch: {nw} walls)")

    tokens_array = np.concatenate(all_tokens, axis=0)
    # Shuffle so wall counts are mixed
    rng_np = np.random.default_rng(args.seed)
    rng_np.shuffle(tokens_array)

    print(f"\nFinal shape: {tokens_array.shape} (expected ({args.n_levels}, {seq_len}))")

    np.save(args.output, tokens_array)
    print(f"Saved to {args.output}")

    # Stats
    nonzero_per_level = (tokens_array[:, :-2] > 0).sum(axis=1)
    print(f"Wall count stats: mean={nonzero_per_level.mean():.1f}, "
          f"std={nonzero_per_level.std():.1f}, "
          f"min={nonzero_per_level.min()}, max={nonzero_per_level.max()}")

    # GCS upload
    if args.gcs_bucket:
        filename = os.path.basename(args.output)
        gcs_path = f"gs://{args.gcs_bucket}/{args.gcs_prefix}/{filename}"
        print(f"\nUploading to {gcs_path}...")
        try:
            subprocess.run(["gsutil", "cp", args.output, gcs_path], check=True)
            print(f"Uploaded successfully.")
        except FileNotFoundError:
            # Try with full gcloud SDK path
            gcloud_gsutil = "/cs/student/project_msc/2025/csml/rhautier/google-cloud-sdk/bin/gsutil"
            subprocess.run([gcloud_gsutil, "cp", args.output, gcs_path], check=True)
            print(f"Uploaded successfully.")


if __name__ == "__main__":
    main()
