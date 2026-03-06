#!/usr/bin/env python3
"""Build a mixed training dataset from vae_og and plr_pvl datasets.

Output format matches MazeAE training expectations:
  - grids: (N, 13, 13, 3) float32

Extra metadata is included for diagnostics:
  - source: uint8 (0=og, 1=plr)
  - score: float32 (NaN for OG rows)
  - wall_count: int32
  - bfs_length: int32 (-1 for OG rows)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--og-train",
        type=Path,
        default=Path("/Users/khyledhanani/Documents/superintelligence/vae/datasets/vae_og/datasets_new_train_200k_envs.npy"),
    )
    p.add_argument(
        "--og-val",
        type=Path,
        default=Path("/Users/khyledhanani/Documents/superintelligence/vae/datasets/vae_og/datasets_new_val_20k_envs.npy"),
    )
    p.add_argument(
        "--plr",
        type=Path,
        default=Path("vae/datasets/plr_pvl_dataset.npz"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("vae/datasets/mixed_og_plr_balanced.npz"),
    )
    p.add_argument(
        "--mode",
        choices=["balanced", "concat"],
        default="balanced",
        help="balanced: sample OG to match PLR count; concat: use all rows from both.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def seqs_to_grids(seqs: np.ndarray, h: int = 13, w: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """Convert CLUTTR (N,52) sequences to (N,H,W,3) grids + wall counts."""
    n = seqs.shape[0]
    obs = seqs[:, :50]
    goal_idx = np.clip(seqs[:, 50] - 1, 0, h * w - 1)
    agent_idx = np.clip(seqs[:, 51] - 1, 0, h * w - 1)

    wall_flat = np.zeros((n, h * w), dtype=bool)
    valid = obs > 0
    rows = np.repeat(np.arange(n), obs.shape[1])
    cols = np.clip(obs.reshape(-1) - 1, 0, h * w - 1)
    wall_flat[rows[valid.reshape(-1)], cols[valid.reshape(-1)]] = True

    # Clear goal/agent cells from wall map.
    wall_flat[np.arange(n), goal_idx] = False
    wall_flat[np.arange(n), agent_idx] = False

    grids = np.zeros((n, h, w, 3), dtype=np.float32)
    walls = wall_flat.reshape(n, h, w)
    grids[..., 0] = walls.astype(np.float32)

    goal_x = goal_idx % w
    goal_y = goal_idx // w
    agent_x = agent_idx % w
    agent_y = agent_idx // w
    grids[np.arange(n), goal_y, goal_x, 1] = 1.0
    grids[np.arange(n), agent_y, agent_x, 2] = 1.0

    wall_counts = walls.reshape(n, -1).sum(axis=1).astype(np.int32)
    return grids, wall_counts


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.og_train.exists() or not args.og_val.exists():
        raise FileNotFoundError("OG dataset files not found.")
    if not args.plr.exists():
        raise FileNotFoundError(f"PLR dataset not found: {args.plr}")

    print("Loading PLR dataset...")
    plr = np.load(args.plr)
    plr_grids = plr["grids"].astype(np.float32)
    plr_scores = plr["scores"].astype(np.float32) if "scores" in plr.files else np.full((len(plr_grids),), np.nan, dtype=np.float32)
    plr_walls = plr["wall_counts"].astype(np.int32) if "wall_counts" in plr.files else plr_grids[..., 0].reshape(len(plr_grids), -1).sum(axis=1).astype(np.int32)
    plr_bfs = plr["bfs_lengths"].astype(np.int32) if "bfs_lengths" in plr.files else np.full((len(plr_grids),), -1, dtype=np.int32)
    n_plr = len(plr_grids)
    print(f"  PLR rows: {n_plr}")

    print("Loading OG sequences...")
    og_train = np.load(args.og_train, mmap_mode="r")
    og_val = np.load(args.og_val, mmap_mode="r")
    og_all = np.concatenate([np.asarray(og_train, dtype=np.int32), np.asarray(og_val, dtype=np.int32)], axis=0)
    n_og_all = len(og_all)
    print(f"  OG rows available: {n_og_all}")

    if args.mode == "balanced":
        n_og_use = min(n_og_all, n_plr)
        idx = rng.choice(n_og_all, size=n_og_use, replace=False)
        og_use = og_all[idx]
        print(f"  Using balanced OG rows: {n_og_use}")
    else:
        og_use = og_all
        n_og_use = len(og_use)
        print(f"  Using all OG rows: {n_og_use}")

    print("Converting OG sequences to grids...")
    og_grids, og_walls = seqs_to_grids(og_use)
    og_scores = np.full((n_og_use,), np.nan, dtype=np.float32)
    og_bfs = np.full((n_og_use,), -1, dtype=np.int32)

    print("Merging + shuffling...")
    grids = np.concatenate([og_grids, plr_grids], axis=0)
    source = np.concatenate(
        [np.zeros((n_og_use,), dtype=np.uint8), np.ones((n_plr,), dtype=np.uint8)],
        axis=0,
    )
    scores = np.concatenate([og_scores, plr_scores], axis=0)
    wall_counts = np.concatenate([og_walls, plr_walls], axis=0)
    bfs_lengths = np.concatenate([og_bfs, plr_bfs], axis=0)

    perm = rng.permutation(len(grids))
    grids = grids[perm]
    source = source[perm]
    scores = scores[perm]
    wall_counts = wall_counts[perm]
    bfs_lengths = bfs_lengths[perm]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        grids=grids,
        source=source,
        scores=scores,
        wall_counts=wall_counts,
        bfs_lengths=bfs_lengths,
    )

    print(f"Saved: {args.out}")
    print(f"  Total rows: {len(grids)}")
    print(f"  Source counts -> OG: {(source == 0).sum()}, PLR: {(source == 1).sum()}")
    print(f"  Wall count mean/std: {wall_counts.mean():.3f} / {wall_counts.std():.3f}")


if __name__ == "__main__":
    main()

