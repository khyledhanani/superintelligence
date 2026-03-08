#!/usr/bin/env python3
"""Build a structurally balanced random-only MazeAE dataset from OG sequences.

This script is task-agnostic: it uses only static maze structure labels
computed from layout geometry (no policy or replay metrics).

Output format matches `vae/train_maze_ae.py` expectations:
  - grids:          (N, 13, 13, 3) float32
  - static_targets: (N, 7)         float32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))
from es.maze_ae import compute_structural_targets_from_grids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--og-train",
        type=Path,
        default=Path("vae/datasets/vae_og/datasets_new_train_200k_envs.npy"),
        help="Path to OG train CLUTTR sequence dataset (.npy).",
    )
    p.add_argument(
        "--og-val",
        type=Path,
        default=Path("vae/datasets/vae_og/datasets_new_val_20k_envs.npy"),
        help="Path to OG val CLUTTR sequence dataset (.npy).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("vae/datasets/structured_random_stage1.npz"),
        help="Output dataset path.",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=100_000,
        help="Requested number of levels in the final balanced dataset.",
    )
    p.add_argument(
        "--target-solvable-ratio",
        type=float,
        default=0.85,
        help="Target fraction of solvable levels in output (0..1).",
    )
    p.add_argument(
        "--bins-per-dim",
        type=int,
        default=6,
        help="Number of quantile bins per selected structural dimension.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size for JAX static-target computation.",
    )
    return p.parse_args()


def seqs_to_grids(seqs: np.ndarray, h: int = 13, w: int = 13) -> np.ndarray:
    """Convert CLUTTR (N,52) sequences to (N,H,W,3) grids."""
    n = seqs.shape[0]
    obs = seqs[:, :50]
    goal_idx = np.clip(seqs[:, 50] - 1, 0, h * w - 1)
    agent_idx = np.clip(seqs[:, 51] - 1, 0, h * w - 1)

    wall_flat = np.zeros((n, h * w), dtype=bool)
    valid = obs > 0
    rows = np.repeat(np.arange(n), obs.shape[1])
    cols = np.clip(obs.reshape(-1) - 1, 0, h * w - 1)
    wall_flat[rows[valid.reshape(-1)], cols[valid.reshape(-1)]] = True

    # Goal/agent cells should remain free.
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
    return grids


def compute_static_targets_np(grids: np.ndarray, chunk_size: int) -> np.ndarray:
    fn = jax.jit(compute_structural_targets_from_grids)
    outs: list[np.ndarray] = []
    n = len(grids)
    for i in tqdm(range(0, n, chunk_size), desc="static-targets", unit="chunk"):
        g = jnp.asarray(grids[i : i + chunk_size], dtype=jnp.float32)
        outs.append(np.asarray(fn(g), dtype=np.float32))
    return np.concatenate(outs, axis=0)


def _quantile_edges(x: np.ndarray, bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    edges = np.quantile(x, q).astype(np.float32)
    # Ensure monotonic strictly-increasing edges for stable digitization.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.float32(np.inf), dtype=np.float32)
    return edges


def _balanced_pick(
    feat: np.ndarray,
    candidate_idx: np.ndarray,
    target_n: int,
    bins_per_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_n <= 0 or len(candidate_idx) == 0:
        return np.empty((0,), dtype=np.int64)
    if target_n >= len(candidate_idx):
        out = candidate_idx.copy()
        rng.shuffle(out)
        return out

    x = feat[candidate_idx]
    d = x.shape[1]
    edges = [_quantile_edges(x[:, k], bins_per_dim) for k in range(d)]

    bins = np.zeros((len(candidate_idx), d), dtype=np.int32)
    for k in range(d):
        # searchsorted on interior edges gives bin ids in [0, bins_per_dim-1]
        bins[:, k] = np.searchsorted(edges[k][1:-1], x[:, k], side="right")
        bins[:, k] = np.clip(bins[:, k], 0, bins_per_dim - 1)

    strides = np.array([bins_per_dim ** i for i in range(d)], dtype=np.int64)
    bin_id = (bins.astype(np.int64) * strides[None, :]).sum(axis=1)

    groups: dict[int, list[int]] = {}
    for pos, b in enumerate(bin_id.tolist()):
        groups.setdefault(b, []).append(pos)

    # Shuffle inside each occupied bin to avoid order bias.
    for b in groups:
        rng.shuffle(groups[b])

    occupied = list(groups.keys())
    rng.shuffle(occupied)

    chosen_pos: list[int] = []
    cursor = {b: 0 for b in occupied}
    while len(chosen_pos) < target_n:
        progressed = False
        for b in occupied:
            c = cursor[b]
            if c < len(groups[b]):
                chosen_pos.append(groups[b][c])
                cursor[b] = c + 1
                progressed = True
                if len(chosen_pos) >= target_n:
                    break
        if not progressed:
            break

    chosen_pos_np = np.array(chosen_pos, dtype=np.int64)
    return candidate_idx[chosen_pos_np]


def main() -> None:
    args = parse_args()
    if not args.og_train.exists() or not args.og_val.exists():
        raise FileNotFoundError("OG sequence dataset files not found.")
    if args.target_size < 1:
        raise ValueError("--target-size must be >= 1.")
    if not (0.0 <= args.target_solvable_ratio <= 1.0):
        raise ValueError("--target-solvable-ratio must be in [0, 1].")
    if args.bins_per_dim < 2:
        raise ValueError("--bins-per-dim must be >= 2.")

    rng = np.random.default_rng(args.seed)

    print("Loading OG sequences...")
    og_train = np.load(args.og_train, mmap_mode="r")
    og_val = np.load(args.og_val, mmap_mode="r")
    seqs = np.concatenate(
        [np.asarray(og_train, dtype=np.int32), np.asarray(og_val, dtype=np.int32)],
        axis=0,
    )
    print(f"  rows: {len(seqs)}")

    print("Converting sequences to grids...")
    grids = seqs_to_grids(seqs)

    print("Computing static structural targets...")
    static_targets = compute_static_targets_np(grids, chunk_size=args.chunk_size)

    # s1..s7: solvable, wall_density, bfs_norm, manhattan_norm, slack_norm,
    #         branch_ratio, dead_end_ratio
    solvable = static_targets[:, 0] > 0.5
    target_solvable_n = int(round(args.target_size * args.target_solvable_ratio))
    target_unsolvable_n = args.target_size - target_solvable_n

    solv_idx = np.flatnonzero(solvable)
    uns_idx = np.flatnonzero(~solvable)
    print(f"  solvable rows: {len(solv_idx)}")
    print(f"  unsolvable rows: {len(uns_idx)}")

    # Balance across task-agnostic structural dimensions.
    # Use s2,s3,s5,s6,s7 for solvable; drop s3/s5 for unsolvable (often saturated).
    feat_solv = static_targets[:, [1, 2, 4, 5, 6]]
    feat_uns = static_targets[:, [1, 3, 5, 6]]

    pick_solv = _balanced_pick(
        feat=feat_solv,
        candidate_idx=solv_idx,
        target_n=target_solvable_n,
        bins_per_dim=args.bins_per_dim,
        rng=rng,
    )
    pick_uns = _balanced_pick(
        feat=feat_uns,
        candidate_idx=uns_idx,
        target_n=target_unsolvable_n,
        bins_per_dim=args.bins_per_dim,
        rng=rng,
    )

    picked = np.concatenate([pick_solv, pick_uns], axis=0)
    if len(picked) < args.target_size:
        # Backfill from remaining rows (still random, no task leakage).
        remaining = np.setdiff1d(np.arange(len(grids), dtype=np.int64), picked, assume_unique=False)
        need = args.target_size - len(picked)
        if need > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
            picked = np.concatenate([picked, extra], axis=0)

    rng.shuffle(picked)
    picked = picked[: args.target_size]

    out_grids = grids[picked].astype(np.float32)
    out_static = static_targets[picked].astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        grids=out_grids,
        static_targets=out_static,
    )

    out_solvable_ratio = float((out_static[:, 0] > 0.5).mean())
    print(f"Saved: {args.out}")
    print(f"  rows: {len(out_grids)}")
    print(f"  solvable_ratio: {out_solvable_ratio:.4f}")
    print(f"  wall_density mean/std: {out_static[:,1].mean():.4f}/{out_static[:,1].std():.4f}")
    print(f"  bfs_norm mean/std: {out_static[:,2].mean():.4f}/{out_static[:,2].std():.4f}")
    print(f"  slack_norm mean/std: {out_static[:,4].mean():.4f}/{out_static[:,4].std():.4f}")


if __name__ == "__main__":
    main()
