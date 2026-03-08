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
        help="Legacy option (ignored by current coverage-based sampler).",
    )
    p.add_argument(
        "--augment-synthetic",
        type=int,
        default=200_000,
        help="Number of additional synthetic random mazes to generate for wider coverage.",
    )
    p.add_argument(
        "--synthetic-max-walls",
        type=int,
        default=120,
        help="Maximum wall count for synthetic mazes (13x13 has 169 cells).",
    )
    p.add_argument(
        "--synthetic-tail-prob",
        type=float,
        default=0.65,
        help="Probability of sampling from high-wall tail for synthetic mazes.",
    )
    p.add_argument(
        "--coverage-min-per-bin-solv",
        type=int,
        default=1000,
        help="Minimum number of solvable samples per wall/bfs/slack bin.",
    )
    p.add_argument(
        "--coverage-min-per-bin-uns",
        type=int,
        default=300,
        help="Minimum number of unsolvable samples per wall-density bin.",
    )
    p.add_argument(
        "--wall-bin-edges",
        type=str,
        default="0.0,0.10,0.16,0.22,0.30,0.40,1.0",
        help="Comma-separated wall_density bin edges (inclusive outer bounds).",
    )
    p.add_argument(
        "--bfs-bin-edges",
        type=str,
        default="0.0,0.03,0.06,0.10,0.16,0.25,1.0",
        help="Comma-separated bfs_norm bin edges for solvable subset.",
    )
    p.add_argument(
        "--slack-bin-edges",
        type=str,
        default="0.0,0.005,0.015,0.03,0.06,0.12,1.0",
        help="Comma-separated slack_norm bin edges for solvable subset.",
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


def parse_edges(text: str) -> np.ndarray:
    edges = np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""], dtype=np.float32)
    if len(edges) < 3:
        raise ValueError("Bin edges must contain at least 3 values.")
    if abs(float(edges[0])) > 1e-6 or abs(float(edges[-1] - 1.0)) > 1e-6:
        raise ValueError("Bin edges must start at 0.0 and end at 1.0.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be strictly increasing.")
    return edges


def _bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    b = np.searchsorted(edges[1:-1], values, side="right")
    return np.clip(b, 0, len(edges) - 2).astype(np.int32)


def _sample_wall_counts(
    rng: np.random.Generator,
    n: int,
    max_walls: int,
    tail_prob: float,
    wall_cap: int,
) -> np.ndarray:
    if not (0.0 <= tail_prob <= 1.0):
        raise ValueError("--synthetic-tail-prob must be in [0,1].")
    max_walls = max(0, min(max_walls, wall_cap))
    if max_walls < 2:
        return np.zeros((n,), dtype=np.int32)
    split = min(max_walls, 60)
    draw_tail = rng.random(n) < tail_prob
    low = rng.integers(0, split + 1, size=n, dtype=np.int32)
    high_lo = min(split, max_walls)
    high = rng.integers(high_lo, max_walls + 1, size=n, dtype=np.int32)
    return np.where(draw_tail, high, low).astype(np.int32)


def generate_synthetic_grids(
    n: int,
    rng: np.random.Generator,
    h: int = 13,
    w: int = 13,
    max_walls: int = 120,
    tail_prob: float = 0.65,
) -> np.ndarray:
    """Generate random mazes with a heavier high-wall tail than OG data."""
    if n <= 0:
        return np.zeros((0, h, w, 3), dtype=np.float32)

    total = h * w
    wall_cap = total - 2  # keep goal and agent free
    wall_counts = _sample_wall_counts(rng, n, max_walls=max_walls, tail_prob=tail_prob, wall_cap=wall_cap)

    grids = np.zeros((n, h, w, 3), dtype=np.float32)
    for i in tqdm(range(n), desc="synthetic-grids", unit="maze"):
        wc = int(wall_counts[i])
        picks = rng.choice(total, size=wc + 2, replace=False)
        g_flat = int(picks[0])
        a_flat = int(picks[1])
        wall_flat = picks[2:]

        gy, gx = divmod(g_flat, w)
        ay, ax = divmod(a_flat, w)
        grids[i, gy, gx, 1] = 1.0
        grids[i, ay, ax, 2] = 1.0

        if wc > 0:
            wy = wall_flat // w
            wx = wall_flat % w
            grids[i, wy, wx, 0] = 1.0

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


def _select_with_min_bin_coverage(
    candidate_idx: np.ndarray,
    bin_arrays: list[np.ndarray],
    n_bins_list: list[int],
    target_n: int,
    min_per_bin: int,
    rng: np.random.Generator,
    tag: str,
) -> np.ndarray:
    """Select indices with guaranteed minimum count per bin for each bin-array."""
    if target_n <= 0 or len(candidate_idx) == 0:
        return np.empty((0,), dtype=np.int64)

    sel_mask = np.zeros((len(bin_arrays[0]),), dtype=bool)
    selected: list[np.ndarray] = []
    counters = [np.zeros((n_bins,), dtype=np.int32) for n_bins in n_bins_list]

    # First satisfy per-bin minima for each requested view.
    for view_id, bins in enumerate(bin_arrays):
        n_bins = n_bins_list[view_id]
        for b in range(n_bins):
            have = int(counters[view_id][b])
            need = max(min_per_bin - have, 0)
            if need == 0:
                continue
            mask = (~sel_mask[candidate_idx]) & (bins[candidate_idx] == b)
            cand = candidate_idx[mask]
            if len(cand) < need:
                raise ValueError(
                    f"{tag}: insufficient data for coverage in view={view_id}, bin={b}. "
                    f"need={need}, available={len(cand)}. Increase --augment-synthetic or "
                    "relax --coverage-min-per-bin-* / bin edges."
                )
            picks = rng.choice(cand, size=need, replace=False)
            sel_mask[picks] = True
            selected.append(picks)
            for k, b_arr in enumerate(bin_arrays):
                np.add.at(counters[k], b_arr[picks], 1)

    picked = np.concatenate(selected, axis=0) if selected else np.empty((0,), dtype=np.int64)
    if len(picked) > target_n:
        rng.shuffle(picked)
        return picked[:target_n]

    # Fill remainder with round-robin over occupied joint bins to keep diversity.
    remain_needed = target_n - len(picked)
    remaining = candidate_idx[~sel_mask[candidate_idx]]
    if remain_needed <= 0:
        return picked
    if len(remaining) < remain_needed:
        raise ValueError(f"{tag}: not enough remaining samples to reach target_n={target_n}.")

    if len(bin_arrays) == 1:
        rng.shuffle(remaining)
        return np.concatenate([picked, remaining[:remain_needed]], axis=0)

    mat = np.stack([b_arr[remaining] for b_arr in bin_arrays], axis=1)
    strides = np.array([int(np.prod(n_bins_list[:k], dtype=np.int64)) for k in range(len(n_bins_list))], dtype=np.int64)
    joint = (mat.astype(np.int64) * strides[None, :]).sum(axis=1)
    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(joint.tolist()):
        groups.setdefault(gid, []).append(i)
    for gid in groups:
        rng.shuffle(groups[gid])
    gids = list(groups.keys())
    rng.shuffle(gids)

    out = list(picked.tolist())
    while len(out) < target_n:
        progressed = False
        for gid in gids:
            if groups[gid]:
                out.append(int(remaining[groups[gid].pop()]))
                progressed = True
                if len(out) >= target_n:
                    break
        if not progressed:
            break
    if len(out) < target_n:
        extra_pool = np.setdiff1d(remaining, np.array(out, dtype=np.int64), assume_unique=False)
        need = target_n - len(out)
        extra = rng.choice(extra_pool, size=need, replace=False)
        out.extend(extra.tolist())
    return np.asarray(out, dtype=np.int64)


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

    wall_edges = parse_edges(args.wall_bin_edges)
    bfs_edges = parse_edges(args.bfs_bin_edges)
    slack_edges = parse_edges(args.slack_bin_edges)

    print("Loading OG sequences...")
    og_train = np.load(args.og_train, mmap_mode="r")
    og_val = np.load(args.og_val, mmap_mode="r")
    seqs = np.concatenate(
        [np.asarray(og_train, dtype=np.int32), np.asarray(og_val, dtype=np.int32)],
        axis=0,
    )
    print(f"  rows: {len(seqs)}")

    print("Converting sequences to grids...")
    og_grids = seqs_to_grids(seqs)

    syn_grids = generate_synthetic_grids(
        args.augment_synthetic,
        rng=rng,
        h=13,
        w=13,
        max_walls=int(args.synthetic_max_walls),
        tail_prob=float(args.synthetic_tail_prob),
    )
    if len(syn_grids) > 0:
        print(f"Generated synthetic mazes: {len(syn_grids)}")

    grids = np.concatenate([og_grids, syn_grids], axis=0) if len(syn_grids) > 0 else og_grids
    print(f"Total candidate mazes: {len(grids)}")

    print("Computing static structural targets for full pool...")
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

    if len(solv_idx) < target_solvable_n:
        raise ValueError(
            f"Not enough solvable mazes ({len(solv_idx)}) for target {target_solvable_n}. "
            "Increase --augment-synthetic or lower --target-solvable-ratio."
        )
    if len(uns_idx) < target_unsolvable_n:
        raise ValueError(
            f"Not enough unsolvable mazes ({len(uns_idx)}) for target {target_unsolvable_n}. "
            "Increase --augment-synthetic or increase difficulty tail."
        )

    wall_bin = _bin_ids(static_targets[:, 1], wall_edges)
    bfs_bin = _bin_ids(static_targets[:, 2], bfs_edges)
    slack_bin = _bin_ids(static_targets[:, 4], slack_edges)

    pick_solv = _select_with_min_bin_coverage(
        candidate_idx=solv_idx,
        bin_arrays=[wall_bin, bfs_bin, slack_bin],
        n_bins_list=[len(wall_edges) - 1, len(bfs_edges) - 1, len(slack_edges) - 1],
        target_n=target_solvable_n,
        min_per_bin=int(args.coverage_min_per_bin_solv),
        rng=rng,
        tag="solvable",
    )
    pick_uns = _select_with_min_bin_coverage(
        candidate_idx=uns_idx,
        bin_arrays=[wall_bin],
        n_bins_list=[len(wall_edges) - 1],
        target_n=target_unsolvable_n,
        min_per_bin=int(args.coverage_min_per_bin_uns),
        rng=rng,
        tag="unsolvable",
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
    out_wall_bin = _bin_ids(out_static[:, 1], wall_edges)
    out_bfs_bin = _bin_ids(out_static[:, 2], bfs_edges)
    out_slack_bin = _bin_ids(out_static[:, 4], slack_edges)
    print(f"  wall_bin_counts: {np.bincount(out_wall_bin, minlength=len(wall_edges)-1).tolist()}")
    print(f"  bfs_bin_counts: {np.bincount(out_bfs_bin, minlength=len(bfs_edges)-1).tolist()}")
    print(f"  slack_bin_counts: {np.bincount(out_slack_bin, minlength=len(slack_edges)-1).tolist()}")


if __name__ == "__main__":
    main()
