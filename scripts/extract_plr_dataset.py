#!/usr/bin/env python3
"""
Extract PLR buffer data from ACCEL Orbax checkpoints into a VAE training dataset.

For each checkpoint step, reads:
  sampler.levels.{wall_map, goal_pos, agent_pos}
  sampler.scores  (PVL or MaxMC difficulty scores)
  sampler.size    (number of valid entries in the ring buffer)

Outputs a single NPZ file with:
  grids              (N, 13, 13, 3)  float32  — [wall_channel, goal_channel, agent_channel]
  scores             (N,)             float32  — raw difficulty scores (not normalised)
  wall_counts        (N,)             int32    — number of walls per maze
  bfs_lengths        (N,)             int32    — shortest navigable path length (169=unsolvable)
  static_targets     (N, 7)           float32  — [s1..s7] task-aware static labels
  p_ema              (N,)             float32  — dynamic success EMA (0 if unavailable)
  l_ema              (N,)             float32  — p_ema * (1 - p_ema)
  success_obs_count  (N,)             float32  — dynamic label confidence counts
  dynamic_weight     (N,)             float32  — min(1, success_obs_count/20)

Usage:
  python scripts/extract_plr_dataset.py \
      --checkpoint_dir checkpoints/accel_baseline_200k_seed0/0/models \
      --output        vae/datasets/plr_pvl_dataset.npz \
      [--step_stride 5]        # only load every Nth checkpoint (saves RAM/time)
      [--max_steps 9999999]    # cap on steps to include

Or aggregate across all runs:
  python scripts/extract_plr_dataset.py \
      --checkpoint_dir checkpoints/ \
      --output        vae/datasets/plr_pvl_dataset.npz
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.type_handlers as _ocp_th

# ---------------------------------------------------------------------------
# Cross-device compatibility: checkpoints saved on CUDA can't restore
# directly on CPU because orbax validates the saved sharding against
# jax.local_devices(). Patch the deserializer to fall back to the first
# local device (CPU on this machine) when the saved device is unavailable.
# ---------------------------------------------------------------------------
_orig_deserialize_sharding = _ocp_th._deserialize_sharding_from_json_string

def _cpu_fallback_sharding(json_string: str):
    try:
        return _orig_deserialize_sharding(json_string)
    except (ValueError, KeyError):
        return jax.sharding.SingleDeviceSharding(jax.devices()[0])

_ocp_th._deserialize_sharding_from_json_string = _cpu_fallback_sharding

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from es.env_bridge import bfs_path_length
from es.maze_ae import compute_structural_targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wall_map_fingerprint(wall_map: np.ndarray) -> bytes:
    """SHA-1 of the flattened boolean wall map for fast deduplication."""
    return hashlib.sha1(wall_map.astype(np.uint8).tobytes()).digest()


def level_to_grid(
    wall_map: np.ndarray,   # (H, W) bool
    goal_pos: np.ndarray,   # (2,) int  [col, row]
    agent_pos: np.ndarray,  # (2,) int  [col, row]
    H: int = 13,
    W: int = 13,
) -> np.ndarray:
    """Pack a Level into a (H, W, 3) float32 grid.

    Channel 0: wall map   (1.0 = wall, 0.0 = free)
    Channel 1: goal       (1.0 at goal cell, 0.0 elsewhere)
    Channel 2: agent      (1.0 at agent cell, 0.0 elsewhere)
    """
    grid = np.zeros((H, W, 3), dtype=np.float32)
    grid[:, :, 0] = wall_map.astype(np.float32)

    goal_row, goal_col = int(goal_pos[1]), int(goal_pos[0])
    agent_row, agent_col = int(agent_pos[1]), int(agent_pos[0])

    if 0 <= goal_row < H and 0 <= goal_col < W:
        grid[goal_row, goal_col, 1] = 1.0
    if 0 <= agent_row < H and 0 <= agent_col < W:
        grid[agent_row, agent_col, 2] = 1.0

    return grid


def compute_bfs_batch(
    wall_maps: np.ndarray,   # (N, H, W) bool
    goal_positions: np.ndarray,   # (N, 2) int  [col, row]
    agent_positions: np.ndarray,  # (N, 2) int  [col, row]
    batch_size: int = 256,
) -> np.ndarray:
    """Compute BFS path lengths for a batch of mazes via JAX vmap."""
    bfs_vmapped = jax.jit(jax.vmap(bfs_path_length))
    results = []
    N = len(wall_maps)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        wm = jnp.array(wall_maps[start:end], dtype=jnp.bool_)
        gp = jnp.array(goal_positions[start:end], dtype=jnp.uint32)
        ap = jnp.array(agent_positions[start:end], dtype=jnp.uint32)
        bfs = bfs_vmapped(wm, ap, gp)
        results.append(np.array(bfs, dtype=np.int32))
        if (start // batch_size) % 10 == 0:
            print(f"  BFS: {end}/{N}", flush=True)
    return np.concatenate(results)


def compute_static_targets_batch(
    wall_maps: np.ndarray,
    goal_positions: np.ndarray,
    agent_positions: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    fn = jax.jit(compute_structural_targets)
    out = []
    n = len(wall_maps)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        wm = jnp.asarray(wall_maps[start:end], dtype=jnp.bool_)
        gp = jnp.asarray(goal_positions[start:end], dtype=jnp.uint32)
        ap = jnp.asarray(agent_positions[start:end], dtype=jnp.uint32)
        out.append(np.asarray(fn(wm, gp, ap), dtype=np.float32))
    return np.concatenate(out, axis=0)


def discover_model_dirs(path: Path) -> list[Path]:
    """Discover Orbax model dirs.

    If `path` is already a models directory, return [path].
    Otherwise, recursively find all directories named `models` under `path`.
    """
    resolved = path.resolve()
    if resolved.name == "models":
        return [resolved]
    return sorted([p.resolve() for p in resolved.rglob("models") if p.is_dir()])


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(
    checkpoint_dir: Path,
    output_path: Path,
    step_stride: int = 1,
    max_steps: int = 10_000_000,
    H: int = 13,
    W: int = 13,
) -> None:
    model_dirs = discover_model_dirs(checkpoint_dir)
    if not model_dirs:
        print(f"ERROR: No `models` directories found under: {checkpoint_dir}")
        sys.exit(1)
    print(f"Discovered {len(model_dirs)} models directories under: {checkpoint_dir}")

    seen_fingerprints: set[bytes] = set()
    all_grids: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_wall_counts: list[np.ndarray] = []
    all_goal_pos: list[np.ndarray] = []
    all_agent_pos: list[np.ndarray] = []
    all_p_ema: list[np.ndarray] = []
    all_success_obs_count: list[np.ndarray] = []

    total_ckpts = 0
    loaded_ckpts = 0
    for model_dir_idx, model_dir in enumerate(model_dirs):
        print(f"\n[{model_dir_idx+1}/{len(model_dirs)}] Scanning: {model_dir}", flush=True)
        ckpt_mgr = ocp.CheckpointManager(
            str(model_dir),
            item_handlers=ocp.StandardCheckpointHandler(),
        )
        all_steps = sorted(ckpt_mgr.all_steps())
        selected_steps = [s for i, s in enumerate(all_steps) if i % step_stride == 0 and s <= max_steps]
        total_ckpts += len(selected_steps)
        print(
            f"  Found {len(all_steps)} checkpoints, loading {len(selected_steps)} "
            f"(stride={step_stride}, max_step={max_steps})"
        )

        for step_idx, step in enumerate(selected_steps):
            print(f"  - Loading step {step} ({step_idx+1}/{len(selected_steps)}) ...", flush=True)
            ckpt = ckpt_mgr.restore(step)
            if ckpt is None:
                print("    restore returned None, skipping.")
                continue

            if "sampler" not in ckpt:
                print("    checkpoint has no `sampler`, skipping.")
                continue
            loaded_ckpts += 1

            sampler = ckpt["sampler"]
            size = int(np.array(sampler["size"]))
            if size == 0:
                print("    buffer empty, skipping.")
                continue

            wall_maps  = np.array(sampler["levels"]["wall_map"][:size], dtype=np.bool_)   # (size, H, W)
            goal_pos   = np.array(sampler["levels"]["goal_pos"][:size],  dtype=np.int32)   # (size, 2)
            agent_pos  = np.array(sampler["levels"]["agent_pos"][:size], dtype=np.int32)  # (size, 2)
            scores     = np.array(sampler["scores"][:size],               dtype=np.float32) # (size,)
            if "levels_extra" in sampler and "success_ema" in sampler["levels_extra"]:
                success_ema = np.array(sampler["levels_extra"]["success_ema"][:size], dtype=np.float32)
                success_obs_count = np.array(sampler["levels_extra"]["success_obs_count"][:size], dtype=np.float32)
            else:
                success_ema = np.zeros((size,), dtype=np.float32)
                success_obs_count = np.zeros((size,), dtype=np.float32)

            new_wall_maps, new_goal_pos, new_agent_pos, new_scores = [], [], [], []
            new_success_ema, new_success_obs_count = [], []
            n_dup = 0
            for i in range(size):
                fp = _wall_map_fingerprint(wall_maps[i])
                if fp in seen_fingerprints:
                    n_dup += 1
                    continue
                seen_fingerprints.add(fp)
                new_wall_maps.append(wall_maps[i])
                new_goal_pos.append(goal_pos[i])
                new_agent_pos.append(agent_pos[i])
                new_scores.append(scores[i])
                new_success_ema.append(success_ema[i])
                new_success_obs_count.append(success_obs_count[i])

            print(
                f"    valid entries: {size}, new unique: {len(new_wall_maps)}, "
                f"duplicates skipped: {n_dup}"
            )
            if not new_wall_maps:
                continue

            new_wall_maps  = np.stack(new_wall_maps)   # (M, H, W)
            new_goal_pos   = np.stack(new_goal_pos)    # (M, 2)
            new_agent_pos  = np.stack(new_agent_pos)   # (M, 2)
            new_scores     = np.array(new_scores)      # (M,)
            new_success_ema = np.array(new_success_ema, dtype=np.float32)
            new_success_obs_count = np.array(new_success_obs_count, dtype=np.float32)

            # Build grid tensors
            grids = np.stack([
                level_to_grid(new_wall_maps[i], new_goal_pos[i], new_agent_pos[i], H, W)
                for i in range(len(new_wall_maps))
            ])  # (M, H, W, 3)

            wall_counts = new_wall_maps.sum(axis=(1, 2)).astype(np.int32)  # (M,)

            all_grids.append(grids)
            all_scores.append(new_scores)
            all_wall_counts.append(wall_counts)
            all_goal_pos.append(new_goal_pos)
            all_agent_pos.append(new_agent_pos)
            all_p_ema.append(new_success_ema)
            all_success_obs_count.append(new_success_obs_count)

    if not all_grids:
        print("ERROR: No data extracted. Check checkpoint directory and buffer contents.")
        sys.exit(1)
    print(f"\nLoaded {loaded_ckpts}/{total_ckpts} checkpoints.")

    grids       = np.concatenate(all_grids,       axis=0)
    scores      = np.concatenate(all_scores,      axis=0)
    wall_counts = np.concatenate(all_wall_counts, axis=0)
    goal_pos    = np.concatenate(all_goal_pos,    axis=0)
    agent_pos   = np.concatenate(all_agent_pos,   axis=0)
    p_ema = np.concatenate(all_p_ema, axis=0)
    success_obs_count = np.concatenate(all_success_obs_count, axis=0)

    N = len(grids)
    print(f"\nTotal unique levels: {N}")
    print(f"Wall count: min={wall_counts.min()}, mean={wall_counts.mean():.1f}, max={wall_counts.max()}")
    print(f"Scores: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")

    # BFS path lengths
    print("\nComputing BFS path lengths ...")
    bfs_lengths = compute_bfs_batch(grids[..., 0].astype(np.bool_), goal_pos, agent_pos)
    solvable_mask = bfs_lengths < (H * W)
    n_solvable = solvable_mask.sum()
    print(f"Solvable: {n_solvable}/{N} ({100*n_solvable/N:.1f}%)")
    print(f"BFS lengths (solvable): min={bfs_lengths[solvable_mask].min()}, "
          f"mean={bfs_lengths[solvable_mask].mean():.1f}, "
          f"p50={np.percentile(bfs_lengths[solvable_mask], 50):.0f}, "
          f"p90={np.percentile(bfs_lengths[solvable_mask], 90):.0f}, "
          f"max={bfs_lengths[solvable_mask].max()}")

    print("\nComputing static targets s1..s7 ...")
    static_targets = compute_static_targets_batch(
        grids[..., 0] > 0.5,
        goal_pos,
        agent_pos,
    )
    l_ema = (p_ema * (1.0 - p_ema)).astype(np.float32)
    dynamic_weight = np.minimum(1.0, success_obs_count / 20.0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        grids=grids,
        scores=scores,
        wall_counts=wall_counts,
        bfs_lengths=bfs_lengths,
        static_targets=static_targets,
        p_ema=p_ema.astype(np.float32),
        l_ema=l_ema,
        success_obs_count=success_obs_count.astype(np.float32),
        dynamic_weight=dynamic_weight,
    )
    print(f"\nSaved dataset → {output_path}")
    print(f"  grids:       {grids.shape}  (float32)")
    print(f"  scores:      {scores.shape}  (float32, raw PVL)")
    print(f"  wall_counts: {wall_counts.shape}  (int32)")
    print(f"  bfs_lengths: {bfs_lengths.shape}  (int32, {H*W}=unsolvable)")
    print(f"  static_targets: {static_targets.shape}  (float32, [s1..s7])")
    print(f"  p_ema: {p_ema.shape}  (float32)")
    print(f"  success_obs_count: {success_obs_count.shape}  (float32)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint_dir", type=Path,
        default=Path("checkpoints/accel_baseline_200k_seed0/0/models"),
        help="Path to the Orbax models directory (contains numbered step folders).",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("vae/datasets/plr_pvl_dataset.npz"),
        help="Output NPZ path.",
    )
    p.add_argument(
        "--step_stride", type=int, default=1,
        help="Load every Nth checkpoint (default 1 = all). Use 2-5 to reduce load time.",
    )
    p.add_argument(
        "--max_steps", type=int, default=10_000_000,
        help="Ignore checkpoints beyond this training step.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        step_stride=args.step_stride,
        max_steps=args.max_steps,
    )
