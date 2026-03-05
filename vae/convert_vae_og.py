#!/usr/bin/env python3
"""
Convert vae_og sequence datasets to the (N, H, W, 3) grid format expected by train_maze_ae.py.

Sequence format: (N, 52) int32
  - cols 0-49: sorted obstacle cell indices (1-169), 0 = empty padding
  - col 50:    goal cell index (1-169)
  - col 51:    agent cell index (1-169)

Grid format: (N, 13, 13, 3) float32
  - channel 0: walls  (1.0 where obstacles are)
  - channel 1: goal   (1.0 at goal cell)
  - channel 2: agent  (1.0 at agent cell)

Cell indices are 1-indexed: row = (idx-1) // 13, col = (idx-1) % 13
"""

from pathlib import Path
import numpy as np

GRID_H, GRID_W = 13, 13
INPUT_DIR = Path(__file__).parent / "datasets" / "vae_og"
OUTPUT_DIR = Path(__file__).parent / "datasets" / "vae_og"

FILES = [
    ("datasets_new_train_200k_envs.npy", "train_200k_grids.npz"),
    ("datasets_new_val_20k_envs.npy",    "val_20k_grids.npz"),
]


def seqs_to_grids(seqs: np.ndarray) -> np.ndarray:
    """Convert (N, 52) int32 sequences to (N, H, W, 3) float32 grids."""
    N = len(seqs)
    grids = np.zeros((N, GRID_H, GRID_W, 3), dtype=np.float32)

    obs_indices  = seqs[:, :50]   # (N, 50) — 0 means empty
    goal_indices = seqs[:, 50]    # (N,)
    agent_indices = seqs[:, 51]   # (N,)

    # Walls: iterate over 50 obstacle slots
    for slot in range(50):
        idx = obs_indices[:, slot]          # (N,)
        valid = idx > 0                     # mask out padding zeros
        rows = (idx - 1) // GRID_W
        cols = (idx - 1) % GRID_W
        grids[valid, rows[valid], cols[valid], 0] = 1.0

    # Goal channel
    g_rows = (goal_indices - 1) // GRID_W
    g_cols = (goal_indices - 1) % GRID_W
    grids[np.arange(N), g_rows, g_cols, 1] = 1.0

    # Agent channel
    a_rows = (agent_indices - 1) // GRID_W
    a_cols = (agent_indices - 1) % GRID_W
    grids[np.arange(N), a_rows, a_cols, 2] = 1.0

    return grids


def main():
    for in_file, out_file in FILES:
        in_path  = INPUT_DIR  / in_file
        out_path = OUTPUT_DIR / out_file

        print(f"Loading {in_path} ...")
        seqs = np.load(in_path)
        print(f"  Input shape: {seqs.shape}, dtype: {seqs.dtype}")

        grids = seqs_to_grids(seqs)
        print(f"  Output shape: {grids.shape}, dtype: {grids.dtype}")

        np.savez_compressed(out_path, grids=grids)
        print(f"  Saved -> {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
