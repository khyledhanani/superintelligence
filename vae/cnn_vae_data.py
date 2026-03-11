"""CLUTR <-> grid data converters for the CNN-LSTM VAE maze project.

CLUTR sequence format (52 int32 values per maze):
  positions 0-49: wall cell positions
    - 1-indexed cell numbers in [1, 169]
    - value 0 = padding (no wall at that slot)
    - left-zero-padded so walls are at the END of the array, sorted ascending
  position 50: goal cell position (1-indexed, always non-zero)
  position 51: agent cell position (1-indexed, always non-zero)

Grid format: (13, 13, 3) float32
  channel 0: walls  — binary 0.0/1.0
  channel 1: goal   — binary one-hot (exactly one cell = 1.0)
  channel 2: agent  — binary one-hot (exactly one cell = 1.0)

Cell indexing:
  cell N (1-indexed) -> row = (N-1) // 13,  col = (N-1) % 13
  (row, col) -> cell number (1-indexed) = row * 13 + col + 1
"""

import numpy as np


def clutr_to_grid(seq: np.ndarray) -> np.ndarray:
    """Convert a CLUTR 52-int sequence to a (13, 13, 3) float32 grid.

    Args:
        seq: np.ndarray of shape (52,) with dtype int32.
             positions 0-49 are wall cell positions (1-indexed, 0=padding),
             position 50 is the goal cell (1-indexed),
             position 51 is the agent cell (1-indexed).

    Returns:
        grid: np.ndarray of shape (13, 13, 3) with dtype float32.
              channel 0 = walls, channel 1 = goal, channel 2 = agent.
    """
    grid = np.zeros((13, 13, 3), dtype=np.float32)

    # Wall cells: positions 0-49 (0 = padding, skip)
    for pos in seq[:50]:
        if pos > 0:
            idx = int(pos) - 1          # convert 1-indexed to 0-indexed
            grid[idx // 13, idx % 13, 0] = 1.0

    # Goal cell: position 50 (always 1-indexed, never 0)
    goal_idx = int(seq[50]) - 1
    grid[goal_idx // 13, goal_idx % 13, 1] = 1.0

    # Agent cell: position 51 (always 1-indexed, never 0)
    agent_idx = int(seq[51]) - 1
    grid[agent_idx // 13, agent_idx % 13, 2] = 1.0

    return grid


def grid_to_clutr(grid: np.ndarray) -> np.ndarray:
    """Convert a (13, 13, 3) float32 grid back to a 52-int CLUTR sequence.

    Wall positions are recovered by thresholding channel 0 at > 0.5,
    then sorted ascending and left-zero-padded into a 50-element array.
    Goal and agent positions are recovered via argmax of their respective
    channels and clamped to [1, 169].

    Args:
        grid: np.ndarray of shape (13, 13, 3) with dtype float32.

    Returns:
        seq: np.ndarray of shape (52,) with dtype int32.
    """
    # --- Wall channel ---
    wall_flat = grid[:, :, 0].flatten()                    # (169,)
    wall_positions_0idx = np.where(wall_flat > 0.5)[0]     # 0-indexed cell positions
    wall_positions = np.sort(wall_positions_0idx + 1).astype(np.int32)  # 1-indexed, sorted

    n_walls = len(wall_positions)
    walls = np.zeros(50, dtype=np.int32)
    if n_walls > 0:
        wall_positions = wall_positions[-min(n_walls, 50):]  # clamp: keep last 50
        n_walls = len(wall_positions)
        walls[50 - n_walls:] = wall_positions               # left-zero-pad

    # --- Goal channel ---
    goal_idx = int(np.argmax(grid[:, :, 1].flatten()))     # 0-indexed argmax
    goal_pos = np.clip(goal_idx + 1, 1, 169).astype(np.int32)  # 1-indexed, clamped

    # --- Agent channel ---
    agent_idx = int(np.argmax(grid[:, :, 2].flatten()))    # 0-indexed argmax
    agent_pos = np.clip(agent_idx + 1, 1, 169).astype(np.int32)  # 1-indexed, clamped

    return np.concatenate([walls, [goal_pos, agent_pos]]).astype(np.int32)
