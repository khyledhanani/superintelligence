"""Generate two 17x17 single-path labyrinth prefabs (spiral + serpentine).

These are hand-designed single-path mazes (no dead ends) in the same style
as the llm-injection-exp branch's Labyrinth21_* prefabs, but at 17x17 so
agents trained at 13x13 can at least see the local structure within the
agent view.

Saves an NPZ compatible with scripts/ood_evaluate.py and also prints each
maze as ASCII for inspection.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jaxued.environments.maze.level import Level  # noqa: E402


def _blank(size: int) -> np.ndarray:
    """Solid wall grid with 1-cell border."""
    g = np.ones((size, size), dtype=np.int32)
    g[1:-1, 1:-1] = 1  # start all walls
    return g


def _carve_line(g, r0, c0, r1, c1):
    """Carve an axis-aligned corridor from (r0,c0) to (r1,c1)."""
    if r0 == r1:
        lo, hi = min(c0, c1), max(c0, c1)
        g[r0, lo : hi + 1] = 0
    elif c0 == c1:
        lo, hi = min(r0, r1), max(r0, r1)
        g[lo : hi + 1, c0] = 0
    else:
        raise ValueError("carve_line requires an axis-aligned segment")


def spiral_labyrinth_17() -> str:
    """A single continuous inward spiral path from (1,1) to (9,7)."""
    size = 17
    g = _blank(size)

    # Draw the spiral as a single connected path by carving axis-aligned
    # segments one after the other. The path turns inward at each corner,
    # never creating a shortcut (each inner layer is 2 cells inside the
    # previous one, so there is always a 1-cell wall between them).
    segments = [
        ((1, 1), (1, 15)),    # right along row 1 (outer top)
        ((1, 15), (15, 15)),  # down col 15 (outer right)
        ((15, 15), (15, 1)),  # left along row 15 (outer bottom)
        ((15, 1), (3, 1)),    # up col 1, stop at row 3 (outer left)
        ((3, 1), (3, 13)),    # right row 3 (layer 2 top)
        ((3, 13), (13, 13)),  # down col 13
        ((13, 13), (13, 3)),  # left row 13
        ((13, 3), (5, 3)),    # up col 3, stop at row 5
        ((5, 3), (5, 11)),    # right row 5 (layer 3 top)
        ((5, 11), (11, 11)),  # down col 11
        ((11, 11), (11, 5)),  # left row 11
        ((11, 5), (7, 5)),    # up col 5, stop at row 7
        ((7, 5), (7, 9)),     # right row 7 (innermost top)
        ((7, 9), (9, 9)),     # down col 9
        ((9, 9), (9, 7)),     # left to goal at (9, 7)
    ]
    for (r0, c0), (r1, c1) in segments:
        _carve_line(g, r0, c0, r1, c1)

    # Goal: end of innermost segment
    gr, gc = 9, 7

    # Build the ASCII string
    lines = []
    for r in range(size):
        row = ""
        for c in range(size):
            if r == 1 and c == 1:
                row += ">"
            elif r == gr and c == gc:
                row += "G"
            else:
                row += "#" if g[r, c] == 1 else "."
        lines.append(row)
    return "\n".join(lines)


def serpentine_labyrinth_17() -> str:
    """Left-right serpentine corridors joined by vertical bridges. Single path."""
    size = 17
    g = _blank(size)

    # Horizontal corridors at odd rows 1, 3, 5, 7, 9, 11, 13, 15
    corridor_rows = list(range(1, size - 1, 2))

    # Each corridor spans columns 1..15
    for r in corridor_rows:
        _carve_line(g, r, 1, r, size - 2)

    # Connect consecutive corridors by a 1-cell vertical bridge,
    # alternating sides so the path zig-zags.
    for i in range(len(corridor_rows) - 1):
        r1 = corridor_rows[i]
        r2 = corridor_rows[i + 1]
        if i % 2 == 0:
            # bridge on the right side
            _carve_line(g, r1, size - 2, r2, size - 2)
            # Close the rest of the right column between corridors
        else:
            _carve_line(g, r1, 1, r2, 1)

    # Wall off the unused side of each corridor above its bridge to
    # guarantee a single solution path. We rebuild walls between non-bridge
    # ends.
    #
    # For corridor i at row r, left end is (r, 1), right end is (r, size-2)
    # Bridge at end "e" connects to corridor i+1. We leave the other end
    # closed (a dead-end? no — single path: we want a long corridor, not
    # dead-ends).  Because each corridor is fully carved 1..15 and the
    # only connections between them are the single bridges, the result
    # IS a single long path without dead-ends (both ends of each corridor
    # are connected only through the bridge; the other end is a wall).
    # Our carving above already creates this structure — good.

    # Start: top-left, Goal: bottom-right of the last corridor
    start_r, start_c = corridor_rows[0], 1
    # Goal: last corridor, end opposite the last bridge
    last_r = corridor_rows[-1]
    last_bridge_side = (len(corridor_rows) - 2) % 2  # bridge into last corridor
    if last_bridge_side == 0:
        # last bridge on right -> goal at left of last corridor
        goal_r, goal_c = last_r, 1
    else:
        goal_r, goal_c = last_r, size - 2

    lines = []
    for r in range(size):
        row = ""
        for c in range(size):
            if r == start_r and c == start_c:
                row += ">"
            elif r == goal_r and c == goal_c:
                row += "G"
            else:
                row += "#" if g[r, c] == 1 else "."
        lines.append(row)
    return "\n".join(lines)


LABYRINTH_17_1 = spiral_labyrinth_17()
LABYRINTH_17_2 = serpentine_labyrinth_17()


PREFABS = [
    ("Labyrinth17_1", LABYRINTH_17_1),
    ("Labyrinth17_2", LABYRINTH_17_2),
]


def verify_solvable(level_str: str) -> tuple[bool, int]:
    """BFS from start to goal. Returns (solvable, path length)."""
    lines = level_str.strip().split("\n")
    h = len(lines)
    w = len(lines[0])
    start = None
    goal = None
    walls = np.zeros((h, w), dtype=bool)
    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            if ch == "#":
                walls[r, c] = True
            elif ch in "><^v":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    assert start is not None and goal is not None

    from collections import deque

    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return True, d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return False, -1


def main():
    output = "eval_levels/large/labyrinth17_custom.npz"
    ascii_output = "eval_levels/large/labyrinth17_custom.txt"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    levels = []
    names = []
    ascii_parts = []
    for name, s in PREFABS:
        ok, d = verify_solvable(s)
        print(f"{name}: solvable={ok}, shortest path = {d}")
        print(s)
        print()
        if not ok:
            raise RuntimeError(f"{name} is not solvable!")
        levels.append(Level.from_str(s.strip()))
        names.append(name)
        ascii_parts.append(f"=== {name} (shortest path = {d}) ===\n{s.strip()}\n")

    stacked = Level.stack(levels)
    np.savez(
        output,
        wall_map=np.asarray(stacked.wall_map),
        goal_pos=np.asarray(stacked.goal_pos),
        agent_pos=np.asarray(stacked.agent_pos),
        agent_dir=np.asarray(stacked.agent_dir),
        width=np.asarray(stacked.width),
        height=np.asarray(stacked.height),
        names=np.asarray(names),
    )
    with open(ascii_output, "w") as f:
        f.write("\n".join(ascii_parts))

    print(f"Saved NPZ: {output}")
    print(f"Saved ASCII: {ascii_output}")


if __name__ == "__main__":
    main()
