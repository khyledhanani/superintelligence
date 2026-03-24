#!/usr/bin/env python3
"""Compare the prototype mutator against random minimax mutation on static mazes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from es.prototype_maze_mutator import (  # noqa: E402
    library_to_jax_arrays,
    load_context_prototype_library,
    mutate_wall_maps_with_prototypes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "vae" / "datasets" / "vae_og" / "train_200k_grids.npz",
    )
    p.add_argument("--dataset-key", type=str, default="grids")
    p.add_argument(
        "--library",
        type=Path,
        default=ROOT / "analysis" / "context_prototype_sweep_patch7_center5" / "best_library.npz",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "prototype_mutator_eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-parents", type=int, default=2048)
    p.add_argument("--num-edits", type=int, default=5)
    p.add_argument("--prototype-wall-biases", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    return p.parse_args()


def load_grids(path: Path, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        grids = data
    elif key in data:
        grids = data[key]
    else:
        candidates = [k for k in data.files if data[k].ndim == 4 and data[k].shape[-1] == 3]
        if not candidates:
            raise ValueError(f"No (N,H,W,3) grid array found in {path}. Keys: {list(data.files)}")
        grids = data[candidates[0]]
    grids = np.asarray(grids, dtype=np.float32)
    if grids.ndim != 4 or grids.shape[-1] != 3:
        raise ValueError(f"Expected (N,H,W,3) grids, got {grids.shape}")
    return grids


def grids_to_components(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    walls = grids[..., 0] > 0.5
    n, h, w = walls.shape
    goal_flat = grids[..., 1].reshape(n, -1).argmax(axis=-1).astype(np.int32)
    agent_flat = grids[..., 2].reshape(n, -1).argmax(axis=-1).astype(np.int32)
    goal_pos = np.stack([goal_flat % w, goal_flat // w], axis=-1).astype(np.uint32)
    agent_pos = np.stack([agent_flat % w, agent_flat // w], axis=-1).astype(np.uint32)
    return walls, goal_pos, agent_pos


def random_toggle_mutation(
    rng: jax.Array,
    wall_maps: jax.Array,
    goal_pos: jax.Array,
    agent_pos: jax.Array,
    num_edits: int,
) -> jax.Array:
    batch, height, width = wall_maps.shape
    flat_size = height * width
    edit_idx = jax.random.randint(rng, (batch, num_edits), 0, flat_size)
    rows = (edit_idx // width).astype(jnp.int32)
    cols = (edit_idx % width).astype(jnp.int32)
    batch_idx = jnp.arange(batch, dtype=jnp.int32)[:, None]
    toggled = wall_maps.at[batch_idx, rows, cols].set(~wall_maps[batch_idx, rows, cols])
    toggled = toggled.at[jnp.arange(batch), goal_pos[:, 1], goal_pos[:, 0]].set(False)
    toggled = toggled.at[jnp.arange(batch), agent_pos[:, 1], agent_pos[:, 0]].set(False)
    return toggled


def compute_solv_bfs(
    walls: np.ndarray,
    goal_pos: np.ndarray,
    agent_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, h, w = walls.shape
    bfs = np.full((n,), h * w, dtype=np.int32)
    solv = np.zeros((n,), dtype=bool)
    for i in range(n):
        wall = walls[i]
        start = (int(agent_pos[i, 1]), int(agent_pos[i, 0]))
        goal = (int(goal_pos[i, 1]), int(goal_pos[i, 0]))
        if wall[start] or wall[goal]:
            continue
        queue = [start]
        dist = {start: 0}
        found = False
        while queue:
            y, x = queue.pop(0)
            if (y, x) == goal:
                bfs[i] = dist[(y, x)]
                solv[i] = True
                found = True
                break
            nd = dist[(y, x)] + 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if wall[ny, nx] or (ny, nx) in dist:
                    continue
                dist[(ny, nx)] = nd
                queue.append((ny, nx))
        if not found:
            bfs[i] = h * w
    return solv, bfs


def summarize_method(
    name: str,
    parent_walls: np.ndarray,
    parent_goal: np.ndarray,
    parent_agent: np.ndarray,
    child_walls: np.ndarray,
    child_goal: np.ndarray,
    child_agent: np.ndarray,
    parent_solv: np.ndarray,
    parent_bfs: np.ndarray,
    child_solv: np.ndarray,
    child_bfs: np.ndarray,
    extra: dict[str, float] | None = None,
) -> dict[str, float | str]:
    wall_hamming = np.logical_xor(parent_walls, child_walls).sum(axis=(1, 2)).astype(np.float32)
    both_solvable = parent_solv & child_solv
    bfs_gain = child_bfs.astype(np.int32) - parent_bfs.astype(np.int32)
    wall_strings = child_walls.reshape(child_walls.shape[0], -1).astype(np.uint8)
    unique_children = len({row.tobytes() for row in wall_strings})
    row: dict[str, float | str] = {
        "method": name,
        "child_solvable_rate": float(child_solv.mean()),
        "solvability_flip_rate": float((parent_solv != child_solv).mean()),
        "both_solvable_rate": float(both_solvable.mean()),
        "exact_match_rate": float((wall_hamming == 0).mean()),
        "mean_wall_hamming": float(wall_hamming.mean()),
        "mean_wall_hamming_frac": float(wall_hamming.mean() / (parent_walls.shape[1] * parent_walls.shape[2])),
        "bfs_up_rate": float(np.mean(bfs_gain > 0)),
        "mean_bfs_gain": float(bfs_gain.mean()),
        "unique_child_rate": float(unique_children / len(child_walls)),
    }
    row["mean_abs_bfs_delta_both_solvable"] = (
        float(np.abs(child_bfs[both_solvable] - parent_bfs[both_solvable]).mean())
        if both_solvable.any()
        else None
    )
    if extra:
        row.update(extra)
    return row


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.library.exists():
        raise FileNotFoundError(f"Library not found: {args.library}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grids = load_grids(args.dataset, args.dataset_key)
    rng_np = np.random.default_rng(args.seed)
    idx = rng_np.choice(len(grids), size=min(args.num_parents, len(grids)), replace=False)
    parent_grids = grids[idx]
    parent_walls, parent_goal, parent_agent = grids_to_components(parent_grids)
    parent_solv, parent_bfs = compute_solv_bfs(parent_walls, parent_goal, parent_agent)

    library = library_to_jax_arrays(load_context_prototype_library(str(args.library)))
    rows = []
    for i, wall_bias in enumerate(args.prototype_wall_biases):
        rng_proto = jax.random.PRNGKey(args.seed + 1 + i)
        proto_walls_jax, proto_info = mutate_wall_maps_with_prototypes(
            rng_proto,
            jnp.asarray(parent_walls, dtype=jnp.bool_),
            jnp.asarray(parent_goal, dtype=jnp.uint32),
            jnp.asarray(parent_agent, dtype=jnp.uint32),
            library,
            deterministic=False,
            prototype_wall_bias=float(wall_bias),
        )
        proto_walls = np.asarray(proto_walls_jax, dtype=bool)
        proto_goal = parent_goal
        proto_agent = parent_agent
        proto_solv, proto_bfs = compute_solv_bfs(proto_walls, proto_goal, proto_agent)
        rows.append(
            summarize_method(
                f"prototype_bias_{wall_bias:g}",
                parent_walls,
                parent_goal,
                parent_agent,
                proto_walls,
                proto_goal,
                proto_agent,
                parent_solv,
                parent_bfs,
                proto_solv,
                proto_bfs,
                extra={
                    "mean_selected_alt_mass": float(np.asarray(proto_info["chosen_alt_mass"]).mean()),
                    "num_codes_used": float(len(np.unique(np.asarray(proto_info["chosen_code"])))),
                },
            )
        )

    rng_random = jax.random.PRNGKey(args.seed + 100)
    rand_walls_jax = random_toggle_mutation(
        rng_random,
        jnp.asarray(parent_walls, dtype=jnp.bool_),
        jnp.asarray(parent_goal, dtype=jnp.uint32),
        jnp.asarray(parent_agent, dtype=jnp.uint32),
        int(args.num_edits),
    )
    rand_walls = np.asarray(rand_walls_jax, dtype=bool)
    rand_goal = parent_goal
    rand_agent = parent_agent
    rand_solv, rand_bfs = compute_solv_bfs(rand_walls, rand_goal, rand_agent)

    rows.append(
        summarize_method(
            "random_toggle",
            parent_walls,
            parent_goal,
            parent_agent,
            rand_walls,
            rand_goal,
            rand_agent,
            parent_solv,
            parent_bfs,
            rand_solv,
            rand_bfs,
        )
    )

    summary = {
        "dataset": str(args.dataset),
        "library": str(args.library),
        "num_parents": int(len(parent_walls)),
        "num_edits": int(args.num_edits),
        "prototype_wall_biases": [float(x) for x in args.prototype_wall_biases],
        "results": rows,
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(args.out_dir / "results.csv", rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation to {args.out_dir}")


if __name__ == "__main__":
    main()
