#!/usr/bin/env python3
"""Measure local smoothness of the CNN maze VAE latent space.

The key question is decoder locality: if we perturb an encoded latent vector
slightly, does the decoded maze change slightly? To isolate that from encoder
reconstruction error, all perturbation comparisons are made against
`decode(mu_parent)` rather than against the original input grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from es.env_bridge import bfs_path_length, flood_fill_solvable
from vae.cnn_vae_level_utils import decode_latent_to_levels_grid
from vae.cnn_vae_model import CnnEncoder, CnnLstmDecoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "vae" / "checkpoints" / "cnn_vae" / "run11_1M",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "vae" / "datasets" / "vae_og" / "train_200k_grids.npz",
    )
    p.add_argument("--dataset-key", type=str, default="grids")
    p.add_argument("--num-parents", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "cnn_vae_locality",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_cnn_vae_params(checkpoint_dir: Path) -> dict:
    handler = ocp.CompositeCheckpointHandler(default=ocp.StandardCheckpointHandler())
    checkpointer = ocp.Checkpointer(handler)
    state = checkpointer.restore(str(checkpoint_dir))
    for _ in range(4):
        if hasattr(state, "keys") and "params" in state:
            break
        if hasattr(state, "keys") and "default" in state:
            state = state["default"]
            continue
        break
    if not (hasattr(state, "keys") and "params" in state):
        keys = list(state.keys()) if hasattr(state, "keys") else []
        raise KeyError(f"Checkpoint missing `params`. Top keys: {keys}")
    return state["params"]


def load_grids(path: Path, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        grids = data
    elif key in data:
        grids = data[key]
    else:
        candidates = [k for k in data.files if data[k].ndim == 4 and data[k].shape[-1] == 3]
        if not candidates:
            raise ValueError(f"No grid array found in {path}. Keys: {list(data.files)}")
        grids = data[candidates[0]]
    grids = np.asarray(grids, dtype=np.float32)
    if grids.ndim != 4 or grids.shape[-1] != 3:
        raise ValueError(f"Expected (N,H,W,3) grids, got {grids.shape}")
    return grids


def extract_components_from_grids(grids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, h, w, _ = grids.shape
    walls = grids[..., 0] > 0.5
    goal_flat = grids[..., 1].reshape(n, -1).argmax(axis=-1)
    agent_flat = grids[..., 2].reshape(n, -1).argmax(axis=-1)
    goal_pos = np.stack([goal_flat % w, goal_flat // w], axis=1).astype(np.uint32)
    agent_pos = np.stack([agent_flat % w, agent_flat // w], axis=1).astype(np.uint32)
    return walls, goal_pos, agent_pos


def encode_mu(params: dict, grids: np.ndarray, batch_size: int) -> np.ndarray:
    encoder = CnnEncoder(name="encoder")
    mean_kernel = jnp.asarray(params["mean_layer"]["kernel"], dtype=jnp.float32)
    mean_bias = jnp.asarray(params["mean_layer"]["bias"], dtype=jnp.float32)

    @jax.jit
    def _encode(batch: jnp.ndarray) -> jnp.ndarray:
        h = encoder.apply({"params": params["encoder"]}, batch)
        return jnp.tanh(h @ mean_kernel + mean_bias) * 4.0

    chunks: list[np.ndarray] = []
    for start in range(0, len(grids), batch_size):
        batch = jnp.asarray(grids[start : start + batch_size], dtype=jnp.float32)
        chunks.append(np.asarray(_encode(batch), dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, mean_kernel.shape[1]), dtype=np.float32)


def decode_components_from_latents(params: dict, z_batch: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent_dim = int(params["mean_layer"]["kernel"].shape[1])
    decoder = CnnLstmDecoder(latent_dim=latent_dim, name="decoder")

    @jax.jit
    def _decode_single(z_single: jnp.ndarray):
        wall_logits, goal_logits, agent_logits = decoder.apply(
            {"params": params["decoder"]}, z_single[None, :]
        )
        return wall_logits[0], goal_logits[0], agent_logits[0]

    levels = decode_latent_to_levels_grid(
        decode_fn=_decode_single,
        z_batch=jnp.asarray(z_batch, dtype=jnp.float32),
        rng=jax.random.PRNGKey(seed),
    )
    return (
        np.asarray(levels.wall_map, dtype=bool),
        np.asarray(levels.goal_pos, dtype=np.uint32),
        np.asarray(levels.agent_pos, dtype=np.uint32),
    )


def compute_solvability_and_bfs(
    walls: np.ndarray, goal_pos: np.ndarray, agent_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    solv = np.asarray(
        jax.vmap(flood_fill_solvable)(
            jnp.asarray(walls),
            jnp.asarray(agent_pos, dtype=jnp.uint32),
            jnp.asarray(goal_pos, dtype=jnp.uint32),
        ),
        dtype=bool,
    )
    bfs = np.asarray(
        jax.vmap(bfs_path_length)(
            jnp.asarray(walls),
            jnp.asarray(agent_pos, dtype=jnp.uint32),
            jnp.asarray(goal_pos, dtype=jnp.uint32),
        ),
        dtype=np.int32,
    )
    return solv, bfs


def compare_layouts(
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
    step_l2: np.ndarray,
) -> dict[str, float]:
    wall_hamming = np.logical_xor(parent_walls, child_walls).sum(axis=(1, 2)).astype(np.float32)
    wall_hamming_frac = wall_hamming / float(parent_walls.shape[1] * parent_walls.shape[2])
    wall_delta = np.abs(
        parent_walls.sum(axis=(1, 2)).astype(np.int32) - child_walls.sum(axis=(1, 2)).astype(np.int32)
    ).astype(np.float32)
    goal_shift = (
        np.abs(parent_goal[:, 0].astype(np.int32) - child_goal[:, 0].astype(np.int32))
        + np.abs(parent_goal[:, 1].astype(np.int32) - child_goal[:, 1].astype(np.int32))
    ).astype(np.float32)
    agent_shift = (
        np.abs(parent_agent[:, 0].astype(np.int32) - child_agent[:, 0].astype(np.int32))
        + np.abs(parent_agent[:, 1].astype(np.int32) - child_agent[:, 1].astype(np.int32))
    ).astype(np.float32)
    exact_match = (wall_hamming == 0) & (goal_shift == 0) & (agent_shift == 0)
    both_solvable = parent_solv & child_solv

    out = {
        "mean_step_l2": float(step_l2.mean()),
        "child_solvable_rate": float(child_solv.mean()),
        "solvability_flip_rate": float((parent_solv != child_solv).mean()),
        "both_solvable_rate": float(both_solvable.mean()),
        "exact_match_rate": float(exact_match.mean()),
        "same_wall_rate": float((wall_hamming == 0).mean()),
        "same_goal_rate": float((goal_shift == 0).mean()),
        "same_agent_rate": float((agent_shift == 0).mean()),
        "mean_wall_hamming": float(wall_hamming.mean()),
        "mean_wall_hamming_frac": float(wall_hamming_frac.mean()),
        "mean_abs_wall_delta": float(wall_delta.mean()),
        "mean_goal_shift": float(goal_shift.mean()),
        "mean_agent_shift": float(agent_shift.mean()),
    }
    if both_solvable.any():
        out["mean_abs_bfs_delta_both_solvable"] = float(
            np.abs(parent_bfs[both_solvable] - child_bfs[both_solvable]).mean()
        )
    else:
        out["mean_abs_bfs_delta_both_solvable"] = None
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    params = load_cnn_vae_params(args.checkpoint)
    grids = load_grids(args.dataset, key=args.dataset_key)
    rng = np.random.default_rng(args.seed)

    parent_n = min(args.num_parents, len(grids))
    parent_idx = rng.choice(len(grids), size=parent_n, replace=False)
    parent_grids = grids[parent_idx]
    orig_walls, orig_goal, orig_agent = extract_components_from_grids(parent_grids)

    z_parent = encode_mu(params, parent_grids, batch_size=args.batch_size)
    base_walls, base_goal, base_agent = decode_components_from_latents(params, z_parent, seed=args.seed + 1)
    orig_solv, orig_bfs = compute_solvability_and_bfs(orig_walls, orig_goal, orig_agent)
    base_solv, base_bfs = compute_solvability_and_bfs(base_walls, base_goal, base_agent)

    recon_metrics = compare_layouts(
        parent_walls=orig_walls,
        parent_goal=orig_goal,
        parent_agent=orig_agent,
        child_walls=base_walls,
        child_goal=base_goal,
        child_agent=base_agent,
        parent_solv=orig_solv,
        parent_bfs=orig_bfs,
        child_solv=base_solv,
        child_bfs=base_bfs,
        step_l2=np.zeros((parent_n,), dtype=np.float32),
    )

    rows: list[dict[str, object]] = []
    key = jax.random.PRNGKey(args.seed + 123)
    latent_dim = z_parent.shape[1]

    for sigma in args.sigmas:
        key, sub = jax.random.split(key)
        noise = np.asarray(jax.random.normal(sub, z_parent.shape), dtype=np.float32)
        z_child = z_parent + float(sigma) * noise
        step_l2 = np.linalg.norm(z_child - z_parent, axis=1)

        child_walls, child_goal, child_agent = decode_components_from_latents(
            params, z_child, seed=args.seed + 1000 + int(round(1000 * float(sigma)))
        )
        child_solv, child_bfs = compute_solvability_and_bfs(child_walls, child_goal, child_agent)

        metrics = compare_layouts(
            parent_walls=base_walls,
            parent_goal=base_goal,
            parent_agent=base_agent,
            child_walls=child_walls,
            child_goal=child_goal,
            child_agent=child_agent,
            parent_solv=base_solv,
            parent_bfs=base_bfs,
            child_solv=child_solv,
            child_bfs=child_bfs,
            step_l2=step_l2,
        )
        metrics["sigma"] = float(sigma)
        metrics["latent_dim"] = int(latent_dim)
        rows.append({"sigma": metrics.pop("sigma"), **metrics, "latent_dim": metrics.pop("latent_dim")})

    write_csv(args.out_dir / "locality_results.csv", rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "dataset_key": args.dataset_key,
        "num_parents": int(parent_n),
        "latent_dim": int(latent_dim),
        "reconstruction_vs_input": recon_metrics,
        "results_csv": str(args.out_dir / "locality_results.csv"),
        "sigmas": [float(s) for s in args.sigmas],
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.out_dir / 'locality_results.csv'}")
    print(f"Wrote: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
