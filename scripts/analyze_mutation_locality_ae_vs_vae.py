#!/usr/bin/env python3
"""Compare AE vs VAE mutation locality on current checkpoints.

No regret is used in model training or scoring here. We evaluate structural
locality and "uphill proxy" rates using path metrics only.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("vae/datasets/plr_pvl_dataset.npz"))
    p.add_argument("--num_parents", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--out_dir", type=Path, default=Path("analysis/mutation_locality_ae_vs_vae"))
    return p.parse_args()


def import_es_modules():
    candidate_roots = [
        ROOT,
        Path("/Users/khyledhanani/Documents/superintelligence"),
    ]
    last_err = None
    for base in candidate_roots:
        if not (base / "es" / "cluttr_encoder.py").exists():
            continue
        sys.path.insert(0, str(base))
        sys.path.insert(0, str(base / "src"))
        try:
            cluttr = importlib.import_module("es.cluttr_encoder")
            vae_decoder = importlib.import_module("es.vae_decoder")
            maze_ae = importlib.import_module("es.maze_ae")
            env_bridge = importlib.import_module("es.env_bridge")
            return cluttr, vae_decoder, maze_ae, env_bridge, base
        except Exception as e:
            last_err = e
    raise ImportError(f"Unable to import ES modules. Last error: {last_err}")


def grids_to_sequences(grids: np.ndarray) -> np.ndarray:
    """Convert (N,13,13,3) grids to CLUTTR (N,52) sequences."""
    n, h, w, _ = grids.shape
    wall = grids[..., 0] > 0.5
    goal_flat = grids[..., 1].reshape(n, -1).argmax(axis=1)
    agent_flat = grids[..., 2].reshape(n, -1).argmax(axis=1)

    seqs = np.zeros((n, 52), dtype=np.int32)
    for i in range(n):
        obs = np.flatnonzero(wall[i].reshape(-1)) + 1  # 1-indexed
        # Cap at 50 obstacles per CLUTTR format.
        obs = obs[:50]
        seqs[i, : len(obs)] = obs
        seqs[i, 50] = int(goal_flat[i] + 1)
        seqs[i, 51] = int(agent_flat[i] + 1)
        # Enforce obstacle sort and clear collisions.
        seqs[i, :50] = np.sort(seqs[i, :50])
        seqs[i, :50] = np.where(seqs[i, :50] == seqs[i, 50], 0, seqs[i, :50])
        seqs[i, :50] = np.where(seqs[i, :50] == seqs[i, 51], 0, seqs[i, :50])
        seqs[i, :50] = np.sort(seqs[i, :50])
    return seqs


def seqs_to_components(seqs: np.ndarray, h: int = 13, w: int = 13):
    """Return walls(bool), goal_pos(uint32[x,y]), agent_pos(uint32[x,y])."""
    n = seqs.shape[0]
    obs = seqs[:, :50]
    goal_idx = np.clip(seqs[:, 50] - 1, 0, h * w - 1)
    agent_idx = np.clip(seqs[:, 51] - 1, 0, h * w - 1)

    wall_flat = np.zeros((n, h * w), dtype=bool)
    valid = obs > 0
    rows = np.repeat(np.arange(n), obs.shape[1])
    cols = np.clip(obs.reshape(-1) - 1, 0, h * w - 1)
    wall_flat[rows[valid.reshape(-1)], cols[valid.reshape(-1)]] = True

    wall_flat[np.arange(n), goal_idx] = False
    wall_flat[np.arange(n), agent_idx] = False

    walls = wall_flat.reshape(n, h, w)
    goal_pos = np.stack([goal_idx % w, goal_idx // w], axis=1).astype(np.uint32)
    agent_pos = np.stack([agent_idx % w, agent_idx // w], axis=1).astype(np.uint32)
    return walls, goal_pos, agent_pos


def components_to_grids(walls: np.ndarray, goal_pos: np.ndarray, agent_pos: np.ndarray):
    n, h, w = walls.shape
    grids = np.zeros((n, h, w, 3), dtype=np.float32)
    grids[..., 0] = walls.astype(np.float32)
    grids[np.arange(n), goal_pos[:, 1], goal_pos[:, 0], 1] = 1.0
    grids[np.arange(n), agent_pos[:, 1], agent_pos[:, 0], 2] = 1.0
    return grids


def deterministic_decode_mazeae(decoder_params: dict, z: jnp.ndarray, maze_decoder_cls):
    """Deterministic decode for MazeAE (argmax heads)."""
    wall_logits, goal_logits, agent_logits = maze_decoder_cls(height=13, width=13).apply(
        {"params": decoder_params}, z
    )
    wall = (jax.nn.sigmoid(wall_logits) > 0.5)
    goal_flat = jnp.argmax(goal_logits, axis=-1)
    # Mask goal for agent argmax to avoid overlap.
    n = goal_logits.shape[0]
    mask = jnp.zeros_like(agent_logits).at[jnp.arange(n), goal_flat].set(-jnp.inf)
    agent_flat = jnp.argmax(agent_logits + mask, axis=-1)

    gx = (goal_flat % 13).astype(jnp.uint32)
    gy = (goal_flat // 13).astype(jnp.uint32)
    ax = (agent_flat % 13).astype(jnp.uint32)
    ay = (agent_flat // 13).astype(jnp.uint32)
    wall = wall.at[jnp.arange(n), gy, gx].set(False)
    wall = wall.at[jnp.arange(n), ay, ax].set(False)
    goal_pos = jnp.stack([gx, gy], axis=1)
    agent_pos = jnp.stack([ax, ay], axis=1)
    return wall, goal_pos, agent_pos


def write_csv(path: Path, rows: list[list[object]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "sigma",
                "solvable_rate",
                "mean_abs_wall_delta",
                "mean_wall_hamming",
                "mean_abs_bfs_delta",
                "mean_abs_path_slack_delta",
                "bfs_up_rate",
                "path_slack_up_rate",
                "mean_bfs_gain",
                "mean_path_slack_gain",
            ]
        )
        w.writerows(rows)


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cluttr_mod, vae_dec_mod, maze_ae_mod, env_bridge_mod, es_root = import_es_modules()

    # Checkpoints
    cluttr_ckpt_candidates = [
        ROOT / "vae/model/checkpoint_420000.pkl",
        es_root / "vae/model/checkpoint_420000.pkl",
    ]
    maze_ckpt_candidates = [
        ROOT / "vae/model_maze_ae/checkpoint_final.pkl",
        es_root / "vae/model_maze_ae/checkpoint_final.pkl",
    ]
    cluttr_ckpt = next((p for p in cluttr_ckpt_candidates if p.exists()), None)
    maze_ckpt = next((p for p in maze_ckpt_candidates if p.exists()), None)
    if cluttr_ckpt is None or maze_ckpt is None:
        raise FileNotFoundError("Missing required checkpoints.")

    # Data
    d = np.load(args.dataset)
    grids = d["grids"].astype(np.float32)
    n = grids.shape[0]
    rng_np = np.random.default_rng(args.seed)
    idx = rng_np.choice(n, size=min(args.num_parents, n), replace=False)
    parent_grids = grids[idx]
    parent_seqs = grids_to_sequences(parent_grids)
    parent_walls, parent_goal, parent_agent = seqs_to_components(parent_seqs)

    # Parent structural metrics
    bfs_fn = jax.jit(jax.vmap(env_bridge_mod.bfs_path_length))
    solv_fn = jax.jit(jax.vmap(env_bridge_mod.flood_fill_solvable))
    parent_bfs = np.asarray(
        bfs_fn(
            jnp.asarray(parent_walls),
            jnp.asarray(parent_agent, dtype=jnp.uint32),
            jnp.asarray(parent_goal, dtype=jnp.uint32),
        )
    ).astype(np.int32)
    parent_manh = (
        np.abs(parent_goal[:, 0].astype(np.int32) - parent_agent[:, 0].astype(np.int32))
        + np.abs(parent_goal[:, 1].astype(np.int32) - parent_agent[:, 1].astype(np.int32))
    ).astype(np.int32)
    parent_slack = (parent_bfs - parent_manh).astype(np.int32)
    parent_wall_count = parent_walls.reshape(parent_walls.shape[0], -1).sum(axis=1)

    # Load models
    cluttr_params_full = vae_dec_mod.load_vae_params(str(cluttr_ckpt))
    cluttr_enc = cluttr_mod.extract_encoder_params(cluttr_params_full)
    cluttr_dec = vae_dec_mod.extract_decoder_params(cluttr_params_full)
    enc_cluttr_fn = jax.jit(lambda x: cluttr_mod.encode_levels_to_latents(cluttr_enc, x))
    dec_cluttr_fn = jax.jit(lambda z: vae_dec_mod.decode_latent_to_env(cluttr_dec, z, rng_key=None, temperature=0.0))
    repair_fn = jax.jit(jax.vmap(vae_dec_mod.repair_cluttr_sequence))

    maze_params_full = maze_ae_mod.load_maze_ae_params(str(maze_ckpt))
    maze_enc = maze_ae_mod.extract_maze_encoder_params(maze_params_full)
    maze_dec = maze_ae_mod.extract_maze_decoder_params(maze_params_full)
    enc_maze_fn = jax.jit(lambda x: maze_ae_mod.encode_maze_levels(maze_enc, x))
    dec_maze_fn = jax.jit(lambda z: deterministic_decode_mazeae(maze_dec, z, maze_ae_mod.MazeDecoder))

    z_parent_vae = np.asarray(enc_cluttr_fn(jnp.asarray(parent_seqs, dtype=jnp.int32)))
    z_parent_ae = np.asarray(enc_maze_fn(jnp.asarray(parent_grids, dtype=jnp.float32)))

    rows = []
    rng_jax = jax.random.PRNGKey(args.seed + 123)

    for model_name, z_parent in [("vae", z_parent_vae), ("ae", z_parent_ae)]:
        for sigma in args.sigmas:
            rng_jax, sub = jax.random.split(rng_jax)
            noise = np.asarray(jax.random.normal(sub, z_parent.shape), dtype=np.float32)
            z_child = z_parent + float(sigma) * noise

            if model_name == "vae":
                seq_child = np.asarray(repair_fn(dec_cluttr_fn(jnp.asarray(z_child, dtype=jnp.float32))), dtype=np.int32)
                cw, cg, ca = seqs_to_components(seq_child)
            else:
                cw_j, cg_j, ca_j = dec_maze_fn(jnp.asarray(z_child, dtype=jnp.float32))
                cw = np.asarray(cw_j, dtype=bool)
                cg = np.asarray(cg_j, dtype=np.uint32)
                ca = np.asarray(ca_j, dtype=np.uint32)

            solv = np.asarray(
                solv_fn(
                    jnp.asarray(cw),
                    jnp.asarray(ca, dtype=jnp.uint32),
                    jnp.asarray(cg, dtype=jnp.uint32),
                ),
                dtype=bool,
            )
            child_bfs = np.asarray(
                bfs_fn(
                    jnp.asarray(cw),
                    jnp.asarray(ca, dtype=jnp.uint32),
                    jnp.asarray(cg, dtype=jnp.uint32),
                )
            ).astype(np.int32)
            child_manh = (
                np.abs(cg[:, 0].astype(np.int32) - ca[:, 0].astype(np.int32))
                + np.abs(cg[:, 1].astype(np.int32) - ca[:, 1].astype(np.int32))
            ).astype(np.int32)
            child_slack = (child_bfs - child_manh).astype(np.int32)
            child_wall_count = cw.reshape(cw.shape[0], -1).sum(axis=1)

            wall_hamming = np.mean(np.not_equal(cw, parent_walls), axis=(1, 2))
            bfs_gain = child_bfs - parent_bfs
            slack_gain = child_slack - parent_slack

            row = [
                model_name,
                float(sigma),
                float(solv.mean()),
                float(np.mean(np.abs(child_wall_count - parent_wall_count))),
                float(wall_hamming.mean()),
                float(np.mean(np.abs(child_bfs - parent_bfs))),
                float(np.mean(np.abs(child_slack - parent_slack))),
                float(np.mean(bfs_gain > 0)),
                float(np.mean(slack_gain > 0)),
                float(np.mean(bfs_gain)),
                float(np.mean(slack_gain)),
            ]
            rows.append(row)

    write_csv(out_dir / "locality_results.csv", rows)

    summary = {
        "dataset": str(args.dataset),
        "num_parents": int(len(parent_grids)),
        "checkpoints": {"vae": str(cluttr_ckpt), "ae": str(maze_ckpt)},
        "sigmas": [float(s) for s in args.sigmas],
        "results_csv": str(out_dir / "locality_results.csv"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_dir / 'locality_results.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
