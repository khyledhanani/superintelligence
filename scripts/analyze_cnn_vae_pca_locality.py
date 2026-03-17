#!/usr/bin/env python3
"""Measure CNN-VAE decoder locality under PCA-aligned latent perturbations."""

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
    p.add_argument("--pca-fit-count", type=int, default=5000)
    p.add_argument("--num-parents", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-pcs", type=int, default=5)
    p.add_argument(
        "--step-multipliers",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "cnn_vae_pca_locality",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_params(path: Path) -> dict:
    handler = ocp.CompositeCheckpointHandler(default=ocp.StandardCheckpointHandler())
    checkpointer = ocp.Checkpointer(handler)
    state = checkpointer.restore(str(path))
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


def decode_components(params: dict, z_batch: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def compute_solv_bfs(
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


def compare(
    ref_walls: np.ndarray,
    ref_goal: np.ndarray,
    ref_agent: np.ndarray,
    ref_solv: np.ndarray,
    ref_bfs: np.ndarray,
    cand_walls: np.ndarray,
    cand_goal: np.ndarray,
    cand_agent: np.ndarray,
    cand_solv: np.ndarray,
    cand_bfs: np.ndarray,
) -> dict[str, float | None]:
    wall_hamming = np.logical_xor(ref_walls, cand_walls).sum(axis=(1, 2)).astype(np.float32)
    wall_hamming_frac = wall_hamming / float(ref_walls.shape[1] * ref_walls.shape[2])
    wall_delta = (
        cand_walls.sum(axis=(1, 2)).astype(np.int32) - ref_walls.sum(axis=(1, 2)).astype(np.int32)
    ).astype(np.float32)
    goal_shift = (
        np.abs(ref_goal[:, 0].astype(np.int32) - cand_goal[:, 0].astype(np.int32))
        + np.abs(ref_goal[:, 1].astype(np.int32) - cand_goal[:, 1].astype(np.int32))
    ).astype(np.float32)
    agent_shift = (
        np.abs(ref_agent[:, 0].astype(np.int32) - cand_agent[:, 0].astype(np.int32))
        + np.abs(ref_agent[:, 1].astype(np.int32) - cand_agent[:, 1].astype(np.int32))
    ).astype(np.float32)
    exact_match = (wall_hamming == 0) & (goal_shift == 0) & (agent_shift == 0)
    both_solvable = ref_solv & cand_solv

    out: dict[str, float | None] = {
        "exact_match_rate": float(exact_match.mean()),
        "same_wall_rate": float((wall_hamming == 0).mean()),
        "same_goal_rate": float((goal_shift == 0).mean()),
        "same_agent_rate": float((agent_shift == 0).mean()),
        "solvability_flip_rate": float((ref_solv != cand_solv).mean()),
        "child_solvable_rate": float(cand_solv.mean()),
        "mean_wall_hamming": float(wall_hamming.mean()),
        "mean_wall_hamming_frac": float(wall_hamming_frac.mean()),
        "mean_signed_wall_delta": float(wall_delta.mean()),
        "mean_abs_wall_delta": float(np.abs(wall_delta).mean()),
        "mean_goal_shift": float(goal_shift.mean()),
        "mean_agent_shift": float(agent_shift.mean()),
    }
    out["mean_abs_bfs_delta_both_solvable"] = (
        float(np.abs(ref_bfs[both_solvable] - cand_bfs[both_solvable]).mean()) if both_solvable.any() else None
    )
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
    rng = np.random.default_rng(args.seed)

    params = load_params(args.checkpoint)
    grids = load_grids(args.dataset, args.dataset_key)

    fit_n = min(args.pca_fit_count, len(grids))
    fit_idx = rng.choice(len(grids), size=fit_n, replace=False)
    fit_mu = encode_mu(params, grids[fit_idx], batch_size=args.batch_size)
    mu_mean = fit_mu.mean(axis=0)
    centered = fit_mu - mu_mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0).astype(np.float32)
    eigvecs = eigvecs[:, order].T.astype(np.float32)
    pc_stds = np.sqrt(np.maximum(eigvals, 1e-8)).astype(np.float32)

    parent_n = min(args.num_parents, len(grids))
    parent_idx = rng.choice(len(grids), size=parent_n, replace=False)
    parent_mu = encode_mu(params, grids[parent_idx], batch_size=args.batch_size)
    base_walls, base_goal, base_agent = decode_components(params, parent_mu, seed=args.seed + 1)
    base_solv, base_bfs = compute_solv_bfs(base_walls, base_goal, base_agent)
    base_wc = base_walls.sum(axis=(1, 2)).astype(np.int32)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    num_pcs = min(args.num_pcs, eigvecs.shape[0])

    for pc_idx in range(num_pcs):
        direction = eigvecs[pc_idx]
        pc_std = float(pc_stds[pc_idx])
        unit_dir = direction / max(np.linalg.norm(direction), 1e-8)
        for mult in args.step_multipliers:
            step = float(mult) * pc_std
            z_plus = parent_mu + step * unit_dir[None, :]
            z_minus = parent_mu - step * unit_dir[None, :]

            plus_w, plus_g, plus_a = decode_components(
                params, z_plus, seed=args.seed + 100 + (pc_idx * 1000) + int(round(mult * 100))
            )
            minus_w, minus_g, minus_a = decode_components(
                params, z_minus, seed=args.seed + 600 + (pc_idx * 1000) + int(round(mult * 100))
            )
            plus_s, plus_b = compute_solv_bfs(plus_w, plus_g, plus_a)
            minus_s, minus_b = compute_solv_bfs(minus_w, minus_g, minus_a)

            plus_metrics = compare(base_walls, base_goal, base_agent, base_solv, base_bfs, plus_w, plus_g, plus_a, plus_s, plus_b)
            minus_metrics = compare(base_walls, base_goal, base_agent, base_solv, base_bfs, minus_w, minus_g, minus_a, minus_s, minus_b)

            plus_wc = plus_w.sum(axis=(1, 2)).astype(np.int32)
            minus_wc = minus_w.sum(axis=(1, 2)).astype(np.int32)
            wall_count_pm = plus_wc - minus_wc
            mean_pm = float(wall_count_pm.mean())
            sign_consistency = (
                float((np.sign(wall_count_pm) == np.sign(mean_pm)).mean()) if abs(mean_pm) > 1e-8 else None
            )

            row = {
                "pc": int(pc_idx + 1),
                "pc_std": pc_std,
                "step_multiplier": float(mult),
                "step_l2": step,
                "plus_exact_match_rate": plus_metrics["exact_match_rate"],
                "minus_exact_match_rate": minus_metrics["exact_match_rate"],
                "plus_same_wall_rate": plus_metrics["same_wall_rate"],
                "minus_same_wall_rate": minus_metrics["same_wall_rate"],
                "plus_mean_wall_hamming": plus_metrics["mean_wall_hamming"],
                "minus_mean_wall_hamming": minus_metrics["mean_wall_hamming"],
                "plus_mean_abs_wall_delta": plus_metrics["mean_abs_wall_delta"],
                "minus_mean_abs_wall_delta": minus_metrics["mean_abs_wall_delta"],
                "plus_mean_signed_wall_delta": plus_metrics["mean_signed_wall_delta"],
                "minus_mean_signed_wall_delta": minus_metrics["mean_signed_wall_delta"],
                "plus_mean_abs_bfs_delta_both_solvable": plus_metrics["mean_abs_bfs_delta_both_solvable"],
                "minus_mean_abs_bfs_delta_both_solvable": minus_metrics["mean_abs_bfs_delta_both_solvable"],
                "plus_solvability_flip_rate": plus_metrics["solvability_flip_rate"],
                "minus_solvability_flip_rate": minus_metrics["solvability_flip_rate"],
                "wall_count_plus_minus_mean_diff": mean_pm,
                "wall_count_plus_minus_median_diff": float(np.median(wall_count_pm)),
                "wall_count_sign_consistency": sign_consistency,
            }
            rows.append(row)

            summary_rows.append(
                {
                    "pc": int(pc_idx + 1),
                    "step_multiplier": float(mult),
                    "step_l2": step,
                    "avg_mean_wall_hamming": float(
                        0.5 * (float(plus_metrics["mean_wall_hamming"]) + float(minus_metrics["mean_wall_hamming"]))
                    ),
                    "avg_exact_match_rate": float(
                        0.5 * (float(plus_metrics["exact_match_rate"]) + float(minus_metrics["exact_match_rate"]))
                    ),
                    "avg_solvability_flip_rate": float(
                        0.5 * (float(plus_metrics["solvability_flip_rate"]) + float(minus_metrics["solvability_flip_rate"]))
                    ),
                    "wall_count_plus_minus_mean_diff": mean_pm,
                    "wall_count_sign_consistency": sign_consistency,
                }
            )

    write_csv(args.out_dir / "locality_results.csv", rows)
    write_csv(args.out_dir / "summary_table.csv", summary_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "dataset_key": args.dataset_key,
        "pca_fit_count": fit_n,
        "num_parents": parent_n,
        "num_pcs": num_pcs,
        "step_multipliers": [float(x) for x in args.step_multipliers],
        "top5_pca_std": [float(x) for x in pc_stds[:5]],
        "top5_pca_var_ratio": [float(x) for x in eigvals[:5] / np.maximum(eigvals.sum(), 1e-8)],
        "results_csv": str(args.out_dir / "locality_results.csv"),
        "summary_table_csv": str(args.out_dir / "summary_table.csv"),
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.out_dir / 'locality_results.csv'}")
    print(f"Wrote: {args.out_dir / 'summary_table.csv'}")
    print(f"Wrote: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
