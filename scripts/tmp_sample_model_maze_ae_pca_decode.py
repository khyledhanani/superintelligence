#!/usr/bin/env python3
"""Sample random latents from model_maze_ae and decode PCA perturbations.

For each random base latent z ~ N(0, I), this script decodes:
- base latent
- base + step along PCA component 1
- base + step along PCA component 2

Default output for 5 samples is 15 decoded mazes total.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from es.env_bridge import bfs_path_length, flood_fill_solvable
from es.maze_ae import (
    decode_maze_latents,
    encode_maze_levels,
    extract_maze_decoder_params,
    extract_maze_encoder_params,
    load_maze_ae_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random model_maze_ae latents and decode PCA perturbations."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "vae" / "model_maze_ae" / "checkpoint_final.pkl",
        help="Path to MazeAE checkpoint (.pkl).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "vae" / "datasets" / "vae_og" / "train_200k_grids.npz",
        help="Grid dataset used to fit PCA directions.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="grids",
        help="Array key for NPZ datasets.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random base latents.")
    parser.add_argument(
        "--pca-fit-count",
        type=int,
        default=5000,
        help="How many dataset items to use to fit PCA.",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=1.0,
        help="Multiplier for PCA step size (scaled by sqrt(eigenvalue)).",
    )
    parser.add_argument(
        "--step-multipliers",
        type=str,
        default="1.0",
        help="Comma-separated step multipliers, e.g. '1.0,2.0'.",
    )
    parser.add_argument(
        "--opposing-directions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, include both + and - perturbation directions.",
    )
    parser.add_argument(
        "--decode-temperature",
        type=float,
        default=0.25,
        help="Gumbel decode temperature for goal/agent sampling.",
    )
    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for wall logits.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size for PCA fit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--height",
        type=int,
        default=13,
        help="Maze height (should match training data).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=13,
        help="Maze width (should match training data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "tmp_model_maze_ae_pca_decode",
        help="Directory for PNG/text/npz outputs.",
    )
    return parser.parse_args()


def load_grids(path: Path, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        grids = data
    else:
        if key in data:
            grids = data[key]
        else:
            candidates = [
                k
                for k in data.files
                if data[k].ndim == 4 and data[k].shape[-1] == 3
            ]
            if not candidates:
                raise ValueError(
                    f"No (N,H,W,3) grid arrays found in {path}. Keys: {list(data.files)}"
                )
            grids = data[candidates[0]]
    grids = np.asarray(grids, dtype=np.float32)
    if grids.ndim != 4 or grids.shape[-1] != 3:
        raise ValueError(f"Expected grids with shape (N,H,W,3), got {grids.shape}")
    return grids


def encode_in_batches(encoder_params: dict, grids: np.ndarray, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(grids), batch_size):
        chunk = jnp.asarray(grids[start : start + batch_size], dtype=jnp.float32)
        z = encode_maze_levels(encoder_params, chunk)
        chunks.append(np.asarray(z, dtype=np.float32))
    if not chunks:
        return np.zeros((0, 64), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def compute_pca(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if latents.shape[0] < 2:
        d = latents.shape[1]
        return np.eye(d, dtype=np.float32), np.ones((d,), dtype=np.float32)

    centered = latents - latents.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0).astype(np.float32)
    eigvecs = eigvecs[:, order].T.astype(np.float32)  # rows are PCs
    return eigvecs, eigvals


def maze_rgb(wall_map: np.ndarray, goal_xy: np.ndarray, agent_xy: np.ndarray) -> np.ndarray:
    h, w = wall_map.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    img[wall_map] = np.array([0.05, 0.05, 0.05], dtype=np.float32)

    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    ax, ay = int(agent_xy[0]), int(agent_xy[1])
    img[gy, gx] = np.array([0.15, 0.80, 0.20], dtype=np.float32)
    img[ay, ax] = np.array([0.90, 0.15, 0.10], dtype=np.float32)
    return img


def maze_ascii(wall_map: np.ndarray, goal_xy: np.ndarray, agent_xy: np.ndarray) -> str:
    grid = np.full(wall_map.shape, ".", dtype="<U1")
    grid[wall_map] = "#"
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    ax, ay = int(agent_xy[0]), int(agent_xy[1])
    grid[gy, gx] = "G"
    grid[ay, ax] = "A"
    return "\n".join("".join(row.tolist()) for row in grid)


def _parse_step_multipliers(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        v = float(p)
        if v <= 0:
            raise ValueError(f"Step multipliers must be > 0, got {v}")
        vals.append(v)
    if not vals:
        vals = [1.0]
    return vals


def _fmt_mult(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "1"


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    print(f"Loading checkpoint: {args.checkpoint}")
    full_params = load_maze_ae_params(str(args.checkpoint))
    encoder_params = extract_maze_encoder_params(full_params)
    decoder_params = extract_maze_decoder_params(full_params)

    print(f"Loading dataset for PCA fit: {args.dataset}")
    grids = load_grids(args.dataset, key=args.dataset_key)

    rng = np.random.default_rng(args.seed)
    fit_n = min(args.pca_fit_count, len(grids))
    fit_idx = rng.choice(len(grids), size=fit_n, replace=False)
    fit_grids = grids[fit_idx]

    print(f"Encoding {fit_n} grids to fit PCA...")
    fit_latents = encode_in_batches(encoder_params, fit_grids, batch_size=args.batch_size)
    eigvecs, eigvals = compute_pca(fit_latents)
    if eigvecs.shape[0] < 2:
        raise RuntimeError("Need at least 2 PCA components for two perturbations.")

    latent_dim = fit_latents.shape[1]
    pc1_step = args.perturb_scale * np.sqrt(max(float(eigvals[0]), 1e-8)) * eigvecs[0]
    pc2_step = args.perturb_scale * np.sqrt(max(float(eigvals[1]), 1e-8)) * eigvecs[1]
    step_multipliers = _parse_step_multipliers(args.step_multipliers)

    base_latents = rng.normal(loc=0.0, scale=1.0, size=(args.num_samples, latent_dim)).astype(np.float32)

    variant_specs: list[tuple[str, np.ndarray]] = [
        ("base", np.zeros((latent_dim,), dtype=np.float32))
    ]
    for pc_name, pc_step in (("pc1", pc1_step), ("pc2", pc2_step)):
        for mult in step_multipliers:
            mult_s = _fmt_mult(mult)
            variant_specs.append(
                (f"{pc_name}_plus_x{mult_s}", (mult * pc_step).astype(np.float32))
            )
            if args.opposing_directions:
                variant_specs.append(
                    (f"{pc_name}_minus_x{mult_s}", (-mult * pc_step).astype(np.float32))
                )

    variant_names = [name for name, _ in variant_specs]
    num_variants = len(variant_specs)
    total = args.num_samples * num_variants
    decode_latents = np.zeros((total, latent_dim), dtype=np.float32)

    for i in range(args.num_samples):
        base = base_latents[i]
        for v_idx, (_, offset) in enumerate(variant_specs):
            decode_latents[i * num_variants + v_idx] = base + offset

    level_keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), total)
    levels = decode_maze_latents(
        decoder_params=decoder_params,
        z=jnp.asarray(decode_latents, dtype=jnp.float32),
        rng_keys=level_keys,
        wall_threshold=float(args.wall_threshold),
        temperature=float(args.decode_temperature),
        height=int(args.height),
        width=int(args.width),
    )

    solvable = np.asarray(
        jax.vmap(flood_fill_solvable)(levels.wall_map, levels.agent_pos, levels.goal_pos),
        dtype=bool,
    )
    path_len = np.asarray(
        jax.vmap(bfs_path_length)(levels.wall_map, levels.agent_pos, levels.goal_pos),
        dtype=np.int32,
    )
    wall_count = np.asarray(levels.wall_map.sum(axis=(1, 2)), dtype=np.int32)

    np.savez_compressed(
        args.out_dir / "decoded_mazes.npz",
        latents=decode_latents,
        wall_map=np.asarray(levels.wall_map, dtype=bool),
        goal_pos=np.asarray(levels.goal_pos, dtype=np.int32),
        agent_pos=np.asarray(levels.agent_pos, dtype=np.int32),
        solvable=solvable,
        path_len=path_len,
        wall_count=wall_count,
        pca_eigvals=eigvals,
    )

    fig, axes = plt.subplots(
        args.num_samples,
        num_variants,
        figsize=(2.6 * num_variants, 2.9 * args.num_samples),
        squeeze=False,
    )
    for s in range(args.num_samples):
        for v, name in enumerate(variant_names):
            idx = s * num_variants + v
            ax = axes[s][v]
            wall = np.asarray(levels.wall_map[idx], dtype=bool)
            goal = np.asarray(levels.goal_pos[idx], dtype=np.int32)
            agent = np.asarray(levels.agent_pos[idx], dtype=np.int32)
            ax.imshow(maze_rgb(wall, goal, agent), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"s{s} {name}\nwall={int(wall_count[idx])} "
                f"solv={'Y' if solvable[idx] else 'N'} bfs={int(path_len[idx])}",
                fontsize=8,
            )
    fig.tight_layout()
    png_path = args.out_dir / "decoded_mazes.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    txt_path = args.out_dir / "decoded_mazes.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in range(args.num_samples):
            for v, name in enumerate(variant_names):
                idx = s * num_variants + v
                wall = np.asarray(levels.wall_map[idx], dtype=bool)
                goal = np.asarray(levels.goal_pos[idx], dtype=np.int32)
                agent = np.asarray(levels.agent_pos[idx], dtype=np.int32)
                header = (
                    f"sample={s} variant={name} wall={int(wall_count[idx])} "
                    f"solvable={bool(solvable[idx])} bfs={int(path_len[idx])}"
                )
                ascii_maze = maze_ascii(wall, goal, agent)
                block = f"{header}\n{ascii_maze}\n"
                f.write(block + "\n")
                print(block)

    print("\nSaved outputs:")
    print(f"- {png_path}")
    print(f"- {txt_path}")
    print(f"- {args.out_dir / 'decoded_mazes.npz'}")
    print(
        "PCA variance (top-2): "
        f"pc1={float(eigvals[0]):.4f}, pc2={float(eigvals[1]):.4f}"
    )


if __name__ == "__main__":
    main()
