#!/usr/bin/env python3
"""Analyze latent embeddings for custom eval mazes across available checkpoints.

Primary model:
- CLUTTR VAE: `vae/model/checkpoint_420000.pkl`

Secondary model:
- Preferred: MazeAE checkpoint with `MazeEncoder_0` params, searched under
  `vae/model_maze_ae/checkpoint*.pkl`
- Fallback: Structured conv maze model
  `vae/model_disentangled/checkpoint_final.pkl`
"""

from __future__ import annotations

import csv
import glob
import json
import math
import pickle
import sys
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from es.cluttr_encoder import encode_levels_to_latents, extract_encoder_params
from es.env_bridge import level_to_cluttr_sequence
from es.maze_ae import (
    encode_maze_levels,
    extract_maze_encoder_params,
    load_maze_ae_params,
    maze_level_to_grid,
)
from es.vae_decoder import load_vae_params
from jaxued.environments.maze.level import Level, prefabs
from vae.train_structured_conv_vae import StructuredConvGridVAE


NEW_LEVEL_NAMES = [
    "NarrowBridge",
    "ForkDeception",
    "PerimeterRun",
    "SpiralPocket",
    "SymmetricCross",
    "ZigZagTunnel",
    "RoomKeyhole",
    "DualLoopChoice",
    "CentralChoke",
    "LongDetour",
    "DeadendFan",
    "OpenFieldBarriers",
    "ParallelCorridors",
    "CornerTrapEscape",
    "SnakeSpine",
]


def bfs_path_length(wall_map: np.ndarray, start_xy: tuple[int, int], goal_xy: tuple[int, int]) -> int:
    """Return shortest 4-neighbor path length; 169 if unreachable."""
    h, w = wall_map.shape
    sx, sy = start_xy
    gx, gy = goal_xy
    q: deque[tuple[int, int, int]] = deque([(sx, sy, 0)])
    seen = {(sx, sy)}

    while q:
        x, y, d = q.popleft()
        if (x, y) == (gx, gy):
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if wall_map[ny, nx]:
                continue
            if (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            q.append((nx, ny, d + 1))
    return h * w


def count_free_neighbors(wall_map: np.ndarray, x: int, y: int) -> int:
    h, w = wall_map.shape
    n = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and (not wall_map[ny, nx]):
            n += 1
    return n


def structural_metrics(wall_map: np.ndarray, agent_xy: tuple[int, int], goal_xy: tuple[int, int]) -> dict[str, float]:
    h, w = wall_map.shape
    free_mask = ~wall_map
    wall_count = int(wall_map.sum())
    free_count = int(free_mask.sum())
    ax, ay = agent_xy
    gx, gy = goal_xy
    manhattan = abs(ax - gx) + abs(ay - gy)
    bfs_len = bfs_path_length(wall_map, agent_xy, goal_xy)
    slack = bfs_len - manhattan if bfs_len < (h * w) else h * w

    branch_points = 0
    dead_ends = 0
    for y in range(h):
        for x in range(w):
            if wall_map[y, x]:
                continue
            deg = count_free_neighbors(wall_map, x, y)
            if deg >= 3:
                branch_points += 1
            if deg == 1:
                dead_ends += 1

    return {
        "wall_count": float(wall_count),
        "free_count": float(free_count),
        "manhattan": float(manhattan),
        "bfs_path_len": float(bfs_len),
        "path_slack": float(slack),
        "branch_points": float(branch_points),
        "dead_ends": float(dead_ends),
    }


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def upper_triangle_values(mat: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(mat.shape[0], k=1)
    return mat[i, j]


def rankdata(x: np.ndarray) -> np.ndarray:
    """Simple rank transform (ties are unlikely for distances)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    if den <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / den)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x0 = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x0, full_matrices=False)
    coords = u[:, :2] * s[:2]
    var = (s * s) / max(1, (x.shape[0] - 1))
    ratio = var / max(1e-12, var.sum())
    return coords, ratio[:2]


def nearest_neighbors(dist: np.ndarray, names: list[str]) -> list[str]:
    out: list[str] = []
    for i in range(dist.shape[0]):
        order = np.argsort(dist[i])
        nn = int(order[1])  # [0] is self
        out.append(names[nn])
    return out


def load_levels(level_names: list[str]) -> tuple[list[Level], list[dict[str, float]]]:
    levels: list[Level] = []
    metrics: list[dict[str, float]] = []
    for name in level_names:
        lvl = Level.from_str(prefabs[name])
        levels.append(lvl)
        wall = np.asarray(lvl.wall_map).astype(bool)
        agent_xy = (int(lvl.agent_pos[0]), int(lvl.agent_pos[1]))
        goal_xy = (int(lvl.goal_pos[0]), int(lvl.goal_pos[1]))
        metrics.append(structural_metrics(wall, agent_xy, goal_xy))
    return levels, metrics


def encode_cluttr_latents(levels: list[Level]) -> tuple[np.ndarray, str]:
    ckpt = ROOT / "vae" / "model" / "checkpoint_420000.pkl"
    full = load_vae_params(str(ckpt))
    encoder_params = extract_encoder_params(full)

    seqs = []
    for lvl in levels:
        seq = level_to_cluttr_sequence(lvl.wall_map, lvl.goal_pos, lvl.agent_pos)
        seqs.append(np.asarray(seq, dtype=np.int32))
    seqs_np = np.stack(seqs, axis=0)

    z = encode_levels_to_latents(encoder_params, jnp.array(seqs_np, dtype=jnp.int32))
    return np.asarray(z, dtype=np.float64), str(ckpt)


def _latest_checkpoint(paths: list[str]) -> str | None:
    if not paths:
        return None
    finals = [p for p in paths if p.endswith("checkpoint_final.pkl")]
    if finals:
        return sorted(finals)[-1]
    return sorted(paths)[-1]


def encode_secondary_latents(levels: list[Level]) -> tuple[np.ndarray, dict[str, str]]:
    # Preferred: MazeAE checkpoint.
    maze_candidates = glob.glob(str(ROOT / "vae" / "model_maze_ae" / "checkpoint*.pkl"))
    maze_ckpt = _latest_checkpoint(maze_candidates)

    wall_maps = jnp.stack([lvl.wall_map for lvl in levels], axis=0)
    goal_pos = jnp.stack([lvl.goal_pos for lvl in levels], axis=0)
    agent_pos = jnp.stack([lvl.agent_pos for lvl in levels], axis=0)
    grids = jax.vmap(maze_level_to_grid)(wall_maps, goal_pos, agent_pos)

    if maze_ckpt is not None:
        full = load_maze_ae_params(maze_ckpt)
        if "MazeEncoder_0" in full:
            enc = extract_maze_encoder_params(full)
            z = encode_maze_levels(enc, grids)
            return np.asarray(z, dtype=np.float64), {
                "model_kind": "maze_ae",
                "checkpoint": maze_ckpt,
                "note": "Using MazeAE checkpoint.",
            }

    # Fallback: structured conv VAE checkpoint.
    fallback_ckpt = ROOT / "vae" / "model_disentangled" / "checkpoint_final.pkl"
    cfg_path = ROOT / "vae" / "disentangled_conv_vae_config.yml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(fallback_ckpt, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    model = StructuredConvGridVAE(
        latent_dim=int(cfg["latent_dim"]),
        n_diff_dims=int(cfg.get("n_diff_dims", 0)),
    )
    z_rng = jax.random.PRNGKey(0)
    _, _, _, mean, _, _ = model.apply({"params": params}, grids, z_rng)
    return np.asarray(mean, dtype=np.float64), {
        "model_kind": "structured_conv_fallback",
        "checkpoint": str(fallback_ckpt),
        "note": (
            "No `vae/model_maze_ae/checkpoint*.pkl` found; "
            "used `vae/model_disentangled/checkpoint_final.pkl` as the second latent model."
        ),
    }


def metric_alignment(latents: np.ndarray, metric_vals: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for metric_name, vals in metric_vals.items():
        best_dim = -1
        best_corr = 0.0
        for d in range(latents.shape[1]):
            corr = pearson(latents[:, d], vals)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_dim = d
        rows.append(
            {
                "metric": metric_name,
                "best_dim": int(best_dim),
                "corr": float(best_corr),
            }
        )
    rows.sort(key=lambda r: abs(float(r["corr"])), reverse=True)
    return rows


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def main() -> None:
    out_dir = ROOT / "analysis" / "latent_eval_levels"
    out_dir.mkdir(parents=True, exist_ok=True)

    levels, metric_rows = load_levels(NEW_LEVEL_NAMES)
    metric_arrays = {k: np.array([m[k] for m in metric_rows], dtype=np.float64) for k in metric_rows[0]}

    z_cluttr, cluttr_ckpt = encode_cluttr_latents(levels)
    z_second, second_meta = encode_secondary_latents(levels)

    d_cluttr = pairwise_l2(z_cluttr)
    d_second = pairwise_l2(z_second)
    dist_rho = spearman(upper_triangle_values(d_cluttr), upper_triangle_values(d_second))

    nn_cluttr = nearest_neighbors(d_cluttr, NEW_LEVEL_NAMES)
    nn_second = nearest_neighbors(d_second, NEW_LEVEL_NAMES)

    iso_cluttr = d_cluttr.mean(axis=1)
    iso_second = d_second.mean(axis=1)
    iso_rank_delta = rankdata(iso_second) - rankdata(iso_cluttr)

    pca_cluttr, pca_var_cluttr = pca2(z_cluttr)
    pca_second, pca_var_second = pca2(z_second)

    align_cluttr = metric_alignment(z_cluttr, metric_arrays)
    align_second = metric_alignment(z_second, metric_arrays)

    # Write per-level table.
    level_table = []
    for i, name in enumerate(NEW_LEVEL_NAMES):
        m = metric_rows[i]
        level_table.append(
            [
                name,
                int(m["wall_count"]),
                int(m["bfs_path_len"]),
                int(m["manhattan"]),
                int(m["path_slack"]),
                int(m["branch_points"]),
                int(m["dead_ends"]),
                float(np.linalg.norm(z_cluttr[i])),
                float(np.linalg.norm(z_second[i])),
                nn_cluttr[i],
                nn_second[i],
                float(iso_rank_delta[i]),
                float(pca_cluttr[i, 0]),
                float(pca_cluttr[i, 1]),
                float(pca_second[i, 0]),
                float(pca_second[i, 1]),
            ]
        )
    write_csv(
        out_dir / "level_latent_summary.csv",
        [
            "level",
            "wall_count",
            "bfs_path_len",
            "manhattan",
            "path_slack",
            "branch_points",
            "dead_ends",
            "cluttr_latent_norm",
            "second_latent_norm",
            "cluttr_nearest",
            "second_nearest",
            "isolation_rank_delta_second_minus_cluttr",
            "cluttr_pc1",
            "cluttr_pc2",
            "second_pc1",
            "second_pc2",
        ],
        level_table,
    )

    # Write pairwise distance matrices.
    dist_header = ["level"] + NEW_LEVEL_NAMES
    dist_rows_cluttr = [[NEW_LEVEL_NAMES[i]] + [float(x) for x in d_cluttr[i]] for i in range(len(NEW_LEVEL_NAMES))]
    dist_rows_second = [[NEW_LEVEL_NAMES[i]] + [float(x) for x in d_second[i]] for i in range(len(NEW_LEVEL_NAMES))]
    write_csv(out_dir / "pairwise_dist_cluttr.csv", dist_header, dist_rows_cluttr)
    write_csv(out_dir / "pairwise_dist_second.csv", dist_header, dist_rows_second)

    summary = {
        "levels": NEW_LEVEL_NAMES,
        "cluttr_checkpoint": cluttr_ckpt,
        "second_model": second_meta,
        "distance_spearman": dist_rho,
        "cluttr_pca_var_ratio": [float(x) for x in pca_var_cluttr],
        "second_pca_var_ratio": [float(x) for x in pca_var_second],
        "metric_alignment_cluttr": align_cluttr,
        "metric_alignment_second": align_second,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Small markdown report.
    top_delta_idx = np.argsort(-np.abs(iso_rank_delta))[:5]
    md = []
    md.append("# Latent Analysis for New Eval Mazes")
    md.append("")
    md.append("## Models")
    md.append(f"- CLUTTR VAE checkpoint: `{cluttr_ckpt}`")
    md.append(f"- Secondary model kind: `{second_meta['model_kind']}`")
    md.append(f"- Secondary checkpoint: `{second_meta['checkpoint']}`")
    md.append(f"- Note: {second_meta['note']}")
    md.append("")
    md.append("## Cross-Model Geometry Agreement")
    md.append(f"- Spearman correlation of pairwise distances: **{dist_rho:.3f}**")
    md.append(
        f"- PCA variance (CLUTTR): PC1={pca_var_cluttr[0]:.3f}, PC2={pca_var_cluttr[1]:.3f}"
    )
    md.append(
        f"- PCA variance (Secondary): PC1={pca_var_second[0]:.3f}, PC2={pca_var_second[1]:.3f}"
    )
    md.append("")
    md.append("## Strongest Structural Alignments")
    md.append("- CLUTTR best metric->latent alignments:")
    for row in align_cluttr[:5]:
        md.append(
            f"  - `{row['metric']}` with latent dim {row['best_dim']}: corr={row['corr']:.3f}"
        )
    md.append("- Secondary best metric->latent alignments:")
    for row in align_second[:5]:
        md.append(
            f"  - `{row['metric']}` with latent dim {row['best_dim']}: corr={row['corr']:.3f}"
        )
    md.append("")
    md.append("## Levels with Biggest Isolation Rank Shift")
    for i in top_delta_idx:
        md.append(
            f"- `{NEW_LEVEL_NAMES[i]}`: delta={iso_rank_delta[i]:+.1f}, "
            f"nn_cluttr=`{nn_cluttr[i]}`, nn_second=`{nn_second[i]}`"
        )
    md.append("")
    md.append("## Artifacts")
    md.append("- `level_latent_summary.csv`")
    md.append("- `pairwise_dist_cluttr.csv`")
    md.append("- `pairwise_dist_second.csv`")
    md.append("- `summary.json`")
    md.append("")
    md.append("Interpretation caution: only 15 levels were analyzed, so dimension-level correlations are directional, not definitive.")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote analysis to: {out_dir}")
    print(f"Distance Spearman correlation: {dist_rho:.3f}")
    print(f"Secondary model: {second_meta['model_kind']} ({second_meta['checkpoint']})")


if __name__ == "__main__":
    main()

