#!/usr/bin/env python3
"""Latent-space comparison on the vae_og dataset.

Compares:
1) CLUTTR VAE encoder from `vae/model/checkpoint_420000.pkl`
2) MazeAE encoder from `vae/model_maze_ae/checkpoint_final.pkl`

Dataset:
- `datasets_new_train_200k_envs.npy`
- `datasets_new_val_20k_envs.npy`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/Users/khyledhanani/Documents/superintelligence/vae/datasets/vae_og"),
        help="Directory containing datasets_new_{train,val}_... files.",
    )
    parser.add_argument("--train_samples", type=int, default=15000)
    parser.add_argument("--val_samples", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pair_samples", type=int, default=200000)
    parser.add_argument("--knn_anchors", type=int, default=500)
    parser.add_argument("--knn_pool", type=int, default=5000)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--bfs_subset", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "analysis" / "latent_vae_og",
    )
    return parser.parse_args()


def import_es_modules():
    """Import required ES modules from an available source tree.

    Prefers this workspace; falls back to the sibling repo where source .py
    files may exist.
    """
    candidate_roots = [
        ROOT,
        Path("/Users/khyledhanani/Documents/superintelligence"),
    ]

    last_err = None
    for base in candidate_roots:
        es_file = base / "es" / "cluttr_encoder.py"
        if not es_file.exists():
            continue
        sys.path.insert(0, str(base))
        sys.path.insert(0, str(base / "src"))
        try:
            cluttr = importlib.import_module("es.cluttr_encoder")
            env_bridge = importlib.import_module("es.env_bridge")
            maze_ae = importlib.import_module("es.maze_ae")
            vae_decoder = importlib.import_module("es.vae_decoder")
            return cluttr, env_bridge, maze_ae, vae_decoder, base
        except Exception as e:  # pragma: no cover - fallback path handling
            last_err = e
            continue

    raise ImportError(
        "Unable to import required ES modules from known roots."
        f" Last error: {last_err}"
    )


def rankdata(x: np.ndarray) -> np.ndarray:
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
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    coords = u[:, :2] * s[:2]
    var = (s * s) / max(1, (x.shape[0] - 1))
    ratio = var / max(1e-12, var.sum())
    return coords, ratio[:2]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def sample_sequences(
    train_path: Path, val_path: Path, train_samples: int, val_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = np.load(train_path, mmap_mode="r")
    val = np.load(val_path, mmap_mode="r")

    n_train = min(train_samples, train.shape[0])
    n_val = min(val_samples, val.shape[0])

    train_idx = rng.choice(train.shape[0], size=n_train, replace=False)
    val_idx = rng.choice(val.shape[0], size=n_val, replace=False)

    seq_train = np.asarray(train[train_idx], dtype=np.int32)
    seq_val = np.asarray(val[val_idx], dtype=np.int32)

    seqs = np.concatenate([seq_train, seq_val], axis=0)
    splits = np.array(["train"] * n_train + ["val"] * n_val)
    source_idx = np.concatenate([train_idx, val_idx], axis=0)
    return seqs, splits, source_idx


def sequences_to_components(
    seqs: np.ndarray, h: int = 13, w: int = 13
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    grids = np.zeros((n, h, w, 3), dtype=np.float32)
    grids[..., 0] = walls.astype(np.float32)
    grids[np.arange(n), goal_pos[:, 1], goal_pos[:, 0], 1] = 1.0
    grids[np.arange(n), agent_pos[:, 1], agent_pos[:, 0], 2] = 1.0

    return walls, goal_pos, agent_pos, grids


def structural_metrics(
    walls: np.ndarray, goal_pos: np.ndarray, agent_pos: np.ndarray
) -> dict[str, np.ndarray]:
    n, h, w = walls.shape
    free = ~walls

    neigh = np.zeros_like(walls, dtype=np.int16)
    neigh[:, 1:, :] += free[:, :-1, :]
    neigh[:, :-1, :] += free[:, 1:, :]
    neigh[:, :, 1:] += free[:, :, :-1]
    neigh[:, :, :-1] += free[:, :, 1:]

    wall_count = walls.reshape(n, -1).sum(axis=1).astype(np.float64)
    free_count = free.reshape(n, -1).sum(axis=1).astype(np.float64)
    branch_points = ((free) & (neigh >= 3)).reshape(n, -1).sum(axis=1).astype(np.float64)
    dead_ends = ((free) & (neigh == 1)).reshape(n, -1).sum(axis=1).astype(np.float64)

    manhattan = (
        np.abs(goal_pos[:, 0].astype(np.int32) - agent_pos[:, 0].astype(np.int32))
        + np.abs(goal_pos[:, 1].astype(np.int32) - agent_pos[:, 1].astype(np.int32))
    ).astype(np.float64)

    return {
        "wall_count": wall_count,
        "free_count": free_count,
        "branch_points": branch_points,
        "dead_ends": dead_ends,
        "manhattan": manhattan,
    }


def encode_in_batches(
    x: np.ndarray, batch_size: int, encode_fn
) -> np.ndarray:
    outs = []
    for i in range(0, x.shape[0], batch_size):
        xb = jnp.asarray(x[i : i + batch_size])
        z = encode_fn(xb)
        outs.append(np.asarray(z, dtype=np.float64))
    return np.concatenate(outs, axis=0)


def sample_pair_distances(
    z1: np.ndarray, z2: np.ndarray, pair_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = z1.shape[0]
    i = rng.integers(0, n, size=pair_samples)
    j = rng.integers(0, n, size=pair_samples)
    mask = i != j
    i = i[mask]
    j = j[mask]

    d1 = np.linalg.norm(z1[i] - z1[j], axis=1)
    d2 = np.linalg.norm(z2[i] - z2[j], axis=1)
    return d1, d2


def knn_overlap(
    z1: np.ndarray,
    z2: np.ndarray,
    anchors: int,
    pool_size: int,
    k: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = z1.shape[0]
    a = min(anchors, n)
    p = min(pool_size, n)
    anchor_idx = rng.choice(n, size=a, replace=False)
    pool_idx = rng.choice(n, size=p, replace=False)

    a1, p1 = z1[anchor_idx], z1[pool_idx]
    a2, p2 = z2[anchor_idx], z2[pool_idx]

    d1 = np.sum((a1[:, None, :] - p1[None, :, :]) ** 2, axis=-1)
    d2 = np.sum((a2[:, None, :] - p2[None, :, :]) ** 2, axis=-1)

    overlap_sum = 0.0
    nn1_same = 0
    for r in range(a):
        order1 = np.argsort(d1[r])
        order2 = np.argsort(d2[r])

        neigh1 = []
        neigh2 = []
        for idx in order1:
            if pool_idx[idx] != anchor_idx[r]:
                neigh1.append(pool_idx[idx])
            if len(neigh1) == k:
                break
        for idx in order2:
            if pool_idx[idx] != anchor_idx[r]:
                neigh2.append(pool_idx[idx])
            if len(neigh2) == k:
                break

        set1 = set(neigh1)
        set2 = set(neigh2)
        overlap_sum += len(set1 & set2) / max(1, k)
        if neigh1 and neigh2 and neigh1[0] == neigh2[0]:
            nn1_same += 1

    return overlap_sum / max(1, a), nn1_same / max(1, a)


def metric_alignment(latents: np.ndarray, metrics: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for name, vals in metrics.items():
        best_dim = -1
        best_corr = 0.0
        for d in range(latents.shape[1]):
            corr = pearson(latents[:, d], vals)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_dim = d
        rows.append({"metric": name, "best_dim": int(best_dim), "corr": float(best_corr)})
    rows.sort(key=lambda x: abs(float(x["corr"])), reverse=True)
    return rows


def compute_bfs_subset(
    walls: np.ndarray,
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    manhattan: np.ndarray,
    subset_n: int,
    seed: int,
    bfs_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = walls.shape[0]
    k = min(subset_n, n)
    idx = rng.choice(n, size=k, replace=False)

    lengths = jax.vmap(bfs_fn)(
        jnp.asarray(walls[idx], dtype=jnp.bool_),
        jnp.asarray(agent_pos[idx], dtype=jnp.uint32),
        jnp.asarray(goal_pos[idx], dtype=jnp.uint32),
    )
    lengths_np = np.asarray(lengths, dtype=np.float64)
    slack = lengths_np - manhattan[idx]
    return idx, lengths_np, slack


def summarize_array(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cluttr_mod, env_bridge_mod, maze_ae_mod, vae_decoder_mod, es_root = import_es_modules()
    encode_levels_to_latents = cluttr_mod.encode_levels_to_latents
    extract_encoder_params = cluttr_mod.extract_encoder_params
    bfs_fn = env_bridge_mod.bfs_path_length
    encode_maze_levels = maze_ae_mod.encode_maze_levels
    extract_maze_encoder_params = maze_ae_mod.extract_maze_encoder_params
    load_maze_ae_params = maze_ae_mod.load_maze_ae_params
    load_vae_params = vae_decoder_mod.load_vae_params

    train_path = args.dataset_root / "datasets_new_train_200k_envs.npy"
    val_path = args.dataset_root / "datasets_new_val_20k_envs.npy"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected dataset files missing under {args.dataset_root}."
        )

    cluttr_candidates = [
        ROOT / "vae" / "model" / "checkpoint_420000.pkl",
        es_root / "vae" / "model" / "checkpoint_420000.pkl",
    ]
    maze_candidates = [
        ROOT / "vae" / "model_maze_ae" / "checkpoint_final.pkl",
        es_root / "vae" / "model_maze_ae" / "checkpoint_final.pkl",
    ]
    cluttr_ckpt = next((p for p in cluttr_candidates if p.exists()), None)
    maze_ckpt = next((p for p in maze_candidates if p.exists()), None)
    if cluttr_ckpt is None:
        raise FileNotFoundError(
            "CLUTTR checkpoint_420000 not found in known locations: "
            + ", ".join(str(p) for p in cluttr_candidates)
        )
    if maze_ckpt is None:
        raise FileNotFoundError(
            "MazeAE checkpoint_final not found in known locations: "
            + ", ".join(str(p) for p in maze_candidates)
        )

    seqs, splits, source_idx = sample_sequences(
        train_path, val_path, args.train_samples, args.val_samples, args.seed
    )
    walls, goal_pos, agent_pos, grids = sequences_to_components(seqs)
    metrics = structural_metrics(walls, goal_pos, agent_pos)

    # Encode CLUTTR latent.
    full_cluttr = load_vae_params(str(cluttr_ckpt))
    enc_cluttr = extract_encoder_params(full_cluttr)
    cluttr_fn = jax.jit(lambda x: encode_levels_to_latents(enc_cluttr, x))
    z_cluttr = encode_in_batches(seqs.astype(np.int32), args.batch_size, cluttr_fn)

    # Encode MazeAE latent.
    full_maze = load_maze_ae_params(str(maze_ckpt))
    enc_maze = extract_maze_encoder_params(full_maze)
    maze_fn = jax.jit(lambda x: encode_maze_levels(enc_maze, x))
    z_maze = encode_in_batches(grids.astype(np.float32), args.batch_size, maze_fn)

    # Global geometry comparisons.
    d1, d2 = sample_pair_distances(z_cluttr, z_maze, args.pair_samples, args.seed + 1)
    pair_pearson = pearson(d1, d2)
    pair_spearman = spearman(d1, d2)

    knn_overlap_k, nn1_agreement = knn_overlap(
        z_cluttr,
        z_maze,
        anchors=args.knn_anchors,
        pool_size=args.knn_pool,
        k=args.knn_k,
        seed=args.seed + 2,
    )

    pca_cluttr, pca_var_cluttr = pca2(z_cluttr)
    pca_maze, pca_var_maze = pca2(z_maze)

    align_cluttr = metric_alignment(z_cluttr, metrics)
    align_maze = metric_alignment(z_maze, metrics)

    bfs_idx, bfs_len, path_slack = compute_bfs_subset(
        walls,
        agent_pos,
        goal_pos,
        metrics["manhattan"],
        args.bfs_subset,
        args.seed + 3,
        bfs_fn,
    )
    bfs_metrics = {
        "bfs_path_len_subset": bfs_len,
        "path_slack_subset": path_slack,
    }
    bfs_align_cluttr = metric_alignment(z_cluttr[bfs_idx], bfs_metrics)
    bfs_align_maze = metric_alignment(z_maze[bfs_idx], bfs_metrics)

    # Outlier shift examples by centroid-distance rank delta.
    dc = np.linalg.norm(z_cluttr - z_cluttr.mean(axis=0, keepdims=True), axis=1)
    dm = np.linalg.norm(z_maze - z_maze.mean(axis=0, keepdims=True), axis=1)
    rank_delta = rankdata(dm) - rankdata(dc)
    pos_idx = np.argsort(-rank_delta)[:10]
    neg_idx = np.argsort(rank_delta)[:10]

    exemplar_rows = []
    for idx in list(pos_idx) + list(neg_idx):
        exemplar_rows.append(
            [
                int(idx),
                splits[idx],
                int(source_idx[idx]),
                float(rank_delta[idx]),
                float(metrics["wall_count"][idx]),
                float(metrics["manhattan"][idx]),
                float(metrics["branch_points"][idx]),
                float(metrics["dead_ends"][idx]),
                float(np.linalg.norm(z_cluttr[idx])),
                float(np.linalg.norm(z_maze[idx])),
                float(pca_cluttr[idx, 0]),
                float(pca_cluttr[idx, 1]),
                float(pca_maze[idx, 0]),
                float(pca_maze[idx, 1]),
            ]
        )
    write_csv(
        out_dir / "rank_shift_exemplars.csv",
        [
            "sample_idx",
            "split",
            "source_idx",
            "rank_delta_maze_minus_cluttr",
            "wall_count",
            "manhattan",
            "branch_points",
            "dead_ends",
            "cluttr_norm",
            "maze_norm",
            "cluttr_pc1",
            "cluttr_pc2",
            "maze_pc1",
            "maze_pc2",
        ],
        exemplar_rows,
    )

    # Aggregate metric CSV.
    summary_rows = []
    for name, vals in metrics.items():
        s = summarize_array(vals)
        summary_rows.append([name, s["mean"], s["std"], s["p05"], s["p50"], s["p95"]])
    s_bfs = summarize_array(bfs_len)
    summary_rows.append(
        ["bfs_path_len_subset", s_bfs["mean"], s_bfs["std"], s_bfs["p05"], s_bfs["p50"], s_bfs["p95"]]
    )
    s_slack = summarize_array(path_slack)
    summary_rows.append(
        ["path_slack_subset", s_slack["mean"], s_slack["std"], s_slack["p05"], s_slack["p50"], s_slack["p95"]]
    )
    write_csv(
        out_dir / "dataset_metric_summary.csv",
        ["metric", "mean", "std", "p05", "p50", "p95"],
        summary_rows,
    )

    latent_summary_rows = [
        [
            "cluttr_norm",
            *summarize_array(np.linalg.norm(z_cluttr, axis=1)).values(),
        ],
        [
            "maze_norm",
            *summarize_array(np.linalg.norm(z_maze, axis=1)).values(),
        ],
    ]
    write_csv(
        out_dir / "latent_norm_summary.csv",
        ["metric", "mean", "std", "p05", "p50", "p95"],
        latent_summary_rows,
    )

    result = {
        "dataset_root": str(args.dataset_root),
        "es_module_root": str(es_root),
        "sample_counts": {
            "train": int(np.sum(splits == "train")),
            "val": int(np.sum(splits == "val")),
            "total": int(len(splits)),
        },
        "checkpoints": {
            "cluttr": str(cluttr_ckpt),
            "maze_ae": str(maze_ckpt),
        },
        "cross_model_geometry": {
            "pair_distance_pearson": float(pair_pearson),
            "pair_distance_spearman": float(pair_spearman),
            "knn_overlap_at_k": float(knn_overlap_k),
            "top1_neighbor_agreement": float(nn1_agreement),
            "pair_samples": int(len(d1)),
        },
        "pca_variance_ratio": {
            "cluttr_pc1": float(pca_var_cluttr[0]),
            "cluttr_pc2": float(pca_var_cluttr[1]),
            "maze_pc1": float(pca_var_maze[0]),
            "maze_pc2": float(pca_var_maze[1]),
        },
        "metric_alignment": {
            "cluttr_all": align_cluttr,
            "maze_all": align_maze,
            "cluttr_bfs_subset": bfs_align_cluttr,
            "maze_bfs_subset": bfs_align_maze,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md = []
    md.append("# Latent Analysis on vae_og Dataset")
    md.append("")
    md.append("## Setup")
    md.append(f"- Dataset root: `{args.dataset_root}`")
    md.append(f"- ES module root: `{es_root}`")
    md.append(
        f"- Sampled {result['sample_counts']['total']} examples "
        f"({result['sample_counts']['train']} train, {result['sample_counts']['val']} val)"
    )
    md.append(f"- CLUTTR checkpoint: `{cluttr_ckpt}`")
    md.append(f"- MazeAE checkpoint: `{maze_ckpt}`")
    md.append("")
    md.append("## Cross-Model Geometry")
    md.append(
        f"- Pairwise distance Pearson: **{pair_pearson:.3f}** "
        f"(from {result['cross_model_geometry']['pair_samples']} sampled pairs)"
    )
    md.append(f"- Pairwise distance Spearman: **{pair_spearman:.3f}**")
    md.append(f"- kNN overlap@{args.knn_k}: **{knn_overlap_k:.3f}**")
    md.append(f"- Top-1 neighbor agreement: **{nn1_agreement:.3f}**")
    md.append("")
    md.append("## PCA Variance")
    md.append(
        f"- CLUTTR: PC1={pca_var_cluttr[0]:.3f}, PC2={pca_var_cluttr[1]:.3f}"
    )
    md.append(
        f"- MazeAE: PC1={pca_var_maze[0]:.3f}, PC2={pca_var_maze[1]:.3f}"
    )
    md.append("")
    md.append("## Strongest Structural Alignments")
    md.append("- CLUTTR (all-sample metrics):")
    for row in align_cluttr[:6]:
        md.append(
            f"  - `{row['metric']}` best dim {row['best_dim']} corr={row['corr']:.3f}"
        )
    md.append("- MazeAE (all-sample metrics):")
    for row in align_maze[:6]:
        md.append(
            f"  - `{row['metric']}` best dim {row['best_dim']} corr={row['corr']:.3f}"
        )
    md.append("- CLUTTR (BFS subset metrics):")
    for row in bfs_align_cluttr:
        md.append(
            f"  - `{row['metric']}` best dim {row['best_dim']} corr={row['corr']:.3f}"
        )
    md.append("- MazeAE (BFS subset metrics):")
    for row in bfs_align_maze:
        md.append(
            f"  - `{row['metric']}` best dim {row['best_dim']} corr={row['corr']:.3f}"
        )
    md.append("")
    md.append("## Artifacts")
    md.append("- `summary.json`")
    md.append("- `dataset_metric_summary.csv`")
    md.append("- `latent_norm_summary.csv`")
    md.append("- `rank_shift_exemplars.csv`")
    md.append("")
    md.append(
        "Interpretation note: correlations identify dominant axes in the chosen sample, not causal factors."
    )

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote analysis to: {out_dir}")
    print(
        f"Pairwise distance agreement -> Pearson: {pair_pearson:.3f}, "
        f"Spearman: {pair_spearman:.3f}"
    )
    print(
        f"kNN overlap@{args.knn_k}: {knn_overlap_k:.3f}, "
        f"top-1 agreement: {nn1_agreement:.3f}"
    )


if __name__ == "__main__":
    main()
