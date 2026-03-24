#!/usr/bin/env python3
"""Probe whether learned local primitive codes can act as editable operators."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vae.local_patch_jepa_poc import (  # noqa: E402
    LocalPatchJepaPoc,
    canonical_center_pattern_ids_np,
    center_pattern_ids_np,
    extract_wall_patches_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--dataset-key", type=str, default=None)
    p.add_argument("--num-mazes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k-sites", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


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


def split_train_val(
    grids: np.ndarray,
    seed: int,
    max_train: int,
    max_val: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(grids))
    val_n = min(max_val, max(1, len(grids) // 10))
    train_n = min(max_train, len(grids) - val_n)
    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    return grids[train_idx], grids[val_idx]


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def enumerate_site_patches(
    grids: np.ndarray,
    patch_size: int,
    num_mazes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_mazes = min(num_mazes, len(grids))
    maze_sel = rng.choice(len(grids), size=num_mazes, replace=False)
    radius = patch_size // 2
    ys: list[int] = []
    xs: list[int] = []
    maze_ids: list[int] = []
    for maze_idx in maze_sel:
        h, w = grids[maze_idx].shape[:2]
        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                maze_ids.append(int(maze_idx))
                ys.append(y)
                xs.append(x)
    maze_ids_arr = np.asarray(maze_ids, dtype=np.int32)
    ys_arr = np.asarray(ys, dtype=np.int32)
    xs_arr = np.asarray(xs, dtype=np.int32)
    patches = extract_wall_patches_np(grids[maze_ids_arr], ys_arr, xs_arr, patch_size=patch_size)
    return patches, maze_ids_arr, ys_arr, xs_arr


def decode_center_patterns_np(center_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center_patches = center_binary[..., None].astype(np.float32)
    raw_ids = center_pattern_ids_np(center_patches, center_size=center_binary.shape[1])
    can_ids = canonical_center_pattern_ids_np(center_patches, center_size=center_binary.shape[1])
    return raw_ids, can_ids


def weighted_top_pattern_prob(code_ids: np.ndarray, pattern_ids: np.ndarray, num_codes: int) -> float:
    total = len(code_ids)
    if total == 0:
        return 0.0
    acc = 0.0
    for code in range(num_codes):
        sel = code_ids == code
        count = int(sel.sum())
        if count == 0:
            continue
        _, counts = np.unique(pattern_ids[sel], return_counts=True)
        acc += float(counts.max())
    return acc / total


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    train_args = ckpt["args"]
    params = ckpt["params"]

    dataset = Path(args.dataset) if args.dataset else Path(train_args["dataset"])
    dataset_key = args.dataset_key or train_args["dataset_key"]
    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = load_grids(dataset, dataset_key)
    _, val_grids = split_train_val(
        grids,
        seed=int(train_args["seed"]),
        max_train=int(train_args["max_train"]),
        max_val=int(train_args["max_val"]),
    )

    patch_size = int(train_args["patch_size"])
    center_size = int(train_args["center_size"])
    num_codes = int(train_args["num_codes"])
    model = LocalPatchJepaPoc(
        patch_size=patch_size,
        center_size=center_size,
        embed_dim=int(train_args["embed_dim"]),
        num_codes=num_codes,
        code_dim=int(train_args["code_dim"]),
        gumbel_temperature=float(train_args["gumbel_temperature"]),
    )

    patches, maze_ids, center_y, center_x = enumerate_site_patches(
        val_grids,
        patch_size=patch_size,
        num_mazes=args.num_mazes,
        seed=args.seed,
    )
    patches_jax = jnp.asarray(patches, dtype=jnp.float32)
    start = (patch_size - center_size) // 2
    end = start + center_size
    original_center = (patches[:, start:end, start:end, 0] > 0.5).astype(np.int32)
    _, original_can_ids = decode_center_patterns_np(original_center)

    @jax.jit
    def decode_for_code(batch: jax.Array, code_index: int):
        code_indices = jnp.full((batch.shape[0],), code_index, dtype=jnp.int32)
        return model.apply(
            {"params": params},
            batch,
            code_indices,
            method=LocalPatchJepaPoc.decode_with_code,
        )

    confidence_matrix = np.zeros((num_codes, len(patches)), dtype=np.float32)
    can_pattern_matrix = np.zeros((num_codes, len(patches)), dtype=np.int32)
    raw_pattern_matrix = np.zeros((num_codes, len(patches)), dtype=np.int32)
    per_code_rows: list[dict[str, float | int]] = []
    top_site_rows: list[dict[str, float | int]] = []

    for code in range(num_codes):
        outputs = decode_for_code(patches_jax, code)
        center_probs = np.asarray(outputs["center_probs"], dtype=np.float32)
        center_conf = np.asarray(outputs["center_confidence"], dtype=np.float32)
        center_binary = (center_probs > 0.5).astype(np.int32)
        raw_ids, can_ids = decode_center_patterns_np(center_binary)

        confidence_matrix[code] = center_conf
        raw_pattern_matrix[code] = raw_ids
        can_pattern_matrix[code] = can_ids

        changed_cells = np.mean(center_binary != original_center, axis=(1, 2))
        changed_pattern = (can_ids != original_can_ids).astype(np.float32)
        unique_ids, unique_counts = np.unique(can_ids, return_counts=True)
        top_idx = int(np.argmax(unique_counts))
        top_can_id = int(unique_ids[top_idx])
        top_can_prob = float(unique_counts[top_idx] / len(can_ids))

        q = max(1, len(center_conf) // 4)
        top_sel = np.argsort(center_conf)[-q:]
        top_q_ids = can_ids[top_sel]
        top_q_unique, top_q_counts = np.unique(top_q_ids, return_counts=True)
        top_q_top_prob = float(top_q_counts.max() / len(top_q_ids))
        per_code_rows.append(
            {
                "code": code,
                "mean_confidence": float(center_conf.mean()),
                "std_confidence": float(center_conf.std()),
                "mean_cell_edit_rate": float(changed_cells.mean()),
                "mean_pattern_change_rate": float(changed_pattern.mean()),
                "top_canonical_pattern_id": top_can_id,
                "top_canonical_pattern_prob_all_sites": top_can_prob,
                "top_canonical_pattern_prob_top_quartile_sites": top_q_top_prob,
                "num_unique_canonical_patterns": int(len(unique_ids)),
            }
        )

        top_k = min(args.top_k_sites, len(center_conf))
        for rank, site_idx in enumerate(np.argsort(center_conf)[-top_k:][::-1], start=1):
            top_site_rows.append(
                {
                    "code": code,
                    "rank": rank,
                    "maze_index": int(maze_ids[site_idx]),
                    "center_y": int(center_y[site_idx]),
                    "center_x": int(center_x[site_idx]),
                    "confidence": float(center_conf[site_idx]),
                    "original_canonical_pattern_id": int(original_can_ids[site_idx]),
                    "decoded_canonical_pattern_id": int(can_ids[site_idx]),
                    "cell_edit_rate": float(changed_cells[site_idx]),
                }
            )

    best_code_per_site = np.argmax(confidence_matrix, axis=0)
    best_code_counts = np.bincount(best_code_per_site, minlength=num_codes)
    best_code_probs = best_code_counts / max(best_code_counts.sum(), 1)
    best_code_perplexity = float(
        np.exp(-np.sum(best_code_probs[best_code_probs > 0] * np.log(best_code_probs[best_code_probs > 0])))
    )
    unique_pattern_count_per_site = np.array(
        [len(np.unique(can_pattern_matrix[:, i])) for i in range(len(patches))],
        dtype=np.float32,
    )
    best_pattern_ids = can_pattern_matrix[best_code_per_site, np.arange(len(patches))]
    best_confidence = confidence_matrix[best_code_per_site, np.arange(len(patches))]

    best_code_rows: list[dict[str, float | int]] = []
    for code in range(num_codes):
        sel = best_code_per_site == code
        count = int(sel.sum())
        if count == 0:
            best_code_rows.append(
                {
                    "code": code,
                    "count": 0,
                    "prob": 0.0,
                    "mean_best_confidence": 0.0,
                    "top_canonical_pattern_id": 0,
                    "top_canonical_pattern_prob": 0.0,
                }
            )
            continue
        ids, counts = np.unique(best_pattern_ids[sel], return_counts=True)
        best_code_rows.append(
            {
                "code": code,
                "count": count,
                "prob": float(count / len(best_code_per_site)),
                "mean_best_confidence": float(best_confidence[sel].mean()),
                "top_canonical_pattern_id": int(ids[np.argmax(counts)]),
                "top_canonical_pattern_prob": float(counts.max() / count),
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset),
        "dataset_key": dataset_key,
        "num_codes": num_codes,
        "num_eval_mazes": int(args.num_mazes),
        "num_eval_sites": int(len(patches)),
        "mean_unique_canonical_patterns_per_site": float(unique_pattern_count_per_site.mean()),
        "fraction_sites_with_multiple_patterns": float((unique_pattern_count_per_site > 1).mean()),
        "mean_confidence_all_code_site_pairs": float(confidence_matrix.mean()),
        "mean_best_site_confidence": float(best_confidence.mean()),
        "best_code_active": int((best_code_counts > 0).sum()),
        "best_code_perplexity": best_code_perplexity,
        "best_code_weighted_top_canonical_pattern_prob": weighted_top_pattern_prob(
            best_code_per_site,
            best_pattern_ids,
            num_codes,
        ),
        "mean_code_pattern_change_rate": float(
            np.mean([float(row["mean_pattern_change_rate"]) for row in per_code_rows])
        ),
        "mean_code_top_canonical_pattern_prob_all_sites": float(
            np.mean([float(row["top_canonical_pattern_prob_all_sites"]) for row in per_code_rows])
        ),
        "mean_code_top_canonical_pattern_prob_top_quartile_sites": float(
            np.mean([float(row["top_canonical_pattern_prob_top_quartile_sites"]) for row in per_code_rows])
        ),
    }

    with open(out_dir / "editability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(out_dir / "editability_per_code.csv", per_code_rows)
    write_csv(out_dir / "editability_best_code.csv", best_code_rows)
    write_csv(out_dir / "editability_top_sites.csv", top_site_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote editability probe to {out_dir}")


if __name__ == "__main__":
    main()
