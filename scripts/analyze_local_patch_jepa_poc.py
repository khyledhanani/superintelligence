#!/usr/bin/env python3
"""Analyze code specialization for a trained local patch JEPA primitive model."""

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
    sample_wall_patch_batch_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--dataset-key", type=str, default=None)
    p.add_argument("--num-batches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
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


def split_train_val(grids: np.ndarray, seed: int, max_train: int, max_val: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(grids))
    val_n = min(max_val, max(1, len(grids) // 10))
    train_n = min(max_train, len(grids) - val_n)
    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    return grids[train_idx], grids[val_idx]


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalized_mutual_info(
    a: np.ndarray,
    b: np.ndarray,
    num_a: int,
) -> float:
    if len(a) == 0:
        return 0.0
    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    contingency = np.zeros((len(a_vals), len(b_vals)), dtype=np.float64)
    for i in range(len(a)):
        contingency[a_inv[i], b_inv[i]] += 1.0
    contingency /= contingency.sum()
    pa = contingency.sum(axis=1, keepdims=True)
    pb = contingency.sum(axis=0, keepdims=True)
    nz = contingency > 0
    mi = float((contingency[nz] * np.log(contingency[nz] / (pa @ pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    if ha <= 1e-12 or hb <= 1e-12:
        return 0.0
    return mi / np.sqrt(ha * hb)


def weighted_top_pattern_prob(
    codes: np.ndarray,
    pattern_ids: np.ndarray,
    num_codes: int,
) -> float:
    total = len(codes)
    if total == 0:
        return 0.0
    acc = 0.0
    for code in range(num_codes):
        sel = codes == code
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

    num_codes = int(train_args["num_codes"])
    patch_size = int(train_args["patch_size"])
    center_size = int(train_args["center_size"])
    model = LocalPatchJepaPoc(
        patch_size=patch_size,
        center_size=center_size,
        embed_dim=int(train_args["embed_dim"]),
        num_codes=num_codes,
        code_dim=int(train_args["code_dim"]),
        gumbel_temperature=float(train_args["gumbel_temperature"]),
    )

    @jax.jit
    def forward(batch: jax.Array):
        return model.apply({"params": params}, batch, deterministic=True)

    host_rng = np.random.default_rng(args.seed)
    count_by_code = np.zeros(num_codes, dtype=np.int64)
    all_codes: list[np.ndarray] = []
    all_raw_ids: list[np.ndarray] = []
    all_can_ids: list[np.ndarray] = []
    all_density: list[np.ndarray] = []

    for _ in range(args.num_batches):
        batch_np = sample_wall_patch_batch_np(
            val_grids,
            host_rng,
            batch_size=args.batch_size,
            patch_size=patch_size,
            augment_dihedral=False,
        )
        outputs = forward(jnp.asarray(batch_np, dtype=jnp.float32))
        codes = np.asarray(outputs["code_indices"], dtype=np.int32)
        raw_ids = center_pattern_ids_np(batch_np, center_size=center_size)
        can_ids = canonical_center_pattern_ids_np(batch_np, center_size=center_size)
        start = (patch_size - center_size) // 2
        end = start + center_size
        density = batch_np[:, start:end, start:end, 0].mean(axis=(1, 2))

        count_by_code += np.bincount(codes, minlength=num_codes)
        all_codes.append(codes)
        all_raw_ids.append(raw_ids)
        all_can_ids.append(can_ids)
        all_density.append(density)

    codes = np.concatenate(all_codes)
    raw_ids = np.concatenate(all_raw_ids)
    can_ids = np.concatenate(all_can_ids)
    density = np.concatenate(all_density)

    probs = count_by_code / max(count_by_code.sum(), 1)
    perplexity = float(np.exp(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))))
    active_codes = int((count_by_code > 0).sum())

    code_rows: list[dict[str, float]] = []
    for code in range(num_codes):
        sel = codes == code
        count = int(sel.sum())
        if count == 0:
            code_rows.append(
                {
                    "code": float(code),
                    "count": 0.0,
                    "prob": 0.0,
                    "mean_center_wall_density": 0.0,
                    "top_raw_pattern_id": 0.0,
                    "top_raw_pattern_prob": 0.0,
                    "top_canonical_pattern_id": 0.0,
                    "top_canonical_pattern_prob": 0.0,
                }
            )
            continue
        raw_vals, raw_counts = np.unique(raw_ids[sel], return_counts=True)
        can_vals, can_counts = np.unique(can_ids[sel], return_counts=True)
        code_rows.append(
            {
                "code": float(code),
                "count": float(count),
                "prob": float(count / len(codes)),
                "mean_center_wall_density": float(density[sel].mean()),
                "top_raw_pattern_id": float(raw_vals[np.argmax(raw_counts)]),
                "top_raw_pattern_prob": float(raw_counts.max() / count),
                "top_canonical_pattern_id": float(can_vals[np.argmax(can_counts)]),
                "top_canonical_pattern_prob": float(can_counts.max() / count),
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset),
        "dataset_key": dataset_key,
        "num_eval_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "num_eval_examples": int(len(codes)),
        "num_codes": num_codes,
        "active_codes": active_codes,
        "assignment_perplexity": perplexity,
        "max_code_prob": float(probs.max(initial=0.0)),
        "weighted_top_raw_pattern_prob": weighted_top_pattern_prob(codes, raw_ids, num_codes),
        "weighted_top_canonical_pattern_prob": weighted_top_pattern_prob(codes, can_ids, num_codes),
        "raw_pattern_nmi": normalized_mutual_info(codes, raw_ids, num_codes),
        "canonical_pattern_nmi": normalized_mutual_info(codes, can_ids, num_codes),
        "center_wall_density_between_group_ratio": float(
            np.var([density[codes == code].mean() for code in range(num_codes) if np.any(codes == code)])
            / max(np.var(density), 1e-12)
        ),
    }

    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(out_dir / "analysis_code_usage.csv", code_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
