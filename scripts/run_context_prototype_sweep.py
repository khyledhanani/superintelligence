#!/usr/bin/env python3
"""Build and evaluate explicit local prototype libraries from static maze patches."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vae.local_patch_jepa_poc import sample_wall_patch_batch_np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "vae" / "datasets" / "vae_og" / "train_200k_grids.npz",
    )
    p.add_argument("--dataset-key", type=str, default="grids")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "context_prototype_sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--center-size", type=int, default=5)
    p.add_argument("--num-train-patches", type=int, default=50000)
    p.add_argument("--num-val-patches", type=int, default=20000)
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-val", type=int, default=10000)
    p.add_argument("--num-iters", type=int, default=15)
    p.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128])
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


def split_patch(patches: np.ndarray, center_size: int) -> tuple[np.ndarray, np.ndarray]:
    start = (patches.shape[1] - center_size) // 2
    end = start + center_size
    center = (patches[:, start:end, start:end, 0] > 0.5).astype(np.float32)
    mask = np.ones((patches.shape[1], patches.shape[2]), dtype=bool)
    mask[start:end, start:end] = False
    context = (patches[:, :, :, 0] > 0.5).astype(np.float32)[:, mask]
    return context, center.reshape(len(patches), -1)


def bit_ids(binary_rows: np.ndarray) -> np.ndarray:
    nbits = binary_rows.shape[1]
    weights = (1 << np.arange(nbits, dtype=np.int64))[None, :]
    return (binary_rows.astype(np.int64) * weights).sum(axis=1)


def pairwise_sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = (x * x).sum(axis=1, keepdims=True)
    y_norm = (y * y).sum(axis=1, keepdims=True).T
    return x_norm + y_norm - 2.0 * x @ y.T


def run_kmeans_binary(
    data: np.ndarray,
    k: int,
    num_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + k)
    centers = data[rng.choice(len(data), size=k, replace=False)].copy()
    assign = np.zeros((len(data),), dtype=np.int32)
    for _ in range(num_iters):
        dists = pairwise_sqdist(data, centers)
        assign = np.argmin(dists, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for idx in range(k):
            sel = assign == idx
            if not np.any(sel):
                new_centers[idx] = data[rng.integers(0, len(data))]
            else:
                new_centers[idx] = (data[sel].mean(axis=0) >= 0.5).astype(np.float32)
        if np.array_equal(new_centers, centers):
            break
        centers = new_centers
    dists = pairwise_sqdist(data, centers)
    assign = np.argmin(dists, axis=1).astype(np.int32)
    return centers, assign


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_sparse_context_probs(
    context_ids: np.ndarray,
    code_assign: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids, inverse = np.unique(context_ids, return_inverse=True)
    counts = np.zeros((len(unique_ids), k), dtype=np.int32)
    np.add.at(counts, (inverse, code_assign), 1)
    probs = counts.astype(np.float32)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1.0)
    return unique_ids.astype(np.int64), probs


def evaluate_library(
    k: int,
    prototypes: np.ndarray,
    train_assign: np.ndarray,
    val_center: np.ndarray,
    val_context_ids: np.ndarray,
    train_context_ids: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    num_contexts = int(train_context_ids.max()) + 1 if len(train_context_ids) else 0
    counts = np.zeros((num_contexts, k), dtype=np.int32)
    np.add.at(counts, (train_context_ids, train_assign), 1)
    global_counts = np.bincount(train_assign, minlength=k).astype(np.float32)
    global_probs = global_counts / max(global_counts.sum(), 1.0)

    val_counts = counts[val_context_ids]
    seen_mask = val_counts.sum(axis=1) > 0
    pred_probs = np.where(
        seen_mask[:, None],
        val_counts / np.maximum(val_counts.sum(axis=1, keepdims=True), 1),
        global_probs[None, :],
    )
    pred_codes = pred_probs.argmax(axis=1).astype(np.int32)
    pred_center = prototypes[pred_codes]

    val_assign = np.argmin(pairwise_sqdist(val_center, prototypes), axis=1).astype(np.int32)
    best_counts = np.bincount(pred_codes, minlength=k).astype(np.float32)
    best_probs = best_counts / max(best_counts.sum(), 1.0)
    perplexity = float(np.exp(-np.sum(best_probs[best_probs > 0] * np.log(best_probs[best_probs > 0]))))
    prototype_ids = bit_ids(prototypes.astype(np.int64))

    summary = {
        "k": int(k),
        "context_seen_rate": float(seen_mask.mean()),
        "best_code_active": int((best_counts > 0).sum()),
        "best_code_perplexity": perplexity,
        "cluster_top1_accuracy": float((pred_codes == val_assign).mean()),
        "prototype_center_cell_acc": float((pred_center == val_center).mean()),
        "prototype_center_exact_match": float(np.all(pred_center == val_center, axis=1).mean()),
        "num_unique_prototypes": int(len(np.unique(prototype_ids))),
    }
    artifacts = {
        "counts": counts,
        "global_probs": global_probs,
        "prototype_ids": prototype_ids,
        "pred_codes": pred_codes,
        "val_assign": val_assign,
        "prototypes": prototypes,
    }
    return summary, artifacts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grids = load_grids(args.dataset, args.dataset_key)
    train_grids, val_grids = split_train_val(grids, args.seed, args.max_train, args.max_val)

    rng = np.random.default_rng(args.seed)
    train_patches = sample_wall_patch_batch_np(
        train_grids,
        rng,
        batch_size=args.num_train_patches,
        patch_size=args.patch_size,
        augment_dihedral=False,
    )
    val_patches = sample_wall_patch_batch_np(
        val_grids,
        rng,
        batch_size=args.num_val_patches,
        patch_size=args.patch_size,
        augment_dihedral=False,
    )
    train_context, train_center = split_patch(train_patches, center_size=args.center_size)
    val_context, val_center = split_patch(val_patches, center_size=args.center_size)
    train_context_ids = bit_ids(train_context.astype(np.int64))
    val_context_ids = bit_ids(val_context.astype(np.int64))

    summaries: list[dict[str, float | int]] = []
    best_payload: dict[str, object] | None = None
    best_score = -1.0
    for k in args.ks:
        prototypes, train_assign = run_kmeans_binary(
            train_center,
            k=k,
            num_iters=args.num_iters,
            seed=args.seed,
        )
        summary, artifacts = evaluate_library(
            k=k,
            prototypes=prototypes,
            train_assign=train_assign,
            val_center=val_center,
            val_context_ids=val_context_ids,
            train_context_ids=train_context_ids,
        )
        summaries.append(summary)
        if float(summary["prototype_center_exact_match"]) > best_score:
            best_score = float(summary["prototype_center_exact_match"])
            best_payload = {
                "summary": summary,
                "artifacts": artifacts,
                "train_assign": train_assign,
            }
        print(json.dumps(summary, indent=2))

    if best_payload is None:
        raise RuntimeError("No prototype library results produced.")

    best_summary = best_payload["summary"]
    best_artifacts = best_payload["artifacts"]
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(args.dataset),
                "dataset_key": args.dataset_key,
                "patch_size": int(args.patch_size),
                "center_size": int(args.center_size),
                "num_train_patches": int(args.num_train_patches),
                "num_val_patches": int(args.num_val_patches),
                "ks": [int(k) for k in args.ks],
                "results": summaries,
                "best_by_exact_match": best_summary,
            },
            f,
            indent=2,
        )
    write_csv(args.out_dir / "results.csv", summaries)
    prototype_rows = []
    for code, proto in enumerate(best_artifacts["prototypes"]):
        prototype_rows.append(
            {
                "code": int(code),
                "prototype_id": int(best_artifacts["prototype_ids"][code]),
                "wall_fraction": float(proto.mean()),
            }
        )
    write_csv(args.out_dir / "best_prototypes.csv", prototype_rows)
    with open(args.out_dir / "best_model.pkl", "wb") as f:
        pickle.dump(
            {
                "config": vars(args),
                "best_summary": best_summary,
                "best_artifacts": best_artifacts,
            },
            f,
        )
    sparse_context_ids, sparse_context_code_probs = build_sparse_context_probs(
        train_context_ids,
        best_payload["train_assign"],
        int(best_summary["k"]),
    )
    np.savez_compressed(
        args.out_dir / "best_library.npz",
        patch_size=np.array(args.patch_size, dtype=np.int32),
        center_size=np.array(args.center_size, dtype=np.int32),
        num_codes=np.array(best_summary["k"], dtype=np.int32),
        context_ids=sparse_context_ids,
        context_code_probs=sparse_context_code_probs.astype(np.float32),
        global_probs=best_artifacts["global_probs"].astype(np.float32),
        prototypes=best_artifacts["prototypes"].astype(np.float32),
        prototype_ids=best_artifacts["prototype_ids"].astype(np.int64),
    )
    print(f"Wrote prototype sweep to {args.out_dir}")


if __name__ == "__main__":
    main()
