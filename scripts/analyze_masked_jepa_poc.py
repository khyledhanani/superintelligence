#!/usr/bin/env python3
"""Analyze code usage and specialization for a trained masked-maze JEPA POC."""

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

from vae.masked_jepa_poc import MaskedMazeJepaPoc, sample_masks  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint_final.pkl written by train_masked_jepa_poc.py",
    )
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--dataset-key", type=str, default=None)
    p.add_argument("--num-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
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


def _between_group_ratio(values: np.ndarray, codes: np.ndarray, num_codes: int) -> float:
    total_var = float(np.var(values))
    if total_var <= 1e-12:
        return 0.0
    mean_all = float(np.mean(values))
    n = len(values)
    between = 0.0
    for code in range(num_codes):
        sel = codes == code
        count = int(sel.sum())
        if count == 0:
            continue
        diff = float(np.mean(values[sel])) - mean_all
        between += count * diff * diff
    return between / max(n * total_var, 1e-12)


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
    embed_dim = int(train_args["embed_dim"])
    code_dim = int(train_args["code_dim"])
    height = val_grids.shape[1]
    width = val_grids.shape[2]
    model = MaskedMazeJepaPoc(
        height=height,
        width=width,
        embed_dim=embed_dim,
        num_codes=num_codes,
        code_dim=code_dim,
        gumbel_temperature=float(train_args["gumbel_temperature"]),
    )

    @jax.jit
    def forward(batch: jax.Array, masks: jax.Array):
        return model.apply({"params": params}, batch, masks, deterministic=True)

    host_rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed)

    count_by_code = np.zeros(num_codes, dtype=np.int64)
    all_codes: list[np.ndarray] = []
    all_wall_density: list[np.ndarray] = []
    all_goal_in_mask: list[np.ndarray] = []
    all_agent_in_mask: list[np.ndarray] = []
    all_mask_area: list[np.ndarray] = []
    all_center_y: list[np.ndarray] = []
    all_center_x: list[np.ndarray] = []

    for _ in range(args.num_batches):
        replace = len(val_grids) < args.batch_size
        idx = host_rng.choice(len(val_grids), size=args.batch_size, replace=replace)
        batch_np = val_grids[idx]
        jax_rng, mask_rng = jax.random.split(jax_rng)
        masks = sample_masks(
            mask_rng,
            batch_size=args.batch_size,
            height=height,
            width=width,
            mask_mode=str(train_args.get("mask_mode", "rect")),
            min_size=int(train_args["mask_min_size"]),
            max_size=int(train_args["mask_max_size"]),
            block_height=int(train_args.get("mask_block_height", 4)),
            block_width=int(train_args.get("mask_block_width", 4)),
            num_regions=int(train_args.get("mask_num_regions", 2)),
        )
        out = forward(jnp.asarray(batch_np), masks)

        codes = np.asarray(out["code_indices"], dtype=np.int32)
        count_by_code += np.bincount(codes, minlength=num_codes)

        mask_2d = np.asarray(masks[..., 0], dtype=np.float32)
        mask_area = np.maximum(mask_2d.sum(axis=(1, 2)), 1.0)
        wall_density = (batch_np[..., 0] * mask_2d).sum(axis=(1, 2)) / mask_area
        goal_targets = batch_np[..., 1].reshape((args.batch_size, -1)).argmax(axis=-1)
        agent_targets = batch_np[..., 2].reshape((args.batch_size, -1)).argmax(axis=-1)
        flat_mask = mask_2d.reshape((args.batch_size, -1))
        goal_in_mask = flat_mask[np.arange(args.batch_size), goal_targets]
        agent_in_mask = flat_mask[np.arange(args.batch_size), agent_targets]
        rows = np.arange(height, dtype=np.float32)[None, :, None]
        cols = np.arange(width, dtype=np.float32)[None, None, :]
        center_y = (mask_2d * rows).sum(axis=(1, 2)) / mask_area
        center_x = (mask_2d * cols).sum(axis=(1, 2)) / mask_area

        all_codes.append(codes)
        all_wall_density.append(wall_density)
        all_goal_in_mask.append(goal_in_mask)
        all_agent_in_mask.append(agent_in_mask)
        all_mask_area.append(mask_area)
        all_center_y.append(center_y)
        all_center_x.append(center_x)

    codes = np.concatenate(all_codes)
    wall_density = np.concatenate(all_wall_density)
    goal_in_mask = np.concatenate(all_goal_in_mask)
    agent_in_mask = np.concatenate(all_agent_in_mask)
    mask_area = np.concatenate(all_mask_area)
    center_y = np.concatenate(all_center_y)
    center_x = np.concatenate(all_center_x)

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
                    "mean_masked_wall_density": 0.0,
                    "goal_in_mask_rate": 0.0,
                    "agent_in_mask_rate": 0.0,
                    "mean_mask_area": 0.0,
                    "mean_mask_center_y": 0.0,
                    "mean_mask_center_x": 0.0,
                }
            )
            continue
        code_rows.append(
            {
                "code": float(code),
                "count": float(count),
                "prob": float(count / len(codes)),
                "mean_masked_wall_density": float(wall_density[sel].mean()),
                "goal_in_mask_rate": float(goal_in_mask[sel].mean()),
                "agent_in_mask_rate": float(agent_in_mask[sel].mean()),
                "mean_mask_area": float(mask_area[sel].mean()),
                "mean_mask_center_y": float(center_y[sel].mean()),
                "mean_mask_center_x": float(center_x[sel].mean()),
            }
        )

    specialization = {
        "wall_density_between_group_ratio": _between_group_ratio(wall_density, codes, num_codes),
        "goal_in_mask_between_group_ratio": _between_group_ratio(goal_in_mask, codes, num_codes),
        "agent_in_mask_between_group_ratio": _between_group_ratio(agent_in_mask, codes, num_codes),
        "mask_area_between_group_ratio": _between_group_ratio(mask_area, codes, num_codes),
        "mask_center_y_between_group_ratio": _between_group_ratio(center_y, codes, num_codes),
        "mask_center_x_between_group_ratio": _between_group_ratio(center_x, codes, num_codes),
    }

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
        "min_nonzero_code_prob": float(probs[probs > 0].min(initial=0.0)),
        "specialization": specialization,
    }

    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(out_dir / "analysis_code_usage.csv", code_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
