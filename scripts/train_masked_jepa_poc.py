#!/usr/bin/env python3
"""Train a masked-maze JEPA proof of concept with discrete latent commands."""

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
import optax
from flax.training import train_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vae.masked_jepa_poc import (  # noqa: E402
    MaskedMazeJepaPoc,
    compute_masked_jepa_loss,
    sample_masks,
    summarize_code_assignments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "vae" / "datasets" / "vae_og" / "train_200k_grids.npz",
    )
    p.add_argument("--dataset-key", type=str, default="grids")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "masked_jepa_poc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--num-eval-batches", type=int, default=8)
    p.add_argument("--max-train", type=int, default=20000)
    p.add_argument("--max-val", type=int, default=4000)
    p.add_argument("--num-codes", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--code-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gumbel-temperature", type=float, default=0.7)
    p.add_argument("--mask-mode", type=str, default="balanced_blocks", choices=["rect", "balanced_blocks"])
    p.add_argument("--mask-min-size", type=int, default=4)
    p.add_argument("--mask-max-size", type=int, default=7)
    p.add_argument("--mask-block-height", type=int, default=4)
    p.add_argument("--mask-block-width", type=int, default=4)
    p.add_argument("--mask-num-regions", type=int, default=2)
    p.add_argument("--jepa-weight", type=float, default=1.0)
    p.add_argument("--wall-weight", type=float, default=1.0)
    p.add_argument("--goal-weight", type=float, default=0.2)
    p.add_argument("--agent-weight", type=float, default=0.2)
    p.add_argument("--sigreg-weight", type=float, default=0.2)
    p.add_argument("--sigreg-num-projections", type=int, default=16)
    p.add_argument("--sigreg-kernel-gamma", type=float, default=1.0)
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


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_code_rows(all_rows: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not all_rows:
        return []
    num_codes = len(all_rows[0])
    agg: list[dict[str, float]] = []
    for code in range(num_codes):
        code_rows = [rows[code] for rows in all_rows]
        total_count = float(sum(row["count"] for row in code_rows))
        out = {"code": float(code), "count": total_count}
        for key in code_rows[0]:
            if key in {"code", "count"}:
                continue
            if total_count <= 0:
                out[key] = 0.0
            else:
                weighted = sum(row[key] * row["count"] for row in code_rows)
                out[key] = float(weighted / total_count)
        agg.append(out)
    return agg


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grids = load_grids(args.dataset, args.dataset_key)
    train_grids, val_grids = split_train_val(grids, args.seed, args.max_train, args.max_val)
    height, width = train_grids.shape[1:3]

    model = MaskedMazeJepaPoc(
        height=height,
        width=width,
        embed_dim=args.embed_dim,
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        gumbel_temperature=args.gumbel_temperature,
    )
    loss_cfg = {
        "jepa_weight": args.jepa_weight,
        "wall_weight": args.wall_weight,
        "goal_weight": args.goal_weight,
        "agent_weight": args.agent_weight,
        "sigreg_weight": args.sigreg_weight,
        "sigreg_num_projections": args.sigreg_num_projections,
        "sigreg_kernel_gamma": args.sigreg_kernel_gamma,
    }

    init_grids = jnp.asarray(train_grids[:1], dtype=jnp.float32)
    init_masks = sample_masks(
        jax.random.PRNGKey(args.seed + 1),
        batch_size=1,
        height=height,
        width=width,
        mask_mode=args.mask_mode,
        min_size=args.mask_min_size,
        max_size=args.mask_max_size,
        block_height=args.mask_block_height,
        block_width=args.mask_block_width,
        num_regions=args.mask_num_regions,
    )
    params = model.init(
        {"params": jax.random.PRNGKey(args.seed), "gumbel": jax.random.PRNGKey(args.seed + 2)},
        init_grids,
        init_masks,
        deterministic=False,
    )["params"]

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr),
    )

    @jax.jit
    def train_step(state: train_state.TrainState, batch: jax.Array, rng_mask: jax.Array, rng_model: jax.Array):
        masks = sample_masks(
            rng_mask,
            batch_size=batch.shape[0],
            height=height,
            width=width,
            mask_mode=args.mask_mode,
            min_size=args.mask_min_size,
            max_size=args.mask_max_size,
            block_height=args.mask_block_height,
            block_width=args.mask_block_width,
            num_regions=args.mask_num_regions,
        )

        def loss_fn(params):
            return compute_masked_jepa_loss(
                params=params,
                model=model,
                grids=batch,
                masks=masks,
                rng=rng_model,
                cfg=loss_cfg,
                deterministic=False,
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    @jax.jit
    def eval_step(params: dict, batch: jax.Array, rng_mask: jax.Array, rng_model: jax.Array):
        masks = sample_masks(
            rng_mask,
            batch_size=batch.shape[0],
            height=height,
            width=width,
            mask_mode=args.mask_mode,
            min_size=args.mask_min_size,
            max_size=args.mask_max_size,
            block_height=args.mask_block_height,
            block_width=args.mask_block_width,
            num_regions=args.mask_num_regions,
        )
        _, metrics = compute_masked_jepa_loss(
            params=params,
            model=model,
            grids=batch,
            masks=masks,
            rng=rng_model,
            cfg=loss_cfg,
            deterministic=True,
        )
        outputs = model.apply(
            {"params": params},
            batch,
            masks,
            deterministic=True,
        )
        return metrics, outputs, masks

    host_rng = np.random.default_rng(args.seed)
    train_key = jax.random.PRNGKey(args.seed + 10)
    eval_key = jax.random.PRNGKey(args.seed + 20)
    history: list[dict[str, float]] = []

    def sample_np_batch(split_grids: np.ndarray) -> jax.Array:
        if len(split_grids) == 0:
            raise ValueError("Split is empty")
        replace = len(split_grids) < args.batch_size
        idx = host_rng.choice(len(split_grids), size=args.batch_size, replace=replace)
        return jnp.asarray(split_grids[idx], dtype=jnp.float32)

    for step in range(1, args.num_steps + 1):
        batch = sample_np_batch(train_grids)
        train_key, rng_mask, rng_model = jax.random.split(train_key, 3)
        state, metrics = train_step(state, batch, rng_mask, rng_model)

        if step % args.eval_every == 0 or step == 1 or step == args.num_steps:
            metric_rows: list[dict[str, float]] = []
            code_batches: list[list[dict[str, float]]] = []
            for _ in range(args.num_eval_batches):
                batch_val = sample_np_batch(val_grids)
                eval_key, rng_mask, rng_model = jax.random.split(eval_key, 3)
                eval_metrics, outputs, masks = eval_step(state.params, batch_val, rng_mask, rng_model)
                metric_rows.append({k: float(v) for k, v in eval_metrics.items()})
                code_batches.append(
                    summarize_code_assignments(
                        outputs=outputs,
                        grids=batch_val,
                        masks=masks,
                        num_codes=args.num_codes,
                    )
                )

            summary = {
                f"val/{k}": float(np.mean([row[k] for row in metric_rows])) for k in metric_rows[0]
            }
            train_metrics = {f"train/{k}": float(v) for k, v in metrics.items()}
            row = {"step": float(step), **train_metrics, **summary}
            history.append(row)
            last_code_rows = aggregate_code_rows(code_batches)
            print(
                f"step={step:5d} "
                f"train_total={train_metrics['train/total']:.4f} "
                f"val_total={summary['val/total']:.4f} "
                f"val_perplexity={summary['val/code_perplexity']:.2f} "
                f"val_wall_acc={summary['val/wall_acc_masked']:.3f}"
            )

    if not history:
        raise RuntimeError("No evaluation outputs were produced")

    summary_payload = {
        "dataset": str(args.dataset),
        "dataset_key": args.dataset_key,
        "train_size": int(len(train_grids)),
        "val_size": int(len(val_grids)),
        "height": int(height),
        "width": int(width),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "num_codes": int(args.num_codes),
        "embed_dim": int(args.embed_dim),
        "code_dim": int(args.code_dim),
        "mask_mode": args.mask_mode,
        "mask_min_size": int(args.mask_min_size),
        "mask_max_size": int(args.mask_max_size),
        "mask_block_height": int(args.mask_block_height),
        "mask_block_width": int(args.mask_block_width),
        "mask_num_regions": int(args.mask_num_regions),
        "final_metrics": history[-1],
    }

    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    write_csv(args.out_dir / "history.csv", history)
    write_csv(args.out_dir / "code_summary.csv", last_code_rows)
    with open(args.out_dir / "checkpoint_final.pkl", "wb") as f:
        pickle.dump({"params": state.params, "args": vars(args)}, f)

    print(f"Wrote summary to {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
