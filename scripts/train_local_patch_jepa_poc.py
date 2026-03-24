#!/usr/bin/env python3
"""Train a local patch JEPA proof of concept for maze rewrite primitives."""

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

from vae.local_patch_jepa_poc import (  # noqa: E402
    LocalPatchJepaPoc,
    compute_local_patch_jepa_loss,
    sample_wall_patch_batch_np,
    summarize_local_code_assignments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "vae" / "datasets" / "vae_og" / "train_200k_grids.npz",
    )
    p.add_argument("--dataset-key", type=str, default="grids")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "local_patch_jepa_poc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--num-eval-batches", type=int, default=16)
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-val", type=int, default=10000)
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--center-size", type=int, default=3)
    p.add_argument("--augment-dihedral", action="store_true", default=False)
    p.add_argument("--num-codes", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--code-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--gumbel-temperature", type=float, default=0.7)
    p.add_argument("--jepa-weight", type=float, default=1.0)
    p.add_argument("--center-weight", type=float, default=1.0)
    p.add_argument("--sigreg-weight", type=float, default=0.2)
    p.add_argument("--sigreg-num-projections", type=int, default=16)
    p.add_argument("--sigreg-kernel-gamma", type=float, default=1.0)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--usage-weight", type=float, default=0.05)
    p.add_argument("--counterfactual-weight", type=float, default=0.0)
    p.add_argument("--counterfactual-margin", type=float, default=0.05)
    p.add_argument("--diversity-weight", type=float, default=0.0)
    p.add_argument("--diversity-margin", type=float, default=0.1)
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

    model = LocalPatchJepaPoc(
        patch_size=args.patch_size,
        center_size=args.center_size,
        embed_dim=args.embed_dim,
        num_codes=args.num_codes,
        code_dim=args.code_dim,
        gumbel_temperature=args.gumbel_temperature,
    )
    loss_cfg = {
        "jepa_weight": args.jepa_weight,
        "center_weight": args.center_weight,
        "sigreg_weight": args.sigreg_weight,
        "sigreg_num_projections": args.sigreg_num_projections,
        "sigreg_kernel_gamma": args.sigreg_kernel_gamma,
        "entropy_weight": args.entropy_weight,
        "usage_weight": args.usage_weight,
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_margin": args.counterfactual_margin,
        "diversity_weight": args.diversity_weight,
        "diversity_margin": args.diversity_margin,
    }

    host_rng = np.random.default_rng(args.seed)
    init_patches = sample_wall_patch_batch_np(
        train_grids,
        host_rng,
        batch_size=1,
        patch_size=args.patch_size,
        augment_dihedral=args.augment_dihedral,
    )
    params = model.init(
        {"params": jax.random.PRNGKey(args.seed), "gumbel": jax.random.PRNGKey(args.seed + 1)},
        jnp.asarray(init_patches),
        deterministic=False,
    )["params"]

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.grad_clip_norm),
            optax.adam(args.lr),
        ),
    )

    @jax.jit
    def train_step(
        state: train_state.TrainState,
        patches: jax.Array,
        rng_model: jax.Array,
    ):
        def loss_fn(params):
            return compute_local_patch_jepa_loss(
                params=params,
                model=model,
                patches=patches,
                rng=rng_model,
                cfg=loss_cfg,
                deterministic=False,
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    @jax.jit
    def eval_step(params: dict, patches: jax.Array, rng_model: jax.Array):
        _, metrics = compute_local_patch_jepa_loss(
            params=params,
            model=model,
            patches=patches,
            rng=rng_model,
            cfg=loss_cfg,
            deterministic=True,
        )
        outputs = model.apply({"params": params}, patches, deterministic=True)
        return metrics, outputs

    train_key = jax.random.PRNGKey(args.seed + 10)
    eval_key = jax.random.PRNGKey(args.seed + 20)
    history: list[dict[str, float]] = []
    last_code_rows: list[dict[str, float]] = []

    for step in range(1, args.num_steps + 1):
        batch_np = sample_wall_patch_batch_np(
            train_grids,
            host_rng,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            augment_dihedral=args.augment_dihedral,
        )
        batch = jnp.asarray(batch_np, dtype=jnp.float32)
        train_key, rng_model = jax.random.split(train_key)
        state, metrics = train_step(state, batch, rng_model)

        if step % args.eval_every == 0 or step == 1 or step == args.num_steps:
            metric_rows: list[dict[str, float]] = []
            code_batches: list[list[dict[str, float]]] = []
            for _ in range(args.num_eval_batches):
                val_np = sample_wall_patch_batch_np(
                    val_grids,
                    host_rng,
                    batch_size=args.batch_size,
                    patch_size=args.patch_size,
                    augment_dihedral=False,
                )
                val_batch = jnp.asarray(val_np, dtype=jnp.float32)
                eval_key, rng_model = jax.random.split(eval_key)
                eval_metrics, outputs = eval_step(state.params, val_batch, rng_model)
                metric_rows.append({k: float(v) for k, v in eval_metrics.items()})
                code_batches.append(
                    summarize_local_code_assignments(
                        code_indices=np.asarray(outputs["code_indices"]),
                        patches=val_np,
                        num_codes=args.num_codes,
                        center_size=args.center_size,
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
                f"val_center_acc={summary['val/center_acc']:.3f}"
            )

    summary_payload = {
        "dataset": str(args.dataset),
        "dataset_key": args.dataset_key,
        "train_size": int(len(train_grids)),
        "val_size": int(len(val_grids)),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "patch_size": int(args.patch_size),
        "center_size": int(args.center_size),
        "augment_dihedral": bool(args.augment_dihedral),
        "num_codes": int(args.num_codes),
        "embed_dim": int(args.embed_dim),
        "code_dim": int(args.code_dim),
        "lr": float(args.lr),
        "grad_clip_norm": float(args.grad_clip_norm),
        "entropy_weight": float(args.entropy_weight),
        "usage_weight": float(args.usage_weight),
        "counterfactual_weight": float(args.counterfactual_weight),
        "counterfactual_margin": float(args.counterfactual_margin),
        "diversity_weight": float(args.diversity_weight),
        "diversity_margin": float(args.diversity_margin),
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
