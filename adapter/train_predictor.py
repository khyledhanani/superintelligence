"""
Standalone regret predictor training: z -> predicted_regret.

No adapter, no reconstruction — just learn the mapping from latent vectors
to regret scores with the best possible generalization.

Usage:
    python adapter/train_predictor.py \
        --data_path /tmp/adapter_data/train_data.npz \
        --output_dir /tmp/adapter_data/ \
        --hidden_dim 256 --n_layers 3 \
        --epochs 500 --lr 1e-3 --weight_decay 1e-3
"""
import argparse
import os
import sys
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.training import train_state

sys.path.insert(0, os.path.dirname(__file__))
from adapter_model import RegretPredictor, create_predictor


def main():
    parser = argparse.ArgumentParser(description="Standalone regret predictor training")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="ADAPTER_TRAINING")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    z_all = jnp.array(data["z"])
    regret_all = jnp.array(data["regret"])
    latent_dim = int(data["latent_dim"])
    n = len(z_all)

    # Also load test set if available
    test_path = args.data_path.replace(".npz", "_test.npz")
    has_test = os.path.exists(test_path)
    if has_test:
        test_data = np.load(test_path, allow_pickle=True)
        z_test = jnp.array(test_data["z"])
        regret_test = jnp.array(test_data["regret"])
        print(f"  Test set: {len(z_test)} samples")

    # Train/val split
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = np.random.RandomState(args.seed).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    z_train, regret_train = z_all[train_idx], regret_all[train_idx]
    z_val, regret_val = z_all[val_idx], regret_all[val_idx]

    print(f"  {n} total, {n_train} train, {n_val} val, latent_dim={latent_dim}")
    print(f"  Regret: mean={regret_all.mean():.4f}, std={regret_all.std():.4f}, "
          f"min={regret_all.min():.4f}, max={regret_all.max():.4f}")
    print(f"  Nonzero: {int((regret_all > 0).sum())}/{n} ({100*(regret_all > 0).mean():.1f}%)")

    # --- wandb ---
    run_name = args.run_name or f"predictor_h{args.hidden_dim}_l{args.n_layers}_wd{args.weight_decay}"
    wandb.init(project=args.project, name=run_name, config=vars(args), tags=["predictor-only"])

    # --- Create model ---
    predictor, pred_params = create_predictor(latent_dim, args.hidden_dim, args.n_layers)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(pred_params))
    print(f"  Predictor params: {n_params:,}")

    tx = optax.adamw(args.lr, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=predictor.apply,
        params=pred_params,
        tx=tx,
    )

    # --- Loss ---
    @jax.jit
    def eval_fn(params, z_batch, regret_batch):
        pred = predictor.apply(params, z_batch)
        mse = jnp.mean((pred - regret_batch) ** 2)
        ss_res = jnp.sum((regret_batch - pred) ** 2)
        ss_tot = jnp.sum((regret_batch - jnp.mean(regret_batch)) ** 2)
        r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-8)
        return mse, r2

    @jax.jit
    def train_step(state, z_batch, regret_batch):
        def compute_loss(params):
            pred = predictor.apply(params, z_batch)
            return jnp.mean((pred - regret_batch) ** 2)
        loss, grads = jax.value_and_grad(compute_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # --- Training loop ---
    best_val_r2 = -float("inf")
    best_params = None
    best_epoch = 0
    epochs_without_improvement = 0
    t0 = time.time()

    print(f"\nTraining predictor (hidden={args.hidden_dim}, layers={args.n_layers}, "
          f"wd={args.weight_decay}, lr={args.lr})...")

    for epoch in range(args.epochs):
        rng, rng_perm = jax.random.split(rng)
        perm_idx = jax.random.permutation(rng_perm, n_train)

        for i in range(0, n_train, args.batch_size):
            end = min(i + args.batch_size, n_train)
            idx = perm_idx[i:end]
            state, _ = train_step(state, z_train[idx], regret_train[idx])

        # Eval
        train_mse, train_r2 = eval_fn(state.params, z_train, regret_train)
        val_mse, val_r2 = eval_fn(state.params, z_val, regret_val)
        train_r2, val_r2 = float(train_r2), float(val_r2)

        log = {
            "epoch": epoch + 1,
            "train/mse": float(train_mse),
            "train/r2": train_r2,
            "val/mse": float(val_mse),
            "val/r2": val_r2,
            "val/best_r2": max(best_val_r2, val_r2),
        }
        if has_test:
            test_mse, test_r2 = eval_fn(state.params, z_test, regret_test)
            log["test/mse"] = float(test_mse)
            log["test/r2"] = float(test_r2)
        wandb.log(log)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no val R² improvement for {args.patience} epochs)")
                break

        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - t0
            line = (f"  Epoch {epoch+1:4d} | "
                    f"train_R²={train_r2:.4f} val_R²={val_r2:.4f}")
            if has_test:
                line += f" test_R²={float(test_r2):.4f}"
            line += f" | mse={float(train_mse):.6f} | best={best_epoch+1} ({elapsed:.1f}s)"
            print(line)

    # --- Final eval ---
    print(f"\nBest model at epoch {best_epoch+1}, val_R²={best_val_r2:.4f}")

    final_val_mse, final_val_r2 = eval_fn(best_params, z_val, regret_val)
    final_train_mse, final_train_r2 = eval_fn(best_params, z_train, regret_train)

    wandb.run.summary["best_val_r2"] = best_val_r2
    wandb.run.summary["best_epoch"] = best_epoch + 1
    wandb.run.summary["final_train_r2"] = float(final_train_r2)

    if has_test:
        final_test_mse, final_test_r2 = eval_fn(best_params, z_test, regret_test)
        print(f"  Test R²={float(final_test_r2):.4f}, MSE={float(final_test_mse):.6f}")
        wandb.run.summary["final_test_r2"] = float(final_test_r2)

    # --- Scatter plot ---
    val_pred = predictor.apply(best_params, z_val)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.array(regret_val), np.array(val_pred), s=2, alpha=0.3)
    rmax = float(regret_val.max())
    ax.plot([0, rmax], [0, rmax], "r--", linewidth=1)
    ax.set_xlabel("Actual regret")
    ax.set_ylabel("Predicted regret")
    ax.set_title(f"Predictor val R²={best_val_r2:.4f}")
    scatter_path = os.path.join(args.output_dir, "predictor_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    wandb.log({"diagnostics/pred_vs_actual": wandb.Image(fig)})
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    if best_val_r2 < 0.3:
        print("\n  WARNING: R² < 0.3 — the latent space may not encode regret smoothly.")

    # --- Save ---
    pred_ckpt = {
        "params": best_params,
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "latent_dim": latent_dim,
        },
        "metrics": {
            "val_r2": best_val_r2,
            "val_mse": float(final_val_mse),
            "train_r2": float(final_train_r2),
            "best_epoch": best_epoch + 1,
            "n_train": n_train,
            "n_val": n_val,
        },
    }
    pred_path = os.path.join(args.output_dir, "predictor.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(pred_ckpt, f)
    print(f"Saved predictor to {pred_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
