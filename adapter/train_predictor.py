"""
Phase 2: Train the regret predictor on (z, regret) pairs.

The predictor learns to map latent vectors to regret scores.
After training, it is frozen and used during adapter training.

Usage:
    python adapter/train_predictor.py \
        --data_path /tmp/adapter_data/train_data.npz \
        --output_path /tmp/adapter_data/predictor.pkl \
        --hidden_dim 128 --n_layers 2 \
        --epochs 500 --lr 1e-3 --batch_size 256
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
from flax.training import train_state

sys.path.insert(0, os.path.dirname(__file__))
from adapter_model import RegretPredictor, create_predictor


def create_train_state(model, params, lr):
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )


@jax.jit
def train_step(state, z_batch, regret_batch):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, z_batch)
        return jnp.mean((pred - regret_batch) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, z_batch, regret_batch):
    pred = state.apply_fn({"params": state.params}, z_batch)
    mse = jnp.mean((pred - regret_batch) ** 2)
    ss_res = jnp.sum((regret_batch - pred) ** 2)
    ss_tot = jnp.sum((regret_batch - jnp.mean(regret_batch)) ** 2)
    r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-8)
    return mse, r2, pred


def main():
    parser = argparse.ArgumentParser(description="Train regret predictor on (z, regret) pairs")
    parser.add_argument("--data_path", type=str, required=True, help="Path to train_data.npz")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save predictor checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="ADAPTER_TRAINING")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # --- wandb init ---
    run_name = args.run_name or f"predictor_h{args.hidden_dim}_l{args.n_layers}"
    wandb.init(
        project=args.project,
        name=run_name,
        config=vars(args),
        tags=["predictor"],
    )

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path, allow_pickle=True)
    z_all = jnp.array(data["z"])
    regret_all = jnp.array(data["regret"])
    latent_dim = int(data["latent_dim"])
    print(f"  {len(z_all)} samples, latent_dim={latent_dim}")
    print(f"  Regret: mean={regret_all.mean():.4f}, std={regret_all.std():.4f}")

    wandb.run.summary["n_samples"] = len(z_all)
    wandb.run.summary["latent_dim"] = latent_dim
    wandb.run.summary["regret_mean"] = float(regret_all.mean())
    wandb.run.summary["regret_std"] = float(regret_all.std())

    # Train/val split
    n = len(z_all)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = np.random.RandomState(args.seed).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    z_train, regret_train = z_all[train_idx], regret_all[train_idx]
    z_val, regret_val = z_all[val_idx], regret_all[val_idx]
    print(f"  Train: {n_train}, Val: {n_val}")

    # Create model
    model, params = create_predictor(latent_dim, args.hidden_dim, args.n_layers)
    state = create_train_state(model, params, args.lr)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Predictor params: {n_params:,}")
    wandb.run.summary["n_params"] = n_params

    # Training loop
    best_val_loss = float("inf")
    best_params = None
    best_epoch = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        rng, rng_perm = jax.random.split(rng)
        perm_idx = jax.random.permutation(rng_perm, n_train)
        z_shuf = z_train[perm_idx]
        r_shuf = regret_train[perm_idx]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            end = min(i + args.batch_size, n_train)
            state, loss = train_step(state, z_shuf[i:end], r_shuf[i:end])
            epoch_loss += float(loss)
            n_batches += 1
        epoch_loss /= n_batches

        val_mse, val_r2, _ = eval_step(state, z_val, regret_val)
        val_mse, val_r2 = float(val_mse), float(val_r2)

        # wandb logging every epoch
        wandb.log({
            "epoch": epoch + 1,
            "predictor/train_mse": epoch_loss,
            "predictor/val_mse": val_mse,
            "predictor/val_r2": val_r2,
            "predictor/best_val_mse": min(best_val_loss, val_mse),
        })

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
            best_epoch = epoch

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d} | train_loss={epoch_loss:.6f} | val_mse={val_mse:.6f} | val_R2={val_r2:.4f} | best_epoch={best_epoch+1} ({elapsed:.1f}s)")

    # Final evaluation with scatter plot
    final_mse, final_r2, val_preds = eval_step(
        state.replace(params=best_params), z_val, regret_val
    )
    print(f"\nBest model (epoch {best_epoch+1}): val_MSE={float(final_mse):.6f}, val_R2={float(final_r2):.4f}")

    wandb.run.summary["best_val_mse"] = float(final_mse)
    wandb.run.summary["best_val_r2"] = float(final_r2)
    wandb.run.summary["best_epoch"] = best_epoch + 1

    # Log predicted vs actual scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.array(regret_val), np.array(val_preds), s=2, alpha=0.3)
    ax.plot([0, float(regret_val.max())], [0, float(regret_val.max())], "r--", linewidth=1)
    ax.set_xlabel("Actual regret")
    ax.set_ylabel("Predicted regret")
    ax.set_title(f"Predictor: R²={float(final_r2):.4f}")
    wandb.log({"predictor/pred_vs_actual": wandb.Image(fig)})
    plt.close()

    if float(final_r2) < 0.3:
        print("\n  WARNING: R^2 < 0.3 — the latent space may lack regret information.")

    # Save
    checkpoint = {
        "params": best_params,
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "latent_dim": latent_dim,
        },
        "metrics": {
            "val_mse": float(final_mse),
            "val_r2": float(final_r2),
            "best_epoch": best_epoch,
            "n_train": n_train,
            "n_val": n_val,
        },
    }
    with open(args.output_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved predictor to {args.output_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
