"""
Train the supervised trajectory encoder to predict regret from raw
position sequences.

Evaluates:
  - MSE / R² on held-out data
  - Per-dimension correlation of learned representation with regret
  - Comparison with handcrafted features baseline

Usage:
    python trajectory/train_encoder.py \
        --data_path /tmp/traj_analysis/data/trajectories_50k.npz \
        --output_dir /tmp/traj_analysis/encoder/50k
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
from flax.training.train_state import TrainState
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from trajectory_encoder import TrajectoryEncoder


def create_train_state(rng, model, lr, weight_decay):
    dummy = jnp.zeros((1, 250), dtype=jnp.int32)
    variables = model.init(rng, dummy, train=False)
    tx = optax.adamw(lr, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def train_step(state, traj_batch, regret_batch, rng):
    def loss_fn(params):
        pred, z = state.apply_fn(
            {"params": params}, traj_batch, train=True,
            rngs={"dropout": rng},
        )
        mse = jnp.mean((pred - regret_batch) ** 2)
        return mse, (pred, z)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (pred, z)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, pred


def eval_step(state, traj_batch, regret_batch):
    pred, z = state.apply_fn(
        {"params": state.params}, traj_batch, train=False,
    )
    mse = jnp.mean((pred - regret_batch) ** 2)
    return mse, pred, z


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-10)


def main():
    parser = argparse.ArgumentParser(description="Train trajectory encoder for regret prediction")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repr_dim", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    # --- Load data ---
    print("[1/4] Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    trajectories = data["trajectories"]
    regret = data["regret"].astype(np.float32)
    reached_goal = data["reached_goal"] if "reached_goal" in data else None
    n = len(trajectories)

    # Normalize regret for training stability
    regret_mean, regret_std = regret.mean(), regret.std()
    regret_norm = (regret - regret_mean) / max(regret_std, 1e-6)

    # Train/val split
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    train_traj = jnp.array(trajectories[perm[:n_train]])
    val_traj = jnp.array(trajectories[perm[n_train:]])
    train_reg = jnp.array(regret_norm[perm[:n_train]])
    val_reg = jnp.array(regret_norm[perm[n_train:]])
    val_reg_raw = regret[perm[n_train:]]

    print(f"  {n} total, {n_train} train, {n_val} val")
    print(f"  Regret: mean={regret_mean:.4f}, std={regret_std:.4f}")

    # --- Create model ---
    print("[2/4] Creating model...")
    model = TrajectoryEncoder(
        vocab_size=170,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
    )

    rng, rng_init = jax.random.split(rng)
    state = create_train_state(rng_init, model, args.lr, args.weight_decay)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  repr_dim={args.repr_dim}, hidden_dim={args.hidden_dim}, embed_dim={args.embed_dim}")
    print(f"  Parameters: {n_params:,}")

    train_step_jit = jax.jit(train_step)
    eval_step_jit = jax.jit(eval_step)

    # --- Training ---
    print("[3/4] Training...")
    best_val_mse = float("inf")
    best_epoch = 0
    best_params = None
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    history = {"train_mse": [], "val_mse": [], "val_r2": []}

    for epoch in range(args.epochs):
        t0 = time.time()

        # Shuffle
        rng, rng_shuf = jax.random.split(rng)
        shuf_idx = jax.random.permutation(rng_shuf, n_train)
        train_traj_shuf = train_traj[shuf_idx]
        train_reg_shuf = train_reg[shuf_idx]

        # Train
        epoch_loss = 0.0
        for b in range(n_batches):
            s = b * args.batch_size
            e = min(s + args.batch_size, n_train)
            rng, rng_step = jax.random.split(rng)
            state, loss, _ = train_step_jit(
                state, train_traj_shuf[s:e], train_reg_shuf[s:e], rng_step
            )
            epoch_loss += float(loss)
        epoch_loss /= n_batches

        # Validate
        val_mse, val_pred_norm, val_z = eval_step_jit(state, val_traj, val_reg)
        val_mse = float(val_mse)
        val_pred = np.array(val_pred_norm) * regret_std + regret_mean
        val_r2 = compute_r2(val_reg_raw, val_pred)

        history["train_mse"].append(epoch_loss)
        history["val_mse"].append(val_mse)
        history["val_r2"].append(val_r2)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)

        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s) | "
                  f"train MSE={epoch_loss:.4f} | val MSE={val_mse:.4f} R²={val_r2:.4f}")

    print(f"\n  Best epoch: {best_epoch+1}, val MSE={best_val_mse:.4f}")

    # --- Evaluate best model ---
    print("[4/4] Evaluating best model...")
    best_state = state.replace(params=jax.tree_util.tree_map(jnp.array, best_params))
    _, val_pred_norm, val_z = eval_step_jit(best_state, val_traj, val_reg)
    val_pred = np.array(val_pred_norm) * regret_std + regret_mean
    val_z = np.array(val_z)
    val_r2 = compute_r2(val_reg_raw, val_pred)
    val_rho, _ = stats.spearmanr(val_pred, val_reg_raw)

    print(f"  Prediction R²: {val_r2:.4f}")
    print(f"  Prediction Spearman ρ: {val_rho:.4f}")

    # Per-dimension representation analysis
    print(f"\n  Representation analysis ({args.repr_dim} dims):")
    dim_corrs = []
    for d in range(args.repr_dim):
        rho, pval = stats.spearmanr(val_z[:, d], val_reg_raw)
        dim_corrs.append((d, rho, pval))
    dim_corrs.sort(key=lambda x: -abs(x[1]))
    for d, rho, pval in dim_corrs[:10]:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    dim {d:3d}: ρ={rho:+.4f} (p={pval:.2e}) {sig}")

    # Ridge on learned repr
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(val_z)
    ridge_r2 = cross_val_score(Ridge(alpha=1.0), z_scaled, val_reg_raw, cv=5, scoring="r2")
    print(f"\n  Ridge R² on learned repr (5-fold CV): {ridge_r2.mean():.4f} ± {ridge_r2.std():.4f}")

    # Solved-only analysis
    if reached_goal is not None:
        val_goal = reached_goal[perm[n_train:]]
        solved = val_goal.astype(bool)
        n_solved = solved.sum()
        if n_solved > 30:
            solved_r2 = compute_r2(val_reg_raw[solved], val_pred[solved])
            solved_rho, _ = stats.spearmanr(val_pred[solved], val_reg_raw[solved])
            ridge_solved = cross_val_score(
                Ridge(alpha=1.0), z_scaled[solved], val_reg_raw[solved], cv=5, scoring="r2"
            )
            print(f"\n  Solved only ({n_solved}):")
            print(f"    Prediction R²: {solved_r2:.4f}")
            print(f"    Prediction Spearman ρ: {solved_rho:.4f}")
            print(f"    Ridge R² on repr: {ridge_solved.mean():.4f} ± {ridge_solved.std():.4f}")

    # --- Plots ---
    # 1. Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_mse"], label="Train")
    ax1.plot(history["val_mse"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE (normalized)")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax2.plot(history["val_r2"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R²")
    ax2.set_title(f"Validation R² (best={val_r2:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    plt.close()

    # 2. Predicted vs actual regret
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(val_reg_raw, val_pred, alpha=0.15, s=5, rasterized=True)
    lims = [min(val_reg_raw.min(), val_pred.min()), max(val_reg_raw.max(), val_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual regret")
    ax.set_ylabel("Predicted regret")
    ax.set_title(f"Predicted vs Actual Regret\nR²={val_r2:.4f}, ρ={val_rho:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()

    # 3. Representation PCA colored by regret
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(val_z)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=val_reg_raw, cmap="viridis",
                    alpha=0.3, s=5, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Learned Trajectory Representation\nPCA colored by regret")
    plt.colorbar(sc, ax=ax, label="regret")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "repr_pca.png"), dpi=150)
    plt.close()

    # 4. Dimension correlation bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = [x[0] for x in dim_corrs]
    rhos = [x[1] for x in dim_corrs]
    colors = ["#2196F3" if r > 0 else "#F44336" for r in rhos]
    ax.barh(range(len(dims)), rhos, color=colors)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"dim {d}" for d in dims], fontsize=8)
    ax.set_xlabel("Spearman ρ with regret")
    ax.set_title(f"Learned Representation: Per-Dim Correlation with Regret")
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "repr_dim_correlations.png"), dpi=150)
    plt.close()

    print(f"\n  Plots saved to {args.output_dir}/")

    # --- Save ---
    with open(os.path.join(args.output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump({
            "params": best_params,
            "config": vars(args),
            "epoch": best_epoch,
            "val_r2": val_r2,
            "regret_mean": float(regret_mean),
            "regret_std": float(regret_std),
        }, f)
    print(f"  Model saved to {args.output_dir}/best_model.pkl")


if __name__ == "__main__":
    main()
