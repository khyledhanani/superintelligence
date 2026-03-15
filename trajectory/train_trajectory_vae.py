"""
Train the trajectory VAE on collected agent trajectories.

Usage:
    python trajectory/train_trajectory_vae.py \
        --data_path trajectory/data/trajectories.npz \
        --output_dir trajectory/checkpoints/run1 \
        --latent_dim 32 \
        --epochs 200 \
        --batch_size 128 \
        --kl_warmup_epochs 20
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

sys.path.insert(0, os.path.dirname(__file__))
from trajectory_vae import TrajectoryVAE, trajectory_vae_loss


def create_train_state(rng, model, lr, weight_decay):
    dummy_input = jnp.zeros((1, model.max_steps), dtype=jnp.int32)
    rng_init, rng_z = jax.random.split(rng)
    variables = model.init(rng_init, dummy_input, rng_z, train=False)
    tx = optax.adamw(lr, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def train_step(state, batch, rng, kl_weight):
    def loss_fn(params):
        rng_z = rng
        logits, mean, logvar = state.apply_fn(
            {"params": params}, batch, rng_z, train=True,
            rngs={"dropout": jax.random.fold_in(rng, 1)},
        )
        total, (recon, kl) = trajectory_vae_loss(logits, batch, mean, logvar, kl_weight)
        return total, (recon, kl, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon, kl, logits)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # Token accuracy on non-padding positions
    mask = (batch > 0).astype(jnp.float32)
    preds = jnp.argmax(logits, axis=-1)
    acc = ((preds == batch) * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    return state, {"loss": loss, "recon": recon, "kl": kl, "acc": acc}


def eval_step(state, batch, rng, kl_weight):
    logits, mean, logvar = state.apply_fn(
        {"params": state.params}, batch, rng, train=False,
    )
    total, (recon, kl) = trajectory_vae_loss(logits, batch, mean, logvar, kl_weight)

    mask = (batch > 0).astype(jnp.float32)
    preds = jnp.argmax(logits, axis=-1)
    acc = ((preds == batch) * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    return {"loss": total, "recon": recon, "kl": kl, "acc": acc}


def main():
    parser = argparse.ArgumentParser(description="Train trajectory VAE")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20,
                        help="Linear KL annealing over this many epochs")
    parser.add_argument("--kl_weight", type=float, default=1.0,
                        help="Max KL weight after warmup")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="trajectory_vae")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    # --- Load data ---
    print("[1/3] Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    trajectories = data["trajectories"]  # (N, max_steps)
    regret = data["regret"]
    max_steps = int(data["max_steps"])
    n = len(trajectories)

    # Train/val split
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    train_traj = jnp.array(trajectories[perm[:n_train]])
    val_traj = jnp.array(trajectories[perm[n_train:]])
    train_regret = regret[perm[:n_train]]
    val_regret = regret[perm[n_train:]]

    print(f"  {n} total, {n_train} train, {n_val} val")
    print(f"  Trajectory shape: {trajectories.shape}, max_steps={max_steps}")
    print(f"  Regret: mean={regret.mean():.4f}, std={regret.std():.4f}")

    # --- Create model ---
    print("[2/3] Creating model...")
    model = TrajectoryVAE(
        vocab_size=170,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        max_steps=max_steps,
    )

    rng, rng_init = jax.random.split(rng)
    state = create_train_state(rng_init, model, args.lr, args.weight_decay)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  Model: latent_dim={args.latent_dim}, embed_dim={args.embed_dim}")
    print(f"  Parameters: {n_params:,}")

    # JIT
    train_step_jit = jax.jit(train_step)
    eval_step_jit = jax.jit(eval_step)

    # Wandb
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # --- Training ---
    print("[3/3] Training...")
    best_val_loss = float("inf")
    best_epoch = 0
    n_train_batches = (n_train + args.batch_size - 1) // args.batch_size

    for epoch in range(args.epochs):
        t0 = time.time()

        # KL annealing
        if args.kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / args.kl_warmup_epochs) * args.kl_weight
        else:
            kl_weight = args.kl_weight

        # Shuffle training data
        rng, rng_shuffle = jax.random.split(rng)
        shuffle_idx = jax.random.permutation(rng_shuffle, n_train)
        train_shuffled = train_traj[shuffle_idx]

        # Train epoch
        train_metrics = {"loss": 0, "recon": 0, "kl": 0, "acc": 0}
        for b in range(n_train_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, n_train)
            batch = train_shuffled[start:end]

            rng, rng_step = jax.random.split(rng)
            state, metrics = train_step_jit(state, batch, rng_step, kl_weight)
            for k in train_metrics:
                train_metrics[k] += float(metrics[k])

        for k in train_metrics:
            train_metrics[k] /= n_train_batches

        # Validate
        rng, rng_val = jax.random.split(rng)
        val_metrics = eval_step_jit(state, val_traj, rng_val, kl_weight)
        val_metrics = {k: float(v) for k, v in val_metrics.items()}

        elapsed = time.time() - t0

        # Track best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s) | "
                  f"train loss={train_metrics['loss']:.4f} recon={train_metrics['recon']:.4f} "
                  f"kl={train_metrics['kl']:.4f} acc={train_metrics['acc']:.3f} | "
                  f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} | "
                  f"kl_w={kl_weight:.3f}")

        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/recon": train_metrics["recon"],
                "train/kl": train_metrics["kl"],
                "train/acc": train_metrics["acc"],
                "val/loss": val_metrics["loss"],
                "val/recon": val_metrics["recon"],
                "val/kl": val_metrics["kl"],
                "val/acc": val_metrics["acc"],
                "kl_weight": kl_weight,
            })

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{epoch+1}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": jax.tree_util.tree_map(lambda x: np.array(x), state.params),
                    "config": vars(args),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                }, f)

    # Save best model
    best_path = os.path.join(args.output_dir, "best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump({
            "params": best_params,
            "config": vars(args),
            "epoch": best_epoch,
            "val_loss": best_val_loss,
        }, f)
    print(f"\nBest model (epoch {best_epoch+1}, val_loss={best_val_loss:.4f}) saved to {best_path}")

    # Save training regret alongside for correlation analysis
    regret_path = os.path.join(args.output_dir, "split_info.npz")
    np.savez(regret_path,
             train_regret=train_regret, val_regret=val_regret,
             train_idx=np.array(perm[:n_train]), val_idx=np.array(perm[n_train:]))
    print(f"Split info saved to {regret_path}")


if __name__ == "__main__":
    main()
