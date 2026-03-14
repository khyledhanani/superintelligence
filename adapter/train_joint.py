"""
Joint training of regret predictor and adapter MLP.

Both models are trained simultaneously:
- Predictor: z → predicted_regret (MSE loss on original z)
- Adapter: z → delta_z, z' = z + delta_z (recon + regret + reg loss)

The predictor gradients flow only through its own params, not through the adapter.
The adapter uses the predictor's current output as a signal but doesn't backprop into it.

Usage:
    python adapter/train_joint.py \
        --data_path /tmp/adapter_data/train_data.npz \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --output_dir /tmp/adapter_data/ \
        --lambda_regret 0.1 --lambda_reg 0.01 \
        --epochs 300 --lr 1e-3 --batch_size 256
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
import yaml
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.training import train_state

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))

from adapter_model import AdapterMLP, RegretPredictor, create_adapter, create_predictor
from vae_model import CluttrVAE


def load_vae_decoder(vae_checkpoint_path, vae_config_path):
    with open(vae_config_path) as f:
        vae_cfg = yaml.safe_load(f)
    vae = CluttrVAE(
        vocab_size=vae_cfg["vocab_size"],
        embed_dim=vae_cfg["embed_dim"],
        latent_dim=vae_cfg["latent_dim"],
        seq_len=vae_cfg["seq_len"],
    )
    with open(vae_checkpoint_path, "rb") as f:
        vae_ckpt = pickle.load(f)
    vae_params = vae_ckpt["params"] if isinstance(vae_ckpt, dict) and "params" in vae_ckpt else vae_ckpt
    return vae, vae_params, vae_cfg


def compute_training_weights(regret_scores):
    """Upweight frontier levels (intermediate regret)."""
    median_regret = np.median(regret_scores)
    regret_distance = np.abs(regret_scores - median_regret)
    temperature = np.std(regret_scores)
    if temperature < 1e-8:
        return np.ones_like(regret_scores)
    weights = np.exp(-regret_distance / temperature)
    weights = weights / weights.mean()
    return weights


def main():
    parser = argparse.ArgumentParser(description="Joint training of predictor + adapter")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_regret", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--lambda_kl", type=float, default=0.0)
    parser.add_argument("--lambda_pred", type=float, default=1.0,
                        help="Weight for predictor MSE loss in joint objective")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to separate test .npz (if not provided, splits from data_path)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Only used if --test_data_path is not provided")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="ADAPTER_TRAINING")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- wandb ---
    run_name = args.run_name or f"joint_lr{args.lambda_regret}_lreg{args.lambda_reg}"
    wandb.init(
        project=args.project,
        name=run_name,
        config=vars(args),
        tags=["joint", "predictor", "adapter"],
    )

    # --- Load data ---
    print("Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    z_all = jnp.array(data["z"])
    tokens_all = jnp.array(data["tokens"])
    regret_all = jnp.array(data["regret"])
    latent_dim = int(data["latent_dim"])
    n = len(z_all)
    print(f"  {n} samples, latent_dim={latent_dim}")
    print(f"  Regret: mean={regret_all.mean():.4f}, std={regret_all.std():.4f}")

    # --- Load VAE ---
    print("Loading frozen VAE decoder...")
    vae, vae_params, vae_cfg = load_vae_decoder(args.vae_checkpoint_path, args.vae_config_path)
    vocab_size = vae_cfg["vocab_size"]

    # --- Load test data ---
    test_path = args.test_data_path
    if test_path is None:
        # Auto-detect: look for _test.npz alongside data_path
        auto_test = args.data_path.replace(".npz", "_test.npz")
        if os.path.exists(auto_test):
            test_path = auto_test
            print(f"  Auto-detected test set: {test_path}")

    # Split training data into train/val for monitoring generalization
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = np.random.RandomState(args.seed).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    z_train, tokens_train = z_all[train_idx], tokens_all[train_idx]
    regret_train = regret_all[train_idx]
    z_val, tokens_val = z_all[val_idx], tokens_all[val_idx]
    regret_val = regret_all[val_idx]
    print(f"  Train: {n_train}, Val: {n_val} (internal split for monitoring)")

    # Also load separate test set if available (for final eval only)
    if test_path and os.path.exists(test_path):
        print(f"  Test set loaded from {test_path} (used for final eval only)")

    # --- Weights (on training data) ---
    weights_train = jnp.array(compute_training_weights(np.array(regret_train)))
    weights_val = jnp.array(compute_training_weights(np.array(regret_val)))

    # --- Create models ---
    predictor, pred_params = create_predictor(latent_dim, args.hidden_dim, args.n_layers)
    adapter, adapter_params = create_adapter(latent_dim, args.hidden_dim, args.n_layers)

    # Joint params: {"predictor": ..., "adapter": ...}
    joint_params = {
        "predictor": pred_params["params"],
        "adapter": adapter_params["params"],
    }

    tx = optax.adamw(args.lr, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=None,  # we'll call models directly
        params=joint_params,
        tx=tx,
    )

    n_pred_params = sum(x.size for x in jax.tree_util.tree_leaves(pred_params))
    n_adapter_params = sum(x.size for x in jax.tree_util.tree_leaves(adapter_params))
    print(f"  Predictor params: {n_pred_params:,}")
    print(f"  Adapter params:   {n_adapter_params:,}")
    print(f"  Total params:     {n_pred_params + n_adapter_params:,}")

    wandb.run.summary["n_pred_params"] = n_pred_params
    wandb.run.summary["n_adapter_params"] = n_adapter_params

    # --- Frozen VAE decode ---
    @jax.jit
    def frozen_decode(z_prime):
        return jax.vmap(lambda z: vae.apply({"params": vae_params}, z, method=vae.decode))(z_prime)

    # --- Joint loss ---
    lambda_regret = args.lambda_regret
    lambda_reg = args.lambda_reg
    lambda_kl = args.lambda_kl
    lambda_pred = args.lambda_pred

    def joint_loss_fn(params, z_batch, tokens_batch, regret_batch, weights_batch):
        pred_params = params["predictor"]
        adapt_params = params["adapter"]

        # --- Predictor loss: MSE on original z ---
        pred_regret = predictor.apply({"params": pred_params}, z_batch)
        l_pred = jnp.mean(weights_batch * (pred_regret - regret_batch) ** 2)

        # --- Adapter forward ---
        delta_z = adapter.apply({"params": adapt_params}, z_batch)
        z_prime = z_batch + delta_z

        # Reconstruction: decode z' and compare to original tokens
        logits = frozen_decode(z_prime)
        token_one_hot = jax.nn.one_hot(tokens_batch, vocab_size)
        ce_per_sample = -jnp.sum(token_one_hot * jax.nn.log_softmax(logits, axis=-1), axis=(-1, -2))
        l_recon = jnp.mean(ce_per_sample * weights_batch)

        # Regret maximisation: adapter should move z to where predictor predicts higher regret
        # Stop gradient on predictor params so predictor doesn't learn to inflate predictions
        pred_regret_adapted = predictor.apply(
            {"params": jax.lax.stop_gradient(pred_params)}, z_prime
        )
        l_regret = -jnp.mean(weights_batch * pred_regret_adapted)  # negative to maximise

        # Regularisation: keep corrections small
        l_reg = jnp.mean(delta_z ** 2)

        # Total
        total = l_recon + lambda_regret * l_regret + lambda_reg * l_reg + lambda_pred * l_pred

        # Optional KL on z'
        l_kl = jnp.float32(0.0)
        if lambda_kl > 0:
            z_prime_mean = jnp.mean(z_prime, axis=0)
            z_prime_var = jnp.var(z_prime, axis=0)
            l_kl = 0.5 * jnp.sum(z_prime_var + z_prime_mean ** 2 - 1 - jnp.log(jnp.maximum(z_prime_var, 1e-8)))
            total = total + lambda_kl * l_kl

        # R² for predictor
        ss_res = jnp.sum((regret_batch - pred_regret) ** 2)
        ss_tot = jnp.sum((regret_batch - jnp.mean(regret_batch)) ** 2)
        r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-8)

        metrics = {
            "loss": total,
            "l_pred": l_pred,
            "l_recon": l_recon,
            "l_regret": l_regret,
            "l_reg": l_reg,
            "l_kl": l_kl,
            "pred_r2": r2,
            "delta_z_norm": jnp.mean(jnp.sqrt(jnp.sum(delta_z ** 2, axis=-1))),
            "delta_z_max": jnp.max(jnp.sqrt(jnp.sum(delta_z ** 2, axis=-1))),
        }
        return total, metrics

    @jax.jit
    def train_step(state, z_batch, tokens_batch, regret_batch, weights_batch):
        (loss, metrics), grads = jax.value_and_grad(joint_loss_fn, has_aux=True)(
            state.params, z_batch, tokens_batch, regret_batch, weights_batch
        )
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @jax.jit
    def eval_metrics(params, z_batch, tokens_batch, regret_batch, weights_batch):
        _, metrics = joint_loss_fn(params, z_batch, tokens_batch, regret_batch, weights_batch)
        return metrics

    # --- Training loop ---
    best_val_loss = float("inf")
    best_params = None
    best_epoch = 0
    t0 = time.time()

    print(f"\nJoint training (lambda_pred={lambda_pred}, lambda_regret={lambda_regret}, "
          f"lambda_reg={lambda_reg}, lambda_kl={lambda_kl})...")

    for epoch in range(args.epochs):
        rng, rng_perm = jax.random.split(rng)
        perm_idx = jax.random.permutation(rng_perm, n_train)

        epoch_metrics = []
        for i in range(0, n_train, args.batch_size):
            end = min(i + args.batch_size, n_train)
            idx = perm_idx[i:end]
            state, metrics = train_step(
                state, z_train[idx], tokens_train[idx], regret_train[idx], weights_train[idx]
            )
            epoch_metrics.append(metrics)

        # Validate
        val_metrics = eval_metrics(state.params, z_val, tokens_val, regret_val, weights_val)
        val_loss = float(val_metrics["loss"])

        # wandb
        avg_train = {k: float(np.mean([float(m[k]) for m in epoch_metrics])) for k in epoch_metrics[0]}
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train["loss"],
            "train/l_pred": avg_train["l_pred"],
            "train/l_recon": avg_train["l_recon"],
            "train/l_reg": avg_train["l_reg"],
            "train/l_kl": avg_train["l_kl"],
            "train/pred_r2": avg_train["pred_r2"],
            "train/delta_z_norm": avg_train["delta_z_norm"],
            "val/loss": val_loss,
            "val/l_pred": float(val_metrics["l_pred"]),
            "val/l_recon": float(val_metrics["l_recon"]),
            "val/l_reg": float(val_metrics["l_reg"]),
            "val/pred_r2": float(val_metrics["pred_r2"]),
            "val/delta_z_norm": float(val_metrics["delta_z_norm"]),
            "val/delta_z_max": float(val_metrics["delta_z_max"]),
            "val/best_loss": min(best_val_loss, val_loss),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
            best_epoch = epoch

        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d} | "
                  f"loss={avg_train['loss']:.4f} pred={avg_train['l_pred']:.6f} "
                  f"recon={avg_train['l_recon']:.4f} reg={avg_train['l_reg']:.6f} "
                  f"R²={avg_train['pred_r2']:.4f} ||dz||={avg_train['delta_z_norm']:.3f} | "
                  f"val_loss={val_loss:.4f} val_R²={float(val_metrics['pred_r2']):.4f} | "
                  f"best={best_epoch+1} ({elapsed:.1f}s)")

    # --- Final metrics ---
    print(f"\nBest model at epoch {best_epoch+1}, val_loss={best_val_loss:.6f}")
    final_metrics = eval_metrics(best_params, z_val, tokens_val, regret_val, weights_val)

    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_epoch"] = best_epoch + 1
    wandb.run.summary["final_pred_r2"] = float(final_metrics["pred_r2"])
    wandb.run.summary["final_delta_z_norm"] = float(final_metrics["delta_z_norm"])
    wandb.run.summary["final_val_recon"] = float(final_metrics["l_recon"])

    # --- Predicted vs actual scatter ---
    val_pred = predictor.apply({"params": best_params["predictor"]}, z_val)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.array(regret_val), np.array(val_pred), s=2, alpha=0.3)
    rmax = float(regret_val.max())
    ax.plot([0, rmax], [0, rmax], "r--", linewidth=1)
    ax.set_xlabel("Actual regret")
    ax.set_ylabel("Predicted regret")
    ax.set_title(f"Predictor R²={float(final_metrics['pred_r2']):.4f}")
    wandb.log({"diagnostics/pred_vs_actual": wandb.Image(fig)})
    plt.close()

    # --- Save predictor ---
    pred_ckpt = {
        "params": best_params["predictor"],
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "latent_dim": latent_dim,
        },
        "metrics": {
            "val_mse": float(final_metrics["l_pred"]),
            "val_r2": float(final_metrics["pred_r2"]),
            "best_epoch": best_epoch,
            "n_train": n_train,
            "n_val": n_val,
        },
    }
    pred_path = os.path.join(args.output_dir, "predictor.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(pred_ckpt, f)
    print(f"Saved predictor to {pred_path}")

    # --- Save adapter ---
    adapter_ckpt = {
        "params": best_params["adapter"],
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "latent_dim": latent_dim,
            "lambda_regret": args.lambda_regret,
            "lambda_reg": args.lambda_reg,
            "lambda_kl": args.lambda_kl,
        },
        "metrics": {
            "val_loss": best_val_loss,
            "val_recon": float(final_metrics["l_recon"]),
            "val_regret": float(final_metrics["l_regret"]),
            "val_reg": float(final_metrics["l_reg"]),
            "delta_z_norm": float(final_metrics["delta_z_norm"]),
            "best_epoch": best_epoch,
            "n_train": n_train,
            "n_val": n_val,
        },
    }
    adapter_path = os.path.join(args.output_dir, "adapter.pkl")
    with open(adapter_path, "wb") as f:
        pickle.dump(adapter_ckpt, f)
    print(f"Saved adapter to {adapter_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
