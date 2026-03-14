"""
Phase 3: Train the adapter MLP with frozen VAE decoder and frozen regret predictor.

Loss = L_recon + lambda_regret * L_regret + lambda_reg * L_reg

Usage:
    python adapter/train_adapter.py \
        --data_path /tmp/adapter_data/train_data.npz \
        --predictor_path /tmp/adapter_data/predictor.pkl \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --output_path /tmp/adapter_data/adapter.pkl \
        --lambda_regret 0.1 --lambda_reg 0.01 \
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
import yaml
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.training import train_state

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))

from adapter_model import AdapterMLP, RegretPredictor, create_adapter
from vae_model import CluttrVAE


def load_predictor(predictor_path):
    with open(predictor_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    model = RegretPredictor(hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"])
    return model, ckpt["params"]


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
    parser = argparse.ArgumentParser(description="Train adapter MLP")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--predictor_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_regret", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--lambda_kl", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="ADAPTER_TRAINING")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # --- wandb init ---
    run_name = args.run_name or f"adapter_lr{args.lambda_regret}_lreg{args.lambda_reg}"
    wandb.init(
        project=args.project,
        name=run_name,
        config=vars(args),
        tags=["adapter"],
    )

    # --- Load everything ---
    print("Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    z_all = jnp.array(data["z"])
    tokens_all = jnp.array(data["tokens"])
    regret_all = jnp.array(data["regret"])
    latent_dim = int(data["latent_dim"])
    n = len(z_all)
    print(f"  {n} samples, latent_dim={latent_dim}")

    print("Loading frozen regret predictor...")
    pred_model, pred_params = load_predictor(args.predictor_path)

    print("Loading frozen VAE decoder...")
    vae, vae_params, vae_cfg = load_vae_decoder(args.vae_checkpoint_path, args.vae_config_path)
    seq_len = vae_cfg["seq_len"]
    vocab_size = vae_cfg["vocab_size"]

    # --- Compute training weights ---
    weights_all = jnp.array(compute_training_weights(np.array(regret_all)))
    wandb.run.summary["weights_mean"] = float(weights_all.mean())
    wandb.run.summary["weights_max"] = float(weights_all.max())

    # --- Train/val split ---
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = np.random.RandomState(args.seed).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    z_train = z_all[train_idx]
    tokens_train = tokens_all[train_idx]
    regret_train = regret_all[train_idx]
    weights_train = weights_all[train_idx]

    z_val = z_all[val_idx]
    tokens_val = tokens_all[val_idx]
    regret_val = regret_all[val_idx]
    weights_val = weights_all[val_idx]

    # --- Create adapter ---
    adapter, adapter_params = create_adapter(latent_dim, args.hidden_dim, args.n_layers)
    tx = optax.adamw(args.lr, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=adapter.apply, params=adapter_params["params"], tx=tx
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(adapter_params))
    print(f"  Adapter params: {n_params:,}")
    wandb.run.summary["n_params"] = n_params

    # --- Build frozen forward functions ---
    @jax.jit
    def frozen_decode(z_prime):
        return jax.vmap(lambda z: vae.apply({"params": vae_params}, z, method=vae.decode))(z_prime)

    @jax.jit
    def frozen_predict(z):
        return pred_model.apply({"params": pred_params}, z)

    # --- Loss function ---
    lambda_regret = args.lambda_regret
    lambda_reg = args.lambda_reg
    lambda_kl = args.lambda_kl

    def loss_fn(adapter_params, z_batch, tokens_batch, regret_batch, weights_batch):
        delta_z = adapter.apply({"params": adapter_params}, z_batch)
        z_prime = z_batch + delta_z

        # 1. Reconstruction
        logits = frozen_decode(z_prime)
        token_one_hot = jax.nn.one_hot(tokens_batch, vocab_size)
        ce_per_sample = -jnp.sum(token_one_hot * jax.nn.log_softmax(logits, axis=-1), axis=(-1, -2))
        l_recon = jnp.mean(ce_per_sample * weights_batch)

        # 2. Regret prediction on ORIGINAL z
        pred_regret = frozen_predict(z_batch)
        l_regret = jnp.mean(weights_batch * (pred_regret - regret_batch) ** 2)

        # 3. Regularisation
        l_reg = jnp.mean(delta_z ** 2)

        total = l_recon + lambda_regret * l_regret + lambda_reg * l_reg

        # 4. Optional KL
        l_kl = jnp.float32(0.0)
        if lambda_kl > 0:
            z_prime_mean = jnp.mean(z_prime, axis=0)
            z_prime_var = jnp.var(z_prime, axis=0)
            l_kl = 0.5 * jnp.sum(z_prime_var + z_prime_mean ** 2 - 1 - jnp.log(jnp.maximum(z_prime_var, 1e-8)))
            total = total + lambda_kl * l_kl

        metrics = {
            "loss": total,
            "l_recon": l_recon,
            "l_regret": l_regret,
            "l_reg": l_reg,
            "l_kl": l_kl,
            "delta_z_norm": jnp.mean(jnp.sqrt(jnp.sum(delta_z ** 2, axis=-1))),
            "delta_z_max": jnp.max(jnp.sqrt(jnp.sum(delta_z ** 2, axis=-1))),
        }
        return total, metrics

    @jax.jit
    def train_step(state, z_batch, tokens_batch, regret_batch, weights_batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, z_batch, tokens_batch, regret_batch, weights_batch
        )
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @jax.jit
    def eval_metrics(state, z_batch, tokens_batch, regret_batch, weights_batch):
        _, metrics = loss_fn(state.params, z_batch, tokens_batch, regret_batch, weights_batch)
        return metrics

    # --- Training loop ---
    best_val_loss = float("inf")
    best_params = None
    best_epoch = 0

    t0 = time.time()
    print(f"\nTraining adapter (lambda_regret={lambda_regret}, lambda_reg={lambda_reg}, lambda_kl={lambda_kl})...")

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
        val_metrics = eval_metrics(state, z_val, tokens_val, regret_val, weights_val)
        val_loss = float(val_metrics["loss"])

        # wandb logging every epoch
        avg_train = {k: float(np.mean([float(m[k]) for m in epoch_metrics])) for k in epoch_metrics[0]}
        wandb.log({
            "epoch": epoch + 1,
            "adapter/train_loss": avg_train["loss"],
            "adapter/train_recon": avg_train["l_recon"],
            "adapter/train_regret": avg_train["l_regret"],
            "adapter/train_reg": avg_train["l_reg"],
            "adapter/train_kl": avg_train["l_kl"],
            "adapter/train_delta_z_norm": avg_train["delta_z_norm"],
            "adapter/val_loss": val_loss,
            "adapter/val_recon": float(val_metrics["l_recon"]),
            "adapter/val_regret": float(val_metrics["l_regret"]),
            "adapter/val_reg": float(val_metrics["l_reg"]),
            "adapter/val_delta_z_norm": float(val_metrics["delta_z_norm"]),
            "adapter/val_delta_z_max": float(val_metrics["delta_z_max"]),
            "adapter/best_val_loss": min(best_val_loss, val_loss),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
            best_epoch = epoch

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d} | "
                  f"train_loss={avg_train['loss']:.4f} recon={avg_train['l_recon']:.4f} "
                  f"regret={avg_train['l_regret']:.6f} reg={avg_train['l_reg']:.6f} "
                  f"||dz||={avg_train['delta_z_norm']:.3f} | "
                  f"val_loss={val_loss:.4f} | best={best_epoch+1} ({elapsed:.1f}s)")

    # --- Final metrics ---
    print(f"\nBest model at epoch {best_epoch+1}, val_loss={best_val_loss:.6f}")
    final_metrics = eval_metrics(
        state.replace(params=best_params), z_val, tokens_val, regret_val, weights_val
    )
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_epoch"] = best_epoch + 1
    wandb.run.summary["final_delta_z_norm"] = float(final_metrics["delta_z_norm"])
    wandb.run.summary["final_val_recon"] = float(final_metrics["l_recon"])

    # --- Save ---
    checkpoint = {
        "params": best_params,
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
    with open(args.output_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved adapter to {args.output_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
