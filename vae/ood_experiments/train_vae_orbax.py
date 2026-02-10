from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from checkpointing import load_model_params, maybe_save_checkpoint
from modeling import CluttrVAE, VAEConfig, kl_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train maze VAE with Orbax checkpointing.")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    parser.add_argument("--num_steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--recon_weight", type=float, default=500.0)
    parser.add_argument("--anneal_steps", type=int, default=100_000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--seq_len", type=int, default=52)
    parser.add_argument("--vocab_size", type=int, default=170)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    return parser.parse_args()


def make_config(args: argparse.Namespace) -> VAEConfig:
    return VAEConfig(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        recon_weight=args.recon_weight,
        anneal_steps=args.anneal_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout_rate=args.dropout_rate,
    )


def build_train_step(model: CluttrVAE, cfg: VAEConfig):
    @jax.jit
    def train_step(
        state: train_state.TrainState,
        batch: jnp.ndarray,
        z_rng: jax.Array,
        dropout_rng: jax.Array,
        step: jnp.ndarray,
    ):
        beta = kl_weight(step=step, anneal_steps=cfg.anneal_steps)

        def loss_fn(params):
            logits, mean, logvar, _ = model.apply(
                {"params": params},
                batch,
                z_rng,
                train=True,
                rngs={"dropout": dropout_rng},
            )
            labels = jax.nn.one_hot(batch, num_classes=cfg.vocab_size)
            recon = optax.softmax_cross_entropy(logits, labels).mean()
            kl = -0.5 * jnp.mean(jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1))
            total = cfg.recon_weight * recon + beta * kl
            metrics = {
                "loss": total,
                "recon": recon,
                "kl": kl,
                "beta": beta,
            }
            return total, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        del loss
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step


def save_config(path: Path, args: argparse.Namespace, cfg: VAEConfig) -> None:
    payload: Dict[str, object] = {
        "args": vars(args),
        "model_config": cfg.__dict__,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()
    cfg = make_config(args)

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(str(data_path))
    if dataset.ndim != 2:
        raise ValueError(f"Expected dataset shape [N, seq_len], got {dataset.shape}")
    if dataset.shape[1] != cfg.seq_len:
        raise ValueError(f"Sequence length mismatch: data has {dataset.shape[1]}, config expects {cfg.seq_len}")

    dataset = jnp.array(dataset, dtype=jnp.int32)

    model = CluttrVAE(cfg)

    key = jax.random.PRNGKey(args.seed)
    key, init_key, z_key, dropout_key = jax.random.split(key, 4)

    if args.resume_checkpoint:
        params, meta = load_model_params(args.resume_checkpoint)
        start_step = int(meta.get("step") or 0)
        print(f"Resumed params from {args.resume_checkpoint} (backend={meta.get('backend')}, step={start_step})")
    else:
        variables = model.init(
            {"params": init_key, "dropout": dropout_key},
            jnp.zeros((1, cfg.seq_len), dtype=jnp.int32),
            z_key,
            train=False,
        )
        params = variables["params"]
        start_step = 0

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(cfg.learning_rate),
    )

    train_step = build_train_step(model, cfg)
    n = int(dataset.shape[0])

    save_config(output_dir / "train_config.json", args, cfg)

    for step in range(start_step, args.num_steps):
        key, idx_key, z_key, dropout_key = jax.random.split(key, 4)
        idx = jax.random.randint(idx_key, (cfg.batch_size,), 0, n)
        batch = dataset[idx]

        state, metrics = train_step(
            state=state,
            batch=batch,
            z_rng=z_key,
            dropout_rng=dropout_key,
            step=jnp.array(step, dtype=jnp.float32),
        )

        if (step + 1) % args.log_every == 0:
            metrics_np = {k: float(v) for k, v in metrics.items()}
            print(
                f"step={step + 1} "
                f"loss={metrics_np['loss']:.4f} recon={metrics_np['recon']:.4f} "
                f"kl={metrics_np['kl']:.4f} beta={metrics_np['beta']:.4f}"
            )

        if (step + 1) % args.save_every == 0:
            saved = maybe_save_checkpoint(
                directory=str(output_dir / "checkpoints"),
                params=state.params,
                step=step + 1,
                config=cfg,
                prefer_orbax=True,
            )
            print(f"Saved checkpoint: {saved}")

    final_path = maybe_save_checkpoint(
        directory=str(output_dir / "checkpoints"),
        params=state.params,
        step=args.num_steps,
        config=cfg,
        prefer_orbax=True,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
