"""
train_vae.py — CLUTTR VAE training script.

Works on:
  • Local GPU (UCL cluster)     →  config: platform: "local"
  • GCP TPU v2-8 / v3-8 / v4   →  config: platform: "gcp"
  • GCP GPU                     →  config: platform: "gcp"

Changes from original:
  1. Multi-device data parallelism via jax.sharding.
  2. GCS I/O via IOManager.
  3. Vertex AI Experiments tracking.
  4. decode() method on CluttrVAE for standalone generation.
  5. Fixed eval_inference argmax axis bug (was axis=1, now axis=-1).
  6. Fixed weight slicing bug (was [:-2:], now [:, :-2]).
  7. Fixed plot labels: clearly distinguish % vs raw values.
  8. New metrics: per-token accuracy, wall IoU, duplicate walls, latent stats.
  9. New plots: latent diagnostics (active units, mean/std distributions).
"""
import math
import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
import yaml
import pickle
import functools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm
from jax.experimental.shard_map import shard_map
import functools
from jax.experimental.shard_map import shard_map

from utils import evaluate_cluttr_metrics, compute_latent_stats
from gcp_io import IOManager
from experiment_tracker import ExperimentTracker
from run_manager import RunManager

# JAX sharding imports
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# ==========================================
# CONFIG
# ==========================================
with open("vae_train_config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)


# ==========================================
# DEVICE SETUP
# ==========================================
def setup_devices():
    """Detect devices and create a 1-D Mesh for data parallelism."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"[Devices] Found {n_devices} device(s): {[str(d) for d in devices]}")
    mesh = Mesh(np.array(devices), axis_names=("batch",))
    return mesh, n_devices


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class HighwayStage(nn.Module):
    dim: int = 300

    def setup(self):
        self.dense_g = nn.Dense(self.dim)
        self.dense_fg = nn.Dense(self.dim)
        self.dense_q1 = nn.Dense(self.dim)
        self.dense_q2 = nn.Dense(self.dim)
        self.dense_gate = nn.Dense(self.dim)

    def __call__(self, x):
        g = nn.relu(self.dense_g(x))
        f_g_x = nn.relu(self.dense_fg(g))
        q_x = self.dense_q2(nn.relu(self.dense_q1(x)))
        gate = nn.sigmoid(self.dense_gate(x))
        return gate * f_g_x + (1.0 - gate) * q_x


class CluttrVAE(nn.Module):

    def setup(self):
        # Encoder
        self.embed = nn.Embed(CONFIG["vocab_size"], CONFIG["embed_dim"])
        self.enc_drop1 = nn.Dropout(rate=0.1)
        self.enc_hw1 = HighwayStage(CONFIG["embed_dim"])
        self.enc_hw2 = HighwayStage(CONFIG["embed_dim"])
        self.enc_bilstm = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)),
            nn.RNN(nn.LSTMCell(300)),
        )
        self.enc_drop2 = nn.Dropout(rate=0.1)
        self.mean_layer = nn.Dense(CONFIG["latent_dim"])
        self.logvar_layer = nn.Dense(CONFIG["latent_dim"])

        # Decoder
        self.dec_bilstm1 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)),
        )
        self.dec_bilstm2 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)),
        )
        self.dec_output = nn.Dense(CONFIG["vocab_size"])

    def __call__(self, x, z_rng, train: bool = True):
        mean, logvar = self.encode(x, train=train)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std
        logits = self.decode(z)
        return logits, mean, logvar

    def encode(self, x, train: bool = True):
        x = self.embed(x)
        x = self.enc_drop1(x, deterministic=not train)
        x = self.enc_hw1(x)
        x = self.enc_hw2(x)
        outputs = self.enc_bilstm(x)
        outputs = self.enc_drop2(outputs, deterministic=not train)

        fwd_out = outputs[:, :, :300]
        bwd_out = outputs[:, :, 300:]
        h = jnp.concatenate([fwd_out[:, -1, :], bwd_out[:, 0, :]], axis=-1)

        #mean = jnp.tanh(self.mean_layer(h)) * 4.0
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

    def decode(self, z):
        squeeze = False
        if z.ndim == 1:
            z = z[jnp.newaxis, :]
            squeeze = True

        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, CONFIG["seq_len"], 1))
        d_out = self.dec_bilstm1(z_seq)
        d_out = self.dec_bilstm2(d_out)
        logits = self.dec_output(d_out)

        if squeeze:
            logits = logits[0]
        return logits


# ==========================================
# STANDALONE GENERATION HELPERS
# ==========================================
def generate_from_latent(params, z, temperature=1.0):
    """
    Generate token sequences from latent vectors.

    Args:
        params: trained model parameters.
        z: (batch, latent_dim) or (latent_dim,) latent vectors.
        temperature: softmax temperature (1.0 = standard, <1 = sharper, >1 = softer).

    Returns:
        tokens: integer token array, same leading dims as z.
    """
    logits = CluttrVAE().apply({"params": params}, z, method=CluttrVAE.decode)
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)
    else:
        return jnp.argmax(logits / temperature, axis=-1)


def sample_prior(key, params, n_samples=1, temperature=1.0):
    """
    Sample from the prior N(0, I) and decode.

    Args:
        key: JAX PRNG key.
        params: trained model parameters.
        n_samples: number of samples to generate.
        temperature: decoding temperature.

    Returns:
        tokens: (n_samples, seq_len) integer array.
    """
    z = jax.random.normal(key, (n_samples, CONFIG["latent_dim"]))
    return generate_from_latent(params, z, temperature=temperature)


def interpolate_latent(params, z1, z2, n_steps=10):
    """
    Linear interpolation between two latent vectors, decoded at each step.

    Args:
        params: trained model parameters.
        z1, z2: (latent_dim,) arrays.
        n_steps: number of interpolation steps.

    Returns:
        tokens: (n_steps, seq_len) integer array.
    """
    alphas = jnp.linspace(0.0, 1.0, n_steps)
    z_interp = jnp.stack([z1 * (1 - a) + z2 * a for a in alphas])
    return generate_from_latent(params, z_interp)


# ==========================================
# TRAINING UTILITIES
# ==========================================
def get_kl_weight(step, scaled_anneal_steps, beta_max = 0.5):
    return jnp.minimum(beta_max, (step / scaled_anneal_steps)*beta_max)


@jax.jit
def get_latents(state, batch, z_rng):
    """Extract mean and logvar from the encoder (for latent diagnostics)."""
    mean, logvar = CluttrVAE().apply(
        {"params": state.params}, batch, method=CluttrVAE.encode, train=False
    )
    return mean, logvar


@jax.jit
def eval_inference(state, batch, z_rng):
    """Argmax decoding for metric evaluation."""
    logits, _, _ = CluttrVAE().apply(
        {"params": state.params}, batch, z_rng, train=False
    )
    # FIX: was axis=1 (wrong — that's the seq dim). Must be axis=-1 (vocab dim).
    return jnp.argmax(logits, axis=-1)


@jax.jit
def eval_loss_step(state, batch, z_rng):
    logits, mean, logvar = CluttrVAE().apply(
        {"params": state.params}, batch, z_rng, train=False
    )
    labels_onehot = jax.nn.one_hot(batch, num_classes=CONFIG["vocab_size"])
    recon_loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    kl_loss = -0.5 * jnp.mean(
        jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)
    )
    return recon_loss, kl_loss


def train_step(state, batch, z_rng, kl_weight):
    z_key, dropout_key = jax.random.split(z_rng)

    def loss_fn(params):
        logits, mean, logvar = CluttrVAE().apply(
            {"params": params}, batch, z_key, train=True,
            rngs={"dropout": dropout_key},
        )
        labels_onehot = jax.nn.one_hot(batch, num_classes=CONFIG["vocab_size"])
        per_token_loss = optax.softmax_cross_entropy(logits, labels_onehot)

        weights = jnp.ones_like(per_token_loss)
        weights = weights.at[:, :-2].set(CONFIG['wall_weight'])
        weights = weights.at[:, -2:].set(CONFIG['agent_goal_weight'])
        weights = weights / jnp.mean(weights, axis=-1, keepdims=True)

        recon_loss = jnp.mean(per_token_loss * weights)
        kl_loss = -0.5 * jnp.mean(
            jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)
        )
        total_loss = CONFIG["recon_weight"] * recon_loss + kl_weight * kl_loss
        return total_loss, (recon_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon, kl)), grads = grad_fn(state.params)

    # --- THE COLLECTIVE OPERATION ---
    # Now that we'll use shard_map, 'batch' is a valid axis name!
    grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), grads)

    # Also average scalar losses across devices so logged values are global averages,
    # not just one device's local-batch estimate.
    loss  = jax.lax.pmean(loss,  axis_name="batch")
    recon = jax.lax.pmean(recon, axis_name="batch")
    kl    = jax.lax.pmean(kl,    axis_name="batch")

    state = state.apply_gradients(grads=grads)
    return state, loss, recon, kl


# ==========================================
# MAIN TRAINING ENGINE
# ==========================================
def run_training():
    # ------------------------------------------------------------------
    # 0. Platform, devices, I/O, run management, tracking
    # ------------------------------------------------------------------
    mesh, n_devices = setup_devices()
    io = IOManager(CONFIG)

    # RunManager handles: run ID, config snapshot, checkpoint discovery, run log
    rm = RunManager(CONFIG, io)
    run_id, effective_config, start_step, restored_params = rm.setup_run()

    # Use the effective config (which may come from a resumed run's saved config,
    # NOT the current vae_train_config.yml — this ensures consistency).
    cfg = effective_config


    global_batch_size = cfg["batch_size"] * n_devices

    # --- HYBRID SCALING FOR VAEs ---
    # 1. DO scale the Learning Rate to match the larger global batch size
    scaled_lr = cfg["learning_rate"] * math.sqrt(n_devices)
    
    # 2. DO NOT scale the steps. The VAE needs all 500k iterations to prevent collapse.
    scaled_num_steps = cfg["num_steps"]
    scaled_anneal_steps = cfg["anneal_steps"]
    
    # 3. DO NOT scale the logging frequencies, since we are running the full 500k steps.
    scaled_plot_freq = cfg["plot_freq"]
    scaled_save_freq = cfg["save_freq"]
    scaled_log_freq = 100

    tracker = ExperimentTracker(cfg)
    print(f"[Training] Run: {run_id}")
    print(f"[Training] Per-device batch: {cfg['batch_size']} | "
          f"Global batch: {global_batch_size} | Devices: {n_devices}")

    tracker.log_params({
        "run_id": run_id,
        "num_steps": cfg["num_steps"],
        "global_batch_size": global_batch_size,
        "per_device_batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "recon_weight": cfg["recon_weight"],
        "latent_dim": cfg["latent_dim"],
        "embed_dim": cfg["embed_dim"],
        "anneal_steps": cfg["anneal_steps"],
        "n_devices": n_devices,
        "platform": cfg["platform"],
    })

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not io.exists(cfg["train_data_path"]):
        print(f"Error: Training data not found at {cfg['train_data_path']}")
        return
    if not io.exists(cfg["validation_data_path"]):
        print(f"Error: Validation data not found at {cfg['validation_data_path']}")
        return

    dataset_np = io.load_npy(cfg["train_data_path"])
    total_size = len(dataset_np)
    print(f"Loaded Train dataset: {dataset_np.shape}")

    val_dataset_np = io.load_npy(cfg["validation_data_path"])
    val_dataset_small_np = val_dataset_np[:1000]
    print(f"Loaded Val dataset (subset): {val_dataset_small_np.shape}")


    # --- Adding algorithmetically generated mazes within the training data --- #
    algo_mix_ratio = cfg.get("algo_mix_ratio", 0.0)

    if algo_mix_ratio > 0.0:
        if not io.exists(cfg["algo_data_path"]):
            print(f"Error: Algo data not found at {cfg['algo_data_path']}")
            return
            
        algo_dataset_full = io.load_npy(cfg["algo_data_path"])
        
        # Calculate exactly how many samples we need
        num_algo = int(total_size * algo_mix_ratio)
        num_std = total_size - num_algo
        
        print(f"[Data] Mixing {algo_mix_ratio*100}% algorithmically generated mazes.")
        print(f"[Data] Keeping {num_std} standard | Injecting {num_algo} algo")
        
        # Take the first N from standard, and the first N from algo
        std_part = dataset_np[:num_std]
        algo_part = algo_dataset_full[:num_algo]
        
        # Combine them to restore the dataset to exactly total_size (e.g. 200k)
        dataset_np = np.concatenate([std_part, algo_part], axis=0)
        np.random.shuffle(dataset_np)


    # ------------------------------------------------------------------
    # 2. Shard data across devices
    # ------------------------------------------------------------------
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated_sharding = NamedSharding(mesh, P())

    def shard_batch(batch_np):
        return jax.device_put(jnp.array(batch_np), data_sharding)

    val_dataset_small = jax.device_put(jnp.array(val_dataset_small_np), replicated_sharding)
    dataset_jnp_1k = jax.device_put(jnp.array(dataset_np[:1000]), replicated_sharding)

    # ------------------------------------------------------------------
    # 3. Initialize model & state
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(0)
    model = CluttrVAE()
    key, init_key, z_key = jax.random.split(key, 3)

    if restored_params is not None:
        # RunManager already loaded params from the latest checkpoint
        params = restored_params
        print(f"Resuming from step {start_step}")
    else:
        params = model.init(
            init_key,
            jnp.zeros((1, cfg["seq_len"]), dtype=jnp.int32),
            z_key,
            train=False,
        )["params"]

    params = jax.device_put(params, replicated_sharding)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        #tx=optax.adam(cfg["learning_rate"]),
        tx=optax.adam(scaled_lr)
    )

    # ------------------------------------------------------------------
    # 4. Training loop data structures
    # ------------------------------------------------------------------
    history = {
        "train_recon": [], "train_kl": [],
        "val_recon": [], "val_kl": [],
        "steps": [], "val_steps": [],
    }

    metric_history = {
        "steps": [],
        "train": {
            "validity": [], "agent_acc": [], "goal_acc": [],
            "wall_err": [], "per_token_acc": [], "wall_iou": [],
            "duplicate_wall_pct": [],
        },
        "val": {
            "validity": [], "agent_acc": [], "goal_acc": [],
            "wall_err": [], "per_token_acc": [], "wall_iou": [],
            "duplicate_wall_pct": [],
        },
    }

    latent_history = {
        "steps": [],
        "active_units": [],
        "active_units_pct": [],
        "mean_avg": [],
        "mean_std": [],
        "std_avg": [],
        "std_min": [],
    }

    EVAL_METRIC_FREQ = cfg.get("eval_metric_freq", 10000)
    scaled_eval_metric_freq = max(1, EVAL_METRIC_FREQ)

    # --- Create figure: 7 subplots ---
    fig, axes = plt.subplots(7, 1, figsize=(12, 35), sharex=True)
    ax_kl, ax_recon, ax_pct, ax_wall_err, ax_wall_iou, ax_latent, ax_active = axes

    # --- DEFINE SHARDED STEP HERE ---
    # Put this right before the 'with mesh:' or 'for step in progress_bar'
    sharded_train_step = shard_map(
        train_step,
        mesh=mesh,
        in_specs=(
            P(),        # state: replicated across all devices
            P("batch"), # batch: sharded across the 'batch' axis of the mesh
            P(),        # z_rng: replicated
            P()         # kl_weight: replicated
        ),
        out_specs=(P(), P(), P(), P()), # loss, recon, kl are returned as replicated values
        check_rep=False
    )
    
    # JIT the sharded function so it's fast
    sharded_train_step_jit = jax.jit(sharded_train_step)



    progress_bar = tqdm(range(start_step, scaled_num_steps), desc=f"Training [{run_id}]")

    with mesh:
        for step in progress_bar:
            # --- TRAINING STEP ---
            key, subkey = jax.random.split(key)
            idx = np.random.randint(0, len(dataset_np), size=(global_batch_size,))
            batch = shard_batch(dataset_np[idx])

            key, z_rng = jax.random.split(key)
            beta_max = cfg.get("beta_max", 0.5)
            kl_w = get_kl_weight(step, scaled_anneal_steps, beta_max=beta_max)

            state, loss, recon, kl = sharded_train_step_jit(state, batch, z_rng, kl_w)

            recon_f = float(recon)
            kl_f = float(kl)

            history["train_recon"].append(recon_f)
            history["train_kl"].append(kl_f)
            history["steps"].append(step)

            # Vertex AI time-series (every 100 steps)
            if step % scaled_log_freq == 0:
                tracker.log_time_series_metrics(
                    {"train_recon_loss": recon_f, "train_kl_loss": kl_f},
                    step=step,
                )

            # --- METRICS EVALUATION (slow, infrequent) ---
            if step % scaled_eval_metric_freq == 0 and step > 0:
                train_preds = eval_inference(state, dataset_jnp_1k, z_rng)
                val_preds = eval_inference(state, val_dataset_small, z_rng)

                # Run reachability check every 5th metric eval (expensive)
                do_reachability = (step % (scaled_eval_metric_freq * 5) == 0)

                train_m = evaluate_cluttr_metrics(
                    dataset_np[:1000], np.array(train_preds), pad_token=0,
                    check_reachability=do_reachability,
                )
                val_m = evaluate_cluttr_metrics(
                    val_dataset_small_np, np.array(val_preds), pad_token=0,
                    check_reachability=do_reachability,
                )

                metric_history["steps"].append(step)
                for split, m in [("train", train_m), ("val", val_m)]:
                    metric_history[split]["validity"].append(m["validity_pct"])
                    metric_history[split]["agent_acc"].append(m["agent_acc_pct"])
                    metric_history[split]["goal_acc"].append(m["goal_acc_pct"])
                    metric_history[split]["wall_err"].append(m["avg_wall_count_error"])
                    metric_history[split]["per_token_acc"].append(m["per_token_acc_pct"])
                    metric_history[split]["wall_iou"].append(m["avg_wall_iou"])
                    metric_history[split]["duplicate_wall_pct"].append(m["duplicate_wall_sample_pct"])

                # --- Latent diagnostics ---
                mean_arr, logvar_arr = get_latents(state, val_dataset_small, z_rng)
                lstats = compute_latent_stats(mean_arr, logvar_arr)

                latent_history["steps"].append(step)
                latent_history["active_units"].append(lstats["active_units"])
                latent_history["active_units_pct"].append(lstats["active_units_pct"])
                latent_history["mean_avg"].append(lstats["latent_mean_avg"])
                latent_history["mean_std"].append(lstats["latent_mean_std"])
                latent_history["std_avg"].append(lstats["latent_std_avg"])
                latent_history["std_min"].append(lstats["latent_std_min"])

                # Log to Vertex AI
                tracker.log_time_series_metrics({
                    "val_validity_pct": val_m["validity_pct"],
                    "val_agent_acc_pct": val_m["agent_acc_pct"],
                    "val_per_token_acc_pct": val_m["per_token_acc_pct"],
                    "val_wall_iou": val_m["avg_wall_iou"],
                    "active_units": float(lstats["active_units"]),
                    "active_units_pct": lstats["active_units_pct"],
                }, step=step)

            # --- VALIDATION LOSS + PLOTTING ---
            if step % scaled_plot_freq == 0 and step > 0:
                val_recon, val_kl = eval_loss_step(state, val_dataset_small, z_rng)
                val_recon_f = float(val_recon)
                val_kl_f = float(val_kl)

                history["val_recon"].append(val_recon_f)
                history["val_kl"].append(val_kl_f)
                history["val_steps"].append(step)

                progress_bar.set_postfix({
                    "Recon": f"{recon_f:.2f}",
                    "ValRecon": f"{val_recon_f:.2f}",
                    "KL": f"{kl_f:.2f}",
                })

                tracker.log_time_series_metrics(
                    {"val_recon_loss": val_recon_f, "val_kl_loss": val_kl_f},
                    step=step,
                )

                # ===================== PLOTTING =====================
                for ax in axes:
                    ax.clear()

                # --- Plot 1: KL Divergence ---
                ax_kl.plot(history["steps"], history["train_kl"],
                           label="Train KL", color="red", alpha=0.7)
                ax_kl.set_ylabel("KL (nats)")
                ax_kl.set_title(f"KL Divergence (\u03b2 = {float(kl_w):.3f})")
                ax_kl.legend()
                ax_kl.grid(True, alpha=0.3)

                # --- Plot 2: Reconstruction Loss ---
                ax_recon.plot(history["steps"], history["train_recon"],
                              label="Train Recon", color="blue", alpha=0.4, linewidth=1)
                ax_recon.plot(history["val_steps"], history["val_recon"],
                              label="Val Recon", color="darkblue", linewidth=2, linestyle="--")
                ax_recon.set_ylabel("Cross-Entropy")
                ax_recon.set_title("Reconstruction Loss")
                ax_recon.legend()
                ax_recon.grid(True, alpha=0.3)

                if len(metric_history["steps"]) > 0:
                    ms = metric_history["steps"]

                    # --- Plot 3: Percentages (all truly %) ---
                    ax_pct.plot(ms, metric_history["train"]["validity"],
                                label="Train Validity", color="green", marker="o")
                    ax_pct.plot(ms, metric_history["train"]["agent_acc"],
                                label="Train Agent Acc", color="orange", marker="x")
                    ax_pct.plot(ms, metric_history["train"]["goal_acc"],
                                label="Train Goal Acc", color="purple", marker="^")
                    ax_pct.plot(ms, metric_history["train"]["per_token_acc"],
                                label="Train Per-Token Acc", color="teal", marker="s")

                    ax_pct.plot(ms, metric_history["val"]["validity"],
                                label="Val Validity", color="green", ls="--", marker="o", alpha=0.7)
                    ax_pct.plot(ms, metric_history["val"]["agent_acc"],
                                label="Val Agent Acc", color="orange", ls="--", marker="x", alpha=0.7)
                    ax_pct.plot(ms, metric_history["val"]["goal_acc"],
                                label="Val Goal Acc", color="purple", ls="--", marker="^", alpha=0.7)
                    ax_pct.plot(ms, metric_history["val"]["per_token_acc"],
                                label="Val Per-Token Acc", color="teal", ls="--", marker="s", alpha=0.7)

                    ax_pct.set_ylabel("Percentage (%)")
                    ax_pct.set_ylim([-5, 105])
                    ax_pct.set_title("Accuracy & Validity (all values are %)")
                    ax_pct.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                    ax_pct.grid(True, alpha=0.3)

                    # --- Plot 4: Wall Count Error (raw count, NOT %) ---
                    ax_wall_err.plot(ms, metric_history["train"]["wall_err"],
                                     label="Train", color="brown", marker=".")
                    ax_wall_err.plot(ms, metric_history["val"]["wall_err"],
                                     label="Val", color="brown", ls="--", marker=".")
                    ax_wall_err.set_ylabel("Avg |pred_walls - true_walls|")
                    ax_wall_err.set_title("Wall Count Error (raw count, lower is better)")
                    ax_wall_err.legend()
                    ax_wall_err.grid(True, alpha=0.3)

                    # --- Plot 5: Wall IoU + Duplicate Rate ---
                    ax_wall_iou.plot(ms, metric_history["train"]["wall_iou"],
                                     label="Train Wall IoU", color="navy", marker="o")
                    ax_wall_iou.plot(ms, metric_history["val"]["wall_iou"],
                                     label="Val Wall IoU", color="navy", ls="--", marker="o", alpha=0.7)
                    # Convert duplicate % to 0-1 for same axis
                    ax_wall_iou.plot(ms, [d / 100.0 for d in metric_history["train"]["duplicate_wall_pct"]],
                                     label="Train Dup Wall Rate", color="crimson", marker="x")
                    ax_wall_iou.plot(ms, [d / 100.0 for d in metric_history["val"]["duplicate_wall_pct"]],
                                     label="Val Dup Wall Rate", color="crimson", ls="--", marker="x", alpha=0.7)
                    ax_wall_iou.set_ylabel("Ratio (0\u20131)")
                    ax_wall_iou.set_ylim([-0.05, 1.05])
                    ax_wall_iou.set_title("Wall Set IoU (higher=better) & Duplicate Rate (lower=better)")
                    ax_wall_iou.legend(fontsize=8)
                    ax_wall_iou.grid(True, alpha=0.3)

                if len(latent_history["steps"]) > 0:
                    ls_steps = latent_history["steps"]

                    # --- Plot 6: Latent Space Health ---
                    ax_latent.plot(ls_steps, latent_history["mean_avg"],
                                   label="Mean(\u03bc)", color="blue", marker="o")
                    ax_latent.plot(ls_steps, latent_history["mean_std"],
                                   label="Std(\u03bc across batch)", color="blue", ls="--", marker=".")
                    ax_latent.plot(ls_steps, latent_history["std_avg"],
                                   label="Mean(\u03c3)", color="red", marker="o")
                    ax_latent.plot(ls_steps, latent_history["std_min"],
                                   label="Min(\u03c3)", color="red", ls=":", marker=".")
                    ax_latent.axhline(y=1.0, color="gray", ls=":", alpha=0.5, label="Prior \u03c3=1")
                    ax_latent.axhline(y=0.0, color="gray", ls=":", alpha=0.3)
                    ax_latent.set_ylabel("Value")
                    ax_latent.set_title("Latent Space Diagnostics (watch for posterior collapse)")
                    ax_latent.legend(fontsize=8)
                    ax_latent.grid(True, alpha=0.3)

                    # --- Plot 7: Active Units ---
                    ax_active.plot(ls_steps, latent_history["active_units"],
                                   label=f"Active Units (of {cfg['latent_dim']})",
                                   color="darkgreen", marker="o")
                    ax_active.axhline(y=cfg["latent_dim"], color="gray", ls=":",
                                      alpha=0.5, label=f"Max = {cfg['latent_dim']}")
                    ax_active.set_ylabel("Count")
                    ax_active.set_ylim([0, cfg["latent_dim"] + 5])
                    ax_active.set_title("Active Latent Units (dims where Var[\u03bc_d] > 0.01)")
                    ax_active.legend()
                    ax_active.grid(True, alpha=0.3)

                # Placeholder text for empty plots
                if len(metric_history["steps"]) == 0:
                    for ax in [ax_pct, ax_wall_err, ax_wall_iou]:
                        ax.text(0.5, 0.5, "Metrics not computed yet...",
                                ha="center", transform=ax.transAxes)
                if len(latent_history["steps"]) == 0:
                    for ax in [ax_latent, ax_active]:
                        ax.text(0.5, 0.5, "Latent stats not computed yet...",
                                ha="center", transform=ax.transAxes)

                ax_active.set_xlabel("Training Steps")
                fig.tight_layout()
                rm.save_plot(fig)

            # --- Checkpoint ---
            if step % scaled_save_freq == 0 and step > 0:
                rm.save_checkpoint(state, step)

    # --- Final save ---
    rm.save_checkpoint(state, "final")
    rm.finalize_run({"final_train_recon": recon_f, "final_train_kl": kl_f})
    tracker.log_metrics({"final_train_recon": recon_f, "final_train_kl": kl_f})
    tracker.end_run()
    plt.close(fig)
    print(f"Training Complete. Run: {run_id}")


if __name__ == "__main__":
    run_training()