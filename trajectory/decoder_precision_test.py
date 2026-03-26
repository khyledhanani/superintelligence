"""
Decoder precision test: check whether small z-perturbations around
high-learnability levels produce meaningful structural changes or noise.

For each high-SFL level in the buffer:
1. Encode it to z
2. Perturb z along each axis with varying epsilon
3. Decode the perturbed z
4. Measure: token edit distance, wall count change, solvability, structural similarity

This tells us whether the VAE decoder is "precise" enough in the region CMA-ES
searches for fine-tuning to potentially help.

Usage:
    python trajectory/decoder_precision_test.py
"""
import os
import sys
import subprocess
import shutil
import pickle
import numpy as np
import yaml

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vae.vae_model import CluttrVAE
from vae.vae_level_utils import tokens_to_level, repair_tokens, level_to_tokens
from jaxued.environments.maze import Maze

# ── Config ──────────────────────────────────────────────────────────────────
GCLOUD_BIN = shutil.which("gcloud") or "/cs/student/project_msc/2025/csml/rhautier/google-cloud-sdk/bin/gcloud"
LOCAL_DIR = "/tmp/decoder_precision"
GCS_BUCKET = "ucl-ued-project-bucket"

# VAE checkpoint (download if needed)
VAE_CKPT_GCS = f"gs://{GCS_BUCKET}/accel/checkpoints/accel_sfl/1/vae_beta10_checkpoint_500000.pkl"
VAE_CKPT_LOCAL = "/tmp/vae_beta10/checkpoint_500000.pkl"
VAE_CONFIG_LOCAL = "/tmp/vae_beta10/config.yaml"

# Buffer (reuse from nonstationarity analysis)
BUFFER_LOCAL = "/tmp/nonstationarity_analysis/buffer_dump_10k.npz"
BUFFER_GCS = f"gs://{GCS_BUCKET}/accel/buffer_dumps/accel_sfl/1/buffer_dump_10k.npz"

# Perturbation settings
EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
NUM_RANDOM_DIRS = 20  # number of random perturbation directions per level
NUM_LEVELS = 200      # number of buffer levels to test


def download_if_needed(gcs_path, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        subprocess.run([GCLOUD_BIN, "storage", "cp", gcs_path, local_path], check=True)


def load_vae():
    with open(VAE_CKPT_LOCAL, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    if os.path.exists(VAE_CONFIG_LOCAL):
        with open(VAE_CONFIG_LOCAL) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"vocab_size": 170, "embed_dim": 300, "latent_dim": 64, "seq_len": 52}

    vae = CluttrVAE(
        vocab_size=cfg.get("vocab_size", 170),
        embed_dim=cfg.get("embed_dim", 300),
        latent_dim=cfg.get("latent_dim", 64),
        seq_len=cfg.get("seq_len", 52),
    )
    return vae, params


def token_edit_distance(t1, t2):
    """Count how many token positions differ."""
    return jnp.sum(t1 != t2)


def wall_count(tokens):
    """Count non-zero wall tokens (first 50 positions)."""
    return jnp.sum(tokens[:50] > 0)


def is_solvable(level):
    """Quick solvability check via BFS on the wall_map."""
    # Use the maze environment's built-in check
    wall_map = level.wall_map
    agent_pos = level.agent_pos
    goal_pos = level.goal_pos

    # Simple flood-fill check
    visited = jnp.zeros_like(wall_map, dtype=jnp.bool_)
    # We can't do BFS in pure JAX easily, so just check if agent/goal are on walls
    agent_on_wall = wall_map[agent_pos[1], agent_pos[0]]
    goal_on_wall = wall_map[goal_pos[1], goal_pos[0]]
    return ~agent_on_wall & ~goal_on_wall  # basic validity, not full solvability


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Download data
    download_if_needed(BUFFER_GCS, BUFFER_LOCAL)

    # Load VAE
    print("Loading VAE...")
    vae, vae_params = load_vae()

    def decode_fn(z):
        return vae.apply({"params": vae_params}, z, method=vae.decode)

    def encode_fn(tokens):
        return vae.apply({"params": vae_params}, tokens, train=False, method=vae.encode)

    # Load buffer
    data = np.load(BUFFER_LOCAL, allow_pickle=True)
    tokens_all = jnp.array(data["tokens"][:int(data["size"])])
    scores = np.array(data["scores"][:int(data["size"])])
    print(f"Buffer: {len(scores)} levels")

    # Select levels: mix of high-SFL and random
    # High SFL = near the learning frontier
    sfl_proxy = scores  # buffer scores approximate SFL
    top_idx = np.argsort(sfl_proxy)[-NUM_LEVELS // 2:]  # top half by score
    rng_np = np.random.default_rng(42)
    rand_idx = rng_np.choice(len(scores), NUM_LEVELS // 2, replace=False)
    selected_idx = np.concatenate([top_idx, rand_idx])
    selected_idx = np.unique(selected_idx)[:NUM_LEVELS]

    selected_tokens = tokens_all[selected_idx]
    print(f"Selected {len(selected_idx)} levels for analysis")

    # Encode selected levels
    print("Encoding levels...")
    means, logvars = encode_fn(selected_tokens)
    print(f"Encoded: mean range [{float(means.min()):.2f}, {float(means.max()):.2f}], "
          f"std range [{float(jnp.exp(0.5*logvars).min()):.3f}, {float(jnp.exp(0.5*logvars).max()):.3f}]")

    # For each level, perturb along random directions and measure changes
    rng = jax.random.PRNGKey(0)

    results = {
        'epsilons': EPSILONS,
        'edit_distances': [],       # (num_levels, num_eps, num_dirs)
        'wall_count_changes': [],   # same
        'token_stability': [],      # fraction of positions unchanged
        'goal_agent_changes': [],   # did goal/agent position change?
    }

    @jax.jit
    def decode_and_compare(z_base, z_perturbed):
        """Decode base and perturbed z, return token-level metrics."""
        logits_base = decode_fn(z_base)
        tokens_base = jnp.argmax(logits_base, axis=-1)
        tokens_base = repair_tokens(tokens_base)

        logits_pert = decode_fn(z_perturbed)
        tokens_pert = jnp.argmax(logits_pert, axis=-1)
        tokens_pert = repair_tokens(tokens_pert)

        edit_dist = token_edit_distance(tokens_base, tokens_pert)
        wall_diff = jnp.abs(wall_count(tokens_base) - wall_count(tokens_pert))
        stable_frac = jnp.mean((tokens_base == tokens_pert).astype(jnp.float32))
        goal_changed = tokens_base[-2] != tokens_pert[-2]
        agent_changed = tokens_base[-1] != tokens_pert[-1]

        return edit_dist, wall_diff, stable_frac, goal_changed, agent_changed

    # Also measure: what's the logit entropy (confidence) at each position?
    @jax.jit
    def decode_entropy(z):
        logits = decode_fn(z)  # (seq_len, vocab_size)
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)  # (seq_len,)
        return entropy

    print("\nMeasuring decoder precision...")
    all_edit_dists = np.zeros((len(selected_idx), len(EPSILONS), NUM_RANDOM_DIRS))
    all_wall_diffs = np.zeros_like(all_edit_dists)
    all_stable_fracs = np.zeros_like(all_edit_dists)
    all_goal_changes = np.zeros_like(all_edit_dists)
    all_agent_changes = np.zeros_like(all_edit_dists)
    all_entropies = []

    for i in range(len(selected_idx)):
        if i % 50 == 0:
            print(f"  Level {i}/{len(selected_idx)}...")

        z_base = means[i]

        # Measure base entropy
        ent = decode_entropy(z_base)
        all_entropies.append(np.array(ent))

        for j, eps in enumerate(EPSILONS):
            for k in range(NUM_RANDOM_DIRS):
                rng, rng_dir = jax.random.split(rng)
                direction = jax.random.normal(rng_dir, shape=(64,))
                direction = direction / jnp.linalg.norm(direction)
                z_pert = z_base + eps * direction

                ed, wd, sf, gc, ac = decode_and_compare(z_base, z_pert)
                all_edit_dists[i, j, k] = float(ed)
                all_wall_diffs[i, j, k] = float(wd)
                all_stable_fracs[i, j, k] = float(sf)
                all_goal_changes[i, j, k] = float(gc)
                all_agent_changes[i, j, k] = float(ac)

    all_entropies = np.array(all_entropies)  # (num_levels, seq_len)

    # Analysis
    print("\n" + "=" * 70)
    print("DECODER PRECISION ANALYSIS")
    print("=" * 70)

    print(f"\nLevels analyzed: {len(selected_idx)}")
    print(f"Random directions per level: {NUM_RANDOM_DIRS}")

    print(f"\nDecoder confidence (logit entropy per token position):")
    print(f"  Mean entropy: {all_entropies.mean():.3f}")
    print(f"  Wall positions (0-49): {all_entropies[:, :50].mean():.3f}")
    print(f"  Goal position (50): {all_entropies[:, 50].mean():.3f}")
    print(f"  Agent position (51): {all_entropies[:, 51].mean():.3f}")
    print(f"  (Lower = more confident, 0 = deterministic, ln(170)={np.log(170):.2f} = uniform)")

    print(f"\nPerturbation sensitivity (mean over levels and directions):")
    print(f"{'Epsilon':>8} {'Edit Dist':>10} {'Wall Diff':>10} {'Stable%':>10} {'Goal Flip%':>12} {'Agent Flip%':>12}")
    print("-" * 65)
    for j, eps in enumerate(EPSILONS):
        ed = all_edit_dists[:, j, :].mean()
        wd = all_wall_diffs[:, j, :].mean()
        sf = all_stable_fracs[:, j, :].mean() * 100
        gc = all_goal_changes[:, j, :].mean() * 100
        ac = all_agent_changes[:, j, :].mean() * 100
        print(f"{eps:>8.2f} {ed:>10.1f} {wd:>10.1f} {sf:>9.1f}% {gc:>11.1f}% {ac:>11.1f}%")

    # Key diagnostic: at what epsilon does the decoder start producing
    # meaningfully different levels?
    print(f"\nInterpretation:")
    for j, eps in enumerate(EPSILONS):
        ed = all_edit_dists[:, j, :].mean()
        if ed > 1.0:
            print(f"  eps={eps}: First epsilon where mean edit distance > 1 token")
            break

    for j, eps in enumerate(EPSILONS):
        sf = all_stable_fracs[:, j, :].mean()
        if sf < 0.95:
            print(f"  eps={eps}: First epsilon where <95% of tokens are stable")
            break

    # CMA-ES context: what's the typical step size?
    print(f"\n  CMA-ES context: sigma typically starts at 1.0 and collapses to ~0.1")
    print(f"  In 64-dim space, a step of sigma=0.1 has norm ~0.1*sqrt(64) = {0.1*np.sqrt(64):.2f}")
    print(f"  A step of sigma=1.0 has norm ~1.0*sqrt(64) = {1.0*np.sqrt(64):.2f}")

    # Save results
    output_dir = os.path.join(LOCAL_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    np.savez(os.path.join(output_dir, "decoder_precision.npz"),
             epsilons=np.array(EPSILONS),
             edit_distances=all_edit_dists,
             wall_diffs=all_wall_diffs,
             stable_fracs=all_stable_fracs,
             goal_changes=all_goal_changes,
             agent_changes=all_agent_changes,
             entropies=all_entropies,
             selected_scores=scores[selected_idx])

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Edit distance vs epsilon
    ax = axes[0, 0]
    means_ed = all_edit_dists.mean(axis=(0, 2))
    p25 = np.percentile(all_edit_dists.mean(axis=2), 25, axis=0)
    p75 = np.percentile(all_edit_dists.mean(axis=2), 75, axis=0)
    ax.plot(EPSILONS, means_ed, 'b-o')
    ax.fill_between(EPSILONS, p25, p75, alpha=0.2)
    ax.set_xlabel('Perturbation Epsilon')
    ax.set_ylabel('Token Edit Distance')
    ax.set_title('Tokens Changed vs Perturbation Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1 token change')
    ax.legend()

    # 2. Token stability vs epsilon
    ax = axes[0, 1]
    means_sf = all_stable_fracs.mean(axis=(0, 2)) * 100
    ax.plot(EPSILONS, means_sf, 'g-o')
    ax.set_xlabel('Perturbation Epsilon')
    ax.set_ylabel('% Tokens Unchanged')
    ax.set_title('Token Stability vs Perturbation Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5)

    # 3. Entropy histogram
    ax = axes[0, 2]
    ax.hist(all_entropies.flatten(), bins=50, alpha=0.7, color='purple')
    ax.set_xlabel('Token Entropy (nats)')
    ax.set_ylabel('Count')
    ax.set_title('Decoder Confidence Distribution')
    ax.axvline(x=np.log(170), color='red', linestyle='--', label=f'Uniform ({np.log(170):.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Entropy by position
    ax = axes[1, 0]
    mean_ent_by_pos = all_entropies.mean(axis=0)
    ax.bar(range(52), mean_ent_by_pos)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Decoder Confidence by Token Position')
    ax.axvspan(50, 51.5, alpha=0.2, color='red', label='Goal/Agent')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Wall count change vs epsilon
    ax = axes[1, 1]
    means_wd = all_wall_diffs.mean(axis=(0, 2))
    ax.plot(EPSILONS, means_wd, 'r-o')
    ax.set_xlabel('Perturbation Epsilon')
    ax.set_ylabel('Wall Count Change')
    ax.set_title('Wall Count Sensitivity')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 6. Goal/Agent flip rate vs epsilon
    ax = axes[1, 2]
    gc_rate = all_goal_changes.mean(axis=(0, 2)) * 100
    ac_rate = all_agent_changes.mean(axis=(0, 2)) * 100
    ax.plot(EPSILONS, gc_rate, 'o-', label='Goal position flip')
    ax.plot(EPSILONS, ac_rate, 's-', label='Agent position flip')
    ax.set_xlabel('Perturbation Epsilon')
    ax.set_ylabel('% Flipped')
    ax.set_title('Goal/Agent Position Sensitivity')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "decoder_precision.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    print(f"Results saved to {output_dir}/decoder_precision.npz")


if __name__ == "__main__":
    main()
