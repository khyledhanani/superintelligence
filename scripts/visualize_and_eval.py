#!/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
"""Visualise ES-discovered mazes and evaluate trained agents on prefab test mazes.

Outputs:
  eval_results/
    {strategy}_decoded_mazes.png    — top mazes decoded from ES mean latents
    {strategy}_prefab_results.json  — solve rate & mean return on each prefab
    prefab_comparison.png           — bar chart comparing strategies
"""
import functools
import os, sys, json, pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "accel_training"))
sys.path.insert(0, _ROOT)

from jaxued.environments.maze import Maze, Level
from jaxued.environments import MazeRenderer
from jaxued.wrappers import AutoReplayWrapper
from agent_loader import ActorCritic
from ued_interface import load_vae, build_eval_fn
from vae_decoder import decode_latent_to_env, repair_cluttr_sequence
from ued_interface import cluttr_sequence_to_level, flood_fill_solvable

import yaml

# ── Config ──────────────────────────────────────────────────────────────
RUNS_DIR = os.path.join(_ROOT, "runs")
OUT_DIR  = os.path.join(_ROOT, "eval_results")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(_ROOT, "accel_training", "config.yml")) as f:
    config = yaml.safe_load(f)

STRATEGIES = ["phase5-cma-es", "phase5-ns-es"]
PREFAB_NAMES = [
    "SixteenRooms", "SixteenRooms2",
    "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
    "StandardMaze", "StandardMaze2", "StandardMaze3",
]
N_EVAL_ROLLOUTS = 10   # rollouts per prefab per strategy
ROLLOUT_STEPS   = 256

# ── Setup ───────────────────────────────────────────────────────────────
print("Loading VAE decoder...")
decoder_params = load_vae(config["vae_checkpoint"])

maze_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
env = AutoReplayWrapper(maze_env)
env_params = env.default_params
renderer = MazeRenderer(maze_env, tile_size=16)
network = ActorCritic(action_dim=7)

print("Building eval function...")
eval_fn, eval_env = build_eval_fn(
    decoder_params, network, env_params,
    latent_dim=64,
    decode_temperature=config["decode_temperature"],
    num_steps=config.get("eval_rollout_steps", 128),
    min_obstacles=config["min_obstacles"],
    min_distance=config["min_distance"],
)

# ── Helper: run agent on a single level for N rollouts ──────────────────
@functools.partial(jax.jit, static_argnums=(3,))
def eval_agent_on_level(rng, params, level, n_steps=256):
    """Run agent on a level. Returns (total_reward, solved)."""
    rng_reset, rng_step = jax.random.split(rng)
    obs, state = env.reset_to_level(rng_reset, level, env_params)
    hstate = ActorCritic.initialize_carry((1,))
    hstate = jax.tree_util.tree_map(lambda x: x.squeeze(0), hstate)
    done = jnp.array(False)
    total_reward = 0.0

    def _step(carry, _):
        rng, hstate, obs, state, done, total_reward = carry
        rng, rng_s = jax.random.split(rng)
        # x needs (seq_len=1, batch=1, ...), hstate needs (batch=1, ...)
        x = jax.tree_util.tree_map(lambda a: a[None, None, ...], (obs, done))
        hstate_batched = jax.tree_util.tree_map(lambda a: a[None, ...], hstate)
        hstate_batched, pi, value = network.apply(params, x, hstate_batched)
        hstate_out = jax.tree_util.tree_map(lambda a: a.squeeze(0), hstate_batched)
        action = jnp.argmax(pi.logits[0, 0])
        obs, state, reward, new_done, info = env.step(rng_s, state, action, env_params)
        total_reward = total_reward + reward
        return (rng, hstate_out, obs, state, new_done, total_reward), reward

    (_, _, _, final_state, final_done, total_reward), rewards = jax.lax.scan(
        _step, (rng_step, hstate, obs, state, done, 0.0), None, length=n_steps
    )
    # Check if agent reached goal (got reward >= 1.0 at some point)
    solved = jnp.any(rewards >= 0.99)
    return total_reward, solved


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Decode mazes from ES mean latent vectors
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Decoding mazes from ES means ===")

for strat in STRATEGIES:
    es_path = os.path.join(RUNS_DIR, strat, "checkpoint_0020000", "es_state.pkl")
    if not os.path.exists(es_path):
        print(f"  {strat}: no checkpoint, skipping")
        continue

    with open(es_path, "rb") as f:
        es = pickle.load(f)

    # Collect mean latent(s)
    if "particles" in es:
        means = [np.array(p["mean"]) for p in es["particles"]]
    else:
        means = [np.array(es["mean"])]

    # For each mean, decode the mean itself + 8 perturbations around it
    rng = jax.random.PRNGKey(42)
    all_images = []
    all_labels = []

    for mi, mean in enumerate(means):
        # 9 mazes: the mean + 8 small perturbations (sigma=0.1)
        latents = [mean]
        for j in range(8):
            rng, rng_p = jax.random.split(rng)
            perturb = np.array(jax.random.normal(rng_p, (64,))) * 0.1
            latents.append(mean + perturb)
        latents = jnp.array(latents)

        rng, rng_dec = jax.random.split(rng)
        seqs = decode_latent_to_env(decoder_params, latents, rng_key=rng_dec, temperature=0.25)
        seqs = jax.vmap(repair_cluttr_sequence)(seqs)
        dir_keys = jax.random.split(rng, 9)
        levels = jax.vmap(cluttr_sequence_to_level)(seqs, dir_keys)
        valid = jax.vmap(flood_fill_solvable)(levels.wall_map, levels.agent_pos, levels.goal_pos)

        for k in range(9):
            level_k = jax.tree_util.tree_map(lambda x: x[k], levels)
            img = renderer.render_level(level_k, env_params)
            all_images.append(img)
            tag = "mean" if k == 0 else f"p{k}"
            v_str = "valid" if bool(valid[k]) else "INVALID"
            prefix = f"p{mi}_" if len(means) > 1 else ""
            all_labels.append(f"{prefix}{tag} ({v_str})")

    # Plot grid
    n_imgs = len(all_images)
    cols = min(9, n_imgs)
    rows = (n_imgs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    for i, (img, label) in enumerate(zip(all_images, all_labels)):
        axes[i].imshow(np.array(img))
        axes[i].set_title(label, fontsize=8)
        axes[i].axis("off")
    for i in range(len(all_images), len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"{strat}: Decoded mazes from ES mean latent(s)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{strat}_decoded_mazes.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {strat}: saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Evaluate agents on prefab test mazes
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Part 2: Evaluating agents on prefab test mazes ===")

prefab_levels = Level.load_prefabs(PREFAB_NAMES)
all_results = {}

for strat in STRATEGIES:
    params_path = os.path.join(RUNS_DIR, strat, "checkpoint_0020000", "agent_params.pkl")
    if not os.path.exists(params_path):
        print(f"  {strat}: no agent params, skipping")
        continue

    with open(params_path, "rb") as f:
        agent_params = pickle.load(f)

    results = {}
    for pi, pname in enumerate(PREFAB_NAMES):
        level = jax.tree_util.tree_map(lambda x: x[pi], prefab_levels)
        solves = []
        returns = []
        for trial in range(N_EVAL_ROLLOUTS):
            rng_eval = jax.random.PRNGKey(trial * 100 + pi)
            ret, solved = eval_agent_on_level(rng_eval, agent_params, level, ROLLOUT_STEPS)
            solves.append(bool(solved))
            returns.append(float(ret))
        results[pname] = {
            "solve_rate": float(np.mean(solves)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        }
        print(f"  {strat} | {pname:20s} | solve={np.mean(solves):.0%} | return={np.mean(returns):.3f}")

    all_results[strat] = results
    json_path = os.path.join(OUT_DIR, f"{strat}_prefab_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Comparison bar chart
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Part 3: Comparison chart ===")

if len(all_results) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(PREFAB_NAMES))
    width = 0.35

    for i, strat in enumerate(STRATEGIES):
        if strat not in all_results:
            continue
        solve_rates = [all_results[strat][p]["solve_rate"] for p in PREFAB_NAMES]
        mean_returns = [all_results[strat][p]["mean_return"] for p in PREFAB_NAMES]
        label = strat.replace("phase5-", "").upper()
        ax1.bar(x + i * width, solve_rates, width, label=label)
        ax2.bar(x + i * width, mean_returns, width, label=label)

    ax1.set_ylabel("Solve Rate")
    ax1.set_title("Solve Rate on Prefab Mazes")
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(PREFAB_NAMES, rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    ax2.set_ylabel("Mean Return")
    ax2.set_title("Mean Return on Prefab Mazes")
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(PREFAB_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.legend()

    fig.tight_layout()
    chart_path = os.path.join(OUT_DIR, "prefab_comparison.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {chart_path}")

# Also render the prefab mazes themselves for reference
fig, axes = plt.subplots(1, len(PREFAB_NAMES), figsize=(len(PREFAB_NAMES) * 2.5, 2.5))
for i, pname in enumerate(PREFAB_NAMES):
    level = jax.tree_util.tree_map(lambda x: x[i], prefab_levels)
    img = renderer.render_level(level, env_params)
    axes[i].imshow(np.array(img))
    axes[i].set_title(pname, fontsize=7)
    axes[i].axis("off")
fig.suptitle("Prefab Test Mazes", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "prefab_mazes.png"), dpi=150)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/prefab_mazes.png")

print("\n=== Done! ===")
print(f"All outputs in: {OUT_DIR}/")
