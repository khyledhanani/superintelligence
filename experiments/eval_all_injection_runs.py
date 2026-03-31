"""
Re-evaluate all injection run checkpoints with more eval attempts (100)
and log results to new wandb runs with aligned num_env_steps.

For each original run, iterates over all saved checkpoint steps,
loads the agent, runs eval on test levels, and logs to a new
wandb run with "_eval100" suffix.

Usage (on TPU/GPU):
    python experiments/eval_all_injection_runs.py
"""
import os
import sys
import json
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

ENTITY = "romain-hautier-university-college-london-ucl-"
PROJECT = "jaxued_llm_injection"

# 10k warmstart offset in env steps
WARMSTART_OFFSET = 40 * 2048000  # 81920000
ENV_STEPS_PER_EVAL = 2048000  # per wandb logging step

RUNS = {
    # run_id: (gcs_checkpoint_path, is_warmstart)
    "d549ivgw": ("gs://ucl-ued-project-bucket/llm-exp/checkpoints/accel_sfl_baseline_13x13/0", False),
    "ery0i079": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_wallflip_e5_20pct_seed0/checkpoints/inject_wallflip_e5_20pct_seed0/0", True),
    "8neejckb": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_wallflip_e5_20pct_seed1/checkpoints/inject_wallflip_e5_20pct_seed1/1", True),
    "7nh043h7": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_wallflip_e5_20pct_seed0/checkpoints/inject_wallflip_e5_20pct_seed0/0", True),
    "nv500ly0": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_10pct_seed0/checkpoints/inject_llm_10pct_seed0/0", True),
    "zjtzu49z": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_10pct_seed1/checkpoints/inject_llm_10pct_seed1/1", True),
    "aruawwr1": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_10pct_seed2/checkpoints/inject_llm_10pct_seed2/2", True),
    "0rcb7bx3": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_15pct_seed1/checkpoints/inject_llm_15pct_seed1/1", True),
    "4yqp5ts6": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_20pct_seed1/checkpoints/inject_llm_20pct_seed1/1", True),
    "905zuf5b": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_25pct_seed1/checkpoints/inject_llm_25pct_seed1/1", True),
    "6nljg27p": ("gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_5pct_seed1/checkpoints/inject_llm_5pct_seed1/1", True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_attempts", type=int, default=100)
    parser.add_argument("--local_dir", type=str, default="/tmp/eval_injection")
    parser.add_argument("--run_ids", nargs="+", default=None,
                        help="Specific run IDs to evaluate (default: all)")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N checkpoint steps")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    run_ids = args.run_ids or list(RUNS.keys())

    # Get original run configs from wandb
    api = wandb.Api()

    for rid in run_ids:
        if rid not in RUNS:
            print(f"Unknown run ID: {rid}, skipping")
            continue

        gcs_path, is_warmstart = RUNS[rid]
        orig_run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        orig_config = orig_run.config
        orig_name = orig_run.name

        print(f"\n{'='*60}")
        print(f"Run: {orig_name} ({rid})")
        print(f"  GCS: {gcs_path}")
        print(f"  Warmstart: {is_warmstart}")
        print(f"{'='*60}")

        # Pull checkpoint directory
        ckpt_local = os.path.join(args.local_dir, rid)
        if not os.path.exists(ckpt_local):
            print(f"  Pulling from GCS...")
            os.makedirs(ckpt_local, exist_ok=True)
            os.system(f"gcloud storage cp --recursive '{gcs_path}/*' '{ckpt_local}/'")
        else:
            print(f"  Using cached checkpoint at {ckpt_local}")

        # Load config
        config_path = os.path.join(ckpt_local, "config.json")
        if not os.path.exists(config_path):
            print(f"  ERROR: No config.json found at {ckpt_local}, skipping")
            continue

        with open(config_path) as f:
            config = json.load(f)

        # Import maze_plr to get the eval function and model
        # We need to set up the environment and model from config
        from maze_plr import make_main
        eval_fn, create_train_state_fn, env_params, eval_levels = make_main(config, return_eval=True)

        # Set up checkpoint manager
        models_dir = os.path.join(ckpt_local, "models")
        ckpt_manager = ocp.CheckpointManager(
            models_dir,
            item_handlers=ocp.StandardCheckpointHandler(),
        )

        all_steps = sorted(ckpt_manager.all_steps())
        eval_steps = all_steps[::args.eval_every]
        # Always include last step
        if all_steps[-1] not in eval_steps:
            eval_steps.append(all_steps[-1])

        print(f"  {len(all_steps)} checkpoints, evaluating {len(eval_steps)} steps")
        print(f"  Steps: {eval_steps[:5]}...{eval_steps[-3:]}")

        # Create wandb run for results
        new_run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"{orig_name}_eval{args.eval_attempts}",
            config={
                **orig_config,
                "eval_num_attempts": args.eval_attempts,
                "original_run_id": rid,
                "is_eval_rerun": True,
                "is_warmstart": is_warmstart,
                "env_steps_offset": WARMSTART_OFFSET if is_warmstart else 0,
            },
            tags=["eval_rerun", f"eval{args.eval_attempts}"],
        )

        rng_init = jax.random.PRNGKey(10000)
        train_state_og = create_train_state_fn(rng_init)

        for step in eval_steps:
            loaded = ckpt_manager.restore(step)
            params = loaded["params"]
            train_state = train_state_og.replace(params=params)

            rng_eval = jax.random.PRNGKey(step + 42)
            # Run eval with many attempts
            _, cum_rewards, episode_lengths = jax.vmap(
                eval_fn, (0, None)
            )(jax.random.split(rng_eval, args.eval_attempts), train_state)

            # cum_rewards: (eval_attempts, num_eval_levels)
            solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
            returns = cum_rewards.mean(axis=0)
            ep_lens = episode_lengths.mean(axis=0)

            # Compute aligned env steps
            env_steps = (step + 1) * ENV_STEPS_PER_EVAL
            if is_warmstart:
                env_steps += WARMSTART_OFFSET

            log_dict = {
                "num_env_steps": env_steps,
                "checkpoint_step": step,
                "solve_rate/mean": float(solve_rates.mean()),
                "return/mean": float(returns.mean()),
                "eval_ep_lengths/mean": float(ep_lens.mean()),
            }
            for i, name in enumerate(eval_levels):
                log_dict[f"solve_rate/{name}"] = float(solve_rates[i])
                log_dict[f"return/{name}"] = float(returns[i])

            new_run.log(log_dict)
            print(f"    Step {step}: solve_rate={float(solve_rates.mean()):.3f}, return={float(returns.mean()):.4f}")

        new_run.finish()
        print(f"  Logged to wandb: {new_run.name} ({new_run.id})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
