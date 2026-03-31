"""Re-log warmstart injection runs to wandb with corrected num_updates offset.

Pulls history from existing runs, adds 10250 PPO update offset (10k warmstart
+ 250 first eval), and creates new runs with '_aligned' suffix.

Usage:
    python experiments/fix_wandb_offset.py
"""
import wandb

ENTITY = "romain-hautier-university-college-london-ucl-"
PROJECT = "JAXUED_LLM_INJECTION"
UPDATE_OFFSET = 10250
NUM_TRAIN_ENVS = 32
NUM_STEPS = 256
ENV_STEPS_PER_UPDATE = NUM_TRAIN_ENVS * NUM_STEPS  # 8192

# Keys to pull and re-log
LOG_KEYS = [
    "num_updates", "num_env_steps",
    "solve_rate/SixteenRooms", "solve_rate/SixteenRooms2",
    "solve_rate/Labyrinth", "solve_rate/LabyrinthFlipped",
    "solve_rate/Labyrinth2", "solve_rate/StandardMaze",
    "solve_rate/StandardMaze2", "solve_rate/StandardMaze3",
    "return/SixteenRooms", "return/SixteenRooms2",
    "return/Labyrinth", "return/LabyrinthFlipped",
    "return/Labyrinth2", "return/StandardMaze",
    "return/StandardMaze2", "return/StandardMaze3",
    "level_sampler/mean_score", "level_sampler/weighted_mean_score",
    "agent/total_loss", "agent/value_loss", "agent/actor_loss", "agent/entropy",
]

api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}")

# Find warmstart injection runs (not eval100, not baseline, not already aligned)
warmstart_runs = []
for run in all_runs:
    name = run.name or ""
    tags = run.tags or []
    if "eval100" in tags or "_eval100" in name or "_aligned" in name:
        continue
    if "aligned" in tags:
        continue
    cfg = run.config or {}
    if cfg.get("resume_checkpoint_dir") or "inject_" in name:
        warmstart_runs.append(run)

print(f"Found {len(warmstart_runs)} warmstart runs to re-align:")
for r in warmstart_runs:
    print(f"  {r.name} ({r.id}) - state={r.state}")

if not warmstart_runs:
    print("No runs to fix.")
    exit(0)

input("\nPress Enter to proceed with re-logging, or Ctrl-C to abort...")

for orig_run in warmstart_runs:
    print(f"\n--- {orig_run.name} ({orig_run.id}) ---")

    history = list(orig_run.scan_history())
    if not history:
        print("  No history, skipping.")
        continue

    print(f"  {len(history)} log entries")

    first_updates = history[0].get("num_updates", 0)
    if first_updates > UPDATE_OFFSET:
        print(f"  Already offset (first num_updates={first_updates}), skipping.")
        continue

    new_name = f"{orig_run.name}_aligned"
    new_run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=new_name,
        group=orig_run.group,
        config={
            **(orig_run.config or {}),
            "is_aligned": True,
            "original_run_id": orig_run.id,
            "original_run_name": orig_run.name,
            "warmstart_update_offset": UPDATE_OFFSET,
        },
        tags=["aligned"] + (orig_run.tags or []),
    )

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")

    for row in history:
        log_dict = {}
        for k, v in row.items():
            if k.startswith("_") or v is None:
                continue
            if k == "num_updates":
                log_dict[k] = v + UPDATE_OFFSET
            elif k == "num_env_steps":
                log_dict[k] = v + UPDATE_OFFSET * ENV_STEPS_PER_UPDATE
            else:
                log_dict[k] = v
        if log_dict:
            new_run.log(log_dict)

    new_run.finish()
    print(f"  -> {new_name} logged ({len(history)} steps)")

print("\nDone. Filter by tag 'aligned' in wandb to see corrected runs.")
