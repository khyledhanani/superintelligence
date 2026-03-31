#!/bin/bash
# Re-evaluate completed injection runs with 100 eval attempts per checkpoint.
# Creates new wandb runs in group "eval100" with aligned num_env_steps.
# Logs to wandb incrementally as each checkpoint is evaluated.
#
# Runs to eval (seed 1 completed + ACCEL baseline):
#   - cosmic-tree-2    (d549ivgw) ACCEL baseline, seed 0, from scratch
#   - lemon-totem-15   (8neejckb) wallflip 20%, seed 1, warmstart
#   - fanciful-disco-25 (zjtzu49z) llm 10%, seed 1, warmstart
#   - royal-microwave-27 (0rcb7bx3) llm 15%, seed 1, warmstart
#   - dulcet-water-28  (4yqp5ts6) llm 20%, seed 1, warmstart
#   - worthy-valley-29 (905zuf5b) llm 25%, seed 1, warmstart
#
# Usage (on GPU/TPU):
#   cd jaxued && bash experiments/eval_all_injection_runs.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
GCS_CP="gcloud storage cp"
LOCAL_DIR="/tmp/eval_injection"
EVAL_ATTEMPTS=100

mkdir -p "$LOCAL_DIR"

# --- Run definitions ---
# Format: wandb_id|name|gcs_path|is_warmstart
RUNS=(
    "d549ivgw|accel_baseline|gs://ucl-ued-project-bucket/llm-exp/checkpoints/accel_sfl_baseline_13x13/0|0"
    "8neejckb|wallflip_20pct_s1|gs://ucl-ued-project-bucket/llm-exp/training/inject_wallflip_e5_20pct_seed1/checkpoints/inject_wallflip_e5_20pct_seed1/1|1"
    "zjtzu49z|llm_10pct_s1|gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_10pct_seed1/checkpoints/inject_llm_10pct_seed1/1|1"
    "0rcb7bx3|llm_15pct_s1|gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_15pct_seed1/checkpoints/inject_llm_15pct_seed1/1|1"
    "4yqp5ts6|llm_20pct_s1|gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_20pct_seed1/checkpoints/inject_llm_20pct_seed1/1|1"
    "905zuf5b|llm_25pct_s1|gs://ucl-ued-project-bucket/llm-exp/training/inject_llm_25pct_seed1/checkpoints/inject_llm_25pct_seed1/1|1"
)

WARMSTART_OFFSET=81920000  # 10k PPO updates * 8192 env_steps/update
ENV_STEPS_PER_CKPT=2048000  # 250 updates * 8192

echo "=== Eval Injection Runs (${EVAL_ATTEMPTS} attempts per checkpoint) ==="
echo "  Runs: ${#RUNS[@]}"
echo "  Output: ${LOCAL_DIR}"
echo ""

# --- Step 1: Pull all checkpoints from GCS ---
echo "[Step 1] Pulling checkpoints..."
for ENTRY in "${RUNS[@]}"; do
    IFS='|' read -r RID RNAME GCS_PATH IS_WS <<< "$ENTRY"
    CKPT_LOCAL="${LOCAL_DIR}/ckpts/${RID}"

    if [ -d "${CKPT_LOCAL}/models" ]; then
        echo "  ${RNAME}: cached."
        continue
    fi

    echo "  ${RNAME}: pulling from ${GCS_PATH}..."
    mkdir -p "${CKPT_LOCAL}"
    $GCS_CP --recursive "${GCS_PATH}/*" "${CKPT_LOCAL}/"
done
echo ""

# --- Step 2: Evaluate + log to wandb incrementally ---
echo "[Step 2] Evaluating checkpoints and logging to wandb..."
for ENTRY in "${RUNS[@]}"; do
    IFS='|' read -r RID RNAME GCS_PATH IS_WS <<< "$ENTRY"
    CKPT_LOCAL="${LOCAL_DIR}/ckpts/${RID}"
    MODELS_DIR="${CKPT_LOCAL}/models"
    RESULTS_DIR="${LOCAL_DIR}/results/${RID}"
    mkdir -p "$RESULTS_DIR"

    if [ ! -d "$MODELS_DIR" ]; then
        echo "  ${RNAME}: no models/ dir, skipping."
        continue
    fi

    # Get sorted checkpoint steps
    STEPS=($(ls -d "${MODELS_DIR}"/*/ 2>/dev/null | xargs -I{} basename {} | sort -n))
    TOTAL=${#STEPS[@]}
    echo "  ${RNAME} (${RID}): ${TOTAL} checkpoints"

    # Evaluate each step and log to wandb incrementally via Python
    $PYTHON << PYEOF
import os, json, numpy as np, subprocess, wandb

ENTITY = "romain-hautier-university-college-london-ucl-"
PROJECT = "jaxued_llm_injection"
EVAL_ATTEMPTS = ${EVAL_ATTEMPTS}
WARMSTART_OFFSET = ${WARMSTART_OFFSET}
ENV_STEPS_PER_CKPT = ${ENV_STEPS_PER_CKPT}

rid = "${RID}"
rname = "${RNAME}"
is_warmstart = bool(${IS_WS})
ckpt_local = "${CKPT_LOCAL}"
results_dir = "${RESULTS_DIR}"
steps = [${STEPS[@]/%/,}]

offset = WARMSTART_OFFSET if is_warmstart else 0

# Load eval levels from config
config_path = os.path.join(ckpt_local, "config.json")
with open(config_path) as f:
    config = json.load(f)
eval_levels = config.get("eval_levels", [])

# Get original run config
api = wandb.Api()
orig = api.run(f"{ENTITY}/{PROJECT}/{rid}")

run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name=f"{rname}_eval{EVAL_ATTEMPTS}",
    group="eval100",
    config={
        **orig.config,
        "eval_num_attempts": EVAL_ATTEMPTS,
        "original_run_id": rid,
        "original_run_name": orig.name,
        "is_eval_rerun": True,
        "is_warmstart": is_warmstart,
        "env_steps_offset": offset,
    },
    tags=["eval100"],
)

for i, step in enumerate(steps):
    result_file = os.path.join(results_dir, f"step_{step}.npz")

    # Run eval if not cached
    if not os.path.exists(result_file):
        subprocess.run([
            "${PYTHON}", "examples/maze_plr.py",
            "--mode", "eval",
            "--checkpoint_directory", ckpt_local,
            "--checkpoint_to_eval", str(step),
            "--eval_num_attempts", str(EVAL_ATTEMPTS),
        ], check=True, capture_output=True)

        # Find and move result file
        for candidate in [
            ckpt_local.replace("checkpoints", "results") + "/results.npz",
            ckpt_local + "/results.npz",
        ]:
            if os.path.exists(candidate):
                os.rename(candidate, result_file)
                break

    if not os.path.exists(result_file):
        print(f"    step {step}: WARNING no result file")
        continue

    # Load and log
    data = np.load(result_file)
    cum_rewards = data["cum_rewards"]  # (eval_attempts, num_levels)
    solve_rates = (cum_rewards > 0).astype(float).mean(axis=0)
    returns = cum_rewards.mean(axis=0)
    env_steps = (step + 1) * ENV_STEPS_PER_CKPT + offset

    log_dict = {
        "num_env_steps": env_steps,
        "checkpoint_step": step,
        "solve_rate/mean": float(solve_rates.mean()),
        "return/mean": float(returns.mean()),
    }
    for j, name in enumerate(eval_levels):
        if j < len(solve_rates):
            log_dict[f"solve_rate/{name}"] = float(solve_rates[j])
            log_dict[f"return/{name}"] = float(returns[j])

    run.log(log_dict)
    print(f"    step {step} ({i+1}/{len(steps)}): solve={solve_rates.mean():.3f} return={returns.mean():.4f}")

run.finish()
print(f"  -> wandb: {rname}_eval{EVAL_ATTEMPTS}")
PYEOF

    echo ""
done

echo "=== Complete ==="
echo "  In wandb: filter group='eval100', x-axis='num_env_steps'"
echo "  Results cached in: ${LOCAL_DIR}/results/"
