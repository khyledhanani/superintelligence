#!/bin/bash
# ==============================================================================
# Bidirectional cross-evaluation:
#   CMA-ES (beta=2.0) agents <-> ACCEL agents on each other's buffers
#
# For each condition's agent (at matching timestep checkpoint):
#   - Evaluate on its OWN buffer (self-eval baseline)
#   - Evaluate on the OTHER condition's buffer (cross-eval)
#
# Evaluates at buffer timesteps 30k, 40k, 50k.
# Results averaged across seeds in the summary CSV.
#
# Usage (run on TPU):
#   bash examples/launch_cross_eval.sh
# ==============================================================================
set -e

BUCKET="ucl-ued-project-bucket"
PREFIX="accel"
SEEDS=(0 1 2)
TIMESTEPS=(20k 30k 40k 50k)
NUM_ATTEMPTS=10
OUTPUT_DIR="results/cross_eval"

CMAES_RUN="cmaes_vae_beta2.0"
ACCEL_RUN="plain_accel"
ALL_RUNS=("$CMAES_RUN" "$ACCEL_RUN")

CKPT_BASE="/tmp/cross_eval_checkpoints"
BUFFER_BASE="/tmp/accel_comparison_data"

mkdir -p "$CKPT_BASE" "$BUFFER_BASE" "$OUTPUT_DIR"

# ==============================================================================
# 1. Download checkpoints for BOTH conditions
# ==============================================================================
echo "============================================"
echo "  Downloading agent checkpoints"
echo "============================================"
for run_name in "${ALL_RUNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dest="$CKPT_BASE/$run_name/$seed"
        if [ -d "$dest/models" ] && [ "$(ls "$dest/models" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [skip] $run_name/seed$seed already downloaded"
        else
            echo "  Downloading $run_name/seed$seed..."
            mkdir -p "$dest/models"
            gcloud storage cp "gs://$BUCKET/$PREFIX/checkpoints/$run_name/$seed/config.json" "$dest/config.json"
            gcloud storage cp -r "gs://$BUCKET/$PREFIX/checkpoints/$run_name/$seed/models/*" "$dest/models/"
        fi
    done
done

# ==============================================================================
# 2. Download buffer dumps for both conditions
# ==============================================================================
echo ""
echo "============================================"
echo "  Downloading buffer dumps"
echo "============================================"
for run_name in "${ALL_RUNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dest="$BUFFER_BASE/$run_name/$seed"
        if [ -d "$dest" ] && ls "$dest"/buffer_dump_*.npz 1>/dev/null 2>&1; then
            echo "  [skip] $run_name/seed$seed buffers already downloaded"
        else
            echo "  Downloading $run_name/seed$seed buffers..."
            mkdir -p "$dest"
            gcloud storage cp "gs://$BUCKET/$PREFIX/buffer_dumps/$run_name/$seed/buffer_dump_*.npz" "$dest/" 2>/dev/null || \
                echo "  [warn] Some buffer dumps missing for $run_name/seed$seed"
        fi
    done
done

# ==============================================================================
# 3. List what we actually have
# ==============================================================================
echo ""
echo "============================================"
echo "  Available data:"
echo "============================================"
echo "Checkpoints:"
for run_name in "${ALL_RUNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dir="$CKPT_BASE/$run_name/$seed/models"
        if [ -d "$dir" ]; then
            echo "  $run_name/seed$seed: $(ls "$dir" | wc -l) checkpoint steps"
        else
            echo "  $run_name/seed$seed: MISSING"
        fi
    done
done
echo ""
echo "Buffer dumps:"
for run_name in "${ALL_RUNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dir="$BUFFER_BASE/$run_name/$seed"
        if [ -d "$dir" ]; then
            echo "  $run_name/seed$seed: $(ls "$dir"/buffer_dump_*.npz 2>/dev/null | wc -l) dumps"
        else
            echo "  $run_name/seed$seed: MISSING"
        fi
    done
done

# ==============================================================================
# 4. Run cross-evaluations (all 4 directions)
#    CMA-ES agent -> CMA-ES buffer (self)
#    CMA-ES agent -> ACCEL buffer  (cross)
#    ACCEL agent  -> ACCEL buffer  (self)
#    ACCEL agent  -> CMA-ES buffer (cross)
# ==============================================================================
echo ""
echo "============================================"
echo "  Running cross-evaluations (bidirectional)"
echo "============================================"

# Count total evaluations: 2 agents x 2 buffers x 3 agent_seeds x 3 buf_seeds x 3 timesteps
TOTAL=0
for ts in "${TIMESTEPS[@]}"; do
    for agent_run in "${ALL_RUNS[@]}"; do
        for agent_seed in "${SEEDS[@]}"; do
            for buf_run in "${ALL_RUNS[@]}"; do
                for buf_seed in "${SEEDS[@]}"; do
                    TOTAL=$((TOTAL + 1))
                done
            done
        done
    done
done

echo "  Total evaluations: $TOTAL"

RUN_NUM=0
for ts in "${TIMESTEPS[@]}"; do
    UPDATES=$(echo "$ts" | sed 's/k//' | awk '{print $1 * 1000}')

    echo ""
    echo "--- Timestep: ${ts} (agent @ ~${UPDATES} updates) ---"

    for agent_run in "${ALL_RUNS[@]}"; do
        for agent_seed in "${SEEDS[@]}"; do
            agent_dir="$CKPT_BASE/$agent_run/$agent_seed"

            if [ ! -f "$agent_dir/config.json" ]; then
                echo "  SKIP $agent_run/s$agent_seed (no checkpoint)"
                RUN_NUM=$((RUN_NUM + ${#ALL_RUNS[@]} * ${#SEEDS[@]}))
                continue
            fi

            for buf_run in "${ALL_RUNS[@]}"; do
                for buf_seed in "${SEEDS[@]}"; do
                    RUN_NUM=$((RUN_NUM + 1))
                    buf_npz="$BUFFER_BASE/$buf_run/$buf_seed/buffer_dump_${ts}.npz"
                    if [ ! -f "$buf_npz" ]; then
                        echo "  [$RUN_NUM/$TOTAL] SKIP (missing buffer $buf_run/s$buf_seed @ $ts)"
                        continue
                    fi

                    if [ "$agent_run" = "$buf_run" ] && [ "$agent_seed" = "$buf_seed" ]; then
                        eval_type="self"
                    elif [ "$agent_run" = "$buf_run" ]; then
                        eval_type="same-cond"
                    else
                        eval_type="cross"
                    fi

                    echo "  [$RUN_NUM/$TOTAL] ($eval_type) ${agent_run}/s${agent_seed} -> ${buf_run}/s${buf_seed} @ ${ts}"
                    PYTHONUNBUFFERED=1 python3 examples/cross_evaluate.py \
                        --agent_checkpoint_dir "$agent_dir" \
                        --agent_updates $UPDATES \
                        --buffer_npz "$buf_npz" \
                        --num_attempts $NUM_ATTEMPTS \
                        --output_dir "$OUTPUT_DIR/${ts}"
                done
            done
        done
    done
done

# ==============================================================================
# 5. Aggregate results
# ==============================================================================
echo ""
echo "============================================"
echo "  Aggregating results"
echo "============================================"

python3 -c "
import os, glob, numpy as np, csv
from collections import defaultdict

output_dir = '$OUTPUT_DIR'
rows = []
for ts_dir in sorted(glob.glob(os.path.join(output_dir, '*k'))):
    ts = os.path.basename(ts_dir)
    for npz_path in sorted(glob.glob(os.path.join(ts_dir, 'cross_eval_*.npz'))):
        data = dict(np.load(npz_path, allow_pickle=True))
        rows.append({
            'timestep': ts,
            'agent_name': str(data.get('agent_name', '?')),
            'agent_seed': str(data.get('agent_seed', '?')),
            'buffer_run_name': str(data.get('buffer_run_name', '?')),
            'buffer_seed': str(data.get('buffer_seed', '?')),
            'mean_solve_rate': float(data['solve_rates'].mean()),
            'num_levels': int(data.get('num_levels', len(data['solve_rates']))),
        })

if not rows:
    print('No results found!')
    exit(0)

# Per-eval results
print(f\"{'TS':>6s} {'Agent':>25s} {'Buffer Run':>22s} {'BufS':>5s} {'Solve%':>8s}\")
print('-' * 70)
for r in rows:
    label = f\"{r['agent_name']}/s{r['agent_seed']}\"
    print(f\"{r['timestep']:>6s} {label:>25s} {r['buffer_run_name']:>22s} {'s'+r['buffer_seed']:>5s} {r['mean_solve_rate']:>7.1%}\")

# Aggregate by (timestep, agent_name, buffer_run_name) — avg across seeds
print()
print('AVERAGED ACROSS SEEDS (agent_name -> buffer_source):')
print(f\"{'TS':>6s} {'Agent Condition':>25s} {'Buffer Source':>25s} {'Mean%':>8s} {'Std':>7s} {'N':>4s}\")
print('-' * 79)

grouped = defaultdict(list)
for r in rows:
    grouped[(r['timestep'], r['agent_name'], r['buffer_run_name'])].append(r['mean_solve_rate'])

for (ts, agent, buf), rates in sorted(grouped.items()):
    rates = np.array(rates)
    print(f\"{ts:>6s} {agent:>25s} {buf:>25s} {rates.mean():>7.1%} {rates.std():>6.1%} {len(rates):>4d}\")

# Save full CSV
csv_path = os.path.join(output_dir, 'cross_eval_all.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'\nSaved: {csv_path}')
"

echo ""
echo "============================================"
echo "  Cross-evaluation complete!"
echo "  Results in: $OUTPUT_DIR/"
echo "============================================"
