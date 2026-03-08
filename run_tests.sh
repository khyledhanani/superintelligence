#!/bin/bash
# Run tests for a specific phase and save results to test_log/phase_N/
#
# Usage:
#   ./run_tests.sh 1        # Run Phase 1 tests
#   ./run_tests.sh 2        # Run Phase 2 tests
#   ./run_tests.sh all      # Run all phases
#   ./run_tests.sh          # Defaults to all

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
PHASE="${1:-all}"

TOTAL_PASS=0
TOTAL_FAIL=0

run_suite() {
    local name="$1"
    local cmd="$2"
    local log_file="$3"

    echo "--- $name ---" | tee -a "$log_file"
    if output=$(eval "$cmd" 2>&1); then
        echo "$output" | tee -a "$log_file"
        echo "RESULT: PASS" | tee -a "$log_file"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo "$output" | tee -a "$log_file"
        echo "RESULT: FAIL" | tee -a "$log_file"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
    echo "" | tee -a "$log_file"
}

run_phase() {
    local phase_num="$1"
    local phase_dir="$SCRIPT_DIR/test_log/phase_${phase_num}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="$phase_dir/${timestamp}.log"

    mkdir -p "$phase_dir"

    # Header
    {
        echo "========================================"
        echo "Phase ${phase_num} Test Results"
        echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Branch: $(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null || echo 'unknown')"
        echo "Commit: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        echo "Python: $($VENV_PYTHON --version 2>&1)"
        echo "========================================"
        echo ""
    } | tee "$log_file"

    case "$phase_num" in
        1)
            run_suite "Trajectory Metrics (9 tests)" \
                "WANDB_MODE=disabled $VENV_PYTHON -m pytest $SCRIPT_DIR/metrics/test_trajectory_metrics.py -v --tb=short 2>&1" \
                "$log_file"

            run_suite "Trajectory Cache (5 tests)" \
                "WANDB_MODE=disabled $VENV_PYTHON -m pytest $SCRIPT_DIR/metrics/test_trajectory_cache.py -v --tb=short 2>&1" \
                "$log_file"

            run_suite "Example Scripts Smoke Tests (3 tests)" \
                "cd $SCRIPT_DIR && WANDB_MODE=disabled $VENV_PYTHON -m pytest tests/test_examples_kinda.py -v --tb=short 2>&1" \
                "$log_file"

            run_suite "Metrics Visualization Plots (4 figures)" \
                "cd $SCRIPT_DIR && $VENV_PYTHON metrics/plot_metrics_demo.py 2>&1" \
                "$log_file"
            ;;
        2)
            echo "Phase 2 tests not yet implemented." | tee -a "$log_file"
            ;;
        3)
            echo "Phase 3 tests not yet implemented." | tee -a "$log_file"
            ;;
        4)
            echo "Phase 4 tests not yet implemented." | tee -a "$log_file"
            ;;
        *)
            echo "Unknown phase: $phase_num" | tee -a "$log_file"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            ;;
    esac

    # Phase summary
    {
        echo "========================================"
        echo "PHASE ${phase_num} SUMMARY"
        echo "  Suites passed: $TOTAL_PASS"
        echo "  Suites failed: $TOTAL_FAIL"
        if [ "$TOTAL_FAIL" -eq 0 ]; then
            echo "  ALL SUITES PASSED"
        else
            echo "  SOME SUITES FAILED"
        fi
        echo "  Log saved to: $log_file"
        echo "========================================"
    } | tee -a "$log_file"
}

if [ "$PHASE" = "all" ]; then
    for p in 1 2 3 4; do
        run_phase "$p"
    done
else
    run_phase "$PHASE"
fi

exit $TOTAL_FAIL
