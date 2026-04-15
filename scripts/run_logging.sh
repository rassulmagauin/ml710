#!/bin/bash

setup_run_logging() {
    export METHOD_NAME="$1"
    export RUN_LABEL="${2:-$METHOD_NAME}"
    export PER_DEVICE_TRAIN_BATCH_SIZE="${3:-1}"
    export GRADIENT_ACCUMULATION_STEPS="${4:-1}"
    export WORLD_SIZE="${5:-1}"
    export RUN_USER="${RUN_USER:-${USER:-unknown_user}}"

    export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
    mkdir -p "$LOG_DIR"
    export RUNS_ROOT="${RUNS_ROOT:-$LOG_DIR/runs/$RUN_USER}"
    mkdir -p "$RUNS_ROOT"

    export RUN_ID="${METHOD_NAME}_$(date +%Y%m%d_%H%M%S)"
    export RUN_DIR="$RUNS_ROOT/$RUN_ID"
    mkdir -p "$RUN_DIR"

    export TRAIN_LOG="$RUN_DIR/train.log"
    export GPU_LOG="$RUN_DIR/gpu.csv"
    export RAM_LOG="$RUN_DIR/ram.log"
    export SUMMARY_TXT="$RUN_DIR/summary.txt"
    export SUMMARY_JSON="$RUN_DIR/summary.json"
    export SUMMARY_CSV="$RUNS_ROOT/${METHOD_NAME}_summary.csv"

    GPU_MONITOR_PID=""
    RAM_MONITOR_PID=""
    export GPU_MONITOR_PID RAM_MONITOR_PID
}

cleanup_run_logging() {
    if [ -n "$GPU_MONITOR_PID" ] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    if [ -n "$RAM_MONITOR_PID" ] && kill -0 "$RAM_MONITOR_PID" 2>/dev/null; then
        kill "$RAM_MONITOR_PID" 2>/dev/null || true
    fi
}

start_run_logging() {
    export START_TS=$(date +%s)

    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv -l 5 > "$GPU_LOG" &
    GPU_MONITOR_PID=$!
    export GPU_MONITOR_PID

    (
        while true; do
            date '+%F %T'
            free -h
            sleep 5
        done
    ) > "$RAM_LOG" &
    RAM_MONITOR_PID=$!
    export RAM_MONITOR_PID
}

finish_run_logging() {
    export TRAIN_EXIT="$1"
    export END_TS=$(date +%s)

    cleanup_run_logging
    python "$PROJECT_DIR/scripts/summarize_run.py"
}
