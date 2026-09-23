#!/usr/bin/env bash
set -euo pipefail
LOG="$HOME/rk3588_tennis_system/outputs/logs/side_run_$(date +%Y%m%d_%H%M%S).log"
cd "$(dirname "$0")/side_training"
python3 run_side_pipeline.py "$@" 2>&1 | tee "$LOG"
