#!/usr/bin/env bash
set -euo pipefail
LOG="$HOME/rk3588_tennis_system/outputs/logs/match_run_$(date +%Y%m%d_%H%M%S).log"
cd "$(dirname "$0")/match_judgement"
bash start_test.sh "$@" 2>&1 | tee "$LOG"
