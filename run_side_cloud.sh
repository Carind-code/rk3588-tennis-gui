#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash cloud_runner.sh 1 bash -c 'cd side_training && python3 run_side_pipeline.py --upload' "$@"
