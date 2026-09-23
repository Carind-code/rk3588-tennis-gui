#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash cloud_runner.sh 2 bash -c 'bash match_judgement/run_cloud_task.sh && cd side_training && python3 run_side_pipeline.py --upload'
