#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run_match_local.sh
bash run_side_local.sh
