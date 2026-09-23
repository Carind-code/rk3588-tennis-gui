#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/body_action"
exec bash start_full.sh "$@"
