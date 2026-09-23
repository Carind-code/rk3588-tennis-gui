#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/match_judgement"
exec bash start_all.sh "$@"
