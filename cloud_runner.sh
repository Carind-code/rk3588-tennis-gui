#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXPECTED="$1"
shift

if [ ! -f "$ROOT/match_judgement/cloud/config.json" ]; then
  echo "缺少match_judgement/cloud/config.json"
  exit 1
fi

rm -f /tmp/tennis_ipc.sock
(
  cd "$ROOT/match_judgement/cloud"
  python3 main.py --expected-tasks "$EXPECTED"
) > "$ROOT/cloud_process.log" 2>&1 &
CLOUD_PID=$!

cleanup() {
  kill "$CLOUD_PID" >/dev/null 2>&1 || true
  rm -f /tmp/tennis_ipc.sock
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 100); do
  [ -S /tmp/tennis_ipc.sock ] && break
  sleep 0.2
done
[ -S /tmp/tennis_ipc.sock ] || { echo "上云进程启动失败，请查看cloud_process.log"; exit 1; }

"$@"
wait "$CLOUD_PID"
echo "上云完成，日志：$ROOT/cloud_process.log"
