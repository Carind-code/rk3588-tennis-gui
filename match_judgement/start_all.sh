#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f cloud/config.json ]; then
  echo "缺少 cloud/config.json，请先填写真实阿里云配置。"
  exit 1
fi

PYTHON_CMD=(python3)
if command -v taskset >/dev/null 2>&1; then
  PYTHON_CMD=(taskset -c 4-7 python3)
fi

echo "启动上云进程..."
(
  cd cloud
  python3 main.py
) > cloud_process.log 2>&1 &
CLOUD_PID=$!

cleanup() {
  if kill -0 "$CLOUD_PID" >/dev/null 2>&1; then
    kill "$CLOUD_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup INT TERM

for _ in $(seq 1 50); do
  if [ -S /tmp/tennis_ipc.sock ]; then
    break
  fi
  sleep 0.2
done

if [ ! -S /tmp/tennis_ipc.sock ]; then
  echo "上云进程未创建 /tmp/tennis_ipc.sock，请查看 cloud_process.log。"
  cleanup
  exit 1
fi

echo "启动 RKNN 推理并上云：V3.7 FP 追踪 + V5.2 MediaPipe/场地上下文 + 落地与判罚。"

"${PYTHON_CMD[@]}" infer_tracknet_rknn_lite.py \
  --model ./model_v37_360x640_b1_sigmoid_fp.rknn \
  --court-model ./court_detector_fp.rknn \
  --video ./input.mp4 \
  --out-csv ./tracknet_rknn_output.csv \
  --out-video ./tracknet_rknn_output.mp4 \
  --workers 3 \
  --cores 0,1,2 \
  --opencv-threads 1 \
  --threshold 0.35 \
  --peak-window 9 \
  --max-dist 220 \
  --dot-radius 5 \
  "$@"

echo ""
echo "推理完成，视频和 CSV 已发送给上云进程。"
echo "上传日志：./cloud_process.log"

wait "$CLOUD_PID"
