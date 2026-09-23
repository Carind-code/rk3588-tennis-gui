#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_CMD=(python3)
if command -v taskset >/dev/null 2>&1; then
  PYTHON_CMD=(taskset -c 4-7 python3)
fi

echo "启动本地测试：V3.7 FP 追踪 + V5.2 MediaPipe/场地上下文 + 落地与判罚，不上云。"

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
  --no-ipc \
  "$@"

echo ""
echo "本地测试完成。"
echo "CSV 输出：./tracknet_rknn_output.csv"
echo "视频输出：./tracknet_rknn_output.mp4"
