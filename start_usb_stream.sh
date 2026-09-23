#!/bin/bash
# USB 相机推流自启脚本（由 cron @reboot 调用）
# 作用：等相机节点出现 -> 设亮度/锐化/曝光 -> 启动 MJPEG 推流
# 浏览器看：http://192.168.0.232:8000
CAM=/dev/video21
PORT=8000
LOG=/tmp/usb_stream.log

# 1. 等 USB 相机枚举（重启后要几秒，最多等 30 秒）
for i in $(seq 1 30); do
    [ -e "$CAM" ] && break
    sleep 1
done
if [ ! -e "$CAM" ]; then
    echo "$(date): 相机节点 $CAM 未出现" >> "$LOG"
    exit 1
fi

# 2. 相机参数（重启后会重置，这里开机重新设）
v4l2-ctl -d "$CAM" --set-ctrl=exposure_auto=3     # 自动曝光
v4l2-ctl -d "$CAM" --set-ctrl=brightness=200       # 亮度
v4l2-ctl -d "$CAM" --set-ctrl=sharpness=5000       # 锐化

# 3. 启动推流（前台运行，交给 cron 托管）
exec python3 -u /home/elf/rk3588_tennis_system/usb_camera_stream.py \
    --width 1920 --height 1080 --fps 60 --focus -1 --port "$PORT" >> "$LOG" 2>&1
