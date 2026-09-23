#!/usr/bin/env python3
"""把板上所有调用旧摄像头(OV13855 ISP)的地方切到 USB AR0234 相机。
替换规则:
  /dev/video11|12 -> /dev/video21
  采集 fourcc UYVY/NV12 -> MJPG (USB 相机只有 MJPG/YUYV，高分辨率必须 MJPG 压缩)
  不支持的分辨率 1024x576 -> 1280x720
运行前会备份原文件到 backups/usb_switch_<时间戳>/"""
import os, shutil, datetime

FILES = [
    "camera_recorder.py",
    "match_judgement/camera_preview.py",
    "roi_editor.py",
    "gui/panels/roi_editor.py",
    "gui/widgets/camera_preprocess.py",
    "gui/camera_recorder.py",
    "gui/widgets/video_widget.py",
    "live_vision/camera_capture.py",
    "models/detect/camera_capture.py",
    "live_vision/live_worker.py",
    "live_vision/ball_engine.py",
    "live_vision/pose_engine.py",
    "models/pose/pose_live.py",
    "models/detect/tennis_ball_detector.py",
]

# 顺序很重要: 先 12 后 11，避免 /dev/video11 替换后影响 /dev/video112 之类(这里无此情况，保险起见)
REPLS = [
    ("/dev/video12", "/dev/video21"),
    ("/dev/video11", "/dev/video21"),
    ("VideoWriter_fourcc(*'UYVY')", "VideoWriter_fourcc(*'MJPG')"),
    ("VideoWriter_fourcc(*\"NV12\")", "VideoWriter_fourcc(*\"MJPG\")"),
    ("width=1024, height=576", "width=1280, height=720"),
]

backup = os.path.join("backups", "usb_switch_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(backup, exist_ok=True)

for f in FILES:
    if not os.path.exists(f):
        print("MISSING:", f)
        continue
    dst = os.path.join(backup, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(f, dst)
    s = open(f).read()
    orig = s
    for old, new in REPLS:
        s = s.replace(old, new)
    if s != orig:
        open(f, "w").write(s)
        print("CHANGED:", f)
    else:
        print("NO-OP  :", f)
print("backup ->", backup)
