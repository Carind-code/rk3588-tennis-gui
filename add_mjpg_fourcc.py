#!/usr/bin/env python3
"""给没有显式设置采集 fourcc 的摄像头文件，在 FRAME_WIDTH 之前插入 MJPG fourcc。
(USB 相机只有 MJPG/YUYV，高分辨率必须 MJPG 压缩才能过 USB2.0 带宽)"""
import re

FILES = [
    "models/detect/camera_capture.py",
    "gui/widgets/camera_preprocess.py",
    "gui/widgets/video_widget.py",
    "roi_editor.py",
    "gui/panels/roi_editor.py",
]

for f in FILES:
    s = open(f).read()
    if "CAP_PROP_FOURCC" in s:
        print("ALREADY HAS FOURCC:", f)
        continue
    lines = s.split("\n")
    out = []
    added = 0
    for line in lines:
        m = re.match(r"^(\s*)(\S+)\.set\(cv2\.CAP_PROP_FRAME_WIDTH", line)
        if m:
            indent, prefix = m.group(1), m.group(2)
            out.append(f'{indent}{prefix}.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))')
            added += 1
        out.append(line)
    if added:
        open(f, "w").write("\n".join(out))
        print(f"ADDED {added} MJPG fourcc:", f)
    else:
        print("NO FRAME_WIDTH MATCH:", f)
