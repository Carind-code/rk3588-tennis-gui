#!/usr/bin/env python3
"""
USB AR0234 摄像头验证 + 录制脚本（无头可用，适合 SSH 远程跑）
用法:
  # 只验证 + 测帧率
  python3 usb_camera_live.py --camera /dev/video21 --width 1920 --height 1080 --fps 60
  # 抓一帧看画面
  python3 usb_camera_live.py --camera /dev/video21 --snapshot /tmp/test.jpg
  # 录 5 秒视频
  python3 usb_camera_live.py --camera /dev/video21 --record /tmp/rec.mp4 --seconds 5
  # 固定对焦到远景(网球推荐，值 800~1000 按清晰度调)
  python3 usb_camera_live.py --camera /dev/video21 --focus 900 --snapshot /tmp/far.jpg
"""
import argparse
import time
import cv2
from usb_camera import USBCamera


def measure_fps(cam, num):
    t0 = time.perf_counter()
    got = 0
    while got < num:
        ret, _ = cam.read()
        if not ret:
            time.sleep(0.005)
            continue
        got += 1
    return got / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="/dev/video21")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--focus", type=int, default=None,
                   help="固定对焦 0~1023(远景用大值); 不传则保持自动对焦")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--snapshot", default=None, help="保存一帧 JPEG 的路径")
    p.add_argument("--record", default=None, help="录制 mp4 的路径")
    p.add_argument("--seconds", type=int, default=5)
    a = p.parse_args()

    cam = USBCamera(a.camera, a.width, a.height, a.fps, focus=a.focus)
    print("[camera]", cam.info())

    if cam.width == 0 or cam.height == 0:
        print("错误: 分辨率协商失败，先用 v4l2-ctl --list-formats-ext 查支持")
        cam.release()
        return

    # 预热(丢弃前几帧，等自动曝光/白平衡稳定)
    for _ in range(10):
        ret, frame = cam.read()

    ret, frame = cam.read()
    if not ret:
        print("错误: 读帧失败")
        cam.release()
        return
    h, w = frame.shape[:2]
    print("[frame] 首帧 %s dtype=%s" % (frame.shape, frame.dtype))

    if a.snapshot:
        cv2.imwrite(a.snapshot, frame)
        print("[snapshot] 已保存 %s" % a.snapshot)

    if a.record:
        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (w, h))
        n = int(a.fps * a.seconds)
        for _ in range(n):
            ret, f = cam.read()
            if ret:
                writer.write(f)
        writer.release()
        print("[record] 已保存 %s (%d 帧)" % (a.record, n))

    fps = measure_fps(cam, a.frames)
    print("[fps] 实测 %.2f fps (请求 %d)" % (fps, a.fps))

    cam.release()


if __name__ == "__main__":
    main()
