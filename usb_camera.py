#!/usr/bin/env python3
"""
USB UVC 摄像头通用封装（Sunplus DCXIN / AR0234 模组实测）
特性:
  - V4L2 + MJPG 高帧率直出（无需 RK3588 ISP）
  - 可选固定对焦（关掉自动对焦，避免网球追踪时的"呼吸对焦"）
  - 可选工频设置（50/60Hz 抗闪烁）
  - 打开后校验"请求 vs 实际"分辨率/格式/帧率

模组能力（v4l2-ctl --list-formats-ext 实测）:
  MJPG: 1920x1200 / 1080 / 720 / 360  @ 120/90/60/30/10 fps
  YUYV: 1920x1200 / 1080 @ 最高 60fps；720/360 @ 最高 120fps
"""
import subprocess
import cv2


def set_uvc_ctrl(device, ctrl, value):
    """通过 v4l2-ctl 设置 UVC 控制项。
    注: cv2 对 UVC 控制项(CAP_PROP_FOCUS 等)映射不可靠，用 v4l2-ctl 更稳。"""
    cmd = ["v4l2-ctl", "-d", device, "--set-ctrl={}={}".format(ctrl, value)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def fix_focus(device, absolute):
    """关自动对焦 -> 手动固定对焦。absolute: 0=近景, 1023=远景"""
    ok1, e1 = set_uvc_ctrl(device, "focus_auto", 0)
    ok2, e2 = set_uvc_ctrl(device, "focus_absolute", absolute)
    return ok1 and ok2, (e1, e2)


def set_power_line_freq(device, hz=50):
    """工频抗闪烁: 50Hz(中国) -> 1, 60Hz -> 2"""
    return set_uvc_ctrl(device, "power_line_frequency", 1 if hz == 50 else 2)


class USBCamera:
    def __init__(self, device, width=1920, height=1080, fps=60,
                 fourcc="MJPG", focus=None, power_hz=50):
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头 %s" % device)

        # 顺序很重要: 先 FOURCC -> 再分辨率 -> 最后帧率(改格式会重置分辨率)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 固定对焦(可选，传 None 则保持自动对焦)
        if focus is not None:
            ok, err = fix_focus(device, focus)
            if not ok:
                print("[warn] 对焦设置失败: %s" % (err,))

        # 工频(可选，传 None 则不设置)
        if power_hz:
            set_power_line_freq(device, power_hz)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = "".join(chr((fcc >> i * 8) & 0xFF) for i in range(4))

    def info(self):
        return "设备 %s: %dx%d %s @ %.2f fps" % (
            self.device, self.width, self.height, self.fourcc, self.fps)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="USB 摄像头快速自检")
    p.add_argument("--camera", default="/dev/video21")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--focus", type=int, default=None)
    a = p.parse_args()
    cam = USBCamera(a.camera, a.width, a.height, a.fps, focus=a.focus)
    print(cam.info())
    ret, f = cam.read()
    print("首帧:", "OK" if ret else "失败", f.shape if ret else "")
    cam.release()
