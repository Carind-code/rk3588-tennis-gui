#!/usr/bin/env python3
"""
USB 摄像头 MJPEG HTTP 推流 —— 在 Windows 浏览器里看实时画面
用法:
  python3 usb_camera_stream.py --camera /dev/video21 --width 1280 --height 720 --fps 60 --port 8000
然后在 Windows 浏览器打开: http://192.168.0.232:8000
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from usb_camera import USBCamera


class CameraStreamer:
    def __init__(self, device, width, height, fps, focus):
        self.cam = USBCamera(device, width, height, fps, focus=focus)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self):
        t0 = time.perf_counter()
        n = 0
        while self.running:
            ret, frame = self.cam.read()
            if not ret:
                time.sleep(0.01)
                continue
            ok, jpg = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            with self.lock:
                self.frame = jpg.tobytes()
            n += 1
            if n % 60 == 0:
                el = time.perf_counter() - t0
                print("[stream] %.1f fps (推流中)" % (n / el))

    def get_frame(self):
        with self.lock:
            return self.frame


class Handler(BaseHTTPRequestHandler):
    streamer = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_index()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_index(self):
        html = ("<html><head><title>USB Camera</title></head>"
                "<body style='background:#111;text-align:center'>"
                "<h3 style='color:#ccc'>USB AR0234 实时画面</h3>"
                "<img src='/stream' style='max-width:100%'></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                frame = Handler.streamer.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *args):
        pass  # 静默，避免刷屏


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="/dev/video21")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--focus", type=int, default=900,
                   help="固定对焦 0~1023; 传 -1 表示保持自动对焦")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    focus = None if a.focus < 0 else a.focus
    streamer = CameraStreamer(a.camera, a.width, a.height, a.fps, focus)
    print("[camera]", streamer.cam.info())
    Handler.streamer = streamer

    server = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print("浏览器打开: http://192.168.0.232:%d" % a.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        streamer.running = False
        streamer.cam.release()
        server.server_close()


if __name__ == "__main__":
    main()
