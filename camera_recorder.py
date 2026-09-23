#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Recorder — RK3588 USB camera recording.

Architecture:
  Producer Thread: V4L2(/dev/video21) → MJPG → BGR (1920×1080)
  Consumer Thread: ffmpeg h264_rkmpp (MPP hardware) with mp4v fallback
  Buffer Pool:     queue.Queue ring for decoupled capture & encode

Encoding note:
  Recording prefers the Rockchip MPP hardware H.264 encoder via
  `ffmpeg -c:v h264_rkmpp`, fed with raw BGR frames on stdin.  ffmpeg
  finalises the MP4 (moov atom) correctly on stdin EOF, unlike the old
  gst-launch appsrc pipeline which never finalised and left orphaned
  processes writing 0-byte files.  If ffmpeg/h264_rkmpp is unavailable,
  it falls back to cv2.VideoWriter with the mp4v codec.

Usage:
    python3 camera_recorder.py                 # preview + record (press space)
    python3 camera_recorder.py --headless      # record only, no window
"""

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import cv2

# ── Constants ────────────────────────────────────────────────────
DEVICE = "/dev/video21"
CAP_W, CAP_H = 1920, 1080
PREVIEW_W, PREVIEW_H = 800, 450
BUFFER_POOL_SIZE = 4
OUTPUT_DIR = os.path.expanduser("~/rk3588_tennis_system/outputs/recordings")

# ── Shutdown ─────────────────────────────────────────────────────
_shutdown = threading.Event()


def _on_signal(s, f):
    _shutdown.set()


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


# ── Filename ─────────────────────────────────────────────────────

def next_filename():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(OUTPUT_DIR, "recording_{}.mp4".format(i))):
        i += 1
    return os.path.join(OUTPUT_DIR, "recording_{}.mp4".format(i))


# ── Camera (Producer) ────────────────────────────────────────────

class CameraProducer(threading.Thread):
    """V4L2 capture thread → frame queue."""

    def __init__(self, device, width, height, pool_size=BUFFER_POOL_SIZE):
        super().__init__(daemon=True)
        self.device = device
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=pool_size)
        self.running = False
        self.frames_captured = 0

    def open(self):
        # The USB camera occasionally fails its first open right after a
        # previous release; retry a few times before giving up.
        for _ in range(3):
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if self.cap.isOpened():
                break
            self.cap.release()
            time.sleep(0.3)
        else:
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Verify actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def run(self):
        self.running = True
        while self.running and not _shutdown.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.002)
                continue
            self.frames_captured += 1
            try:
                self.frame_queue.put(frame, timeout=0.5)
            except queue.Full:
                # Consumer too slow, drop oldest by reading one out
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass

    def stop(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()


# ── Recorder (Consumer) ──────────────────────────────────────────

class RecorderConsumer:
    """Encodes frames from the queue. Uses ffmpeg h264_rkmpp (MPP hardware)
    when available, otherwise cv2.VideoWriter (mp4v)."""

    def __init__(self, output_path, width, height, fps=30, codec="h264"):
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.output_path = output_path
        self.writer = None        # cv2.VideoWriter fallback
        self.ffmpeg = None        # ffmpeg subprocess
        self.frames_written = 0
        self.recording = False

    def _has_rkmpp(self):
        """True if ffmpeg ships the Rockchip MPP H.264 encoder."""
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10)
        except Exception:
            return False
        return "h264_rkmpp" in (out.stdout or "")

    def _ffmpeg_cmd(self):
        return [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(self.width, self.height),
            "-r", str(self.fps), "-i", "pipe:0",
            "-c:v", "h264_rkmpp", "-b:v", "12000000", "-g", "30",
            "-r", str(self.fps),
            "-movflags", "+faststart",
            self.output_path,
        ]

    def start(self):
        self.recording = True
        # Prefer ffmpeg + MPP hardware encoder.
        if self.codec == "h264" and self._has_rkmpp():
            try:
                self.ffmpeg = subprocess.Popen(
                    self._ffmpeg_cmd(), stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                self.ffmpeg = None
            if self.ffmpeg:
                print("[REC] MPP hardware encoder (h264_rkmpp)")
                return True
        # Fallback: cv2.VideoWriter mp4v.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            self.recording = False
            return False
        print("[REC] Software encoder (mp4v) {}x{}".format(
            self.width, self.height))
        return True

    def write(self, frame):
        if not self.recording:
            return
        if frame is None:
            return
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            # A size mismatch silently corrupts the output; skip such frames.
            return
        if self.ffmpeg:
            if self.ffmpeg.poll() is not None:
                print("[REC] encoder process exited unexpectedly")
                self.recording = False
                return
            try:
                self.ffmpeg.stdin.write(frame.tobytes())
                self.frames_written += 1
            except (BrokenPipeError, OSError):
                print("[REC] encoder pipe broken")
                self.recording = False
        elif self.writer:
            self.writer.write(frame)
            self.frames_written += 1

    def stop(self):
        self.recording = False
        if self.ffmpeg:
            try:
                self.ffmpeg.stdin.close()
            except Exception:
                pass
            try:
                self.ffmpeg.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.ffmpeg.kill()
                self.ffmpeg.wait()
            self.ffmpeg = None
        if self.writer:
            self.writer.release()
            self.writer = None
        if os.path.exists(self.output_path) and \
                os.path.getsize(self.output_path) > 0:
            sz = os.path.getsize(self.output_path) / (1024 * 1024)
            print("[REC] Saved: {} ({:.1f} MB, {} frames)".format(
                os.path.basename(self.output_path), sz, self.frames_written))
        else:
            print("[REC] No valid recording (0 frames)")


# ── Preview (Consumer) ───────────────────────────────────────────

class PreviewDisplay:
    """Shows frames in a tkinter/PIL window, runs in main thread."""

    def __init__(self, frame_queue, width=PREVIEW_W, height=PREVIEW_H):
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self.root = None
        self.panel = None
        self.running = False
        self._last_frame = None
        self._recorder = None

    def set_recorder(self, rec):
        self._recorder = rec

    def start(self):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.root = tk.Tk()
        self.root.title("Camera Recorder — 摄像头")
        self.root.configure(bg="#0a0e14")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.panel = tk.Label(self.root, bg="#0d1117")
        self.panel.pack(padx=4, pady=(4, 0))

        bar = tk.Frame(self.root, bg="#0a0e14")
        bar.pack(fill=tk.X, padx=8, pady=4)

        self.btn_rec = tk.Button(bar, text="● 开始录制", bg="#1a7f37", fg="white",
                                 font=("", 11, "bold"), relief=tk.FLAT,
                                 command=self._toggle_record)
        self.btn_rec.pack(side=tk.LEFT, padx=2)

        tk.Button(bar, text="✕ 退出", bg="#da3633", fg="white",
                  font=("", 10), relief=tk.FLAT,
                  command=self._close).pack(side=tk.RIGHT, padx=2)

        self.lbl = tk.Label(bar, text="就绪 | 0帧", bg="#0a0e14",
                            fg="#8b949e", font=("", 9))
        self.lbl.pack(side=tk.RIGHT, padx=8)

        self.root.bind('<space>', lambda e: self._toggle_record())
        self.root.bind('<Escape>', lambda e: self._close())
        self.root.bind('q', lambda e: self._close())
        self.root.geometry("{}x{}".format(self.width + 16, self.height + 60))
        self.running = True

    def _toggle_record(self):
        if self._recorder is None:
            return
        if self._recorder.recording:
            self._recorder.stop()
            self.btn_rec.configure(text="● 开始录制", bg="#1a7f37")
        else:
            if self._recorder.start():
                self.btn_rec.configure(text="■ 停止", bg="#da3633")

    def _close(self):
        self.running = False
        _shutdown.set()
        if self._recorder and self._recorder.recording:
            self._recorder.stop()
        if self.root:
            self.root.quit()
            self.root.destroy()

    def refresh(self):
        """Called periodically from main thread."""
        if not self.running:
            return
        frame = None
        drained = 0
        while drained < 3:
            try:
                frame = self.frame_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if frame is not None:
            self._last_frame = frame
            preview = cv2.resize(frame, (self.width, self.height))
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)
            # Status
            parts = []
            if self._recorder and self._recorder.recording:
                parts.append("● REC {}".format(self._recorder.frames_written))
            self.lbl.configure(text=" | ".join(parts) if parts else "预览")
        if self.running:
            self.root.after(33, self.refresh)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RK3588 Camera Recorder")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--codec", choices=["h264", "h265", "mp4v"],
                        default="h264",
                        help="recording codec (h264 = MPP hardware; "
                             "h265/mp4v fall back to mp4v software)")
    parser.add_argument("--width", type=int, default=CAP_W)
    parser.add_argument("--height", type=int, default=CAP_H)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    output_path = next_filename()

    # Producer
    producer = CameraProducer(args.device, args.width, args.height)
    if not producer.open():
        print("ERROR: Cannot open {}".format(args.device))
        return 1
    print("[CAM] {} {}x{}".format(args.device, producer.width, producer.height))

    # Consumer: Recorder
    recorder = RecorderConsumer(output_path, producer.width, producer.height,
                                fps=30, codec=args.codec)

    if args.headless:
        # Headless: auto-record
        recorder.start()
        producer.start()
        print("[REC] Headless → {}".format(output_path))
        try:
            while producer.is_alive() and not _shutdown.is_set():
                try:
                    frame = producer.frame_queue.get(timeout=0.1)
                    recorder.write(frame)
                except queue.Empty:
                    if not producer.running:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            _shutdown.set()
            recorder.stop()
            producer.stop()
            producer.join(timeout=3)
    else:
        # Preview mode
        preview = PreviewDisplay(producer.frame_queue)
        preview.set_recorder(recorder)
        producer.start()
        preview.start()
        preview.root.after(100, preview.refresh)
        preview.root.mainloop()
        recorder.stop()
        producer.stop()
        producer.join(timeout=3)

    dur = producer.frames_captured / 30.0 if producer.frames_captured else 0
    print("[DONE] {} frames captured, ~{:.0f}s".format(
        producer.frames_captured, dur))
    return 0


if __name__ == "__main__":
    sys.exit(main())
