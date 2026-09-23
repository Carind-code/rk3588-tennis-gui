#!/usr/bin/env python3
"""Newest-frame V4L2 capture for the camera realtime detector."""

import threading
import time

import cv2


class LatestCamera:
    def __init__(self, device="/dev/video21", width=1920, height=1080, fps=30):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.frames = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open camera {}".format(self.device))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.sequence += 1
                self.frames += 1
                self.latest = (self.sequence, time.perf_counter(), frame)

    def next_after(self, sequence):
        with self.lock:
            if self.latest is None or self.latest[0] <= sequence:
                return None
            frame_id, stamp, frame = self.latest
            return frame_id, stamp, frame.copy()

    def actual_size(self):
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height),
        )

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

