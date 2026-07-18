#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Preprocessing Pipeline — RK3588 optimized.

Flow:
  V4L2 (/dev/video11 ISP) → NV12 1920×1080
  → Resize ONCE to 640×360 BGR (OpenCV ~2ms)
  → Ring Buffer stores RESIZED frames (3 frames: t, t-1, t-2)
  → Tensor: direct slice copy (no re-resize) → 9-channel uint8
  → Feed to RKNN TrackNet model

Key optimization: resize at buffer entry, not at tensor build time.
"""

import time
import cv2
import numpy as np

INPUT_W = 640
INPUT_H = 360
INPUT_CHANNELS = 9


class RingBuffer:
    """Fixed-size ring buffer storing pre-resized 640x360 BGR frames."""

    def __init__(self, size=3):
        self._buf = [None] * size
        self._idx = 0
        self._size = size
        self._filled = False
        # Pre-allocate all buffer slots
        for i in range(size):
            self._buf[i] = np.empty((INPUT_H, INPUT_W, 3), dtype=np.uint8)

    def put(self, frame_bgr):
        """Store a pre-resized 640x360 BGR frame. Overwrites oldest."""
        np.copyto(self._buf[self._idx], frame_bgr)
        self._idx = (self._idx + 1) % self._size
        if self._idx == 0:
            self._filled = True

    def get(self, offset=0):
        """Get frame at offset (0=current, 1=prev, 2=preprev)."""
        if not self._filled and offset >= self._idx:
            return None
        return self._buf[(self._idx - 1 - offset) % self._size]

    @property
    def ready(self):
        return self._filled


class CameraPipeline:
    """Live camera → optimized tensor assembly."""

    def __init__(self, device="/dev/video11"):
        self.device = device
        self.cap = None
        self.running = False
        self.ring = RingBuffer(size=3)

        # Pre-allocate resize buffer and tensor
        self._resized = np.empty((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        self._tensor = np.empty((1, INPUT_H, INPUT_W, INPUT_CHANNELS),
                                dtype=np.uint8)
        # Raw frame reference for recording
        self._raw_frame = None

        self.frame_count = 0
        self.prep_ms = 0.0

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def start(self):
        if not self.open_camera():
            return False
        self.running = True
        return True

    def process_frame(self):
        """Read → resize ONCE → ring → tensor. Returns tensor or None."""
        if not self.running:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_count += 1
        self._raw_frame = frame

        # Resize ONCE (to be stored and reused)
        cv2.resize(frame, (INPUT_W, INPUT_H), dst=self._resized,
                   interpolation=cv2.INTER_LINEAR)
        self.ring.put(self._resized)

        if not self.ring.ready:
            return None
        return self._build_tensor()

    def _build_tensor(self):
        """Direct slice copy — no resize needed."""
        t0 = time.perf_counter()
        self._tensor[0, :, :, 0:3] = self.ring.get(0)
        self._tensor[0, :, :, 3:6] = self.ring.get(1)
        self._tensor[0, :, :, 6:9] = self.ring.get(2)
        self.prep_ms = (time.perf_counter() - t0) * 1000.0
        return self._tensor

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_raw_frame(self):
        return self._raw_frame

    def get_display_frame(self):
        f = self.ring.get(0)
        return cv2.resize(f, (800, 450)) if f is not None else None
