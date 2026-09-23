"""Low-latency V4L2 camera capture for RK3588 live display."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class LatestCamera:
    """Continuously read a V4L2 camera and expose only its newest frame."""

    def __init__(self, device: str, width: int, height: int, fps: float, buffer_size: int = 1) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._timestamp = 0.0
        self.read_failures = 0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.read_failures += 1
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._timestamp = time.perf_counter()

    def latest_after(self, frame_id: int) -> Optional[Tuple[int, float, np.ndarray]]:
        with self._lock:
            if self._frame is None or self._frame_id <= frame_id:
                return None
            return self._frame_id, self._timestamp, self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
