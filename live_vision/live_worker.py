"""One-camera, two-model, low-latency live recognition worker for QiuWu AI."""

import math
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from rknnlite.api import RKNNLite

from camera_capture import LatestCamera
from pose_engine import PoseRuntime, LatestPoseWorker, draw_overlay
from ball_engine import LatestDetection, GreenTrail, hsv_ball_candidate


class LiveVisionWorker(QThread):
    """Captures the camera once and publishes annotated latest frames.

    Both inference workers use a one-item queue: when inference is slower than
    the camera, stale frames are discarded instead of increasing display lag.
    """
    frame_ready = pyqtSignal(object)
    metrics_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, camera="/dev/video21", width=1280, height=720, fps=30,
                 parent=None):
        super().__init__(parent)
        self.camera_device, self.width, self.height, self.fps = camera, width, height, fps
        self._stopping = False

    def stop(self):
        self._stopping = True
        self.wait(3500)

    def run(self):
        camera = pose = pose_worker = ball = None
        try:
            root = Path(__file__).resolve().parent
            # Pose receives cores 0+1; ball detection receives core 2.
            pose = PoseRuntime(root / "yolov8n_pose_i8.rknn", .50, .45, 1,
                               core_mask=RKNNLite.NPU_CORE_0_1)
            pose_worker = LatestPoseWorker(pose)
            ball = LatestDetection(str(root / "yolo_tennis_ball_fp.rknn"), .35,
                                   core_mask=RKNNLite.NPU_CORE_2)
            ball.start()
            camera = LatestCamera(self.camera_device, self.width, self.height, self.fps)
            camera.start()
            last_id, shown, started, last_report = 0, 0, time.perf_counter(), 0.0
            trail = GreenTrail(length=22, max_jump=280.0)
            while not self._stopping:
                packet = camera.latest_after(last_id)
                if packet is None:
                    self.msleep(1)
                    continue
                frame_id, stamp, frame = packet
                last_id = frame_id
                pose_worker.submit(frame_id, stamp, frame)
                ball.submit(frame_id, stamp, frame)

                rendered = draw_overlay(frame, pose_worker.latest(), .35, .40)
                latest = ball.result()
                point = box = None
                source = "SEARCHING"
                if latest and latest["point"] and stamp - latest["stamp"] <= .35:
                    point, box, source = latest["point"], latest["box"], "YOLO"
                candidate = hsv_ball_candidate(frame, 20, frame.shape[0] * frame.shape[1] * .02, point)
                if candidate:
                    cpoint, cbox, _, _ = candidate
                    # A valid HSV/circularity candidate may bootstrap tracking
                    # when YOLO is below its confidence threshold. Previously
                    # this path was unreachable unless YOLO had already found
                    # the ball, so tracking could never recover from a miss.
                    if point is None:
                        point, box, source = cpoint, cbox, "COLOR"
                    elif math.hypot(cpoint[0] - point[0], cpoint[1] - point[1]) <= 100:
                        point, box, source = cpoint, cbox, "YOLO+COLOR"
                point = trail.update(point, stamp)
                elapsed = max(.001, time.perf_counter() - started)
                trail.draw(rendered, point, box, source, shown / elapsed)
                cv2.rectangle(rendered, (0, rendered.shape[0] - 38), (rendered.shape[1], rendered.shape[0]), (11, 18, 32), -1)
                cv2.putText(rendered, "LIVE  |  Pose + Ball tracking  |  press Stop to end", (16, rendered.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, .52, (231, 243, 75), 1, cv2.LINE_AA)
                self.frame_ready.emit(rendered)
                shown += 1
                now = time.perf_counter()
                if now - last_report >= 1.0:
                    pose_packet = pose_worker.latest()
                    self.metrics_ready.emit({
                        "display_fps": round(shown / elapsed, 1),
                        "pose_ms": round(pose_packet.npu_ms, 1) if pose_packet else 0.0,
                        "ball_ms": round(ball.npu_ms / max(1, ball.count), 1),
                        "ball_source": source,
                    })
                    last_report = now
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if ball:
                ball.close()
            if pose_worker:
                pose_worker.stop()
            if pose:
                pose.close()
            if camera:
                camera.stop()
