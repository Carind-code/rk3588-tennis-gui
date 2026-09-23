#!/usr/bin/env python3
"""RK3588 real-time single-person skeleton overlay using YOLOv8n-pose RKNN."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise SystemExit("Missing RKNNLite2. Run this program on the RK3588 board environment.") from exc

from camera_capture import LatestCamera

INPUT_SIZE = 640
NUM_KEYPOINTS = 17
SKELETON: Tuple[Tuple[int, int], ...] = (
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
)
KP_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),
    (255, 128, 0), (255, 128, 0), (255, 51, 255), (255, 51, 255), (255, 51, 255),
    (255, 51, 255), (255, 51, 255),
)


@dataclass
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    src_w: int
    src_h: int


@dataclass
class PoseResult:
    box: np.ndarray
    keypoints: np.ndarray


@dataclass
class InferencePacket:
    frame_id: int
    timestamp: float
    poses: List[PoseResult]
    npu_ms: float
    pipeline_ms: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def letterbox_bgr(frame: np.ndarray, size: int = INPUT_SIZE) -> Tuple[np.ndarray, Letterbox]:
    src_h, src_w = frame.shape[:2]
    scale = min(size / src_w, size / src_h)
    new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 56, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas[..., ::-1], Letterbox(scale, pad_x, pad_y, src_w, src_h)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


def nms(indexed: List[Tuple[np.ndarray, np.ndarray]], threshold: float, max_persons: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    ordered = sorted(indexed, key=lambda item: float(item[0][4]), reverse=True)
    kept: List[Tuple[np.ndarray, np.ndarray]] = []
    while ordered and len(kept) < max_persons:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [item for item in ordered if iou(best[0], item[0]) < threshold]
    return kept


def normalize_head(head: np.ndarray) -> np.ndarray:
    head = np.asarray(head, dtype=np.float32)
    if head.ndim != 4:
        raise ValueError(f"Unexpected detection head shape: {head.shape}")
    if head.shape[1] == 65:
        return head
    if head.shape[-1] == 65:
        return head.transpose(0, 3, 1, 2)
    raise ValueError(f"Cannot locate 65 detection channels in shape: {head.shape}")


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3:
        raise ValueError(f"Unexpected keypoint output shape: {keypoints.shape}")
    if keypoints.shape[1] == NUM_KEYPOINTS * 3:
        return keypoints
    if keypoints.shape[-1] == NUM_KEYPOINTS * 3:
        return keypoints.transpose(0, 2, 1)
    raise ValueError(f"Cannot locate 51 keypoint channels in shape: {keypoints.shape}")


class PoseRuntime:
    def __init__(self, model_path: Path, person_threshold: float, nms_threshold: float, max_persons: int) -> None:
        self.person_threshold = person_threshold
        self.nms_threshold = nms_threshold
        self.max_persons = max_persons
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model {model_path}: ret={ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize NPU runtime: ret={ret}")

    def close(self) -> None:
        self.rknn.release()

    def infer(self, frame: np.ndarray) -> Tuple[List[PoseResult], float]:
        input_rgb, meta = letterbox_bgr(frame)
        started = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_rgb], data_format=["nhwc"])
        npu_ms = (time.perf_counter() - started) * 1000.0
        return self._decode(outputs, meta), npu_ms

    def _decode(self, outputs: Sequence[np.ndarray], meta: Letterbox) -> List[PoseResult]:
        if len(outputs) != 4:
            raise RuntimeError(f"Expected four YOLOv8-pose outputs, received {len(outputs)}")
        heads = [normalize_head(output) for output in outputs[:3]]
        keypoints = normalize_keypoints(outputs[3])
        candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        anchor_offset = 0
        for head in heads:
            _, _, height, width = head.shape
            stride = INPUT_SIZE // height
            flat = head.reshape(1, 65, -1)
            confidence = sigmoid(flat[0, 64])
            for anchor in np.flatnonzero(confidence >= self.person_threshold):
                gy, gx = divmod(int(anchor), width)
                dfl = flat[0, :64, anchor].reshape(4, 16)
                distance = (softmax(dfl, axis=1) * np.arange(16, dtype=np.float32)).sum(axis=1)
                box = np.array([
                    (gx + 0.5 - distance[0]) * stride,
                    (gy + 0.5 - distance[1]) * stride,
                    (gx + 0.5 + distance[2]) * stride,
                    (gy + 0.5 + distance[3]) * stride,
                    confidence[anchor],
                ], dtype=np.float32)
                kp = keypoints[0, :, anchor_offset + int(anchor)].reshape(NUM_KEYPOINTS, 3).copy()
                candidates.append((box, kp))
            anchor_offset += height * width
        selected: List[PoseResult] = []
        for box, kp in nms(candidates, self.nms_threshold, self.max_persons):
            box = box.copy()
            box[:4:2] = np.clip((box[:4:2] - meta.pad_x) / meta.scale, 0, meta.src_w - 1)
            box[1:4:2] = np.clip((box[1:4:2] - meta.pad_y) / meta.scale, 0, meta.src_h - 1)
            kp[:, 0] = np.clip((kp[:, 0] - meta.pad_x) / meta.scale, 0, meta.src_w - 1)
            kp[:, 1] = np.clip((kp[:, 1] - meta.pad_y) / meta.scale, 0, meta.src_h - 1)
            selected.append(PoseResult(box=box, keypoints=kp))
        return selected


class LatestPoseWorker:
    """One NPU worker. Any queued stale frame is replaced with the newest one."""

    def __init__(self, runtime: PoseRuntime) -> None:
        self.runtime = runtime
        self._jobs: "queue.Queue[Tuple[int, float, np.ndarray]]" = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest: Optional[InferencePacket] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pose-rknn", daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, timestamp: float, frame: np.ndarray) -> None:
        job = (frame_id, timestamp, frame.copy())
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                return
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                return

    def latest(self) -> Optional[InferencePacket]:
        with self._result_lock:
            return self._latest

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame_id, timestamp, frame = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                poses, npu_ms = self.runtime.infer(frame)
                packet = InferencePacket(frame_id, timestamp, poses, npu_ms, (time.perf_counter() - started) * 1000.0)
                with self._result_lock:
                    self._latest = packet
            except Exception as exc:
                print(f"POSE_WORKER_ERROR {exc}", file=sys.stderr, flush=True)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def draw_overlay(frame: np.ndarray, packet: Optional[InferencePacket], keypoint_threshold: float, stale_after: float) -> np.ndarray:
    canvas = frame.copy()
    if packet is None or time.perf_counter() - packet.timestamp > stale_after:
        cv2.putText(canvas, "POSE: waiting", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
        return canvas
    for pose in packet.poses:
        x1, y1, x2, y2, score = pose.box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (55, 255, 90), 2)
        cv2.putText(canvas, f"person {score:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 255, 90), 2, cv2.LINE_AA)
        for start, end in SKELETON:
            a, b = pose.keypoints[start - 1], pose.keypoints[end - 1]
            if a[2] >= keypoint_threshold and b[2] >= keypoint_threshold:
                cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (80, 255, 80), 3, cv2.LINE_AA)
        for index, point in enumerate(pose.keypoints):
            if point[2] >= keypoint_threshold:
                cv2.circle(canvas, (int(point[0]), int(point[1])), 4, KP_COLORS[index], -1, cv2.LINE_AA)
    cv2.putText(canvas, f"POSE: {len(packet.poses)}  NPU {packet.npu_ms:.1f} ms", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 255, 90), 2, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="RK3588 YOLOv8n-pose real-time skeleton overlay")
    parser.add_argument("--model", type=Path, default=root / "models" / "yolov8n_pose_i8.rknn")
    parser.add_argument("--source", choices=("camera", "video"), default="camera")
    parser.add_argument("--camera", default="/dev/video21")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--video", type=Path, help="Required only with --source video")
    parser.add_argument("--out-video", type=Path, help="Optional annotated MP4 output for --source video")
    parser.add_argument("--person-threshold", type=float, default=0.50)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--keypoint-threshold", type=float, default=0.35)
    parser.add_argument("--max-persons", type=int, default=1)
    parser.add_argument("--stale-after", type=float, default=0.35)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def print_summary(title: str, frames: int, elapsed: float, npu_samples: List[float]) -> None:
    avg_npu = float(np.mean(npu_samples)) if npu_samples else 0.0
    print("=" * 62)
    print(title)
    print(f"Frames displayed: {frames}")
    print(f"Average NPU inference: {avg_npu:.2f} ms, theoretical pose FPS: {1000.0 / avg_npu if avg_npu else 0.0:.2f}")
    print(f"End-to-end visual FPS: {frames / elapsed if elapsed else 0.0:.2f}")
    print("=" * 62)


def run_camera(args: argparse.Namespace, runtime: PoseRuntime) -> None:
    camera = LatestCamera(args.camera, args.camera_width, args.camera_height, args.camera_fps)
    camera.start()
    worker = LatestPoseWorker(runtime)
    print(f"POSE_READY source={args.camera} requested={args.camera_width}x{args.camera_height}@{args.camera_fps:g} model={args.model.name}")
    print("Press q or Esc in the preview window to stop.")
    frame_id, displayed, last_packet_id = 0, 0, -1
    npu_samples: List[float] = []
    started = last_report = time.perf_counter()
    try:
        while True:
            incoming = camera.latest_after(frame_id)
            if incoming is None:
                time.sleep(0.001)
                continue
            frame_id, timestamp, frame = incoming
            worker.submit(frame_id, timestamp, frame)
            packet = worker.latest()
            if packet is not None and packet.frame_id != last_packet_id:
                npu_samples.append(packet.npu_ms)
                last_packet_id = packet.frame_id
            rendered = draw_overlay(frame, packet, args.keypoint_threshold, args.stale_after)
            displayed += 1
            if not args.no_display:
                cv2.imshow("QiuWu AI | RK3588 Live Pose", rendered)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            now = time.perf_counter()
            if now - last_report >= 5.0:
                recent = float(np.mean(npu_samples[-30:])) if npu_samples else 0.0
                print(f"LIVE capture={frame_id} display={displayed / (now - started):.2f}fps pose={1000.0 / recent if recent else 0.0:.2f}fps avg_npu={recent:.1f}ms")
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - started
        worker.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print_summary("RK3588 camera live pose statistics", displayed, elapsed, npu_samples)


def run_video(args: argparse.Namespace, runtime: PoseRuntime) -> None:
    if args.video is None:
        raise ValueError("--video is required with --source video")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer: Optional[cv2.VideoWriter] = None
    if args.out_video:
        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames, npu_samples = 0, []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            poses, npu_ms = runtime.infer(frame)
            npu_samples.append(npu_ms)
            rendered = draw_overlay(frame, InferencePacket(frames, time.perf_counter(), poses, npu_ms, npu_ms), args.keypoint_threshold, 99.0)
            if writer is not None:
                writer.write(rendered)
            if not args.no_display:
                cv2.imshow("QiuWu AI | RK3588 Pose Video", rendered)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print_summary("RK3588 pose video statistics", frames, time.perf_counter() - started, npu_samples)


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"RKNN model not found: {args.model}")
    runtime = PoseRuntime(args.model, args.person_threshold, args.nms_threshold, args.max_persons)
    try:
        run_camera(args, runtime) if args.source == "camera" else run_video(args, runtime)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
