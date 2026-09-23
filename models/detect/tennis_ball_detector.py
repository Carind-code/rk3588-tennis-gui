#!/usr/bin/env python3
"""RK3588 close-range tennis-ball detection and fluorescent-green tracking.

The NPU runs one INT8 YOLOv8 tennis-ball detector.  A bounded latest-frame
worker prevents detector latency from accumulating.  Between detector results,
an HSV/circularity candidate tracker keeps close yellow-green tennis balls
smoothly visualized at camera frame rate.
"""

import argparse
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from camera_capture import LatestCamera


ROOT = Path(__file__).resolve().parent
MODEL_SIZE = 640
CROP_SIZE = 384
NEON_GREEN = (55, 255, 55)


def project_path(value):
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def letterbox(image):
    height, width = image.shape[:2]
    scale = min(MODEL_SIZE / height, MODEL_SIZE / width)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    pad_x, pad_y = (MODEL_SIZE - resized_w) // 2, (MODEL_SIZE - resized_h) // 2
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    return canvas, scale, pad_x, pad_y


def xywh_to_xyxy(values):
    result = np.empty_like(values)
    result[:, 0] = values[:, 0] - values[:, 2] / 2.0
    result[:, 1] = values[:, 1] - values[:, 3] / 2.0
    result[:, 2] = values[:, 0] + values[:, 2] / 2.0
    result[:, 3] = values[:, 1] + values[:, 3] / 2.0
    return result


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - inter + 1e-6
        order = order[1:][inter / union <= iou_threshold]
    return keep


def decode_yolo(output, scale, pad_x, pad_y, width, height, confidence):
    prediction = np.asarray(output).squeeze().astype(np.float32)
    if prediction.ndim != 2:
        prediction = prediction.reshape(prediction.shape[0], -1)
    if prediction.shape[0] <= 16 and prediction.shape[1] > prediction.shape[0]:
        prediction = prediction.T
    if prediction.shape[1] < 5:
        return []
    scores = prediction[:, 4] if prediction.shape[1] == 5 else np.max(prediction[:, 4:], axis=1)
    valid = scores >= confidence
    if not np.any(valid):
        return []
    boxes = xywh_to_xyxy(prediction[valid, :4])
    scores = scores[valid]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return [(boxes[index], float(scores[index])) for index in nms(boxes, scores)]


def hsv_ball_candidate(frame, min_area, max_area, near_point=None):
    height, width = frame.shape[:2]
    # HSV segmentation runs at full resolution so a far-baseline ball
    # (~4px wide at 1920) stays detectable; the candidate is mapped back
    # to the original image afterwards.  The 1920 cap only guards against
    # sources wider than 1920px.
    resize_scale = min(1.0, 1920.0 / max(width, height))
    if resize_scale < 1.0:
        work = cv2.resize(frame, (int(round(width * resize_scale)), int(round(height * resize_scale))), interpolation=cv2.INTER_AREA)
    else:
        work = frame
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    # Tennis ball yellow-green. The model remains the semantic authority; this
    # only follows the latest confirmed ball between detector frames.
    mask = cv2.inRange(hsv, (18, 75, 75), (58, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area * resize_scale * resize_scale or area > max_area * resize_scale * resize_scale:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = min(w, h) / float(max(w, h))
        if ratio < 0.55:
            continue
        center = ((x + w / 2.0) / resize_scale, (y + h / 2.0) / resize_scale)
        candidates.append((center, (x / resize_scale, y / resize_scale, (x + w) / resize_scale, (y + h) / resize_scale), circularity, area / (resize_scale * resize_scale)))
    if not candidates:
        return None
    if near_point is None:
        return max(candidates, key=lambda value: value[2] * value[3])
    return min(candidates, key=lambda value: math.hypot(value[0][0] - near_point[0], value[0][1] - near_point[1]))


class LatestDetection:
    def __init__(self, model_path, confidence, crop_size=CROP_SIZE, min_area=8.0, max_area_fraction=0.18):
        self.model_path = model_path
        self.confidence = confidence
        self.crop_size = crop_size
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction
        self.jobs = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.count = 0
        self.npu_ms = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, frame_id, stamp, frame):
        try:
            while True:
                self.jobs.get_nowait()
        except queue.Empty:
            pass
        try:
            self.jobs.put_nowait((frame_id, stamp, frame))
        except queue.Full:
            pass

    def result(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop_event.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)

    def _run(self):
        runtime = None
        try:
            runtime = RKNNLite()
            if runtime.load_rknn(self.model_path) != 0:
                raise RuntimeError("load_rknn failed")
            if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                raise RuntimeError("init_runtime failed")
            while not self.stop_event.is_set():
                try:
                    job = self.jobs.get(timeout=0.1)
                except queue.Empty:
                    continue
                if job is None:
                    break
                frame_id, stamp, frame = job
                height, width = frame.shape[:2]
                # Crop-and-zoom: a far-baseline ball shrinks to ~1px when the
                # whole frame is letterboxed to 640, so YOLO cannot see it.
                # Find a full-resolution HSV candidate, crop a window around it,
                # and run YOLO on that window instead -- the ball then spans a
                # usable number of pixels and YOLO confirms it semantically.
                with self.lock:
                    last = self.latest
                near = last["point"] if last and last.get("point") is not None else None
                color_point = hsv_ball_candidate(frame, self.min_area, width * height * self.max_area_fraction, near)
                if color_point is not None:
                    cx, cy = color_point[0]
                    crop = min(self.crop_size, width, height)
                    x0 = int(round(min(max(0.0, cx - crop / 2.0), float(width - crop))))
                    y0 = int(round(min(max(0.0, cy - crop / 2.0), float(height - crop))))
                    region = frame[y0:y0 + crop, x0:x0 + crop]
                    model_input, scale, pad_x, pad_y = letterbox(region)
                    crop_w, crop_h, origin_x, origin_y = crop, crop, x0, y0
                else:
                    model_input, scale, pad_x, pad_y = letterbox(frame)
                    crop_w, crop_h, origin_x, origin_y = width, height, 0, 0
                start = time.perf_counter()
                output = runtime.inference(inputs=[model_input[None]], data_format=["nhwc"])[0]
                npu_ms = (time.perf_counter() - start) * 1000.0
                boxes = decode_yolo(output, scale, pad_x, pad_y, crop_w, crop_h, self.confidence)
                if origin_x or origin_y:
                    shifted = []
                    for box, score in boxes:
                        box = box.copy()
                        box[0] += origin_x
                        box[1] += origin_y
                        box[2] += origin_x
                        box[3] += origin_y
                        shifted.append((box, score))
                    boxes = shifted
                if boxes:
                    box, score = max(boxes, key=lambda item: item[1])
                    x1, y1, x2, y2 = box.tolist()
                    point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    detection = {"frame_id": frame_id, "stamp": stamp, "point": point, "box": (x1, y1, x2, y2), "score": score}
                else:
                    detection = {"frame_id": frame_id, "stamp": stamp, "point": None, "box": None, "score": 0.0}
                with self.lock:
                    self.latest = detection
                    self.count += 1
                    self.npu_ms += npu_ms
        except Exception as exc:
            self.error = str(exc)
        finally:
            if runtime is not None:
                runtime.release()


class GreenTrail:
    def __init__(self, length=22, max_jump=280.0):
        self.points = deque(maxlen=length)
        self.max_jump = float(max_jump)
        self.last = None
        self.last_seen = 0.0

    def update(self, point, stamp):
        if point is None:
            if stamp - self.last_seen > 0.45:
                self.points.clear()
                self.last = None
            return None
        if self.last is not None and math.hypot(point[0] - self.last[0], point[1] - self.last[1]) > self.max_jump:
            self.points.clear()
        self.last = point
        self.last_seen = stamp
        self.points.append(point)
        return point

    def draw(self, frame, point, box, source, fps):
        points = list(self.points)
        for index in range(1, len(points)):
            fade = index / max(1.0, len(points) - 1.0)
            color = (int(20 + 40 * fade), int(100 + 145 * fade), int(30 + 35 * fade))
            cv2.line(frame, tuple(map(int, points[index - 1])), tuple(map(int, points[index])), color, max(1, int(1 + 3 * fade)), cv2.LINE_AA)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), NEON_GREEN, 2, cv2.LINE_AA)
        if point is not None:
            center = tuple(map(int, point))
            cv2.circle(frame, center, 9, (10, 35, 10), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 6, NEON_GREEN, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 10, NEON_GREEN, 1, cv2.LINE_AA)
        label = "TENNIS DETECT  {:.1f} FPS  {}".format(fps, source)
        cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 470), 49), (10, 23, 10), -1)
        cv2.putText(frame, label, (22, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.58, NEON_GREEN, 2, cv2.LINE_AA)


def run_camera(args):
    camera = LatestCamera(args.camera, args.camera_width, args.camera_height, args.camera_fps)
    camera.start()
    width, height = camera.actual_size()
    detector = LatestDetection(project_path(args.model), args.confidence, min_area=args.min_area, max_area_fraction=args.max_area_fraction)
    detector.start()
    trail = GreenTrail(args.trail, args.max_jump)
    sequence = 0
    displayed = 0
    start = time.perf_counter()
    last_report = start
    cv2.namedWindow("QIUWU AI - Tennis Ball Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QIUWU AI - Tennis Ball Detection", min(args.display_width, width), min(720, height))
    try:
        while True:
            packet = camera.next_after(sequence)
            now = time.perf_counter()
            if detector.error:
                raise RuntimeError(detector.error)
            if packet is None:
                if now - start > args.camera_timeout and camera.frames == 0:
                    raise RuntimeError("camera returned no frames; check camera USB connection and power")
                cv2.waitKey(1)
                continue
            sequence, stamp, frame = packet
            detector.submit(sequence, stamp, frame)
            latest = detector.result()
            confirmed = None
            box = None
            source = "SEARCHING"
            if latest and latest["point"] is not None and stamp - latest["stamp"] <= args.detector_hold:
                confirmed = latest["point"]
                box = latest["box"]
                source = "YOLO"
            # The near-camera scene normally has a single yellow-green sphere.
            # YOLO is preferred when available. The optional bootstrap lets a
            # constrained HSV/circularity candidate recover the ball after a
            # transient YOLO miss, which keeps handheld demos responsive.
            color_candidate = hsv_ball_candidate(frame, args.min_area, width * height * args.max_area_fraction, confirmed)
            if color_candidate is not None and (confirmed is not None or args.allow_color_bootstrap):
                candidate_point, candidate_box, _, _ = color_candidate
                if confirmed is None or math.hypot(candidate_point[0] - confirmed[0], candidate_point[1] - confirmed[1]) <= args.color_gate:
                    confirmed, box = candidate_point, candidate_box
                    source = "YOLO+COLOR" if latest and latest["point"] else "COLOR"
            point = trail.update(confirmed, stamp)
            displayed += 1
            fps = displayed / max(0.001, now - start)
            trail.draw(frame, point, box, source, fps)
            display = frame
            if frame.shape[1] > args.display_width:
                display = cv2.resize(frame, (args.display_width, int(frame.shape[0] * args.display_width / frame.shape[1])))
            cv2.imshow("QIUWU AI - Tennis Ball Detection", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if now - last_report >= 5.0:
                detector_fps = detector.count / max(0.001, now - start)
                print("LIVE capture={:.2f}FPS visual={:.2f}FPS yolo={:.2f}FPS avg_yolo={:.2f}ms".format(camera.frames / (now - start), fps, detector_fps, detector.npu_ms / max(1, detector.count)), flush=True)
                last_report = now
    finally:
        total = time.perf_counter() - start
        detector.close()
        camera.close()
        cv2.destroyAllWindows()
        print("\n==========================================================")
        print("RK3588 close-range tennis detection")
        print("camera frames: {}  visualized: {}".format(camera.frames, displayed))
        print("camera + visual FPS: {:.2f}".format(displayed / max(0.001, total)))
        print("YOLO inference FPS: {:.2f}".format(detector.count / max(0.001, total)))
        print("YOLO average inference: {:.2f} ms".format(detector.npu_ms / max(1, detector.count)))
        print("total: {:.2f} s".format(total))
        print("==========================================================")


def run_video(args):
    """Deterministic local-video validation. It does not use TrackNet."""
    source = project_path(args.video)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError("cannot open video {}".format(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    runtime = RKNNLite()
    if runtime.load_rknn(project_path(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("init_runtime failed")
    writer = None
    if args.out_video:
        out_path = project_path(args.out_video)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("cannot open VideoWriter {}".format(out_path))
    trail = GreenTrail(args.trail, args.max_jump)
    frames = yolo_detections = color_detections = 0
    infer_s = visual_s = write_s = 0.0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stamp = time.perf_counter()
            prep_start = stamp
            model_input, scale, pad_x, pad_y = letterbox(frame)
            prep_s = time.perf_counter() - prep_start
            infer_start = time.perf_counter()
            output = runtime.inference(inputs=[model_input[None]], data_format=["nhwc"])[0]
            infer_s += time.perf_counter() - infer_start
            boxes = decode_yolo(output, scale, pad_x, pad_y, width, height, args.confidence)
            point = box = None
            source_label = "MISS"
            if boxes:
                box, score = max(boxes, key=lambda item: item[1])
                x1, y1, x2, y2 = box.tolist()
                point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                yolo_detections += 1
                source_label = "YOLO {:.2f}".format(score)
            color_candidate = hsv_ball_candidate(frame, args.min_area, width * height * args.max_area_fraction, point)
            if color_candidate is not None and (point is not None or args.allow_color_bootstrap):
                candidate_point, candidate_box, _, _ = color_candidate
                if point is None or math.hypot(candidate_point[0] - point[0], candidate_point[1] - point[1]) <= args.color_gate:
                    point, box = candidate_point, candidate_box
                    source_label = "YOLO+COLOR" if source_label.startswith("YOLO") else "COLOR"
                    color_detections += 1
            visual_start = time.perf_counter()
            tracked = trail.update(point, stamp)
            elapsed = time.perf_counter() - start
            trail.draw(frame, tracked, box, source_label, (frames + 1) / max(elapsed, 1e-6))
            visual_s += time.perf_counter() - visual_start
            if writer is not None:
                write_start = time.perf_counter()
                writer.write(frame)
                write_s += time.perf_counter() - write_start
            if not args.no_display:
                shown = frame if width <= args.display_width else cv2.resize(frame, (args.display_width, int(height * args.display_width / width)))
                cv2.imshow("QIUWU AI - Tennis Ball Detection", shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frames += 1
            if frames % 30 == 0:
                print("VIDEO {}/{} yolo={:.2f}FPS overall={:.2f}FPS".format(frames, frame_total or "?", frames / max(infer_s, 1e-6), frames / max(time.perf_counter() - start, 1e-6)), flush=True)
    finally:
        total = time.perf_counter() - start
        capture.release()
        if writer is not None:
            writer.release()
        runtime.release()
        cv2.destroyAllWindows()
    print("\n==========================================================")
    print("RK3588 pure tennis-ball detection: local video")
    print("frames: {}  YOLO detections: {}  color-assisted detections: {}".format(frames, yolo_detections, color_detections))
    print("model inference: {:.3f}s, {:.2f}FPS".format(infer_s, frames / max(infer_s, 1e-6)))
    print("preprocess + overlay: {:.3f}s".format(visual_s))
    print("video writing: {:.3f}s".format(write_s))
    print("full end-to-end: {:.3f}s, {:.2f}FPS".format(total, frames / max(total, 1e-6)))
    if writer is not None:
        print("output: {}".format(project_path(args.out_video)))
    print("==========================================================")


def parse_args():
    parser = argparse.ArgumentParser(description="RK3588 close-range tennis ball detection")
    parser.add_argument("--model", default="models/yolo_tennis_ball_fp.rknn")
    parser.add_argument("--source", choices=("camera", "video"), default="camera")
    parser.add_argument("--video", default="input.mp4")
    parser.add_argument("--out-video", default="outputs/tennis_ball_detect.mp4")
    parser.add_argument("--no-write", dest="out_video", action="store_const", const="")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--camera", default="/dev/video21")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--detector-hold", type=float, default=0.35)
    parser.add_argument("--color-gate", type=float, default=180.0)
    parser.add_argument("--min-area", type=float, default=8.0)
    parser.add_argument("--max-area-fraction", type=float, default=0.18)
    parser.add_argument("--no-color-bootstrap", dest="allow_color_bootstrap", action="store_false")
    parser.set_defaults(allow_color_bootstrap=True)
    parser.add_argument("--trail", type=int, default=22)
    parser.add_argument("--max-jump", type=float, default=280.0)
    parser.add_argument("--display-width", type=int, default=1024)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    if options.source == "video":
        run_video(options)
    else:
        run_camera(options)
