# -*- coding: utf-8 -*-
"""
AI Overlay — RKNN models on camera frames, drawn in video widget.
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
from pathlib import Path
import sys
import math

MODEL_DIR = "/home/elf/rk3588_tennis_system/models"
MODEL_SIZE = 640
# BGR colors.  The live overlay deliberately shares the formal side-view
# visual language, while detection and tracking logic remain untouched.
NEON_GREEN = (55, 255, 110)
TRAIL_GREEN = (45, 205, 90)
TRAIL_DARK = (20, 95, 45)
MARKER_DARK = (12, 45, 22)

MODELS = {
    "live_ball": {
        # Six-output INT8 head: the realtime model supplied with detect.7z.
        "rknn": f"{MODEL_DIR}/detect/models/yolo_tennis_ball_head_i8.rknn",
    },
    "pose": {
        "rknn": f"{MODEL_DIR}/pose/models/yolov8n_pose_i8.rknn",
    },
}


class AIOverlay:
    """Loads RKNN models and draws predictions on live frames."""

    def __init__(self):
        self._models = {}
        self._active = set()
        self._trail = []
        self._live_ball_misses = 0
        self._pose_module = None
        self._log = lambda msg: None

    def load(self, name):
        if name in self._models:
            return True
        cfg = MODELS.get(name, {})
        path = cfg.get("rknn", "")
        if not path:
            return False
        try:
            if name == "pose":
                engine_dir = "/home/elf/rk3588_tennis_system/live_vision"
                if engine_dir not in sys.path:
                    sys.path.insert(0, engine_dir)
                import pose_engine
                self._models[name] = pose_engine.PoseRuntime(Path(path), .50, .45, 1)
                self._pose_module = pose_engine
                self._log("[AI] pose model loaded OK")
                return True
            rknn = RKNNLite()
            if rknn.load_rknn(path) != 0:
                raise RuntimeError("load_rknn failed")
            if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                raise RuntimeError("init_runtime failed")
            self._models[name] = rknn
            self._log(f"[AI] {name} model loaded OK")
            return True
        except Exception as e:
            self._log(f"[AI] {name} load failed: {e}")
            return False

    def toggle(self, name):
        if name in self._active:
            self._active.discard(name)
            if name == "live_ball":
                self._reset_live_ball_visual()
            return False
        if self.load(name):
            if name == "live_ball":
                self._reset_live_ball_visual()
            self._active.add(name)
            return True
        return False

    @property
    def active_names(self):
        return self._active

    def process(self, frame):
        """Run all active models, return annotated frame."""
        result = frame.copy()
        h, w = result.shape[:2]

        if "live_ball" in self._active:
            result = self._live_ball(result, frame, w, h)
        if "pose" in self._active:
            result = self._pose(result, frame, w, h)

        return result

    def _reset_live_ball_visual(self):
        self._trail = []
        self._live_ball_misses = 0

    def _mark_live_ball_miss(self):
        self._live_ball_misses += 1
        if self._live_ball_misses >= 3:
            self._trail = []

    # ── YOLO (exact copy of tennis_ball_detector.py logic) ─────

    @staticmethod
    def _letterbox(image):
        h, w = image.shape[:2]
        scale = min(MODEL_SIZE / h, MODEL_SIZE / w)
        rw, rh = int(round(w * scale)), int(round(h * scale))
        pad_x, pad_y = (MODEL_SIZE - rw) // 2, (MODEL_SIZE - rh) // 2
        canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y+rh, pad_x:pad_x+rw] = cv2.resize(image, (rw, rh))
        # The lightweight INT8 head was exported from the RGB YOLO model.
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale, pad_x, pad_y

    @staticmethod
    def _xywh2xyxy(v):
        r = np.empty_like(v)
        r[:,0]=v[:,0]-v[:,2]/2; r[:,1]=v[:,1]-v[:,3]/2
        r[:,2]=v[:,0]+v[:,2]/2; r[:,3]=v[:,1]+v[:,3]/2
        return r

    @staticmethod
    def _nms(boxes, scores, thresh=0.45):
        if len(boxes) == 0:
            return []
        x1,y1,x2,y2 = boxes.T
        areas = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = int(order[0]); keep.append(i)
            if order.size == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1)*np.maximum(0, yy2-yy1)
            iou = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order = order[1:][iou < thresh]
        return keep

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values.astype(np.float32), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _as_nchw(value, channels):
        value = np.asarray(value).astype(np.float32)
        if value.ndim != 4:
            raise ValueError("unexpected YOLO head shape: {}".format(value.shape))
        if value.shape[1] == channels:
            return value
        if value.shape[-1] == channels:
            return value.transpose(0, 3, 1, 2)
        raise ValueError("cannot locate {} channels in {}".format(channels, value.shape))

    def _decode_yolo_heads(self, outputs, scale, pad_x, pad_y, w, h):
        """Decode the 3 DFL box heads + 3 class heads of head_i8.rknn."""
        boxes_all, scores_all = [], []
        bins = np.arange(16, dtype=np.float32).reshape(1, 1, 16)
        for branch in range(3):
            position = self._as_nchw(outputs[branch * 2], 64)
            classes = self._as_nchw(outputs[branch * 2 + 1], 1)
            _, _, grid_h, grid_w = position.shape
            score_map = self._sigmoid(classes[0, 0])
            rows, cols = np.where(score_map >= .25)
            if not rows.size:
                continue
            selected = position[0].transpose(1, 2, 0)[rows, cols].reshape(-1, 4, 16)
            distribution = selected - np.max(selected, axis=2, keepdims=True)
            distribution = np.exp(distribution)
            distribution /= np.sum(distribution, axis=2, keepdims=True)
            distances = np.sum(distribution * bins, axis=2)
            stride_x, stride_y = MODEL_SIZE / grid_w, MODEL_SIZE / grid_h
            center_x, center_y = cols.astype(np.float32) + .5, rows.astype(np.float32) + .5
            boxes_all.append(np.stack(((center_x - distances[:, 0]) * stride_x,
                                       (center_y - distances[:, 1]) * stride_y,
                                       (center_x + distances[:, 2]) * stride_x,
                                       (center_y + distances[:, 3]) * stride_y), axis=-1))
            scores_all.append(score_map[rows, cols])
        if not boxes_all:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        boxes = np.concatenate(boxes_all)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        return boxes, np.concatenate(scores_all)

    def _live_ball(self, display, frame, w, h):
        rknn = self._models.get("live_ball")
        if rknn is None:
            return display

        cx = cy = None
        ball_diameter = 0

        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        inp = canvas[None]  # (1,640,640,3), RGB uint8, NHWC

        output = rknn.inference(inputs=[inp], data_format=["nhwc"])
        if output:
            try:
                if len(output) == 6:
                    boxes, scores = self._decode_yolo_heads(output, scale, pad_x, pad_y, w, h)
                else:
                    pred = np.asarray(output[0]).squeeze().astype(np.float32)
                    if pred.ndim != 2:
                        pred = pred.reshape(pred.shape[0], -1)
                    if pred.shape[0] <= 16 and pred.shape[1] > pred.shape[0]:
                        pred = pred.T
                    if pred.shape[1] >= 5:
                        scores = pred[:,4] if pred.shape[1] == 5 else np.max(pred[:,4:], axis=1)
                        valid = scores >= .25
                        boxes = self._xywh2xyxy(pred[valid, :4])
                        scores = scores[valid]
                        boxes[:,[0,2]] = (boxes[:,[0,2]]-pad_x)/scale
                        boxes[:,[1,3]] = (boxes[:,[1,3]]-pad_y)/scale
                    else:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        scores = np.empty((0,), dtype=np.float32)
                if len(scores):
                    idx = self._nms(boxes, scores, 0.45)
                    if idx:
                        best = max(idx, key=lambda i: scores[i])
                        x1,y1,x2,y2 = boxes[best].astype(int)
                        x1,y1 = max(0,x1), max(0,y1)
                        x2,y2 = min(w,x2), min(h,y2)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        ball_diameter = max(2, min(x2 - x1, y2 - y1))
            except (ValueError, IndexError) as error:
                self._log("[AI] live ball output decode failed: {}".format(error))

        if cx is None:
            # YOLO missed (a far-baseline ball is sub-pixel in the 640px
            # letterbox): fall back to a full-resolution HSV search so the
            # ball stays visible.
            color_point = self._hsv_ball_candidate(frame)
            if color_point is not None:
                cx, cy = color_point
                ball_diameter = 8

        if cx is None:
            self._mark_live_ball_miss()
            return display

        self._live_ball_misses = 0
        self._trail.append((cx,cy))
        if len(self._trail) > 18:
            self._trail = self._trail[-18:]
        # Scale the marker from the detected ball diameter.  This makes a
        # near, large ball visibly larger without letting a noisy box fill the
        # entire preview.
        radius = max(7, min(24, int(round(ball_diameter * .65))))
        # Rendering only: use a short, fading green ribbon.  The trail still
        # uses the detector's raw centres and the marker still uses the raw
        # detection diameter, so no inference or post-processing is changed.
        points = self._trail
        visual_scale = max(0.85, min(1.65, radius / 10.0))
        for index in range(1, len(points)):
            fade = index / max(1, len(points) - 1)
            outer = max(2, int(round((1.5 + 3.0 * fade) * visual_scale)))
            inner = max(1, int(round((0.8 + 1.5 * fade) * visual_scale)))
            green = int(120 + 120 * fade)
            cv2.line(display, points[index - 1], points[index], TRAIL_DARK,
                     outer + 3, cv2.LINE_AA)
            cv2.line(display, points[index - 1], points[index],
                     (35, green, 75), outer, cv2.LINE_AA)
            cv2.line(display, points[index - 1], points[index], TRAIL_GREEN,
                     inner, cv2.LINE_AA)
        for age, point in enumerate(reversed(points[:-1])):
            fade = 1.0 - age / max(1, len(points) - 1)
            dot_radius = max(1, int(round(radius * (0.15 + 0.20 * fade))))
            cv2.circle(display, point, dot_radius, TRAIL_GREEN, -1, cv2.LINE_AA)
        center = (cx, cy)
        cv2.circle(display, center, radius + 6, MARKER_DARK, -1, cv2.LINE_AA)
        cv2.circle(display, center, radius + 3, TRAIL_DARK, -1, cv2.LINE_AA)
        cv2.circle(display, center, radius, NEON_GREEN, -1, cv2.LINE_AA)
        cv2.circle(display, center, max(2, radius // 3), (220, 255, 230), -1,
                   cv2.LINE_AA)
        return display

    @staticmethod
    def _hsv_ball_candidate(frame, min_area=8, max_area=40000):
        """Full-resolution HSV fallback for far balls YOLO cannot resolve."""
        height, width = frame.shape[:2]
        resize_scale = min(1.0, 1920.0 / max(width, height))
        if resize_scale < 1.0:
            work = cv2.resize(frame, (int(round(width * resize_scale)), int(round(height * resize_scale))), interpolation=cv2.INTER_AREA)
        else:
            work = frame
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (18, 75, 75), (58, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
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
            x, y, bw, bh = cv2.boundingRect(contour)
            ratio = min(bw, bh) / float(max(bw, bh))
            if ratio < 0.55:
                continue
            score = circularity * area
            if score > best_score:
                best_score = score
                best = (int((x + bw / 2.0) / resize_scale), int((y + bh / 2.0) / resize_scale))
        return best

    # ── Pose (NHWC BGR uint8 as per pose_live.py) ────────────

    def _pose(self, display, frame, w, h):
        runtime = self._models.get("pose")
        if runtime is None or self._pose_module is None:
            return display
        poses, npu_ms = runtime.infer(frame)
        # Compose the wide glow separately so it remains vivid on both dark
        # indoor scenes and bright court scenes without changing inference.
        glow = np.zeros_like(display)
        for pose in poses:
            # Rendering only: preserve the real-time pose result, then make a
            # screen-readable skeleton whose proportions follow person size.
            person_height = max(1.0, float(pose.box[3] - pose.box[1]))
            scale = max(0.80, min(2.20, person_height / 260.0))
            limb_glow = max(7, int(round(11.0 * scale)))
            limb_outline = max(5, int(round(7.0 * scale)))
            limb_core = max(3, int(round(4.0 * scale)))
            joint_radius = max(5, int(round(7.0 * scale)))
            for start, end in self._pose_module.SKELETON:
                a, b = pose.keypoints[start - 1], pose.keypoints[end - 1]
                if a[2] >= .35 and b[2] >= .35:
                    pa, pb = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
                    cv2.line(glow, pa, pb, (30, 210, 80), limb_glow,
                             cv2.LINE_AA)
                    cv2.line(display, pa, pb, (18, 55, 28), limb_outline,
                             cv2.LINE_AA)
                    cv2.line(display, pa, pb, (70, 255, 125), limb_core,
                             cv2.LINE_AA)
                    cv2.line(display, pa, pb, (200, 255, 215), 1,
                             cv2.LINE_AA)
            for index, point in enumerate(pose.keypoints):
                if point[2] >= .35:
                    center = (int(point[0]), int(point[1]))
                    cv2.circle(glow, center, joint_radius + 6, (0, 210, 255),
                               -1, cv2.LINE_AA)
                    cv2.circle(display, center, joint_radius + 3, (18, 55, 28),
                               -1, cv2.LINE_AA)
                    cv2.circle(display, center, joint_radius, (0, 225, 255),
                               -1, cv2.LINE_AA)
                    cv2.circle(display, center, max(2, joint_radius // 2),
                               (220, 255, 255), -1, cv2.LINE_AA)
        # The alpha is intentionally restrained: it reads as a glow rather
        # than a translucent mask over the player.
        display[:] = cv2.addWeighted(display, 1.0, glow, 0.28, 0.0)
        cv2.putText(display, "Pose {}  {:.0f} ms".format(len(poses), npu_ms), (8, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)
        return display
