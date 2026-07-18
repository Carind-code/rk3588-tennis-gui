# -*- coding: utf-8 -*-
"""
AI Overlay — RKNN models on camera frames, drawn in video widget.
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
from pathlib import Path
import sys

MODEL_DIR = "/home/elf/rk3588_tennis_system/models"
MODEL_SIZE = 640

MODELS = {
    "yolo": {
        # Six-output INT8 head: the realtime model supplied with detect.7z.
        "rknn": f"{MODEL_DIR}/detect/models/yolo_tennis_ball_head_i8.rknn",
    },
    "tracknet": {
        "rknn": f"{MODEL_DIR}/track/side_tracknet_360x640_sigmoid_fp.rknn",
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
        self._track_frames = []
        self._track_point = None
        self._track_trail = []
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
            return False
        if self.load(name):
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

        if "yolo" in self._active:
            result = self._yolo(result, frame, w, h)
        if "tracknet" in self._active:
            result = self._tracknet(result, frame, w, h)
        if "pose" in self._active:
            result = self._pose(result, frame, w, h)

        return result

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

    def _yolo(self, display, frame, w, h):
        rknn = self._models.get("yolo")
        if rknn is None:
            return display

        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        inp = canvas[None]  # (1,640,640,3), RGB uint8, NHWC

        output = rknn.inference(inputs=[inp], data_format=["nhwc"])
        if not output:
            return display

        try:
            if len(output) == 6:
                boxes, scores = self._decode_yolo_heads(output, scale, pad_x, pad_y, w, h)
            else:
                pred = np.asarray(output[0]).squeeze().astype(np.float32)
                if pred.ndim != 2:
                    pred = pred.reshape(pred.shape[0], -1)
                if pred.shape[0] <= 16 and pred.shape[1] > pred.shape[0]:
                    pred = pred.T
                if pred.shape[1] < 5:
                    return display
                scores = pred[:,4] if pred.shape[1] == 5 else np.max(pred[:,4:], axis=1)
                valid = scores >= .25
                boxes = self._xywh2xyxy(pred[valid, :4])
                scores = scores[valid]
                boxes[:,[0,2]] = (boxes[:,[0,2]]-pad_x)/scale
                boxes[:,[1,3]] = (boxes[:,[1,3]]-pad_y)/scale
        except (ValueError, IndexError) as error:
            self._log("[AI] YOLO output decode failed: {}".format(error))
            return display

        if not len(scores):
            return display

        idx = self._nms(boxes, scores, 0.45)
        if not idx:
            return display

        # Match the formal side-view postprocess visual: no detection box or
        # label; only a red current point, white halo, and fading yellow dots.
        best = max(idx, key=lambda i: scores[i])
        x1,y1,x2,y2 = boxes[best].astype(int)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        cx, cy = (x1+x2)//2, (y1+y2)//2

        self._trail.append((cx,cy))
        if len(self._trail) > 25:
            self._trail = self._trail[-25:]
        # Scale the marker from the detected ball diameter.  This makes a
        # near, large ball visibly larger without letting a noisy box fill the
        # entire preview.
        ball_diameter = max(2, min(x2 - x1, y2 - y1))
        radius = max(7, min(24, int(round(ball_diameter * .65))))
        for age, point in enumerate(reversed(self._trail)):
            dot_radius = max(1, radius - age // 4)
            color = (0, 0, 255) if age == 0 else (0, 220, 255)
            cv2.circle(display, point, dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(display, (cx, cy), max(5, radius + 2), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(display, (cx, cy), max(8, radius + 5), (255, 255, 255), 1, cv2.LINE_AA)
        return display

    # ── TrackNet (HSV fallback) ───────────────────────────────

    def _tracknet(self, display, frame, w, h):
        rknn = self._models.get("tracknet")
        if rknn is None:
            return display
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        self._track_frames.insert(0, small)
        self._track_frames = self._track_frames[:3]
        if len(self._track_frames) < 3:
            cv2.putText(display, "TrackNet: warming up", (8, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            return display
        current, previous, preprevious = self._track_frames
        model_input = np.concatenate((current, previous, preprevious), axis=2)[None].astype(np.uint8)
        output = rknn.inference(inputs=[model_input], data_format=["nhwc"])
        if not output:
            return display
        heatmap = np.asarray(output[0]).squeeze().astype(np.float32).reshape(360, 640)
        if float(heatmap.min()) < 0.0 or float(heatmap.max()) > 1.0:
            heatmap = 1.0 / (1.0 + np.exp(-np.clip(heatmap, -60.0, 60.0)))
        _, score, _, location = cv2.minMaxLoc(heatmap)
        point = None
        if score >= .18:
            candidate = (location[0] * w / 640.0, location[1] * h / 360.0)
            if self._track_point is None or np.hypot(candidate[0] - self._track_point[0], candidate[1] - self._track_point[1]) <= max(120.0, w * .30):
                point = candidate
                self._track_point = point
        if point is not None:
            center = (int(point[0]), int(point[1]))
            self._track_trail.append(center)
            self._track_trail = self._track_trail[-25:]
            for index in range(1, len(self._track_trail)):
                cv2.line(display, self._track_trail[index - 1], self._track_trail[index], (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display, center, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(display, center, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, "TrackNet {:.0f}%".format(score * 100), (8, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        return display

    # ── Pose (NHWC BGR uint8 as per pose_live.py) ────────────

    def _pose(self, display, frame, w, h):
        runtime = self._models.get("pose")
        if runtime is None or self._pose_module is None:
            return display
        poses, npu_ms = runtime.infer(frame)
        for pose in poses:
            x1, y1, x2, y2, confidence = pose.box.astype(int)
            for start, end in self._pose_module.SKELETON:
                a, b = pose.keypoints[start - 1], pose.keypoints[end - 1]
                if a[2] >= .35 and b[2] >= .35:
                    cv2.line(display, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), 1, cv2.LINE_AA)
            for index, point in enumerate(pose.keypoints):
                if point[2] >= .35:
                    cv2.circle(display, (int(point[0]), int(point[1])), 2,
                               (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(display, "Pose {}  {:.0f} ms".format(len(poses), npu_ms), (8, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 2, cv2.LINE_AA)
        return display
