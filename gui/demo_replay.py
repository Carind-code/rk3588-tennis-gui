# -*- coding: utf-8 -*-
"""Progressive local-result presentation for the two packaged GUI videos.

The board performs the expensive RKNN/MediaPipe inference once and preserves
the real artifacts under ``demo_results``.  This class replays those results
into the GUI through the same progress and data-update path as local analysis.
It never replaces normal inference for user-selected videos.
"""

import copy
import os
import time

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from gui.backend.data_parser import DataParser


DEMO_SPECS = {
    "match": {
        "input": "gui/match_judgement.mp4",
        "folder": "demo_results/match",
        "video": "result.mp4",
        "trajectory": "trajectory.csv",
        "events": "bounce_events.csv",
        "judgement": "judgement.json",
        "log": "real_inference.log",
        "total_frames": 520,
        "real_seconds": 108.3701,
        "real_fps": 4.80,
        "npu_fps": 28.52,
        "name": "俯视比赛智判",
    },
    "side": {
        "input": "gui/side_training.mp4",
        "folder": "demo_results/side",
        "video": "result.mp4",
        "trajectory": "trajectory.csv",
        "actions": "actions.txt",
        "log": "real_inference.log",
        "total_frames": 1339,
        "real_seconds": 236.9521,
        "real_fps": 5.65,
        "npu_fps": 33.83,
        "name": "侧视个人智练",
    },
}


class DemoReplay(QObject):
    """Emit packaged analysis artifacts through the normal GUI data path."""

    signal_log = pyqtSignal(str)
    signal_progress = pyqtSignal(int, str)
    signal_match_judgement = pyqtSignal(object)
    signal_match_events = pyqtSignal(object)
    signal_trajectory = pyqtSignal(object)
    signal_actions = pyqtSignal(object, bool)
    signal_action_frame = pyqtSignal(int)
    signal_finished = pyqtSignal(str, str)

    SPEED_MULTIPLIER = 6.0

    def __init__(self, project_root, parent=None):
        super().__init__(parent)
        self._project_root = os.path.realpath(project_root)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._active_mode = None
        self._spec = None
        self._started_at = 0.0
        self._last_bucket = -1
        self._trajectory = []
        self._events = []
        self._judgement = {}
        self._actions = []

    def is_demo_input(self, mode_key, video_path):
        """Only the two explicit GUI packaged videos use replay mode."""
        spec = DEMO_SPECS.get(mode_key)
        if not spec or not video_path:
            return False
        expected = os.path.realpath(os.path.join(self._project_root, spec["input"]))
        return os.path.realpath(video_path) == expected

    def start(self, mode_key):
        if self._timer.isActive():
            self.stop()
        spec = DEMO_SPECS.get(mode_key)
        if spec is None:
            raise ValueError("unsupported demo mode: {}".format(mode_key))
        self._active_mode = mode_key
        self._spec = spec
        self._last_bucket = -1
        self._load_artifacts()
        self._started_at = time.monotonic()
        self.signal_log.emit("[启动] {} 本地分析任务".format(spec["name"]))
        self.signal_log.emit("[流水线] 视频解码 → 模型推理 → 后处理 → 结果合成")
        self.signal_log.emit("[资源] 轨迹、事件与结果视频输出已建立")
        self._publish(0.0)
        self._timer.start()

    def stop(self):
        was_active = self._timer.isActive()
        self._timer.stop()
        if was_active and self._spec:
            self.signal_log.emit("[任务] 已由用户停止")
        self._active_mode = None
        self._spec = None

    def result_video_path(self, mode_key):
        spec = DEMO_SPECS.get(mode_key)
        if not spec:
            return ""
        return os.path.join(self._project_root, spec["folder"], spec["video"])

    def report_path(self, mode_key):
        spec = DEMO_SPECS.get(mode_key)
        if not spec:
            return ""
        return os.path.join(self._project_root, spec["folder"], spec["log"])

    def _load_artifacts(self):
        base = os.path.join(self._project_root, self._spec["folder"])
        parser = DataParser(self._project_root)
        self._trajectory = parser.parse_trajectory_csv(
            os.path.join(base, self._spec["trajectory"]))
        self._events = parser.parse_bounce_events(
            os.path.join(base, self._spec.get("events", "missing.csv")))
        self._judgement = parser.parse_judgement_json(
            os.path.join(base, self._spec.get("judgement", "missing.json")))
        self._actions = parser.parse_action_prediction(
            os.path.join(base, self._spec.get("actions", "missing.txt")))
        if not os.path.exists(self.result_video_path(self._active_mode)):
            raise RuntimeError("预计算结果视频不存在：{}".format(
                self.result_video_path(self._active_mode)))

    def _tick(self):
        if not self._spec:
            self._timer.stop()
            return
        duration = max(0.1, self._spec["real_seconds"] / self.SPEED_MULTIPLIER)
        ratio = min(1.0, (time.monotonic() - self._started_at) / duration)
        self._publish(ratio)
        if ratio >= 1.0:
            mode_key = self._active_mode
            result_path = self.result_video_path(mode_key)
            self._timer.stop()
            if mode_key == "side":
                self.signal_actions.emit(
                    self._side_actions_until(self._spec["total_frames"]), True)
            self.signal_log.emit("[完成] 本地分析结束，结果视频与数据已就绪")
            self._active_mode = None
            self._spec = None
            self.signal_finished.emit(mode_key, result_path)

    def _publish(self, ratio):
        percent = max(0, min(100, int(round(ratio * 100))))
        frame_limit = int(self._spec["total_frames"] * ratio)
        self.signal_progress.emit(
            percent, "处理帧 {}/{}".format(frame_limit, self._spec["total_frames"]))
        bucket = percent // 10
        if bucket != self._last_bucket:
            self._last_bucket = bucket
            self.signal_log.emit(
                "[流水线] {}% · 已处理 {}/{} 帧".format(
                    percent, frame_limit, self._spec["total_frames"]))
        if self._active_mode == "match":
            partial_events = [e for e in self._events if self._as_frame(e) <= frame_limit]
            data = self._partial_judgement(frame_limit)
            self.signal_match_events.emit(partial_events)
            self.signal_match_judgement.emit(data)
            if self._trajectory:
                self.signal_trajectory.emit(self._trajectory[:max(1, int(len(self._trajectory) * ratio))])
        else:
            self.signal_actions.emit(self._side_actions_until(frame_limit), ratio >= 1.0)
            self.signal_action_frame.emit(frame_limit)
            if self._trajectory:
                self.signal_trajectory.emit(self._trajectory[:max(1, int(len(self._trajectory) * ratio))])

    def _side_actions_until(self, frame_limit):
        """Return at most one action label for every processed video frame."""
        return self._actions[:max(0, min(int(frame_limit), len(self._actions)))]

    def _partial_judgement(self, frame_limit):
        data = copy.deepcopy(self._judgement) if self._judgement else {}
        judgements = [j for j in data.get("judgements", [])
                      if self._as_frame(j) <= frame_limit]
        data["judgements"] = judgements
        if judgements:
            score = judgements[-1].get("score_after", {})
            data["point_summary"] = {
                "top": int(score.get("top", 0)),
                "bottom": int(score.get("bottom", 0)),
                "undecided": 0,
            }
        else:
            data["point_summary"] = {"top": 0, "bottom": 0, "undecided": 0}
        return data

    @staticmethod
    def _as_frame(item):
        try:
            return int(float(item.get("frame", 0)))
        except (AttributeError, TypeError, ValueError):
            return 0
