#!/usr/bin/env python3
import argparse
import csv
import datetime
import json
import math
import os
import platform
import queue
import socket
import statistics
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from court_runtime import calibrate_court, projected_court_lines, transform_point
from event_runtime import analyze_events, append_court_columns
from heatmap_candidates import topk_heatmap_candidates
from track_video import (
    interpolate_guarded,
    remove_short_fragments,
    remove_temporal_spikes,
    stabilize_v37,
)


INPUT_WIDTH = 640
INPUT_HEIGHT = 360
INPUT_FORMAT = "nhwc"
DEFAULT_IPC_SOCKET = "/tmp/tennis_ipc.sock"
DEVICE_COMPATIBLE_NODE = "/proc/device-tree/compatible"


def now_ms(start):
    return (time.perf_counter() - start) * 1000.0


def make_task_id(device_id):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return "{}_{}".format(device_id, stamp)


def get_host():
    if platform.system() + "-" + platform.machine() != "Linux-aarch64":
        return platform.system() + "-" + platform.machine()
    try:
        with open(DEVICE_COMPATIBLE_NODE, "r", encoding="latin-1") as f:
            compatible = f.read().lower()
    except OSError:
        return "Linux-aarch64"
    if "rk3588" in compatible:
        return "RK3588"
    if "rk3576" in compatible:
        return "RK3576"
    return "Linux-aarch64"


def core_mask_from_arg(core):
    mapping = {
        "auto": "NPU_CORE_AUTO",
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "01": "NPU_CORE_0_1",
        "012": "NPU_CORE_0_1_2",
        "all": "NPU_CORE_ALL",
    }
    attr = mapping.get(core.lower())
    if attr is None:
        raise ValueError("unsupported core mask: {}".format(core))
    return getattr(RKNNLite, attr, None)


def parse_cores(cores, workers):
    parsed = [x.strip() for x in cores.split(",") if x.strip()]
    if not parsed:
        parsed = ["0", "1", "2"]
    while len(parsed) < workers:
        parsed.extend(parsed)
    return parsed[:workers]


def init_rknn(model_path, core):
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError("load_rknn failed: {}".format(ret))

    host = get_host()
    mask = core_mask_from_arg(core)
    if host in ("RK3588", "RK3576") and mask is not None:
        ret = rknn.init_runtime(core_mask=mask)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("init_runtime failed: {}".format(ret))
    return rknn, host


def resize_bgr(frame):
    return cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)


def make_input(frame, prev, preprev):
    stacked = np.concatenate((resize_bgr(frame), resize_bgr(prev), resize_bgr(preprev)), axis=2)
    return stacked[None, :, :, :].astype(np.uint8)


def make_input_from_resized(frame, prev, preprev):
    stacked = np.concatenate((frame, prev, preprev), axis=2)
    return stacked[None, :, :, :].astype(np.uint8)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess_heatmap(output, scale_xy, threshold, peak_window):
    heat = np.asarray(output).squeeze().astype(np.float32)
    if heat.shape != (INPUT_HEIGHT, INPUT_WIDTH):
        heat = heat.reshape((INPUT_HEIGHT, INPUT_WIDTH))
    if float(np.nanmin(heat)) < 0.0 or float(np.nanmax(heat)) > 1.0:
        heat = sigmoid(heat)

    _, max_score, _, max_loc = cv2.minMaxLoc(heat)
    max_score = float(max_score)
    if max_score < threshold:
        return None, None, max_score

    peak_window = max(3, int(peak_window))
    if peak_window % 2 == 0:
        peak_window += 1
    radius = peak_window // 2
    peak_x, peak_y = max_loc
    x0 = max(0, peak_x - radius)
    x1 = min(heat.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heat.shape[0], peak_y + radius + 1)

    crop = heat[y0:y1, x0:x1]
    weights = np.maximum(crop - threshold, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(peak_x) * scale_xy[0], float(peak_y) * scale_xy[1], max_score

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    x_model = float(np.sum(weights * xs[None, :]) / weight_sum)
    y_model = float(np.sum(weights * ys[:, None]) / weight_sum)
    return x_model * scale_xy[0], y_model * scale_xy[1], max_score


def postprocess_candidates(output, scale_xy, threshold, peak_window, top_k=5, suppress_window=15):
    heat = np.asarray(output).squeeze().astype(np.float32)
    if heat.shape != (INPUT_HEIGHT, INPUT_WIDTH):
        heat = heat.reshape((INPUT_HEIGHT, INPUT_WIDTH))
    if float(np.nanmin(heat)) < 0.0 or float(np.nanmax(heat)) > 1.0:
        heat = sigmoid(heat)
    args = SimpleNamespace(
        threshold=float(threshold),
        peak_window=int(peak_window),
        top_k=int(top_k),
        suppress_window=int(suppress_window),
        enable_hough_quality=False,
    )
    return topk_heatmap_candidates(heat, args, scale_xy[0], scale_xy[1])


def infer_worker(worker_id, core, model_path, jobs, results, ready, threshold, peak_window, scale_xy, mode):
    rknn = None
    try:
        rknn, host = init_rknn(model_path, core)
        ready.put((worker_id, core, host))
        while True:
            job = jobs.get()
            if job is None:
                return

            frame_idx = job["frame"]
            t_total = time.perf_counter()

            if mode == "frames":
                t0 = time.perf_counter()
                frames = job["frames"]
                inp = make_input(frames[frame_idx], frames[frame_idx - 1], frames[frame_idx - 2])
                prep_ms = now_ms(t0)
            else:
                inp = job["input"]
                prep_ms = float(job["prep_ms"])

            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[inp], data_format=[INPUT_FORMAT])
            npu_ms = now_ms(t0)
            if outputs is None:
                raise RuntimeError("rknn inference returned None")

            t0 = time.perf_counter()
            candidates = postprocess_candidates(outputs[0], scale_xy, threshold, peak_window)
            primary = candidates[0] if candidates else {"x": None, "y": None, "score": 0.0}
            x, y, score = primary["x"], primary["y"], primary["score"]
            post_ms = now_ms(t0)

            results.put(
                {
                    "frame": frame_idx,
                    "x": x,
                    "y": y,
                    "score": score,
                    "candidates": candidates,
                    "worker": worker_id,
                    "core": core,
                    "prep_ms": prep_ms,
                    "npu_ms": npu_ms,
                    "post_ms": post_ms,
                    "total_ms": now_ms(t_total),
                    "error": "",
                }
            )
    except Exception as exc:
        results.put({"frame": -1, "error": "{}".format(exc), "worker": worker_id, "core": core})
    finally:
        if rknn is not None:
            rknn.release()


def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("failed to open video: {}".format(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded from: {}".format(path))
    return frames, fps


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("failed to open video: {}".format(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, frames


def collect_results(results, expected, progress, start_time=None):
    frame_results = {}
    t0 = start_time if start_time is not None else time.perf_counter()
    while len(frame_results) < expected:
        item = results.get()
        if item.get("error"):
            raise RuntimeError("worker {} core {} failed: {}".format(item.get("worker"), item.get("core"), item["error"]))
        frame_results[item["frame"]] = item
        done = len(frame_results)
        if progress and (done % 20 == 0 or done == expected):
            fps = done / max(time.perf_counter() - t0, 1e-6)
            print(
                "进度：{}/{} ({:.2f}%)，当前推理吞吐：{:.2f} FPS".format(
                    done, expected, done * 100.0 / max(expected, 1), fps
                ),
                flush=True,
            )
    return frame_results, now_ms(t0)


def infer_fast_frames(args, frame_w, frame_h):
    decode_t0 = time.perf_counter()
    frames, fps = read_video(args.video)
    decode_ms = now_ms(decode_t0)
    expected = max(0, len(frames) - 2)
    scale_xy = (frame_w / float(INPUT_WIDTH), frame_h / float(INPUT_HEIGHT))

    jobs = queue.Queue(maxsize=args.workers * 2)
    results = queue.Queue()
    ready = queue.Queue()
    cores = parse_cores(args.cores, args.workers)
    threads = []
    for worker_id, core in enumerate(cores):
        thread = threading.Thread(
            target=infer_worker,
            args=(worker_id, core, args.model, jobs, results, ready, args.threshold, args.peak_window, scale_xy, "frames"),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for _ in cores:
        worker_id, core, host = ready.get()
        if not args.no_progress:
            print("WORKER_READY worker={} core={} host={}".format(worker_id, core, host), flush=True)

    infer_t0 = time.perf_counter()
    for frame_idx in range(2, len(frames)):
        jobs.put({"frame": frame_idx, "frames": frames})
    for _ in threads:
        jobs.put(None)

    frame_results, _ = collect_results(results, expected, not args.no_progress, infer_t0)
    infer_wall_ms = now_ms(infer_t0)
    for thread in threads:
        thread.join()
    return frames, fps, frame_results, decode_ms, infer_wall_ms, cores


def infer_streaming(args, frame_w, frame_h, fps, total_frames):
    expected_hint = max(0, total_frames - 2)
    scale_xy = (frame_w / float(INPUT_WIDTH), frame_h / float(INPUT_HEIGHT))

    jobs = queue.Queue(maxsize=args.workers * 4)
    results = queue.Queue()
    ready = queue.Queue()
    cores = parse_cores(args.cores, args.workers)
    threads = []
    for worker_id, core in enumerate(cores):
        thread = threading.Thread(
            target=infer_worker,
            args=(worker_id, core, args.model, jobs, results, ready, args.threshold, args.peak_window, scale_xy, "input"),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for _ in cores:
        worker_id, core, host = ready.get()
        if not args.no_progress:
            print("WORKER_READY worker={} core={} host={}".format(worker_id, core, host), flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("failed to open video: {}".format(args.video))

    submitted_count = [0]
    decoded_count = [0]
    producer_done = threading.Event()

    def producer():
        prev2 = None
        prev1 = None
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                small = resize_bgr(frame)
                if prev2 is not None and prev1 is not None:
                    inp = make_input_from_resized(small, prev1, prev2)
                    jobs.put({"frame": idx, "input": inp, "prep_ms": now_ms(t0)})
                    submitted_count[0] += 1
                prev2 = prev1
                prev1 = small
                idx += 1
        finally:
            decoded_count[0] = idx
            cap.release()
            for _ in threads:
                jobs.put(None)
            producer_done.set()

    infer_t0 = time.perf_counter()
    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()
    frame_results = {}
    last_reported = 0
    while True:
        try:
            item = results.get(timeout=0.1)
        except queue.Empty:
            if producer_done.is_set() and len(frame_results) >= submitted_count[0]:
                break
            continue
        if item.get("error"):
            raise RuntimeError("worker {} core {} failed: {}".format(item.get("worker"), item.get("core"), item["error"]))
        frame_results[item["frame"]] = item
        done = len(frame_results)
        expected = submitted_count[0] if producer_done.is_set() else expected_hint
        if not args.no_progress and done % 20 == 0:
            infer_fps = done / max(time.perf_counter() - infer_t0, 1e-6)
            print(
                "进度：{}/{} ({:.2f}%)，当前推理吞吐：{:.2f} FPS".format(
                    done, max(expected, done), done * 100.0 / max(expected, done, 1), infer_fps
                ),
                flush=True,
            )
            last_reported = done
    prod_thread.join()
    for thread in threads:
        thread.join()
    if not args.no_progress and len(frame_results) != last_reported:
        infer_fps = len(frame_results) / max(time.perf_counter() - infer_t0, 1e-6)
        print(
            "进度：{}/{} (100.00%)，当前推理吞吐：{:.2f} FPS".format(
                len(frame_results), len(frame_results), infer_fps
            ),
            flush=True,
        )
    return fps, decoded_count[0], frame_results, now_ms(infer_t0), cores


def make_tracks(frame_count, frame_results, frame_w, frame_h, fps, max_dist):
    candidate_rows = [frame_results.get(idx, {}).get("candidates", []) for idx in range(frame_count)]
    args = SimpleNamespace(
        fps_adaptive=True,
        gate_min=60.0,
        gate_max=float(max_dist),
        speed_window=5,
        speed_gate_factor=2.2,
        gate_margin=25.0,
        strong_gate_factor=1.5,
        weak_gate_factor=0.85,
        low_threshold=0.35,
        high_threshold=0.88,
        seed_threshold=0.96,
        min_weak_shape=0.05,
        shape_weight=0.05,
        hough_weight=0.0,
        motion_weight=0.20,
        velocity_alpha=0.65,
        weak_velocity_alpha=0.35,
        turn_reset_angle=70.0,
        extreme_turn_angle=120.0,
        max_accel_ratio=3.0,
        min_speed_for_accel_check=5.0,
        seed_max_gap=3,
        weak_reconnect_gap=4,
        max_reconnect_gap=8,
        interp_gap=3,
        min_fragment_length=3,
        isolated_static_max_length=3,
        isolated_static_radius=8.0,
        static_fragment_max_length=8,
        static_radius=5.0,
        spike_return_radius=35.0,
        spike_distance=80.0,
    )
    stable = stabilize_v37(candidate_rows, frame_w, frame_h, fps, args)
    stable = remove_temporal_spikes(stable, args)
    stable = interpolate_guarded(stable, args.interp_gap, args.gate_max)
    stable = remove_short_fragments(stable, args)
    return stable["tracks"], stable["scores"], stable


def recompute_dists(tracks):
    dists = [-1.0] * len(tracks)
    last = None
    for idx, point in enumerate(tracks):
        if point is None or point[0] is None:
            continue
        if last is not None:
            dists[idx] = math.hypot(point[0] - last[0], point[1] - last[1])
        last = point
    return dists


def write_csv(path, tracks, scores, dists, frame_results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "x", "y", "raw_x", "raw_y", "score", "dist", "worker", "core", "prep_ms", "npu_ms", "post_ms", "total_ms"]
        )
        for idx, point in enumerate(tracks):
            item = frame_results.get(idx, {})
            raw_x = item.get("x")
            raw_y = item.get("y")
            writer.writerow(
                [
                    idx,
                    "" if (point is None or point[0] is None) else "{:.3f}".format(point[0]),
                    "" if (point is None or point[1] is None) else "{:.3f}".format(point[1]),
                    "" if raw_x is None else "{:.3f}".format(raw_x),
                    "" if raw_y is None else "{:.3f}".format(raw_y),
                    "" if scores[idx] is None else "{:.6f}".format(scores[idx]),
                    "{:.3f}".format(dists[idx]),
                    item.get("worker", ""),
                    item.get("core", ""),
                    fmt_ms(item.get("prep_ms")),
                    fmt_ms(item.get("npu_ms")),
                    fmt_ms(item.get("post_ms")),
                    fmt_ms(item.get("total_ms")),
                ]
            )


def fmt_ms(value):
    return "" if value is None else "{:.3f}".format(float(value))


def valid_point(point):
    return point is not None and point[0] is not None and point[1] is not None


def draw_court_lines(frame, frame_id, court_lines, court_start_frame):
    if not court_lines or frame_id < court_start_frame:
        return frame
    for start, end in court_lines:
        cv2.line(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_reference_main(frame, trace, events, radius, trace_min_radius=1, trace_decay=5):
    points = [pt for pt in trace if valid_point(pt)]
    for age, point in enumerate(reversed(points)):
        dot_radius = max(trace_min_radius, radius - age // max(1, trace_decay))
        color = (0, 0, 255) if age == 0 else (0, 220, 255)
        cv2.circle(frame, (int(point[0]), int(point[1])), dot_radius, color, -1, cv2.LINE_AA)
    if points:
        point = points[-1]
        cv2.circle(frame, (int(point[0]), int(point[1])), max(5, radius + 2), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(point[0]), int(point[1])), max(8, radius + 5), (255, 255, 255), 1, cv2.LINE_AA)
    for event in events:
        color = (0, 80, 255) if getattr(event, "event_type", "") == "hit" else (255, 255, 0)
        label = "H" if getattr(event, "event_type", "") == "hit" else "B"
        center = (int(round(event.x)), int(round(event.y)))
        cv2.circle(frame, center, 14, color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "{}{}".format(label, event.frame),
            (center[0] + 12, max(18, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def scale_mini_point(point, origin_x, origin_y, court_w, court_h):
    x = origin_x + int(max(0.0, min(1000.0, point[0])) / 1000.0 * court_w)
    y = origin_y + int(max(0.0, min(2168.0, point[1])) / 2168.0 * court_h)
    return x, y


def draw_mini_court(frame, court_trace, events, court_matrix):
    if court_matrix is None:
        return frame
    height, width = frame.shape[:2]
    panel_h = max(180, int(height * 0.32))
    panel_w = int(panel_h * 1000 / 2168)
    x0 = max(20, width - panel_w - 28)
    y0 = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 10, y0 - 10), (x0 + panel_w + 10, y0 + panel_h + 10), (45, 40, 35), -1)
    cv2.addWeighted(overlay, 0.36, frame, 0.64, 0, frame)
    cv2.rectangle(frame, (x0 - 10, y0 - 10), (x0 + panel_w + 10, y0 + panel_h + 10), (220, 220, 220), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (188, 145, 103), -1)
    alley = int(panel_w * (1.37 / 10.97))
    service = int(panel_h * (5.48 / 23.77))
    mid_x = x0 + panel_w // 2
    mid_y = y0 + panel_h // 2
    line = (245, 245, 245)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), line, 2)
    cv2.line(frame, (x0, mid_y), (x0 + panel_w, mid_y), line, 1)
    cv2.rectangle(frame, (x0 + alley, y0), (x0 + panel_w - alley, y0 + panel_h), line, 1)
    cv2.line(frame, (x0 + alley, y0 + service), (x0 + panel_w - alley, y0 + service), line, 1)
    cv2.line(frame, (x0 + alley, y0 + panel_h - service), (x0 + panel_w - alley, y0 + panel_h - service), line, 1)
    cv2.line(frame, (mid_x, y0 + service), (mid_x, y0 + panel_h - service), line, 1)

    points = [pt for pt in court_trace if valid_point(pt)]
    for age, point in enumerate(reversed(points)):
        dot_radius = 4 if age == 0 else max(1, 3 - age // 5)
        color = (0, 0, 255) if age == 0 else (0, 220, 255)
        cv2.circle(frame, scale_mini_point(point, x0, y0, panel_w, panel_h), dot_radius, color, -1, cv2.LINE_AA)

    for event in events:
        court_event = transform_point(court_matrix, (event.x, event.y))
        if court_event is None:
            continue
        color = (0, 80, 255) if getattr(event, "event_type", "") == "hit" else (255, 255, 0)
        cv2.circle(frame, scale_mini_point(court_event, x0, y0, panel_w, panel_h), 7, color, 2, cv2.LINE_AA)
    return frame


def update_score_state(score_state, latest_judgement, frame_judgements):
    for item in frame_judgements:
        latest_judgement = item
        score_after = item.get("score_after") or {}
        score_state["top"] = int(score_after.get("top", score_state["top"]))
        score_state["bottom"] = int(score_after.get("bottom", score_state["bottom"]))
    return latest_judgement


def draw_scoreboard(frame, score_state, latest_judgement):
    height, width = frame.shape[:2]
    scale = max(0.72, min(1.25, min(width / 1920.0, height / 1080.0)))
    panel_w = max(270, int(390 * scale))
    panel_h = max(120, int(170 * scale))
    x0, y0 = max(12, int(22 * scale)), max(12, int(18 * scale))
    x1, y1 = min(width - 12, x0 + panel_w), min(height - 12, y0 + panel_h)
    panel_w, panel_h = x1 - x0, y1 - y0

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (232, 190, 72), max(1, int(2 * scale)), cv2.LINE_AA)
    header_h = max(30, int(42 * scale))
    cv2.rectangle(frame, (x0, y0), (x1, y0 + header_h), (44, 67, 86), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "MATCH SCORE", (x0 + int(15 * scale), y0 + int(29 * scale)), font, 0.68 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)

    winner = latest_judgement.get("winner") if latest_judgement else None
    rows = (("TOP PLAYER", "top"), ("BOTTOM PLAYER", "bottom"))
    row_start = y0 + header_h + int(30 * scale)
    row_gap = max(28, int(37 * scale))
    for row_index, (label, side) in enumerate(rows):
        baseline = row_start + row_index * row_gap
        color = (90, 230, 150) if winner == side else (235, 235, 235)
        cv2.putText(frame, label, (x0 + int(15 * scale), baseline), font, 0.58 * scale, color, max(1, int(2 * scale)), cv2.LINE_AA)
        score_text = str(int(score_state.get(side, 0)))
        score_size = cv2.getTextSize(score_text, font, 0.78 * scale, max(1, int(2 * scale)))[0]
        cv2.putText(frame, score_text, (x1 - score_size[0] - int(18 * scale), baseline), font, 0.78 * scale, (232, 190, 72), max(1, int(2 * scale)), cv2.LINE_AA)

    if latest_judgement:
        status = str(latest_judgement.get("status", ""))
        status_y = min(y1 - int(9 * scale), row_start + 2 * row_gap)
        cv2.putText(frame, status, (x0 + int(15 * scale), status_y), font, 0.38 * scale, (185, 205, 220), 1, cv2.LINE_AA)
    return frame


def write_video_from_frames(
    frames, tracks, path, fps, radius, fourcc, court_lines, court_start_frame,
    court_matrix, draw_projected_court, events, judgements,
):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter: {}".format(path))
    events_by_frame = {}
    judgements_by_frame = {}
    for event in events:
        events_by_frame.setdefault(event.frame, []).append(event)
    for item in judgements:
        judgements_by_frame.setdefault(item["frame"], []).append(item)
    trace = deque(maxlen=14)
    court_trace = deque(maxlen=14)
    score_state = {"top": 0, "bottom": 0}
    latest_judgement = None
    for idx, frame in enumerate(frames):
        point = tracks[idx] if idx < len(tracks) else (None, None)
        trace.append(point if valid_point(point) else None)
        court_trace.append(transform_point(court_matrix, point) if valid_point(point) else None)
        output = frame.copy()
        if draw_projected_court:
            draw_court_lines(output, idx, court_lines, court_start_frame)
        draw_reference_main(output, trace, events_by_frame.get(idx, []), radius)
        draw_mini_court(output, court_trace, events_by_frame.get(idx, []), court_matrix)
        latest_judgement = update_score_state(
            score_state, latest_judgement, judgements_by_frame.get(idx, [])
        )
        draw_scoreboard(output, score_state, latest_judgement)
        writer.write(output)
    writer.release()


def write_video_streaming(
    video_path, tracks, out_path, fps, radius, fourcc, court_lines, court_start_frame,
    court_matrix, draw_projected_court, events, judgements,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("failed to open video: {}".format(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("failed to open VideoWriter: {}".format(out_path))
    events_by_frame = {}
    judgements_by_frame = {}
    for event in events:
        events_by_frame.setdefault(event.frame, []).append(event)
    for item in judgements:
        judgements_by_frame.setdefault(item["frame"], []).append(item)
    trace = deque(maxlen=14)
    court_trace = deque(maxlen=14)
    score_state = {"top": 0, "bottom": 0}
    latest_judgement = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        point = tracks[idx] if idx < len(tracks) else (None, None)
        trace.append(point if valid_point(point) else None)
        court_trace.append(transform_point(court_matrix, point) if valid_point(point) else None)
        if draw_projected_court:
            draw_court_lines(frame, idx, court_lines, court_start_frame)
        draw_reference_main(frame, trace, events_by_frame.get(idx, []), radius)
        draw_mini_court(frame, court_trace, events_by_frame.get(idx, []), court_matrix)
        latest_judgement = update_score_state(
            score_state, latest_judgement, judgements_by_frame.get(idx, [])
        )
        draw_scoreboard(frame, score_state, latest_judgement)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def write_tracking_only_video(video_path, frames, tracks, out_path, fps, radius, fourcc):
    if frames is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("failed to open video: {}".format(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        cap = None
        height, width = frames[0].shape[:2]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        if cap is not None:
            cap.release()
        raise RuntimeError("failed to open VideoWriter: {}".format(out_path))

    trace = deque(maxlen=14)
    idx = 0
    while idx < len(tracks):
        if frames is None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = frames[idx].copy()
        point = tracks[idx]
        trace.append(point if valid_point(point) else None)
        draw_reference_main(frame, trace, [], radius)
        writer.write(frame)
        idx += 1
    if cap is not None:
        cap.release()
    writer.release()


def timing_summary(frame_results):
    values = list(frame_results.values())
    prep = [float(x["prep_ms"]) for x in values]
    npu = [float(x["npu_ms"]) for x in values]
    post = [float(x["post_ms"]) for x in values]
    return {
        "prep_avg": sum(prep) / len(prep) if prep else 0.0,
        "npu_avg": sum(npu) / len(npu) if npu else 0.0,
        "post_avg": sum(post) / len(post) if post else 0.0,
        "npu_p50": statistics.median(npu) if npu else 0.0,
        "npu_p90": sorted(npu)[int(round((len(npu) - 1) * 0.9))] if npu else 0.0,
        "npu_max": max(npu) if npu else 0.0,
        "prep_total": sum(prep),
        "npu_total": sum(npu),
        "post_total": sum(post),
    }


def send_ipc(socket_path, payload):
    if not socket_path or not os.path.exists(socket_path):
        return False
    try:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            client.connect(socket_path)
            client.sendall(data)
        return True
    except OSError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNet V3 RKNNLite2 inference.")
    parser.add_argument("--model", default="./model_v37_360x640_b1_sigmoid_fp.rknn")
    parser.add_argument("--court-model", default="./court_detector_fp.rknn")
    parser.add_argument("--video", default="./input.mp4")
    parser.add_argument("--out-video", default="./tracknet_rknn_output.mp4")
    parser.add_argument("--out-csv", default="./tracknet_rknn_output.csv")
    parser.add_argument("--event-csv", default="./bounce_events.csv")
    parser.add_argument("--judgement-json", default="./judgement.json")
    parser.add_argument("--player-crops", default="./broadcast_court_only.json")
    parser.add_argument("--court-mode", choices=("singles", "doubles"), default="singles")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true", help="disable V5.2 MediaPipe player context")
    parser.add_argument("--mediapipe-stride", type=int, default=1, help="V5.2 player-context sampling stride")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="low-memory mode for 4K/long videos")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", default="0,1,2")
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--peak-window", type=int, default=9)
    parser.add_argument("--max-dist", type=float, default=220.0)
    parser.add_argument("--dot-radius", type=int, default=5)
    parser.add_argument("--draw-court-lines", dest="draw_court_lines", action="store_true", default=True)
    parser.add_argument("--no-draw-court-lines", dest="draw_court_lines", action="store_false")
    parser.add_argument("--tracking-only", action="store_true", help="skip court/event/judgement and output only the ball trace")
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--device-id", default="RK3588_01")
    parser.add_argument("--ipc-socket", default=DEFAULT_IPC_SOCKET)
    parser.add_argument("--no-ipc", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)
    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    total_t0 = time.perf_counter()
    frame_w, frame_h, fps_hint, total_hint = video_info(args.video)

    court_calibration = (
        {"available": False, "elapsed_ms": 0.0, "scan_frames": 0}
        if args.tracking_only
        else calibrate_court(args.video, args.court_model)
    )
    if court_calibration.get("available"):
        print(
            "COURT_READY frame={} points={}/14 error={:.2f}px scans={}".format(
                court_calibration["frame"],
                court_calibration["valid_points"],
                court_calibration["error_px"],
                court_calibration["scan_frames"],
            ),
            flush=True,
        )
    else:
        best_error = court_calibration.get("best_error_px")
        print(
            "COURT_UNAVAILABLE scans={} best_points={} best_error={}".format(
                court_calibration.get("scan_frames", 0),
                court_calibration.get("best_valid_points", 0),
                "N/A" if best_error is None else "{:.2f}px".format(best_error),
            ),
            flush=True,
        )

    if args.streaming:
        frames = None
        fps, frame_count, frame_results, infer_wall_ms, cores = infer_streaming(
            args, frame_w, frame_h, fps_hint, total_hint
        )
        decode_ms = 0.0
    else:
        frames, fps, frame_results, decode_ms, infer_wall_ms, cores = infer_fast_frames(args, frame_w, frame_h)
        frame_count = len(frames)

    raw_tracks = [(None, None)] * frame_count
    for idx, item in frame_results.items():
        raw_tracks[idx] = (item["x"], item["y"])
    raw_valid = sum(1 for t in raw_tracks if t is not None and t[0] is not None)

    tracks, scores, tracking_stable = make_tracks(
        frame_count, frame_results, frame_w, frame_h, fps, args.max_dist
    )
    dists = recompute_dists(tracks)
    final_valid = sum(1 for t in tracks if t is not None and t[0] is not None)

    write_csv(args.out_csv, tracks, scores, dists, frame_results)

    events = []
    event_payload = {"judgements": [], "bounce_count": 0, "elapsed_ms": 0.0}
    if not args.no_events and not args.tracking_only:
        events, event_payload, stable = analyze_events(
            args.video,
            tracks,
            scores,
            fps,
            frame_w,
            frame_h,
            court_calibration,
            args.player_crops,
            args.event_csv,
            args.judgement_json,
            court_mode=args.court_mode,
            use_mediapipe=not args.no_mediapipe,
            mediapipe_stride=args.mediapipe_stride,
            tracking_stable=tracking_stable,
        )
        tracks = [(None, None) if point is None else point for point in stable["tracks"]]
        final_valid = sum(1 for point in tracks if point[0] is not None)
    court_matrix = court_calibration.get("image_to_court") if court_calibration.get("available") else None
    if not args.tracking_only:
        append_court_columns(args.out_csv, tracks, court_matrix)
    court_lines = projected_court_lines(
        court_calibration.get("reference_to_image") if court_calibration.get("available") else None
    )
    court_start_frame = int(court_calibration.get("frame", 0)) if court_calibration.get("available") else 0

    write_ms = 0.0
    if not args.no_video:
        t0 = time.perf_counter()
        if args.tracking_only:
            write_tracking_only_video(
                args.video, frames, tracks, args.out_video, fps, args.dot_radius, args.fourcc
            )
        elif args.streaming:
            write_video_streaming(
                args.video, tracks, args.out_video, fps, args.dot_radius, args.fourcc,
                court_lines, court_start_frame, court_matrix, args.draw_court_lines,
                events, event_payload["judgements"],
            )
        else:
            write_video_from_frames(
                frames, tracks, args.out_video, fps, args.dot_radius, args.fourcc,
                court_lines, court_start_frame, court_matrix, args.draw_court_lines,
                events, event_payload["judgements"],
            )
        write_ms = now_ms(t0)

    total_ms = now_ms(total_t0)
    timing = timing_summary(frame_results)
    infer_frames = len(frame_results)
    infer_fps = infer_frames / (infer_wall_ms / 1000.0) if infer_wall_ms > 0 else 0.0
    end_fps = frame_count / (total_ms / 1000.0) if total_ms > 0 else 0.0
    task_id = args.task_id or make_task_id(args.device_id)

    if not args.no_ipc:
        send_ipc(
            args.ipc_socket,
            {
                "type": "video_task",
                "task_id": task_id,
                "video_path": "" if args.no_video else os.path.abspath(args.out_video),
                "csv_path": os.path.abspath(args.out_csv),
                "event_csv_path": "" if (args.no_events or args.tracking_only) else os.path.abspath(args.event_csv),
                "judgement_json_path": "" if (args.no_events or args.tracking_only) else os.path.abspath(args.judgement_json),
                "frames": frame_count,
                "valid_points": final_valid,
                "fps": end_fps,
                "bounce_count": len(events),
                "judgement_count": len(event_payload["judgements"]),
                "score": event_payload.get("point_summary", {"top": 0, "bottom": 0, "undecided": 0}),
                "timestamp": time.time(),
            },
        )

    prep_fps = 1000.0 / timing["prep_avg"] if timing["prep_avg"] > 0 else 0.0
    npu_single_fps = 1000.0 / timing["npu_avg"] if timing["npu_avg"] > 0 else 0.0
    post_fps = 1000.0 / timing["post_avg"] if timing["post_avg"] > 0 else 0.0

    print("")
    print("=" * 58)
    print("推理设备与耗时统计")
    print("=" * 58)
    print("模式：{}".format("侧视追踪" if args.tracking_only else ("低内存三核流式" if args.streaming else "三核高速本地视频")))
    print("使用设备：RK3588 NPU 三核({}) + RKNNLite2".format(",".join(cores)))
    print("任务编号：{}".format(task_id))
    print("视频分辨率：{}x{}，模型输入：{}x{} NHWC".format(frame_w, frame_h, INPUT_WIDTH, INPUT_HEIGHT))
    print("视频总帧数：{}，参与推理帧数：{}".format(frame_count, infer_frames))
    print("有效检出点数：过滤前 {}，过滤后 {}".format(raw_valid, final_valid))
    print("球场标定：{}，耗时 {:.2f} ms".format("跳过" if args.tracking_only else ("成功" if court_calibration.get("available") else "失败"), court_calibration.get("elapsed_ms", 0.0)))
    print("落地事件：{}，判罚记录：{}，事件处理耗时 {:.2f} ms".format(len(events), len(event_payload["judgements"]), event_payload.get("elapsed_ms", 0.0)))
    player_stats = event_payload.get("player_context", {})
    mediapipe_state = "启用" if player_stats.get("available") else "未启用/不可用"
    mediapipe_error = player_stats.get("error") or ""
    print(
        "MediaPipe球员上下文：{}，采样帧 {}，stride {}，耗时 {:.2f} ms{}".format(
            mediapipe_state,
            player_stats.get("sampled_frames", 0),
            args.mediapipe_stride,
            float(player_stats.get("elapsed_ms", 0.0) or 0.0),
            "" if not mediapipe_error else "，原因：{}".format(mediapipe_error),
        )
    )
    print("-" * 58)
    print("前处理单帧平均耗时：{:.2f} ms/frame".format(timing["prep_avg"]))
    print("前处理单帧理论 FPS：{:.2f}".format(prep_fps))
    print("-" * 58)
    print("三核模型推理累计耗时：{:.4f} s".format(timing["npu_total"] / 1000.0))
    print("模型推理单帧平均耗时：{:.2f} ms/frame".format(timing["npu_avg"]))
    print("模型推理单帧理论 FPS：{:.2f}".format(npu_single_fps))
    print("三核推理阶段总 FPS：{:.2f}".format(infer_fps))
    print("-" * 58)
    print("后处理单帧平均耗时：{:.2f} ms/frame".format(timing["post_avg"]))
    print("后处理单帧理论 FPS：{:.2f}".format(post_fps))
    print("-" * 58)
    print("脚本总耗时：{:.4f} s".format(total_ms / 1000.0))
    print("端到端实际完整 FPS：{:.2f}".format(end_fps))
    print("CSV 输出：{}".format(args.out_csv))
    if not args.no_events:
        print("落地事件输出：{}".format(args.event_csv))
        print("判罚输出：{}".format(args.judgement_json))
    if not args.no_video:
        print("视频输出：{}".format(args.out_video))
    print("=" * 58)

if __name__ == "__main__":
    main()
