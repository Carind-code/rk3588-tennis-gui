#!/usr/bin/env python3
"""RK3588 pure TrackNet ball tracker.

Only performs ball tracking and writes a red-dot trajectory video.  It has no
court calibration, event detection, action analysis, CSV output, IPC, or cloud
reporting dependencies.
"""

import argparse
import os
import platform
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
try:
    from rknnlite.api import RKNNLite
except ModuleNotFoundError:
    RKNNLite = None

from sideview_postprocess import process_sideview_candidates


INPUT_WIDTH = 640
INPUT_HEIGHT = 360
INPUT_FORMAT = "nhwc"
DEFAULT_MODEL = "side_tracknet_360x640_sigmoid_fp.rknn"
PROJECT_DIR = Path(__file__).resolve().parent


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0


def project_path(value):
    """Use project-local defaults even when this script is started from another directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def host_name():
    if platform.system() + "-" + platform.machine() != "Linux-aarch64":
        return platform.system() + "-" + platform.machine()
    try:
        with open("/proc/device-tree/compatible", "r", encoding="latin-1") as file:
            compatible = file.read().lower()
    except OSError:
        return "Linux-aarch64"
    return "RK3588" if "rk3588" in compatible else "Linux-aarch64"


def core_mask(core):
    names = {
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "auto": "NPU_CORE_AUTO",
        "all": "NPU_CORE_ALL",
    }
    name = names.get(str(core).lower())
    if name is None:
        raise ValueError("unsupported NPU core: {}".format(core))
    return getattr(RKNNLite, name, None)


def init_runtime(model_path, core):
    if RKNNLite is None:
        raise RuntimeError("缺少 rknn-toolkit-lite2；请在已安装RKNNLite2的RK3588板端运行。")
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError("load_rknn failed: {}".format(ret))
    mask = core_mask(core)
    if host_name() == "RK3588" and mask is not None:
        ret = rknn.init_runtime(core_mask=mask)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("init_runtime failed: {}".format(ret))
    return rknn


def resize_bgr(frame):
    return cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)


def make_input(current, previous, preprevious):
    return np.concatenate((current, previous, preprevious), axis=2)[None].astype(np.uint8)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def extract_candidates(output, scale_x, scale_y, threshold, peak_window, top_k=5, suppress_window=15):
    """Extract several heatmap peaks so the temporal tracker can reject distractors."""
    heatmap = np.asarray(output).squeeze().astype(np.float32)
    if heatmap.shape != (INPUT_HEIGHT, INPUT_WIDTH):
        heatmap = heatmap.reshape((INPUT_HEIGHT, INPUT_WIDTH))
    if float(np.nanmin(heatmap)) < 0.0 or float(np.nanmax(heatmap)) > 1.0:
        heatmap = sigmoid(heatmap)

    peak_window = max(3, int(peak_window) | 1)
    suppress_window = max(3, int(suppress_window) | 1)
    work = heatmap.copy()
    candidates = []
    for _ in range(max(1, int(top_k))):
        _, score, _, location = cv2.minMaxLoc(work)
        score = float(score)
        if score < threshold:
            break
        peak_x, peak_y = location
        radius = peak_window // 2
        x0, x1 = max(0, peak_x - radius), min(INPUT_WIDTH, peak_x + radius + 1)
        y0, y1 = max(0, peak_y - radius), min(INPUT_HEIGHT, peak_y + radius + 1)
        crop = heatmap[y0:y1, x0:x1]
        weights = np.maximum(crop - threshold, 0.0)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            x = float((weights * xs[None, :]).sum() / weight_sum)
            y = float((weights * ys[:, None]).sum() / weight_sum)
        else:
            x, y = float(peak_x), float(peak_y)
        candidates.append({"x": x * scale_x, "y": y * scale_y, "score": score})

        radius = suppress_window // 2
        x0, x1 = max(0, peak_x - radius), min(INPUT_WIDTH, peak_x + radius + 1)
        y0, y1 = max(0, peak_y - radius), min(INPUT_HEIGHT, peak_y + radius + 1)
        work[y0:y1, x0:x1] = -1.0
    return candidates


def worker(worker_id, core, model, jobs, results, ready, scale_x, scale_y, args):
    runtime = None
    announced_ready = False
    try:
        runtime = init_runtime(model, core)
        ready.put({"worker": worker_id, "core": core, "host": host_name()})
        announced_ready = True
        while True:
            job = jobs.get()
            if job is None:
                return
            npu_start = time.perf_counter()
            outputs = runtime.inference(inputs=[job["input"]], data_format=[INPUT_FORMAT])
            npu_ms = elapsed_ms(npu_start)
            if not outputs:
                raise RuntimeError("rknn inference returned no output")
            post_start = time.perf_counter()
            candidates = extract_candidates(
                outputs[0], scale_x, scale_y, args.threshold, args.peak_window,
                args.top_k, args.suppress_window,
            )
            results.put({
                "frame": job["frame"],
                "candidates": candidates,
                "prep_ms": job["prep_ms"],
                "npu_ms": npu_ms,
                "post_ms": elapsed_ms(post_start),
                "worker": worker_id,
                "core": core,
                "error": "",
            })
    except Exception as exc:
        if not announced_ready:
            ready.put({"worker": worker_id, "core": core, "host": "ERROR", "error": str(exc)})
        results.put({"frame": -1, "worker": worker_id, "core": core, "error": str(exc)})
    finally:
        if runtime is not None:
            runtime.release()


def start_workers(model, width, height, args):
    jobs = queue.Queue(maxsize=max(3, args.workers * 2))
    results = queue.Queue()
    ready = queue.Queue()
    threads = []
    cores = [str(item).strip() for item in args.cores.split(",") if str(item).strip()]
    if not cores:
        cores = ["0", "1", "2"]
    while len(cores) < args.workers:
        cores.extend(cores)
    cores = cores[:args.workers]
    for worker_id, core in enumerate(cores):
        thread = threading.Thread(
            target=worker,
            args=(worker_id, core, model, jobs, results, ready, width / INPUT_WIDTH, height / INPUT_HEIGHT, args),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    ready_items = [ready.get() for _ in threads]
    failed = next((item for item in ready_items if item.get("error")), None)
    if failed is not None:
        raise RuntimeError("worker {} core {} init failed: {}".format(
            failed["worker"], failed["core"], failed["error"]
        ))
    for item in sorted(ready_items, key=lambda value: value["worker"]):
        print("WORKER_READY worker={worker} core={core} host={host}".format(**item), flush=True)
    return jobs, results, threads, cores


def video_metadata(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("cannot open video: {}".format(source))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid video dimensions: {}".format(source))
    return width, height, fps, frames


def track_video(video_path, model_path, args):
    width, height, fps, frame_hint = video_metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    jobs, results, threads, cores = start_workers(model_path, width, height, args)
    start = time.perf_counter()
    previous = preprevious = None
    submitted = 0
    read_frames = 0
    prep_total = 0.0
    candidate_rows = {}
    timing_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id = read_frames
        read_frames += 1
        prep_start = time.perf_counter()
        small = resize_bgr(frame)
        if previous is not None and preprevious is not None:
            prep_ms = elapsed_ms(prep_start)
            prep_total += prep_ms
            jobs.put({"frame": frame_id, "input": make_input(small, previous, preprevious), "prep_ms": prep_ms})
            submitted += 1
        preprevious, previous = previous, small

    cap.release()
    for _ in threads:
        jobs.put(None)

    completed = 0
    while completed < submitted:
        item = results.get()
        if item["error"]:
            raise RuntimeError("worker {} core {} failed: {}".format(item["worker"], item["core"], item["error"]))
        candidate_rows[item["frame"]] = item["candidates"]
        timing_rows.append(item)
        completed += 1
        if not args.no_progress and (completed % 20 == 0 or completed == submitted):
            fps_now = completed / max(time.perf_counter() - start, 1e-6)
            print("进度：{}/{} ({:.2f}%)，当前追踪吞吐：{:.2f} FPS".format(
                completed, submitted, completed * 100.0 / max(submitted, 1), fps_now
            ), flush=True)
    for thread in threads:
        thread.join()

    candidate_sequence = [candidate_rows.get(index, []) for index in range(read_frames)]
    post_start = time.perf_counter()
    stable = process_sideview_candidates(candidate_sequence, width)
    tracking_ms = elapsed_ms(start)
    smooth_ms = elapsed_ms(post_start)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": read_frames,
        "submitted": submitted,
        "tracks": stable["tracks"],
        "breaks": stable.get("breaks", [False] * read_frames),
        "cores": cores,
        "timing_rows": timing_rows,
        "prep_total_ms": prep_total,
        "smooth_ms": smooth_ms,
        "tracking_ms": tracking_ms,
        "frame_hint": frame_hint,
    }


def valid(point):
    return point is not None and point[0] is not None and point[1] is not None


def draw_track(frame, trace, point, radius):
    """Match the formal Qiuwu visual language: red current point + yellow short tail."""
    points = [item for item in trace if valid(item)]
    for age, item in enumerate(reversed(points)):
        dot_radius = max(1, radius - age // 4)
        color = (0, 0, 255) if age == 0 else (0, 220, 255)
        cv2.circle(frame, (int(round(item[0])), int(round(item[1]))), dot_radius, color, -1, cv2.LINE_AA)
    if valid(point):
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(frame, center, max(5, radius + 2), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, max(8, radius + 5), (255, 255, 255), 1, cv2.LINE_AA)


def write_video(video_path, output_path, result, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot reopen input video: {}".format(video_path))
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*args.fourcc),
        result["fps"],
        (result["width"], result["height"]),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("cannot open output video: {}".format(output_path))
    start = time.perf_counter()
    trace = deque(maxlen=args.trace_length)
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        point = result["tracks"][frame_id] if frame_id < len(result["tracks"]) else None
        if frame_id < len(result["breaks"]) and result["breaks"][frame_id]:
            trace.clear()
        trace.append(point if valid(point) else None)
        draw_track(frame, trace, point, args.dot_radius)
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.release()
    return elapsed_ms(start), frame_id


def choose_live_point(candidates, previous, width):
    """Low-latency association for camera mode; offline video keeps full side-view smoothing."""
    valid_candidates = [item for item in candidates if item["score"] >= 0.18]
    if not valid_candidates:
        return None
    if previous is None:
        return (valid_candidates[0]["x"], valid_candidates[0]["y"])
    gate = max(120.0, float(width) * 0.30)
    ranked = sorted(
        valid_candidates,
        key=lambda item: (item["x"] - previous[0]) ** 2 + (item["y"] - previous[1]) ** 2,
    )
    point = ranked[0]
    distance = float(np.hypot(point["x"] - previous[0], point["y"] - previous[1]))
    return (point["x"], point["y"]) if distance <= gate else None


def track_camera(camera_path, model_path, args):
    cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera: {}".format(camera_path))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("camera returned no frame: {}".format(camera_path))
    height, width = first.shape[:2]
    jobs, results, threads, cores = start_workers(model_path, width, height, args)
    writer = None
    if not args.no_write:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*args.fourcc), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("cannot open output video: {}".format(args.out))

    start = time.perf_counter()
    preprevious = previous = None
    frame_id = 0
    submitted = 0
    completed = 0
    next_output = 2
    buffered_frames = {}
    completed_rows = {}
    trace = deque(maxlen=args.trace_length)
    last_point = None
    timing_rows = []
    raw_valid = 0
    written = 0

    def render_ready():
        nonlocal next_output, last_point, written
        while next_output in completed_rows:
            item = completed_rows.pop(next_output)
            frame = buffered_frames.pop(next_output)
            point = choose_live_point(item["candidates"], last_point, width)
            if point is not None:
                last_point = point
            elif len(trace) >= 3:
                last_point = None
            trace.append(point)
            if writer is not None:
                draw_track(frame, trace, point, args.dot_radius)
                writer.write(frame)
                written += 1
            next_output += 1

    # First two frames initialize TrackNet's temporal input. They are written without a marker.
    for warmup in (first,):
        if writer is not None:
            writer.write(warmup)
            written += 1
        frame_id += 1
        previous = resize_bgr(warmup)

    max_frames = int(args.camera_frames)
    while max_frames <= 0 or frame_id < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        small = resize_bgr(frame)
        if preprevious is None:
            preprevious = previous
            previous = small
            if writer is not None:
                writer.write(frame)
                written += 1
            frame_id += 1
            continue
        prep_start = time.perf_counter()
        input_data = make_input(small, previous, preprevious)
        buffered_frames[frame_id] = frame
        jobs.put({"frame": frame_id, "input": input_data, "prep_ms": elapsed_ms(prep_start)})
        submitted += 1
        preprevious, previous = previous, small
        frame_id += 1

        while True:
            try:
                item = results.get_nowait()
            except queue.Empty:
                break
            if item["error"]:
                raise RuntimeError("worker {} core {} failed: {}".format(item["worker"], item["core"], item["error"]))
            completed_rows[item["frame"]] = item
            timing_rows.append(item)
            raw_valid += int(bool(item["candidates"]))
            completed += 1
        render_ready()

    for _ in threads:
        jobs.put(None)
    while completed < submitted:
        item = results.get()
        if item["error"]:
            raise RuntimeError("worker {} core {} failed: {}".format(item["worker"], item["core"], item["error"]))
        completed_rows[item["frame"]] = item
        timing_rows.append(item)
        raw_valid += int(bool(item["candidates"]))
        completed += 1
        render_ready()
    for thread in threads:
        thread.join()
    if writer is not None:
        writer.release()
    cap.release()
    total_ms = elapsed_ms(start)
    print("\n" + "=" * 62)
    print("RK3588实时摄像头纯网球追踪")
    print("=" * 62)
    print("摄像头：{}，采集分辨率：{}x{}".format(camera_path, width, height))
    print("使用设备：RK3588 NPU 三核({}) + RKNNLite2".format(",".join(cores)))
    print("采集帧数：{}，参与推理帧数：{}，原始有效候选：{}".format(frame_id, submitted, raw_valid))
    camera_label = "不含写视频实时FPS" if args.no_write else "包含写视频实时FPS"
    print("实时处理总耗时：{:.4f} s，{}：{:.2f}".format(
        total_ms / 1000.0, camera_label, frame_id / max(total_ms / 1000.0, 1e-6)
    ))
    if writer is not None:
        print("追踪视频输出：{}".format(args.out))
    print("=" * 62)


def print_summary(result, write_ms, written_frames, output_path):
    rows = result["timing_rows"]
    avg = lambda name: sum(float(row[name]) for row in rows) / len(rows) if rows else 0.0
    npu_sum = sum(float(row["npu_ms"]) for row in rows)
    raw_valid = sum(1 for row in rows if row["candidates"])
    final_valid = sum(1 for point in result["tracks"] if valid(point))
    tracking_fps = result["frames"] / max(result["tracking_ms"] / 1000.0, 1e-6)
    inference_fps = result["submitted"] / max(result["tracking_ms"] / 1000.0, 1e-6)
    total_ms = result["tracking_ms"] + write_ms
    full_fps = result["frames"] / max(total_ms / 1000.0, 1e-6)
    print("\n" + "=" * 62)
    print("RK3588 纯网球追踪性能统计")
    print("=" * 62)
    print("使用设备：RK3588 NPU 三核({}) + RKNNLite2".format(",".join(result["cores"])))
    print("视频：{}x{}，模型输入：{}x{} NHWC".format(result["width"], result["height"], INPUT_WIDTH, INPUT_HEIGHT))
    print("视频帧数：{}，参与推理帧数：{}".format(result["frames"], result["submitted"]))
    print("有效追踪点：原始 {}，轨迹过滤后 {}".format(raw_valid, final_valid))
    print("-" * 62)
    print("前处理平均耗时：{:.2f} ms/frame".format(avg("prep_ms")))
    print("NPU单实例平均耗时：{:.2f} ms/frame".format(avg("npu_ms")))
    print("热图候选提取平均耗时：{:.2f} ms/frame".format(avg("post_ms")))
    print("轨迹过滤与平滑耗时：{:.2f} ms".format(result["smooth_ms"]))
    print("三核NPU累计忙碌时间：{:.4f} s".format(npu_sum / 1000.0))
    print("-" * 62)
    print("不含写视频：纯追踪总耗时 {:.4f} s，追踪FPS {:.2f}".format(result["tracking_ms"] / 1000.0, tracking_fps))
    print("不含写视频：三核推理阶段吞吐 {:.2f} FPS".format(inference_fps))
    if write_ms > 0:
        print("视频写入：{} 帧，耗时 {:.4f} s，写入FPS {:.2f}".format(
            written_frames, write_ms / 1000.0, written_frames / max(write_ms / 1000.0, 1e-6)
        ))
        print("包含写视频：完整总耗时 {:.4f} s，完整FPS {:.2f}".format(total_ms / 1000.0, full_fps))
        print("追踪视频输出：{}".format(output_path))
    print("=" * 62)


def parse_args():
    parser = argparse.ArgumentParser(description="RK3588 pure TrackNet ball tracking.")
    parser.add_argument("--video", default="input.mp4", help="local input video; default: input.mp4")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RKNN TrackNet model")
    parser.add_argument("--out", default="track_output.mp4", help="trajectory video output")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", default="0,1,2")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--peak-window", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--suppress-window", type=int, default=15)
    parser.add_argument("--dot-radius", type=int, default=5)
    parser.add_argument("--trace-length", type=int, default=7)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--no-write", action="store_true", help="only benchmark tracking; do not write video")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--camera", default="", help="V4L2 camera device, for example /dev/video11")
    parser.add_argument("--camera-width", type=int, default=1024)
    parser.add_argument("--camera-height", type=int, default=576)
    parser.add_argument("--camera-frames", type=int, default=0, help="0 means run until Ctrl+C")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.model = str(project_path(args.model))
    args.video = str(project_path(args.video))
    args.out = str(project_path(args.out))
    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)
    if not args.camera and not os.path.exists(args.video):
        raise FileNotFoundError(args.video)
    cv2.setNumThreads(1)
    if args.camera:
        track_camera(args.camera, args.model, args)
        return
    result = track_video(args.video, args.model, args)
    write_ms = 0.0
    written_frames = 0
    if not args.no_write:
        write_ms, written_frames = write_video(args.video, args.out, result, args)
    print_summary(result, write_ms, written_frames, args.out)


if __name__ == "__main__":
    main()
