import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from rknnlite.api import RKNNLite


COLORS = {
    "serve": (55, 96, 255),
    "forehand": (250, 183, 50),
    "backhand": (83, 179, 36),
    "background": (24, 24, 24),
}

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


def load_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split()
            mapping[int(idx)] = name
    return mapping


def normalize_pose_keypoints(keypoints):
    keypoints = keypoints.astype(np.float32)
    normalized = keypoints.copy()

    left_shoulder_idx = 11
    right_shoulder_idx = 12
    left_hip_idx = 23
    right_hip_idx = 24
    eps = 1e-6

    for i in range(normalized.shape[0]):
        frame = normalized[i]
        if np.allclose(frame, 0):
            continue

        left_hip = frame[left_hip_idx]
        right_hip = frame[right_hip_idx]
        hip_center = (left_hip + right_hip) / 2.0

        left_shoulder = frame[left_shoulder_idx]
        right_shoulder = frame[right_shoulder_idx]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        frame = frame - hip_center
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length < eps:
            torso_length = np.linalg.norm(left_shoulder - right_shoulder)
        if torso_length < eps:
            torso_length = 1.0
        normalized[i] = frame / torso_length

    return normalized


def pose_to_feature(pose):
    pose = normalize_pose_keypoints(pose)
    t, v, c = pose.shape
    return pose.reshape(t, v * c).T.astype(np.float32)


def extract_pose_features(video_path, model_complexity):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    all_keypoints = []
    valid_pose = 0
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                valid_pose += 1
                frame_keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            else:
                frame_keypoints = [[0.0, 0.0, 0.0] for _ in range(33)]
            all_keypoints.append(frame_keypoints)

    cap.release()
    pose_array = np.asarray(all_keypoints, dtype=np.float32)
    return pose_array, pose_to_feature(pose_array), valid_pose


def run_fixed_rknn(rknn_path, features, mapping, fixed_time, core_mask):
    if features.ndim != 2 or features.shape[0] != 99:
        raise ValueError(f"Expected feature shape (99, T), got {features.shape}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    init_kwargs = {}
    if core_mask:
        init_kwargs["core_mask"] = getattr(RKNNLite, core_mask)
    ret = rknn.init_runtime(**init_kwargs)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    frame_count = features.shape[1]
    labels = []
    inference_calls = 0
    start = time.perf_counter()

    for start_idx in range(0, frame_count, fixed_time):
        chunk = features[:, start_idx:start_idx + fixed_time]
        valid_len = chunk.shape[1]
        if valid_len < fixed_time:
            padded = np.zeros((99, fixed_time), dtype=np.float32)
            padded[:, :valid_len] = chunk
            chunk = padded

        input_x = chunk[None, :, :].astype(np.float32)
        logits = rknn.inference(inputs=[input_x])[0]
        pred = np.argmax(logits, axis=1).reshape(-1)[:valid_len]
        labels.extend(mapping[int(i)] for i in pred)
        inference_calls += 1

    infer_s = time.perf_counter() - start
    rknn.release()
    return labels, infer_s, inference_calls


def save_prediction(prediction_path, labels):
    prediction_path.write_text(
        "### Frame level recognition: ###\n" + " ".join(labels),
        encoding="utf-8",
    )


def draw_pose(frame, pose_frame):
    height, width = frame.shape[:2]
    points = []
    for x, y, _z in pose_frame:
        if x == 0 and y == 0:
            points.append(None)
        else:
            points.append((int(x * width), int(y * height)))

    for a, b in POSE_CONNECTIONS:
        if points[a] is not None and points[b] is not None:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 1, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 2, (0, 255, 255), -1, cv2.LINE_AA)


def draw_timeline(frame, labels, frame_idx):
    height, width = frame.shape[:2]
    x0, y0 = 24, height - 28
    bar_w, bar_h = min(width - 48, 760), 12
    total = max(1, len(labels))
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)

    last_x = x0
    last_label = labels[0] if labels else "background"
    for idx, label in enumerate(labels):
        x = x0 + int(bar_w * (idx + 1) / total)
        if label != last_label or idx == total - 1:
            cv2.rectangle(frame, (last_x, y0), (x, y0 + bar_h), COLORS.get(last_label, (80, 80, 80)), -1)
            last_x = x
            last_label = label

    marker_x = x0 + int(bar_w * min(frame_idx, total - 1) / total)
    cv2.line(frame, (marker_x, y0 - 4), (marker_x, y0 + bar_h + 4), (255, 255, 255), 2)


def overlay_video(base_video_path, output_path, labels, pose_array, draw_skeleton):
    cap = cv2.VideoCapture(str(base_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {base_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(labels):
            break

        label = labels[frame_idx]
        color = COLORS.get(label, (80, 80, 80))
        if draw_skeleton and frame_idx < len(pose_array):
            draw_pose(frame, pose_array[frame_idx])

        # Slightly larger, high-legibility action card for the pose result
        # video.  Keep it inside the existing top-left safe area.
        x, y = 24, 24
        w, h = 410, 108
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.00, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Action: {label}", (x + 20, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 1.12, (255, 255, 255), 2, cv2.LINE_AA)
        # Keep the result video clean: action details stay in the right-side
        # GUI panel instead of adding a burned-in bottom-left timeline.

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return frame_count, frame_idx


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


def main():
    parser = argparse.ArgumentParser(description="RK3588 side-view ball tracking and body-action pipeline.")
    parser.add_argument("--video", default="input.mp4")
    parser.add_argument("--tracknet", default="side_tracknet_360x640_sigmoid_fp.rknn")
    parser.add_argument("--action-rknn", default="mstcn_tennis_t459_fp.rknn")
    parser.add_argument("--mapping", default="mapping.txt")
    parser.add_argument("--out-video", default="side_training_output.mp4")
    parser.add_argument("--out-csv", default="side_tracknet_output.csv")
    parser.add_argument("--prediction", default="body_action_prediction.txt")
    parser.add_argument("--fixed-time", type=int, default=459)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--no-skeleton", action="store_true")
    parser.add_argument("--ipc-socket", default="/tmp/tennis_ipc.sock")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--task-id", default="")
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent
    match_dir = work_dir.parent / "match_judgement"
    video_path = (work_dir / args.video).resolve()
    tracknet_path = (work_dir / args.tracknet).resolve()
    rknn_path = (work_dir / args.action_rknn).resolve()
    mapping_path = (work_dir / args.mapping).resolve()
    video_output_path = (work_dir / args.out_video).resolve()
    csv_output_path = (work_dir / args.out_csv).resolve()
    prediction_path = (work_dir / args.prediction).resolve()
    ball_video_path = work_dir / ".side_ball_overlay.tmp.mp4"

    total_start = time.perf_counter()

    track_start = time.perf_counter()
    track_cmd = [
        sys.executable,
        str(match_dir / "infer_tracknet_rknn_lite.py"),
        "--model", str(tracknet_path),
        "--video", str(video_path),
        "--out-video", str(ball_video_path),
        "--out-csv", str(csv_output_path),
        "--workers", "3",
        "--cores", "0,1,2",
        "--opencv-threads", "1",
        "--threshold", "0.35",
        "--peak-window", "9",
        "--max-dist", "220",
        "--dot-radius", "5",
        "--streaming",
        "--tracking-only",
        "--no-events",
        "--no-ipc",
    ]
    subprocess.run(track_cmd, cwd=str(match_dir), check=True)
    track_s = time.perf_counter() - track_start

    t0 = time.perf_counter()
    pose_array, features, valid_pose = extract_pose_features(video_path, args.model_complexity)
    pose_s = time.perf_counter() - t0

    mapping = load_mapping(mapping_path)

    labels, infer_s, inference_calls = run_fixed_rknn(
        rknn_path=rknn_path,
        features=features,
        mapping=mapping,
        fixed_time=args.fixed_time,
        core_mask="",
    )
    save_prediction(prediction_path, labels)

    source_frames = int(pose_array.shape[0])
    t0 = time.perf_counter()
    source_frames, written_frames = overlay_video(
        base_video_path=ball_video_path,
        output_path=video_output_path,
        labels=labels,
        pose_array=pose_array,
        draw_skeleton=not args.no_skeleton,
    )
    write_s = time.perf_counter() - t0
    ball_video_path.unlink(missing_ok=True)

    total_s = time.perf_counter() - total_start

    task_id = args.task_id or "RK3588_SIDE_{}".format(time.strftime("%Y%m%d_%H%M%S"))
    if args.upload:
        ok = send_ipc(
            args.ipc_socket,
            {
                "type": "video_task",
                "task_id": task_id,
                "video_path": str(video_output_path),
                "csv_path": str(csv_output_path),
                "artifacts": [
                    {"path": str(prediction_path), "file_type": "action_prediction", "result_key": "action_prediction_url"}
                ],
                "frames": source_frames,
                "fps": source_frames / total_s if total_s > 0 else 0.0,
                "mode": "side_training",
                "timestamp": time.time(),
            },
        )
        if not ok:
            raise RuntimeError("cloud IPC is unavailable: {}".format(args.ipc_socket))

    print("=" * 58)
    print("RK3588侧视训练协同分析")
    print("=" * 58)
    print(f"输入视频：{video_path}")
    print(f"视频帧数：{source_frames}，有效姿态帧：{valid_pose}")
    print("-" * 58)
    print(f"侧视TrackNet追踪：{track_s:.4f} s")
    print(f"MediaPipe姿态与特征：{pose_s:.4f} s，FPS：{source_frames / pose_s if pose_s > 0 else 0.0:.2f}")
    print(f"RKNN动作分类：{infer_s:.4f} s，调用次数：{inference_calls}")
    print(f"合成视频写入：{write_s:.4f} s，FPS：{written_frames / write_s if write_s > 0 else 0.0:.2f}")
    print(f"总耗时：{total_s:.4f} s，端到端FPS：{source_frames / total_s if total_s > 0 else 0.0:.2f}")
    print("-" * 58)
    print(f"协同视频：{video_output_path}")
    print(f"球轨迹CSV：{csv_output_path}")
    print(f"动作结果：{prediction_path}")
    if args.upload:
        print(f"上云任务：{task_id}")
    print("=" * 58)


if __name__ == "__main__":
    main()
