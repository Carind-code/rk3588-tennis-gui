import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


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


def extract_pose_keypoints(video_path, model_complexity):
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
    return np.asarray(all_keypoints, dtype=np.float32), valid_pose


def pose_to_feature(pose):
    pose = normalize_pose_keypoints(pose)
    t, v, c = pose.shape
    return pose.reshape(t, v * c).T.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose features for the tennis action model.")
    parser.add_argument("--video", default="../samples/sample_video.mp4")
    parser.add_argument("--output", default="../samples/sample_video.npy")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    video_path = (script_dir / args.video).resolve()
    output_path = (script_dir / args.output).resolve()

    start = time.perf_counter()
    pose, valid_pose = extract_pose_keypoints(video_path, args.model_complexity)
    feature = pose_to_feature(pose)
    elapsed = time.perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature)

    fps = pose.shape[0] / elapsed if elapsed > 0 else 0.0
    print(f"视频帧数: {pose.shape[0]}")
    print(f"有效人体姿态帧数: {valid_pose}")
    print(f"输出特征: {feature.shape} -> {output_path}")
    print(f"MediaPipe+特征提取耗时: {elapsed:.3f} s, FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
