import argparse
from pathlib import Path

import cv2


COLORS = {
    "serve": (55, 96, 255),
    "forehand": (250, 183, 50),
    "backhand": (83, 179, 36),
    "background": (0, 0, 0),
}


def load_labels(prediction_path):
    lines = Path(prediction_path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"No frame-level prediction line found in {prediction_path}")
    return lines[1].split()


def main():
    parser = argparse.ArgumentParser(description="Overlay tennis action labels on video.")
    parser.add_argument("--video", default="../samples/sample_video.mp4")
    parser.add_argument("--prediction", default="../samples/rknn_prediction.txt")
    parser.add_argument("--output", default="../samples/sample_video_overlay.mp4")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    video_path = (script_dir / args.video).resolve()
    prediction_path = (script_dir / args.prediction).resolve()
    output_path = (script_dir / args.output).resolve()

    labels = load_labels(prediction_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

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

        x, y = 24, 24
        w, h = 360, 92
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (x + 18, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Action: {label}", (x + 18, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"视频帧数: {frame_count}, 标注帧数: {len(labels)}, 写出帧数: {frame_idx}")
    print(f"输出视频: {output_path}")


if __name__ == "__main__":
    main()
