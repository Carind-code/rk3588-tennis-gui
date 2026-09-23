import argparse
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


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


def fit_time(features, fixed_time):
    original_time = features.shape[1]
    if original_time == fixed_time:
        return features, original_time
    if original_time > fixed_time:
        return features[:, :fixed_time], fixed_time

    padded = np.zeros((features.shape[0], fixed_time), dtype=np.float32)
    padded[:, :original_time] = features
    return padded, original_time


def run_chunked_inference(rknn, features, fixed_time, mapping):
    labels = []
    calls = 0
    start = time.perf_counter()

    for start_idx in range(0, features.shape[1], fixed_time):
        chunk = features[:, start_idx:start_idx + fixed_time]
        valid_len = chunk.shape[1]
        fitted, _ = fit_time(chunk, fixed_time)
        input_x = fitted[None, :, :].astype(np.float32)
        logits = rknn.inference(inputs=[input_x])[0]
        pred = np.argmax(logits, axis=1).reshape(-1)[:valid_len]
        labels.extend(mapping[int(i)] for i in pred)
        calls += 1

    infer_ms = (time.perf_counter() - start) * 1000.0
    return labels, infer_ms, calls


def main():
    parser = argparse.ArgumentParser(description="Run fixed-shape MS-TCN RKNN inference on RK3588.")
    parser.add_argument("--rknn", default="../models/mstcn_tennis_t459_fp.rknn")
    parser.add_argument("--feature", default="../samples/test_video3.npy")
    parser.add_argument("--mapping", default="../data/mapping.txt")
    parser.add_argument("--output", default="../samples/rknn_prediction.txt")
    parser.add_argument("--fixed-time", type=int, default=459)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    rknn_path = (script_dir / args.rknn).resolve()
    feature_path = (script_dir / args.feature).resolve()
    mapping_path = (script_dir / args.mapping).resolve()
    output_path = (script_dir / args.output).resolve()

    features = np.load(feature_path).astype(np.float32)
    if features.ndim != 2 or features.shape[0] != 99:
        raise ValueError(f"Expected feature shape (99, T), got {features.shape}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError("load_rknn failed")

    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("init_runtime failed")

    mapping = load_mapping(mapping_path)
    labels, infer_ms, calls = run_chunked_inference(rknn, features, args.fixed_time, mapping)
    rknn.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "### Frame level recognition: ###\n" + " ".join(labels),
        encoding="utf-8",
    )

    fps = len(labels) / (infer_ms / 1000.0) if infer_ms > 0 else 0.0
    print(f"Input feature: {features.shape}")
    print(f"Fixed RKNN input per call: (1, 99, {args.fixed_time})")
    print(f"RKNN calls: {calls}, output frames: {len(labels)}")
    print(f"Model inference time: {infer_ms:.3f} ms, sequence FPS: {fps:.2f}")
    print(f"Prediction output: {output_path}")


if __name__ == "__main__":
    main()
