#!/usr/bin/env python3
"""Official RKNN Model Zoo compatible YOLOv8n-pose RK3588 hybrid INT8 converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


CUSTOM_HYBRID = [
    "/model.22/cv4.0/cv4.0.2/Conv_output_0",
    "/model.22/cv4.0/cv4.0.2/Conv_output_0_1",
    "/model.22/cv4.0/cv4.0.2/Conv_output_0_2",
    "/model.22/cv4.0/cv4.0.2/Conv_output_0_3",
    "/model.22/Concat_6_output_0",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLOv8n-pose ONNX to RK3588 hybrid INT8 RKNN")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Official Model Zoo calibration dataset.txt")
    parser.add_argument("--rknn", type=Path, required=True)
    args = parser.parse_args()

    rknn = RKNN(verbose=True)
    try:
        assert rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588") == 0
        assert rknn.load_onnx(model=str(args.onnx)) == 0
        assert rknn.hybrid_quantization_step1(dataset=str(args.dataset), proposal=False, custom_hybrid=CUSTOM_HYBRID) == 0
        stem = args.onnx.stem
        assert rknn.hybrid_quantization_step2(
            model_input=f"{stem}.model",
            data_input=f"{stem}.data",
            model_quantization_cfg=f"{stem}.quantization.cfg",
        ) == 0
        args.rknn.parent.mkdir(parents=True, exist_ok=True)
        assert rknn.export_rknn(str(args.rknn)) == 0
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
