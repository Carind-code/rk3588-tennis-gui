#!/usr/bin/env python3
"""Export the fixed 640x640 YOLO ONNX model as RK3588 FP RKNN."""

import argparse
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--rknn", required=True)
    args = parser.parse_args()

    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    if rknn.load_onnx(model=args.onnx, inputs=["images"], input_size_list=[[1, 3, 640, 640]]) != 0:
        raise RuntimeError("load_onnx failed")
    if rknn.build(do_quantization=False) != 0:
        raise RuntimeError("FP build failed")
    if rknn.export_rknn(args.rknn) != 0:
        raise RuntimeError("export_rknn failed")
    rknn.release()


if __name__ == "__main__":
    main()
