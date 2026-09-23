# Version

## V3 - Court, Bounce and Judgement Integration

- Version: `v3.0.0`
- Date: 2026-06-15
- Replaced the V2 270x480 model with the V3.7 360x640 FP RKNN model.
- Reused the upstream V3.7 candidate stabilization and trajectory cleanup path.
- Added one-time RKNN court calibration and fixed image-to-court homography.
- Integrated the upstream V5.2 bounce detector, trajectory cleanup and contact timing refinement.
- Added optional low-rate MediaPipe player context using `broadcast_court_only.json`.
- Added singles/doubles IN/OUT and basic rally point judgement.
- Added `bounce_events.csv` and `judgement.json` to OSS/MQTT result delivery.
- Preserved V2 at tag and branch `v2-board-cloud-baseline-20260615`.

## V2

- Version: `v2.0.0`
- Package: `rk3588-tennis-detection`
- Date: 2026-05-31
- Default model: `model_v3_270x480_b1_sigmoid_i8.rknn`
- Optional FP model: `model_v3_270x480_b1_sigmoid_fp.rknn`
- Board runtime: RK3588 + RKNNLite2
- NPU mode: three RKNNLite workers bound to cores `0,1,2`
- IPC: Unix Domain Socket `/tmp/tennis_ipc.sock`
- Cloud flow: MQTT task notification + OSS video/CSV upload

## Recovery Points

The first stable board-cloud implementation is tagged as:

```text
v1
board-cloud-v1
```

This version should be tagged as:

```text
v2
v2.0.0
```

## V2 Changes

1. Updated the board inference path to TrackNet V3 RKNN.
2. Set INT8 RKNN as the default model for local and cloud startup.
3. Kept FP RKNN for quick comparison through `--model`.
4. Used three RKNNLite workers bound to NPU cores `0,1,2`.
5. Switched runtime input to NHWC to avoid RKNN input format conversion warnings.
6. Aligned heatmap postprocess with V3 peak-window refinement.
7. Added raw coordinates to CSV for debugging jump points.
8. Kept `start_test.sh` for local output and `start_all.sh` for inference plus cloud upload.
