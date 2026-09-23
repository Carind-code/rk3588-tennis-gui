# RK3588 Body Action Recognition

This project is the RK3588-adapted version of `body_model`. It restores the PC pipeline as much as possible while replacing the MS-TCN action model inference with RKNN.

## What It Does

Input:

- A tennis video, default: `samples/sample_video.mp4`

Full pipeline:

```text
video frames
-> MediaPipe Pose extracts 33 body landmarks
-> normalize landmarks into feature tensor (99,T)
-> RKNN MS-TCN model predicts frame-level action labels
-> save txt/csv/json results
-> optionally overlay labels, skeleton and action timeline onto video
```

Output:

- `outputs/body_action_features.npy`: normalized feature, shape `(99,T)`
- `outputs/body_action_pose.npy`: raw MediaPipe landmarks, shape `(T,33,3)`
- `outputs/body_action_prediction.txt`: one action label for each frame
- `outputs/body_action_segments.csv`: continuous action segments
- `outputs/body_action_summary.json`: timing and output summary
- `outputs/body_action_overlay.mp4`: visualized video

Classes:

- `serve`
- `forehand`
- `backhand`
- `background`

## RK3588 Adaptation Points

- PC original inference uses ONNX. This project uses `models/mstcn_tennis_t459_fp.rknn`.
- PC original scripts were split and path-hardcoded. This project adds `scripts/run_body_action_full.py` as a default-runnable full pipeline.
- The fixed RKNN model input is `1x99x459`. For videos longer than 459 frames, the script runs chunked inference and concatenates frame labels. The last chunk is zero-padded and then trimmed back to the real frame count.
- MediaPipe Pose is still used for body landmark extraction. This is the main board-side dependency and likely the main runtime bottleneck.
- A dynamic RKNN model is also included: `models/mstcn_tennis_dynamic_t256_459_512_fp.rknn`, but it only supports declared lengths `256/459/512` and should be treated as compatibility testing, not the default path.

## Board Usage

Run the full restored pipeline:

```bash
cd body_model_rknn_test
bash start_full.sh
```

Run without writing overlay video, for faster timing:

```bash
cd body_model_rknn_test
bash start_fast_no_video.sh
```

Use a custom video:

```bash
cd body_model_rknn_test/scripts
python3 run_body_action_full.py --video /path/to/input.mp4
```

## Dependencies

The board must already have:

- `rknn-lite2`
- `opencv-python`
- `numpy`
- `mediapipe`

If MediaPipe is not installed, the RKNN model can still run on existing `.npy` features, but the full PC-equivalent video pipeline cannot run.

## Single Step Debug Commands

Extract features only:

```bash
cd body_model_rknn_test/scripts
python3 extract_pose_features.py --video ../samples/sample_video.mp4 --output ../outputs/body_action_features.npy
```

Run RKNN on an existing feature file:

```bash
cd body_model_rknn_test/scripts
python3 rknn_infer_fixed.py --feature ../outputs/body_action_features.npy --output ../outputs/body_action_prediction.txt
```

Overlay an existing prediction:

```bash
cd body_model_rknn_test/scripts
python3 overlay_video.py --video ../samples/sample_video.mp4 --prediction ../outputs/body_action_prediction.txt --output ../outputs/body_action_overlay.mp4
```

## Current Scope

This is a human action recognition module. It does not output tennis ball coordinates and does not replace TrackNet. It can be integrated with the tennis judging pipeline as an auxiliary human-action signal, such as serve/forehand/backhand/background stage recognition.
