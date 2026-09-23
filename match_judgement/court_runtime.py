import time
import gc

import cv2
import numpy as np


INPUT_W = 640
INPUT_H = 360
LOW_THRESH = 170

REFERENCE_KEYPOINTS = np.asarray(
    [
        (286, 561), (1379, 561),
        (286, 2935), (1379, 2935),
        (423, 561), (423, 2935),
        (1242, 561), (1242, 2935),
        (423, 1110), (1242, 1110),
        (423, 2386), (1242, 2386),
        (832, 1110), (832, 2386),
    ],
    dtype=np.float32,
)

COURT_CONFS = [
    [0, 1, 2, 3], [4, 6, 5, 7], [4, 1, 5, 3], [0, 6, 2, 7],
    [8, 9, 10, 11], [8, 9, 5, 7], [4, 6, 10, 11], [6, 1, 7, 3],
    [0, 4, 2, 5], [8, 12, 10, 13], [12, 9, 13, 11], [10, 11, 5, 7],
]

COURT_LINES = [
    ((286, 561), (1379, 561)), ((286, 2935), (1379, 2935)),
    ((286, 1748), (1379, 1748)), ((286, 561), (286, 2935)),
    ((1379, 561), (1379, 2935)), ((423, 561), (423, 2935)),
    ((1242, 561), (1242, 2935)), ((832, 1110), (832, 2386)),
    ((423, 1110), (1242, 1110)), ((423, 2386), (1242, 2386)),
]


def _decode_peak(heatmap):
    heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(heat_u8, LOW_THRESH, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=2, minRadius=10, maxRadius=30,
    )
    score = float(np.max(heatmap))
    if circles is not None:
        return float(circles[0][0][0]), float(circles[0][0][1]), score
    if score >= LOW_THRESH / 255.0:
        y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        return float(x), float(y), score
    return None, None, score


def _decode_points(output, width, height):
    heatmaps = np.asarray(output)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    if heatmaps.shape[-1] == 15:
        heatmaps = np.transpose(heatmaps, (2, 0, 1))
    sx = width / float(INPUT_W)
    sy = height / float(INPUT_H)
    points = []
    for idx in range(14):
        x, y, score = _decode_peak(heatmaps[idx])
        points.append(None if x is None else (x * sx, y * sy, score))
    return points


def _best_reference_to_image(points):
    detected = np.asarray(
        [[p[0], p[1]] if p is not None else [np.nan, np.nan] for p in points],
        dtype=np.float32,
    )
    best_matrix = None
    best_error = float("inf")
    for conf in COURT_CONFS:
        if np.isnan(detected[conf]).any():
            continue
        matrix, _ = cv2.findHomography(REFERENCE_KEYPOINTS[conf], detected[conf], method=0)
        if matrix is None:
            continue
        projected = cv2.perspectiveTransform(REFERENCE_KEYPOINTS.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        mask = ~np.isnan(detected[:, 0])
        extra = np.ones(len(points), dtype=bool)
        extra[conf] = False
        eval_mask = mask & extra
        error = 0.0 if not np.any(eval_mask) else float(
            np.mean(np.linalg.norm(projected[eval_mask] - detected[eval_mask], axis=1))
        )
        if error < best_error:
            best_matrix = matrix
            best_error = error
    return best_matrix, best_error


def _image_to_court(reference_to_image):
    outer_reference = REFERENCE_KEYPOINTS[[0, 1, 2, 3]].reshape(-1, 1, 2)
    image_corners = cv2.perspectiveTransform(outer_reference, reference_to_image).reshape(-1, 2)
    court_corners = np.asarray(
        [[0.0, 0.0], [1000.0, 0.0], [0.0, 2168.0], [1000.0, 2168.0]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(image_corners.astype(np.float32), court_corners), image_corners


def _valid_geometry(image_corners, width, height):
    polygon = image_corners[[0, 1, 3, 2]].astype(np.float32)
    area = abs(float(cv2.contourArea(polygon)))
    if area < width * height * 0.06:
        return False
    return not (
        np.any(image_corners[:, 0] < -0.5 * width)
        or np.any(image_corners[:, 0] > 1.5 * width)
        or np.any(image_corners[:, 1] < -0.5 * height)
        or np.any(image_corners[:, 1] > 1.5 * height)
    )


def calibrate_court(video_path, model_path, stride=15, min_points=8, max_error=90.0, max_scans=30):
    from rknnlite.api import RKNNLite

    started = time.perf_counter()
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        raise RuntimeError("failed to load court RKNN: {}".format(model_path))
    core_auto = getattr(RKNNLite, "NPU_CORE_AUTO", None)
    ret = rknn.init_runtime(core_mask=core_auto) if core_auto is not None else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("failed to initialize court RKNN")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rknn.release()
        gc.collect()
        raise RuntimeError("failed to open court calibration video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    best = None
    accepted = None
    frame_id = 0
    scans = 0
    try:
        while total <= 0 or frame_id < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            output = rknn.inference(inputs=[small[None].astype(np.uint8)], data_format=["nhwc"])[0]
            points = _decode_points(output, width, height)
            valid = sum(point is not None for point in points)
            reference_to_image, error = _best_reference_to_image(points)
            scans += 1
            if scans >= max_scans:
                break
            if reference_to_image is not None:
                image_to_court, corners = _image_to_court(reference_to_image)
                geometry_ok = _valid_geometry(corners, width, height)
                candidate = {
                    "frame": frame_id,
                    "points": points,
                    "valid_points": valid,
                    "error_px": error,
                    "reference_to_image": reference_to_image,
                    "image_to_court": image_to_court,
                    "image_corners": corners,
                    "geometry_ok": geometry_ok,
                }
                if best is None or (geometry_ok, valid, -error) > (
                    best["geometry_ok"], best["valid_points"], -best["error_px"]
                ):
                    best = candidate
                if geometry_ok and valid >= min_points and error <= max_error:
                    accepted = candidate
                    break
            frame_id += max(1, stride)
    finally:
        cap.release()
        rknn.release()
        gc.collect()

    if accepted is None:
        return {
            "available": False,
            "scan_frames": scans,
            "best_valid_points": 0 if best is None else best["valid_points"],
            "best_error_px": None if best is None else best["error_px"],
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    accepted.update(
        {
            "available": True,
            "scan_frames": scans,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return accepted


def transform_point(matrix, point):
    if matrix is None or point is None or point[0] is None:
        return None
    src = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)[0][0]
    return float(dst[0]), float(dst[1])


def projected_court_lines(reference_to_image):
    if reference_to_image is None:
        return []
    lines = []
    for start, end in COURT_LINES:
        src = np.asarray([[start, end]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, reference_to_image)[0]
        lines.append((tuple(dst[0]), tuple(dst[1])))
    return lines
