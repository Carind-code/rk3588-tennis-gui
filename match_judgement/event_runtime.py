import csv
import json
import os
import sys
import time

import cv2
import numpy as np

from court_runtime import transform_point


VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_v52")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from cleanup import clean_bidirectional_outliers  # noqa: E402
from detector import detect_events, write_event_csv  # noqa: E402
from player import (  # noqa: E402
    MediaPipePoseProvider,
    empty_player_context,
    nearest_player_distance,
    stabilize_player_contexts,
)
from timing_refiner import refine_contact_timing  # noqa: E402


SINGLES_LEFT = 1000.0 * 1.37 / 10.97
SINGLES_RIGHT = 1000.0 - SINGLES_LEFT
COURT_TOP = 0.0
COURT_BOTTOM = 2168.0
COURT_NET = COURT_BOTTOM / 2.0


def build_player_contexts(video_path, crop_json, stride=3, complexity=0):
    started = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("failed to open video for player context")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    contexts = {}
    provider = None
    frame_id = 0
    available = True
    error = ""
    try:
        provider = MediaPipePoseProvider(width, height, complexity, crop_json)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % max(1, stride) == 0:
                contexts[frame_id] = provider.process(frame)
            frame_id += 1
    except Exception as exc:
        available = False
        error = str(exc)
        contexts = {}
    finally:
        if provider is not None:
            provider.close()
        cap.release()
    if contexts:
        contexts = stabilize_player_contexts(contexts, width, height, max_gap=max(3, stride + 1))
    return contexts, {
        "available": available and bool(contexts),
        "sampled_frames": len(contexts),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "error": error,
    }


def _stable_track(tracks, scores):
    accepted = [1 if point is not None and point[0] is not None else 0 for point in tracks]
    normalized = [
        None if point is None or point[0] is None else (float(point[0]), float(point[1]))
        for point in tracks
    ]
    stable = {
        "tracks": normalized,
        "scores": list(scores),
        "accepted": accepted,
        "source": ["accepted" if value else "missing" for value in accepted],
    }
    return stable


def _interpolate_short_gaps(stable, max_gap=4):
    tracks = list(stable["tracks"])
    scores = list(stable["scores"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    idx = 0
    while idx < len(tracks):
        if tracks[idx] is not None:
            idx += 1
            continue
        start = idx
        while idx < len(tracks) and tracks[idx] is None:
            idx += 1
        end = idx - 1
        if end - start + 1 > max_gap or start == 0 or idx >= len(tracks):
            continue
        left = tracks[start - 1]
        right = tracks[idx]
        if left is None or right is None:
            continue
        span = idx - (start - 1)
        for frame in range(start, idx):
            ratio = (frame - (start - 1)) / float(span)
            tracks[frame] = (
                left[0] * (1.0 - ratio) + right[0] * ratio,
                left[1] * (1.0 - ratio) + right[1] * ratio,
            )
            scores[frame] = min(
                scores[start - 1] if scores[start - 1] is not None else 0.0,
                scores[idx] if scores[idx] is not None else 0.0,
            )
            accepted[frame] = 2
            sources[frame] = "interpolated"
    result = dict(stable)
    result.update({"tracks": tracks, "scores": scores, "accepted": accepted, "source": sources})
    return result


def _nearest_context(contexts, frame, max_gap=4):
    if frame in contexts:
        return contexts[frame]
    for gap in range(1, max_gap + 1):
        if frame - gap in contexts:
            return contexts[frame - gap]
        if frame + gap in contexts:
            return contexts[frame + gap]
    return empty_player_context()


def _contact_side(point, ctx, frame_height, threshold=0.18):
    _, near = nearest_player_distance(point, ctx)
    if near < threshold or not ctx.centers:
        return None, near
    center = min(ctx.centers, key=lambda value: (value[0] - point[0]) ** 2 + (value[1] - point[1]) ** 2)
    return ("top" if center[1] < frame_height * 0.5 else "bottom"), near


def infer_contacts(tracks, contexts, frame_height, cooldown=5):
    contacts = []
    last_frame = -10**9
    last_side = None
    for frame, point in enumerate(tracks):
        if point is None:
            continue
        side, score = _contact_side(point, _nearest_context(contexts, frame), frame_height)
        if side is None:
            continue
        if frame - last_frame <= cooldown and side == last_side:
            if contacts and score > contacts[-1]["score"]:
                contacts[-1] = {"frame": frame, "side": side, "score": score}
            continue
        contacts.append({"frame": frame, "side": side, "score": score})
        last_frame = frame
        last_side = side
    return contacts


def _opponent(side):
    if side == "top":
        return "bottom"
    if side == "bottom":
        return "top"
    return None


def _court_side(point):
    if point is None:
        return None
    return "top" if point[1] < COURT_NET else "bottom"


def _inside(point, mode):
    if point is None:
        return None
    x, y = point
    left = 0.0 if mode == "doubles" else SINGLES_LEFT
    right = 1000.0 if mode == "doubles" else SINGLES_RIGHT
    return left <= x <= right and COURT_TOP <= y <= COURT_BOTTOM


def _trajectory_hitter(court_tracks, frame, lookback=8):
    samples = []
    start = max(1, int(frame) - int(lookback))
    for idx in range(start, min(int(frame) + 1, len(court_tracks))):
        previous = court_tracks[idx - 1]
        current = court_tracks[idx]
        if previous is not None and current is not None:
            samples.append(float(current[1] - previous[1]))
    if not samples:
        return None
    velocity_y = float(np.median(samples))
    if abs(velocity_y) < 1.0:
        return None
    # Positive court-y means the ball is travelling from the top player
    # towards the bottom player, so the most recent hitter is on the top side.
    return "top" if velocity_y > 0.0 else "bottom"


def build_judgements(events, contacts, image_to_court, court_tracks, fps, court_mode="singles"):
    judgements = []
    contacts = sorted(contacts, key=lambda value: value["frame"])
    last_hitter = None
    contact_pos = 0
    bounces_since_contact = 0
    last_bounce_side = None
    last_contact_frame = -10**9
    score_locked = False
    last_point_frame = -10**9
    point_statuses = {"OUT_POINT", "OWN_SIDE_POINT_LOST", "DOUBLE_BOUNCE_POINT"}
    for event in sorted(events, key=lambda value: value.frame):
        while contact_pos < len(contacts) and contacts[contact_pos]["frame"] <= event.frame:
            contact_frame = contacts[contact_pos]["frame"]
            if score_locked and contact_frame > last_point_frame:
                score_locked = False
            last_hitter = contacts[contact_pos]["side"]
            last_contact_frame = contact_frame
            bounces_since_contact = 0
            last_bounce_side = None
            contact_pos += 1
        hitter_source = "player_contact" if last_hitter is not None else "unknown"
        inferred_hitter = _trajectory_hitter(court_tracks, event.frame)
        contact_is_stale = event.frame - last_contact_frame > max(5, int(round(fps * 1.5)))
        if inferred_hitter is not None and (last_hitter is None or contact_is_stale):
            if inferred_hitter != last_hitter:
                bounces_since_contact = 0
                last_bounce_side = None
            last_hitter = inferred_hitter
            hitter_source = "trajectory_direction"
        court_point = transform_point(image_to_court, (event.x, event.y))
        inside = _inside(court_point, court_mode)
        bounce_side = _court_side(court_point)
        bounces_since_contact += 1
        status = "UNDECIDED"
        winner = None
        reason = "missing_player_contact" if last_hitter is None else ""
        if inside is False:
            status = "OUT_POINT"
            winner = _opponent(last_hitter)
            reason = "first_bounce_out"
        elif last_hitter is not None and bounce_side == last_hitter:
            status = "OWN_SIDE_POINT_LOST"
            winner = _opponent(last_hitter)
            reason = "ball_bounced_on_hitter_side"
        elif inside is True and bounces_since_contact >= 2 and bounce_side == last_bounce_side:
            status = "DOUBLE_BOUNCE_POINT"
            winner = last_hitter
            reason = "second_bounce_before_return"
        elif inside is True:
            status = "IN_CONTINUE"
            reason = "first_bounce_in"
        point_candidate = status in point_statuses and winner in ("top", "bottom")
        point_awarded = bool(point_candidate and not score_locked)
        if point_awarded:
            score_locked = True
            last_point_frame = event.frame
        judgements.append(
            {
                "frame": event.frame,
                "image_x": round(event.x, 3),
                "image_y": round(event.y, 3),
                "court_x": None if court_point is None else round(court_point[0], 3),
                "court_y": None if court_point is None else round(court_point[1], 3),
                "bounce_side": bounce_side,
                "last_hitter": last_hitter,
                "hitter_source": hitter_source,
                "inside": inside,
                "status": status,
                "winner": winner,
                "reason": reason,
                "point_awarded": point_awarded,
                "score_suppressed": bool(point_candidate and not point_awarded),
                "bounce_score": round(float(event.score), 6),
                "timing_confidence": round(float(event.timing_confidence), 6),
            }
        )
        last_bounce_side = bounce_side
    return judgements


def attach_running_scores(judgements):
    scores = {"top": 0, "bottom": 0}
    undecided = 0
    point_statuses = {"OUT_POINT", "OWN_SIDE_POINT_LOST", "DOUBLE_BOUNCE_POINT"}
    for item in judgements:
        before = dict(scores)
        winner = item.get("winner")
        point_candidate = item.get("status") in point_statuses
        score_changed = bool(item.get("point_awarded", point_candidate and winner in scores))
        if score_changed and winner in scores:
            scores[winner] += 1
        elif point_candidate and winner not in scores:
            undecided += 1
        item["score_before"] = before
        item["score_after"] = dict(scores)
        item["score_changed"] = bool(score_changed and winner in scores)
    return {"top": scores["top"], "bottom": scores["bottom"], "undecided": undecided}


def analyze_events(
    video_path,
    tracks,
    scores,
    fps,
    width,
    height,
    court_calibration,
    crop_json,
    event_csv_path,
    judgement_json_path,
    court_mode="singles",
    use_mediapipe=True,
    mediapipe_stride=1,
    tracking_stable=None,
):
    started = time.perf_counter()
    if use_mediapipe:
        contexts, player_stats = build_player_contexts(video_path, crop_json, stride=mediapipe_stride)
    else:
        contexts = {}
        player_stats = {
            "available": False,
            "sampled_frames": 0,
            "elapsed_ms": 0.0,
            "error": "disabled",
        }
    stable = dict(tracking_stable) if tracking_stable is not None else _interpolate_short_gaps(
        _stable_track(tracks, scores), max_gap=4
    )
    stable = clean_bidirectional_outliers(stable, fps)
    matrix = court_calibration.get("image_to_court") if court_calibration.get("available") else None
    court_tracks = [transform_point(matrix, point) for point in stable["tracks"]]
    court_quality = 0.95 if matrix is not None else 0.0
    court_qualities = [court_quality] * len(stable["tracks"])
    events = detect_events(
        stable["tracks"],
        stable["scores"],
        contexts,
        [],
        fps,
        court_tracks=court_tracks,
        court_qualities=court_qualities,
        bounce_threshold=0.55,
        min_event_gap_seconds=0.60,
        event_mode="v37_bounce",
        frame_width=width,
        frame_height=height,
        contact_threshold=0.18,
        min_flight_seconds=0.15,
        suppress_serve_toss=True,
        track_sources=stable["source"],
        accepted_flags=stable["accepted"],
    )
    events = refine_contact_timing(
        events,
        stable["tracks"],
        stable["accepted"],
        stable["source"],
        court_tracks,
        fps,
    )
    write_event_csv(event_csv_path, events)
    contacts = infer_contacts(stable["tracks"], contexts, height)
    judgements = build_judgements(
        events,
        contacts,
        matrix,
        court_tracks,
        fps,
        court_mode=court_mode,
    )
    point_summary = attach_running_scores(judgements)
    payload = {
        "court_mode": court_mode,
        "court_available": bool(matrix is not None),
        "player_context": player_stats,
        "contacts": contacts,
        "bounce_count": len(events),
        "point_summary": point_summary,
        "judgements": judgements,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    with open(judgement_json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return events, payload, stable


def append_court_columns(csv_path, tracks, court_matrix):
    temp_path = csv_path + ".court.tmp"
    with open(csv_path, "r", newline="", encoding="utf-8") as src, open(
        temp_path, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or []) + ["court_x", "court_y"]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(reader):
            point = tracks[idx] if idx < len(tracks) else None
            row["x"] = "" if point is None or point[0] is None else "{:.3f}".format(point[0])
            row["y"] = "" if point is None or point[0] is None else "{:.3f}".format(point[1])
            court_point = transform_point(court_matrix, point)
            row["court_x"] = "" if court_point is None else "{:.3f}".format(court_point[0])
            row["court_y"] = "" if court_point is None else "{:.3f}".format(court_point[1])
            writer.writerow(row)
    os.replace(temp_path, csv_path)
