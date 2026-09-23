import math


RAW_SCORE_THRESHOLD = 0.18
INTERP_GAP = 10
MAX_INTERP_STEP = 260.0
SMOOTH_WINDOW = 9
RENDER_FRAME_LEAD = 2
CONSISTENCY_JUMP_DISTANCE = 200.0
CONSISTENCY_X_FRACTION = 0.36
CONSISTENCY_MIN_SEGMENT_LEN = 2
CONSISTENCY_PRE_RESEED_HIDE_MAX_LEN = 12
CONSISTENCY_RESEED_CONFIRM_LEN = 4
CONSISTENCY_RESEED_GAP = 3
CONSISTENCY_SEGMENT_GAP = 3
BREAK_JUMP_DISTANCE = 200.0
BREAK_X_FRACTION = 0.42
EDGE_ZONE_FRACTION = 0.18


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _raw_tracks(candidate_rows):
    tracks = []
    scores = []
    source = []
    for candidates in candidate_rows:
        chosen = next(
            (candidate for candidate in candidates if candidate["score"] >= RAW_SCORE_THRESHOLD),
            None,
        )
        if chosen is None:
            tracks.append(None)
            scores.append(None)
            source.append("missing")
        else:
            tracks.append((chosen["x"], chosen["y"]))
            scores.append(chosen["score"])
            source.append("raw")
    return {"tracks": tracks, "scores": scores, "source": source}


def _is_jump(previous, current, frame_gap, width):
    if previous is None or current is None:
        return False
    dt = max(1, int(frame_gap))
    dx = abs(float(current[0]) - float(previous[0]))
    step = _distance(previous, current)
    if step >= CONSISTENCY_JUMP_DISTANCE * dt:
        return True
    return dx >= float(width) * CONSISTENCY_X_FRACTION and step >= CONSISTENCY_JUMP_DISTANCE


def _consistency_filter(stable, width):
    tracks = list(stable["tracks"])
    scores = list(stable["scores"])
    source = list(stable["source"])
    breaks = [False] * len(tracks)
    segments = []
    current = []
    previous_frame = None
    previous_point = None

    for frame_id, point in enumerate(tracks):
        if point is None:
            continue
        frame_gap = 0 if previous_frame is None else frame_id - previous_frame
        gap_break = previous_frame is not None and frame_gap > CONSISTENCY_SEGMENT_GAP
        jump_break = previous_point is not None and _is_jump(
            previous_point, point, frame_gap, width
        )
        if gap_break or jump_break:
            if current:
                segments.append(current)
            current = [frame_id]
            breaks[frame_id] = True
        else:
            current.append(frame_id)
        previous_frame = frame_id
        previous_point = point
    if current:
        segments.append(current)

    hidden = set()
    for index, segment in enumerate(segments):
        length = len(segment)
        if length < CONSISTENCY_MIN_SEGMENT_LEN:
            hidden.update(segment)
            continue
        if index + 1 >= len(segments):
            continue
        next_segment = segments[index + 1]
        gap = next_segment[0] - segment[-1]
        jump = _distance(tracks[segment[-1]], tracks[next_segment[0]])
        if (
            length <= CONSISTENCY_PRE_RESEED_HIDE_MAX_LEN
            and len(next_segment) >= CONSISTENCY_RESEED_CONFIRM_LEN
            and gap <= CONSISTENCY_RESEED_GAP
            and jump >= CONSISTENCY_JUMP_DISTANCE
        ):
            hidden.update(segment)

    for frame_id in hidden:
        tracks[frame_id] = None
        scores[frame_id] = None
        source[frame_id] = "filtered_consistency"
        if frame_id + 1 < len(breaks):
            breaks[frame_id + 1] = True

    return {
        **stable,
        "tracks": tracks,
        "scores": scores,
        "source": source,
        "breaks": breaks,
        "filtered_count": len(hidden),
        "segments_before_filter": len(segments),
    }


def _interpolate(stable):
    tracks = list(stable["tracks"])
    scores = list(stable["scores"])
    source = list(stable["source"])
    breaks = list(stable["breaks"])
    index = 0
    while index < len(tracks):
        if tracks[index] is not None:
            index += 1
            continue
        start = index
        while index < len(tracks) and tracks[index] is None:
            index += 1
        end = index - 1
        gap = end - start + 1
        previous = start - 1
        following = index
        if gap > INTERP_GAP or previous < 0 or following >= len(tracks):
            continue
        if tracks[previous] is None or tracks[following] is None or breaks[following]:
            continue
        if any(source[frame_id].startswith("filtered") for frame_id in range(start, end + 1)):
            continue
        if _distance(tracks[previous], tracks[following]) > MAX_INTERP_STEP * (gap + 1):
            continue
        span = following - previous
        score0 = scores[previous] or 0.0
        score1 = scores[following] or score0
        for frame_id in range(start, end + 1):
            ratio = (frame_id - previous) / float(span)
            tracks[frame_id] = (
                tracks[previous][0] * (1.0 - ratio) + tracks[following][0] * ratio,
                tracks[previous][1] * (1.0 - ratio) + tracks[following][1] * ratio,
            )
            scores[frame_id] = score0 * (1.0 - ratio) + score1 * ratio
            source[frame_id] = "interpolated"
    return {**stable, "tracks": tracks, "scores": scores, "source": source}


def _smooth(tracks, breaks):
    half = max(1, SMOOTH_WINDOW // 2)
    smoothed = list(tracks)
    index = 0
    while index < len(tracks):
        if tracks[index] is None:
            index += 1
            continue
        start = index
        while index < len(tracks) and tracks[index] is not None:
            if index > start and breaks[index]:
                break
            index += 1
        end = index
        if end - start >= 3:
            for frame_id in range(start, end):
                left = max(start, frame_id - half)
                right = min(end, frame_id + half + 1)
                weighted_x = weighted_y = total_weight = 0.0
                for neighbor in range(left, right):
                    weight = float(half + 1 - abs(neighbor - frame_id))
                    weighted_x += tracks[neighbor][0] * weight
                    weighted_y += tracks[neighbor][1] * weight
                    total_weight += weight
                smoothed[frame_id] = (weighted_x / total_weight, weighted_y / total_weight)
        if index < len(tracks) and tracks[index] is not None and breaks[index]:
            continue
    return smoothed


def _should_break(previous, current, width):
    if previous is None or current is None:
        return True
    dx = abs(float(current[0]) - float(previous[0]))
    dy = abs(float(current[1]) - float(previous[1]))
    if math.hypot(dx, dy) >= BREAK_JUMP_DISTANCE:
        return True
    if dx >= float(width) * BREAK_X_FRACTION:
        return True
    left_zone = float(width) * EDGE_ZONE_FRACTION
    right_zone = float(width) * (1.0 - EDGE_ZONE_FRACTION)
    return bool(
        (previous[0] <= left_zone and current[0] >= right_zone)
        or (previous[0] >= right_zone and current[0] <= left_zone)
    )


def _shift(values, lead, fill=None):
    return [values[index + lead] if index + lead < len(values) else fill for index in range(len(values))]


def process_sideview_candidates(candidate_rows, width):
    stable = _raw_tracks(candidate_rows)
    stable["raw_tracks"] = list(stable["tracks"])
    stable = _consistency_filter(stable, width)
    stable = _interpolate(stable)
    render_tracks = _smooth(stable["tracks"], stable["breaks"])
    render_breaks = list(stable["breaks"])
    previous = None
    for frame_id, point in enumerate(render_tracks):
        if point is None:
            previous = None
            continue
        if previous is not None and _should_break(previous, point, width):
            render_breaks[frame_id] = True
        previous = point

    stable["unshifted_tracks"] = render_tracks
    stable["tracks"] = _shift(render_tracks, RENDER_FRAME_LEAD)
    stable["scores"] = _shift(stable["scores"], RENDER_FRAME_LEAD)
    stable["source"] = _shift(stable["source"], RENDER_FRAME_LEAD, "missing")
    stable["breaks"] = _shift(render_breaks, RENDER_FRAME_LEAD, True)
    stable["profile"] = "sideview_v6"
    stable["render_frame_lead"] = RENDER_FRAME_LEAD
    return stable
