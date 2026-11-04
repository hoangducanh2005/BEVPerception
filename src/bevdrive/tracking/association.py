"""Class-aware Hungarian association for metric 3D detections."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_centers(
    track_centers_xy: Sequence[np.ndarray],
    track_classes: Sequence[str],
    detection_centers_xy: Sequence[np.ndarray],
    detection_classes: Sequence[str],
    *,
    max_distance_m: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate detections to predicted tracks with a class-aware distance gate."""

    n_tracks = len(track_centers_xy)
    n_detections = len(detection_centers_xy)
    if not n_tracks or not n_detections:
        return [], list(range(n_tracks)), list(range(n_detections))
    cost = np.full((n_tracks, n_detections), 1e6, dtype=np.float64)
    for track_index, (center, class_name) in enumerate(zip(track_centers_xy, track_classes)):
        for detection_index, (detection, detection_class) in enumerate(zip(detection_centers_xy, detection_classes)):
            if class_name != detection_class:
                continue
            distance = float(np.linalg.norm(np.asarray(center)[:2] - np.asarray(detection)[:2]))
            if distance <= max_distance_m:
                cost[track_index, detection_index] = distance
    rows, columns = linear_sum_assignment(cost)
    matches = [(int(row), int(column)) for row, column in zip(rows, columns) if cost[row, column] < 1e6]
    matched_tracks = {row for row, _ in matches}
    matched_detections = {column for _, column in matches}
    return matches, [i for i in range(n_tracks) if i not in matched_tracks], [i for i in range(n_detections) if i not in matched_detections]
