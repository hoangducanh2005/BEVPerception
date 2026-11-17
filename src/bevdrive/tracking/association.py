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


def associate_query_centers(
    track_centers_xy: Sequence[np.ndarray],
    track_classes: Sequence[str],
    track_embeddings: Sequence[np.ndarray | None],
    detection_centers_xy: Sequence[np.ndarray],
    detection_classes: Sequence[str],
    detection_embeddings: Sequence[np.ndarray | None],
    *,
    max_distance_m: float,
    motion_weight: float,
    appearance_weight: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate with gated motion cost plus optional cosine appearance cost."""

    n_tracks = len(track_centers_xy)
    n_detections = len(detection_centers_xy)
    if not n_tracks or not n_detections:
        return [], list(range(n_tracks)), list(range(n_detections))
    cost = np.full((n_tracks, n_detections), 1e6, dtype=np.float64)
    for track_index, (center, class_name, track_embedding) in enumerate(zip(track_centers_xy, track_classes, track_embeddings)):
        for detection_index, (detection, detection_class, detection_embedding) in enumerate(zip(detection_centers_xy, detection_classes, detection_embeddings)):
            if class_name != detection_class:
                continue
            distance = float(np.linalg.norm(np.asarray(center)[:2] - np.asarray(detection)[:2]))
            if distance > max_distance_m:
                continue
            motion_cost = distance / max(max_distance_m, 1e-8)
            appearance_cost = 0.0
            if track_embedding is not None and detection_embedding is not None:
                appearance_cost = 1.0 - float(np.clip(np.dot(track_embedding, detection_embedding), -1.0, 1.0))
            cost[track_index, detection_index] = motion_weight * motion_cost + appearance_weight * appearance_cost
    rows, columns = linear_sum_assignment(cost)
    matches = [(int(row), int(column)) for row, column in zip(rows, columns) if cost[row, column] < 1e6]
    matched_tracks = {row for row, _ in matches}
    matched_detections = {column for _, column in matches}
    return matches, [i for i in range(n_tracks) if i not in matched_tracks], [i for i in range(n_detections) if i not in matched_detections]
