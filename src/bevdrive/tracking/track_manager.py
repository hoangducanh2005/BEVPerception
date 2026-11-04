"""Online scene-safe 3D tracking baseline and temporal-query state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .association import associate_centers
from .kalman import ConstantVelocityKalman


@dataclass(frozen=True, slots=True)
class DetectionMeasurement:
    """One detector output expressed in a metric world/odometry frame."""

    class_name: str
    score: float
    center_world_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class TrackAssignment:
    """Association result for one detection at the current timestamp."""

    detection_index: int
    track_id: int
    class_name: str
    center_world_xyz: tuple[float, float, float]
    velocity_world_xy_mps: tuple[float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float
    score: float
    age: int
    hits: int
    status: str
    history_world_xy: tuple[tuple[float, float], ...]
    query_state_norm: float | None = None


@dataclass(slots=True)
class _Track:
    track_id: int
    class_name: str
    score: float
    size_lwh: tuple[float, float, float]
    yaw_rad: float
    filter: ConstantVelocityKalman
    last_timestamp_us: int
    age: int = 1
    hits: int = 1
    missed: int = 0
    history_world_xy: list[tuple[float, float]] = field(default_factory=list)
    query_state: np.ndarray | None = None


class TrackManager:
    """Class-aware online tracker with explicit lifecycle and scene reset."""

    def __init__(
        self,
        *,
        max_distance_m: float = 3.0,
        max_missed_frames: int = 3,
        min_score: float = 0.2,
        history_size: int = 12,
        query_ema: float | None = None,
    ) -> None:
        self.max_distance_m = float(max_distance_m)
        self.max_missed_frames = int(max_missed_frames)
        self.min_score = float(min_score)
        self.history_size = int(history_size)
        self.query_ema = query_ema
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.scene_id: str | None = None

    def reset(self, scene_id: str | None = None) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.scene_id = scene_id

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        detections: Sequence[DetectionMeasurement],
        *,
        timestamp_us: int,
        scene_id: str,
    ) -> list[TrackAssignment]:
        if self.scene_id != scene_id:
            self.reset(scene_id)
        current_timestamp = int(timestamp_us)
        valid_detections = [d for d in detections if float(d.score) >= self.min_score]
        tracks = list(self._tracks.values())
        for track in tracks:
            dt = max((current_timestamp - track.last_timestamp_us) / 1_000_000.0, 1e-3)
            track.filter.predict(dt)
            track.age += 1
            track.missed += 1

        track_centers = [track.filter.state[:2] for track in tracks]
        track_classes = [track.class_name for track in tracks]
        detection_centers = [np.asarray(d.center_world_xyz[:2], dtype=np.float64) for d in valid_detections]
        detection_classes = [d.class_name for d in valid_detections]
        matches, unmatched_tracks, unmatched_detections = associate_centers(
            track_centers,
            track_classes,
            detection_centers,
            detection_classes,
            max_distance_m=self.max_distance_m,
        )
        assignments: dict[int, TrackAssignment] = {}
        for track_index, detection_index in matches:
            track = tracks[track_index]
            detection = valid_detections[detection_index]
            track.filter.update(detection_centers[detection_index])
            if self.query_ema is not None:
                measurement_query = self._query_vector(detection)
                if track.query_state is None:
                    track.query_state = measurement_query
                else:
                    alpha = float(np.clip(self.query_ema, 0.0, 1.0))
                    track.query_state = alpha * track.query_state + (1.0 - alpha) * measurement_query
            track.class_name = detection.class_name
            track.score = float(detection.score)
            track.size_lwh = detection.size_lwh
            track.yaw_rad = float(detection.yaw_rad)
            track.last_timestamp_us = current_timestamp
            track.hits += 1
            track.missed = 0
            center = track.filter.state[:2].copy()
            track.history_world_xy.append((float(center[0]), float(center[1])))
            track.history_world_xy = track.history_world_xy[-self.history_size :]
            assignments[detection_index] = self._assignment(track, detection_index, detection, status="matched")

        for track_index in unmatched_tracks:
            track = tracks[track_index]
            if track.missed > self.max_missed_frames:
                self._tracks.pop(track.track_id, None)

        for detection_index in unmatched_detections:
            detection = valid_detections[detection_index]
            position = np.asarray(detection.center_world_xyz[:2], dtype=np.float64)
            track = _Track(
                track_id=self._next_id,
                class_name=detection.class_name,
                score=float(detection.score),
                size_lwh=detection.size_lwh,
                yaw_rad=float(detection.yaw_rad),
                filter=ConstantVelocityKalman(position),
                last_timestamp_us=current_timestamp,
                history_world_xy=[(float(position[0]), float(position[1]))],
                query_state=self._query_vector(detection) if self.query_ema is not None else None,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1
            assignments[detection_index] = self._assignment(track, detection_index, detection, status="new")

        # Detections were filtered by score, so map assignments back to the caller's indices.
        result: list[TrackAssignment] = []
        valid_index = 0
        for original_index, detection in enumerate(detections):
            if float(detection.score) < self.min_score:
                continue
            item = assignments[valid_index]
            result.append(
                TrackAssignment(
                    detection_index=original_index,
                    track_id=item.track_id,
                    class_name=item.class_name,
                    center_world_xyz=item.center_world_xyz,
                    velocity_world_xy_mps=item.velocity_world_xy_mps,
                    size_lwh=item.size_lwh,
                    yaw_rad=item.yaw_rad,
                    score=item.score,
                    age=item.age,
                    hits=item.hits,
                    status=item.status,
                    history_world_xy=item.history_world_xy,
                    query_state_norm=item.query_state_norm,
                )
            )
            valid_index += 1
        return result

    def _assignment(self, track: _Track, index: int, detection: DetectionMeasurement, *, status: str) -> TrackAssignment:
        state = track.filter.state
        return TrackAssignment(
            detection_index=index,
            track_id=track.track_id,
            class_name=detection.class_name,
            center_world_xyz=(float(state[0]), float(state[1]), float(detection.center_world_xyz[2])),
            velocity_world_xy_mps=(float(state[2]), float(state[3])),
            size_lwh=detection.size_lwh,
            yaw_rad=float(detection.yaw_rad),
            score=float(detection.score),
            age=track.age,
            hits=track.hits,
            status=status,
            history_world_xy=tuple(track.history_world_xy),
            query_state_norm=float(np.linalg.norm(track.query_state)) if track.query_state is not None else None,
        )

    @staticmethod
    def _query_vector(detection: DetectionMeasurement) -> np.ndarray:
        """Compact state used by the Phase 2 temporal-query prototype.

        This is deliberately an explicit non-neural query state. It validates
        the persistent-query interface before a learned TrackFormer head is
        trained on BEV features.
        """

        x, y, z = detection.center_world_xyz
        length, width, height = detection.size_lwh
        return np.asarray(
            [x, y, z, length, width, height, np.sin(detection.yaw_rad), np.cos(detection.yaw_rad), detection.score],
            dtype=np.float64,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active_tracks": len(self._tracks),
            "next_track_id": self._next_id,
            "track_ages": {str(t.track_id): t.age for t in self._tracks.values()},
            "track_hits": {str(t.track_id): t.hits for t in self._tracks.values()},
        }
