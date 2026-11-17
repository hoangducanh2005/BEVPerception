"""Optional non-neural appearance-assisted association for frozen BEV features."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .association import associate_query_centers
from .track_manager import DetectionMeasurement, TrackAssignment, TrackManager, _Track


class QueryAssociationTrackManager(TrackManager):
    """TrackManager variant with motion-gated BEV query similarity as a cost term.

    The default ``TrackManager`` remains unchanged. This class is an optional
    inference-only ablation and never uses ground-truth identities.
    """

    def __init__(
        self,
        *,
        max_distance_m: float = 3.0,
        max_missed_frames: int = 3,
        min_score: float = 0.2,
        history_size: int = 12,
        query_ema: float = 0.7,
        motion_weight: float = 0.7,
        appearance_weight: float = 0.3,
    ) -> None:
        super().__init__(max_distance_m=max_distance_m, max_missed_frames=max_missed_frames, min_score=min_score, history_size=history_size, query_ema=query_ema)
        total = float(motion_weight) + float(appearance_weight)
        if total <= 0.0:
            raise ValueError("motion_weight + appearance_weight must be positive")
        self.motion_weight = float(motion_weight) / total
        self.appearance_weight = float(appearance_weight) / total

    def update_with_embeddings(
        self,
        detections: Sequence[DetectionMeasurement],
        embeddings: Sequence[np.ndarray | None],
        *,
        timestamp_us: int,
        scene_id: str,
    ) -> list[TrackAssignment]:
        if len(embeddings) != len(detections):
            raise ValueError("embeddings must have one entry per detection")
        if self.scene_id != scene_id:
            self.reset(scene_id)
        current_timestamp = int(timestamp_us)
        valid_pairs = [(index, detection, embeddings[index]) for index, detection in enumerate(detections) if float(detection.score) >= self.min_score]
        tracks = list(self._tracks.values())
        for track in tracks:
            dt = max((current_timestamp - track.last_timestamp_us) / 1_000_000.0, 1e-3)
            track.filter.predict(dt)
            track.age += 1
            track.missed += 1
        track_centers = [track.filter.state[:2] for track in tracks]
        track_classes = [track.class_name for track in tracks]
        track_embeddings = [track.query_state for track in tracks]
        detection_centers = [np.asarray(detection.center_world_xyz[:2], dtype=np.float64) for _, detection, _ in valid_pairs]
        detection_classes = [detection.class_name for _, detection, _ in valid_pairs]
        detection_embeddings = [self._normalize_embedding(embedding) for _, _, embedding in valid_pairs]
        matches, unmatched_tracks, unmatched_detections = associate_query_centers(
            track_centers,
            track_classes,
            track_embeddings,
            detection_centers,
            detection_classes,
            detection_embeddings,
            max_distance_m=self.max_distance_m,
            motion_weight=self.motion_weight,
            appearance_weight=self.appearance_weight,
        )
        assignments: dict[int, TrackAssignment] = {}
        for track_index, detection_index in matches:
            track = tracks[track_index]
            original_index, detection, _ = valid_pairs[detection_index]
            track.filter.update(detection_centers[detection_index])
            embedding = detection_embeddings[detection_index]
            if embedding is not None:
                if track.query_state is None:
                    track.query_state = embedding
                else:
                    alpha = float(np.clip(self.query_ema or 0.7, 0.0, 1.0))
                    track.query_state = self._normalize_embedding(alpha * track.query_state + (1.0 - alpha) * embedding)
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
            assignments[original_index] = self._assignment(track, original_index, detection, status="matched")
        for track_index in unmatched_tracks:
            track = tracks[track_index]
            if track.missed > self.max_missed_frames:
                self._tracks.pop(track.track_id, None)
        for detection_index in unmatched_detections:
            original_index, detection, _ = valid_pairs[detection_index]
            position = np.asarray(detection.center_world_xyz[:2], dtype=np.float64)
            embedding = detection_embeddings[detection_index]
            track = _Track(
                track_id=self._next_id,
                class_name=detection.class_name,
                score=float(detection.score),
                size_lwh=detection.size_lwh,
                yaw_rad=float(detection.yaw_rad),
                filter=self._new_filter(position),
                last_timestamp_us=current_timestamp,
                history_world_xy=[(float(position[0]), float(position[1]))],
                query_state=embedding,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1
            assignments[original_index] = self._assignment(track, original_index, detection, status="new")
        return [assignments[index] for index, detection in enumerate(detections) if float(detection.score) >= self.min_score]

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None

    @staticmethod
    def _new_filter(position: np.ndarray) -> Any:
        from .kalman import ConstantVelocityKalman

        return ConstantVelocityKalman(position)
