"""Baseline A: tracking-by-detection with a Kalman filter + Hungarian association.

BEVFusion boxes -> global frame (ego-motion compensated by construction) ->
per-class constant-velocity Kalman prediction -> class-aware, distance-gated
Hungarian matching (two passes: confident then low-score detections) ->
track lifecycle (birth, update, coasting through occlusion, death).

Identity is decided **only** by geometry here; there is no learned component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import ConstantVelocityKalman

DEFAULT_GATES_M = {"pedestrian": 1.5, "bicycle": 3.0, "motorcycle": 4.0, "car": 4.0, "truck": 4.0, "bus": 5.5, "trailer": 3.0}
DEFAULT_ACCEL_STD = {"pedestrian": 2.0, "bicycle": 3.0, "motorcycle": 4.0, "car": 4.0, "truck": 3.0, "bus": 3.0, "trailer": 3.0}


@dataclass
class KFTrackerConfig:
    classes: tuple[str, ...] = ("bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck")
    det_score_min: float = 0.1  # detections below are ignored entirely
    high_score: float = 0.3  # first association pass / allowed to start tracks
    gates_m: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GATES_M))
    accel_std: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACCEL_STD))
    max_age: int = 3  # frames a track may coast unobserved (occlusion handling)
    min_hits: int = 1  # hits before a track is reported
    use_detector_velocity: bool = True
    position_std: float = 0.5
    velocity_std: float = 1.0
    report_coasted: bool = False
    coast_score_decay: float = 0.5


@dataclass
class _Track:
    track_id: int
    name: str
    kf: ConstantVelocityKalman
    box_lidar: np.ndarray
    score: float
    last_ts: float
    hits: int = 1
    misses: int = 0
    history_global: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TrackOutput:
    track_id: int
    name: str
    score: float
    box_lidar: np.ndarray  # 9-dof, velocity = filtered velocity in LiDAR frame
    velocity_global: np.ndarray
    history_global: np.ndarray
    status: str  # "matched" | "new" | "coasted"


class KFTracker:
    def __init__(self, config: KFTrackerConfig | None = None) -> None:
        self.cfg = config or KFTrackerConfig()
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.scene: str | None = None

    def reset(self, scene: str | None = None) -> None:
        self.tracks = []
        self.scene = scene

    def step(
        self,
        boxes_lidar: np.ndarray,
        scores: np.ndarray,
        names: Sequence[str],
        lidar2global: np.ndarray,
        timestamp: float,
        scene: str,
    ) -> list[TrackOutput]:
        cfg = self.cfg
        if scene != self.scene:
            self.reset(scene)
        rot = lidar2global[:3, :3]
        keep = [i for i, (n, s) in enumerate(zip(names, scores)) if n in cfg.classes and s >= cfg.det_score_min]
        det_boxes = np.asarray(boxes_lidar, dtype=np.float64)[keep].reshape(-1, 9)
        det_scores = np.asarray(scores, dtype=np.float64)[keep]
        det_names = [names[i] for i in keep]
        det_xy = (det_boxes[:, :3] @ rot.T + lidar2global[:3, 3])[:, :2] if len(keep) else np.zeros((0, 2))
        det_v = (np.c_[det_boxes[:, 7:9], np.zeros(len(keep))] @ rot.T)[:, :2] if len(keep) else np.zeros((0, 2))

        for track in self.tracks:
            track.kf.predict(max(timestamp - track.last_ts, 1e-3), process_accel_std=cfg.accel_std.get(track.name, 3.0))
            track.misses += 1
        matched_det: dict[int, int] = {}
        unmatched_tracks = list(range(len(self.tracks)))
        high = [i for i in range(len(det_scores)) if det_scores[i] >= cfg.high_score]
        low = [i for i in range(len(det_scores)) if det_scores[i] < cfg.high_score]
        for pool in (high, low):
            pairs = self._associate(unmatched_tracks, pool, det_xy, det_names)
            for t, d in pairs:
                matched_det[d] = t
            used = {t for t, _ in pairs}
            unmatched_tracks = [t for t in unmatched_tracks if t not in used]

        outputs: list[TrackOutput] = []
        for d, t in matched_det.items():
            track = self.tracks[t]
            if cfg.use_detector_velocity:
                track.kf.update_position_velocity(det_xy[d], det_v[d], position_std=cfg.position_std, velocity_std=cfg.velocity_std)
            else:
                track.kf.update(det_xy[d], measurement_std=cfg.position_std)
            track.box_lidar, track.score, track.name = det_boxes[d].copy(), float(det_scores[d]), det_names[d]
            track.last_ts, track.hits, track.misses = timestamp, track.hits + 1, 0
            track.history_global.append(tuple(track.kf.state[:2]))
            if track.hits >= cfg.min_hits:
                outputs.append(self._output(track, rot, "matched"))
        for d in range(len(det_scores)):
            if d in matched_det or det_scores[d] < cfg.high_score:
                continue
            kf = ConstantVelocityKalman(det_xy[d], position_var=cfg.position_std**2)
            if cfg.use_detector_velocity:
                kf.state[2:] = det_v[d]
                kf.covariance[2:, 2:] = np.eye(2) * cfg.velocity_std**2 * 4
            track = _Track(self.next_id, det_names[d], kf, det_boxes[d].copy(), float(det_scores[d]), timestamp, history_global=[tuple(det_xy[d])])
            self.next_id += 1
            self.tracks.append(track)
            if cfg.min_hits <= 1:
                outputs.append(self._output(track, rot, "new"))
        alive = []
        for track in self.tracks:
            if track.misses > cfg.max_age:
                continue
            alive.append(track)
            if track.misses > 0 and cfg.report_coasted and track.hits >= cfg.min_hits:
                out = self._output(track, rot, "coasted", lidar2global=lidar2global)
                out.score = track.score * cfg.coast_score_decay ** track.misses
                outputs.append(out)
        self.tracks = alive
        return outputs

    def _associate(self, track_idx: list[int], det_idx: list[int], det_xy: np.ndarray, det_names: list[str]) -> list[tuple[int, int]]:
        if not track_idx or not det_idx:
            return []
        big = 1e6
        cost = np.full((len(track_idx), len(det_idx)), big)
        for a, t in enumerate(track_idx):
            track = self.tracks[t]
            gate = self.cfg.gates_m.get(track.name, 3.0)
            for b, d in enumerate(det_idx):
                if det_names[d] != track.name:
                    continue
                dist = float(np.hypot(*(track.kf.state[:2] - det_xy[d])))
                if dist <= gate:
                    cost[a, b] = dist / gate
        rows, cols = linear_sum_assignment(cost)
        return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols) if cost[r, c] < big]

    def _output(self, track: _Track, rot: np.ndarray, status: str, lidar2global: np.ndarray | None = None) -> TrackOutput:
        box = track.box_lidar.copy()
        v_global = track.kf.state[2:].copy()
        box[7:9] = (rot.T @ np.r_[v_global, 0.0])[:2]
        if status == "coasted" and lidar2global is not None:
            p = np.linalg.inv(lidar2global) @ np.r_[track.kf.state[:2], 0.0, 1.0]
            box[:2] = p[:2]
        return TrackOutput(track.track_id, track.name, track.score, box, v_global, np.asarray(track.history_global[-12:]), status)
