"""Convert LiDAR-frame boxes to official nuScenes detection/tracking submissions.

Box layout in: ``x, y, z_bottom, dx(length), dy(width), dz(height), yaw, vx, vy``
in the keyframe LiDAR frame (mmdet3d >= 1.0 convention). The conversion matches
``mmdet3d.evaluation.metrics.nuscenes_metric.output_to_nusc_box`` (wlh = dy, dx, dz;
gravity-centre z) followed by LiDAR -> ego -> global.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_ATTRIBUTE = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}


def attribute_for(name: str, velocity_xy: Sequence[float]) -> str:
    """Same speed-based attribute heuristic as mmdet3d's NuScenesMetric."""

    speed = math.hypot(float(velocity_xy[0]), float(velocity_xy[1]))
    if speed > 0.2:
        if name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
            return "vehicle.moving"
        if name in ("bicycle", "motorcycle"):
            return "cycle.with_rider"
    return DEFAULT_ATTRIBUTE[name]


def lidar_boxes_to_global(boxes: np.ndarray, lidar2global: np.ndarray) -> list[dict[str, Any]]:
    """Return translation/size/rotation/velocity dicts in the global frame."""

    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
    rotation = Quaternion(matrix=lidar2global[:3, :3], atol=1e-6)
    out = []
    for row in boxes:
        x, y, z, dx, dy, dz, yaw, vx, vy = row
        box = Box(
            center=[x, y, z + dz / 2.0],
            size=[dy, dx, dz],
            orientation=Quaternion(axis=[0, 0, 1], radians=yaw),
            velocity=(vx, vy, 0.0),
        )
        box.rotate(rotation)
        box.translate(lidar2global[:3, 3])
        out.append(
            {
                "translation": [float(v) for v in box.center],
                "size": [float(v) for v in box.wlh],
                "rotation": [float(v) for v in box.orientation.elements],
                "velocity": [float(box.velocity[0]), float(box.velocity[1])],
            }
        )
    return out


def detection_entries(
    sample_token: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    names: Sequence[str],
    lidar2global: np.ndarray,
    *,
    keep_classes: Iterable[str] | None = None,
    max_boxes: int = 500,
) -> list[dict[str, Any]]:
    keep = set(keep_classes) if keep_classes is not None else None
    order = np.argsort(-np.asarray(scores))[:max_boxes]
    globals_ = lidar_boxes_to_global(np.asarray(boxes)[order], lidar2global)
    entries = []
    for idx, g in zip(order, globals_):
        name = str(names[idx])
        if keep is not None and name not in keep:
            continue
        entries.append(
            {
                "sample_token": sample_token,
                **g,
                "detection_name": name,
                "detection_score": float(scores[idx]),
                "attribute_name": attribute_for(name, g["velocity"]),
            }
        )
    return entries


def tracking_entries(
    sample_token: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    names: Sequence[str],
    track_ids: Sequence[int],
    lidar2global: np.ndarray,
    *,
    keep_classes: Iterable[str],
) -> list[dict[str, Any]]:
    keep = set(keep_classes)
    globals_ = lidar_boxes_to_global(np.asarray(boxes), lidar2global)
    entries = []
    for g, score, name, tid in zip(globals_, scores, names, track_ids):
        if str(name) not in keep:
            continue
        entries.append(
            {
                "sample_token": sample_token,
                **g,
                "tracking_name": str(name),
                "tracking_score": float(score),
                "tracking_id": str(int(tid)),
            }
        )
    return entries


def write_submission(path: Path, results: dict[str, list[dict[str, Any]]], *, use_lidar: bool = True, use_camera: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"use_camera": use_camera, "use_lidar": use_lidar, "use_radar": False, "use_map": False, "use_external": False}
    with path.open("w") as handle:
        json.dump({"meta": meta, "results": results}, handle)
