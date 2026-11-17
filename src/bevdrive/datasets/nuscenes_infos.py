"""Build per-keyframe nuScenes records for BEVFusion inference and tracking.

The records follow the mmdet3d v1.x "v2 info" conventions that the BEVFusion
test pipeline consumes (``images``/``lidar_points``/``lidar_sweeps``), plus
ground truth, instance tokens and poses needed by tracking and evaluation.

Fixes relative to checkpoint 1 (see reports/phase_00_audit.md):

* ``lidar_sweeps`` are populated (9 previous sweeps + keyframe = 10, matching
  the BEVFusion training config). Without them mmdet3d silently duplicates
  the keyframe because ``pad_empty_sweeps=True``.
* ``lidar2cam`` composes through each camera's *own* ego pose, so the
  ~0-80 ms LiDAR/camera time offset is compensated exactly as in the
  official mmdet3d nuScenes converter.
* Timestamps fed to the pipeline are in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# mmdet3d / BEVFusion nuScenes class order (label index -> name).
DETECTION_CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
TRACKING_CLASSES: tuple[str, ...] = ("bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck")
VRU_CLASSES: tuple[str, ...] = ("pedestrian", "bicycle", "motorcycle")
CAMERA_NAMES: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def rigid(rotation_wxyz: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform from a nuScenes quaternion and translation."""

    from pyquaternion import Quaternion

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(list(rotation_wxyz)).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def sensor_to_global(nusc: Any, sample_data_token: str) -> np.ndarray:
    """sensor -> ego(t_sensor) -> global for one sample_data record."""

    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return rigid(pose["rotation"], pose["translation"]) @ rigid(cs["rotation"], cs["translation"])


def detection_name(category_name: str) -> str | None:
    from nuscenes.eval.detection.utils import category_to_detection_name

    return category_to_detection_name(category_name)


@dataclass(frozen=True, slots=True)
class SceneSelection:
    version: str
    data_root: Path
    scene_names: tuple[str, ...]


def resolve_scenes(nusc: Any, split_or_scenes: Iterable[str]) -> list[str]:
    """Expand split names (``mini_val``, ``val`` ...) and/or explicit scene names."""

    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    available = {scene["name"] for scene in nusc.scene}
    names: list[str] = []
    for item in split_or_scenes:
        expanded = splits[item] if item in splits else [item]
        names.extend(name for name in expanded if name in available)
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def scene_is_complete(nusc: Any, data_root: Path, scene_name: str, max_sweeps: int = 9) -> bool:
    """True when every keyframe image/LiDAR file and its sweeps exist on disk."""

    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        for channel in ("LIDAR_TOP", *CAMERA_NAMES):
            sd = nusc.get("sample_data", sample["data"][channel])
            if not (data_root / sd["filename"]).is_file():
                return False
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        for _ in range(max_sweeps):
            if not sd["prev"]:
                break
            sd = nusc.get("sample_data", sd["prev"])
            if not (data_root / sd["filename"]).is_file():
                return False
        token = sample["next"]
    return True


def _lidar_sweeps(nusc: Any, data_root: Path, lidar_sd_token: str, global_to_lidar: np.ndarray, max_sweeps: int) -> list[dict[str, Any]]:
    sweeps: list[dict[str, Any]] = []
    sd = nusc.get("sample_data", lidar_sd_token)
    while len(sweeps) < max_sweeps and sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
        sweep_to_lidar = global_to_lidar @ sensor_to_global(nusc, sd["token"])
        rotation = sweep_to_lidar[:3, :3]
        translation = sweep_to_lidar[:3, 3]
        # mmdet3d LoadPointsFromMultiSweeps applies  p @ M[:3,:3] - M[:3,3].
        # Encode M so that this equals the exact rigid transform R p + t.
        lidar2sensor = np.eye(4, dtype=np.float64)
        lidar2sensor[:3, :3] = rotation.T
        lidar2sensor[:3, 3] = -translation
        sweeps.append(
            {
                "lidar_points": {
                    "lidar_path": str(data_root / sd["filename"]),
                    "lidar2sensor": lidar2sensor.astype(np.float32),
                },
                "timestamp": sd["timestamp"] / 1e6,
                "sweep_to_lidar": sweep_to_lidar,
            }
        )
    return sweeps


def build_frame_records(
    nusc: Any,
    data_root: Path | str,
    scene_names: Sequence[str],
    *,
    max_sweeps: int = 9,
    frame_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return one record per keyframe, in temporal order, scene by scene."""

    data_root = Path(data_root)
    scenes = {scene["name"]: scene for scene in nusc.scene}
    records: list[dict[str, Any]] = []
    for scene_name in scene_names:
        scene = scenes[scene_name]
        token = scene["first_sample_token"]
        frame_index = 0
        while token and (frame_limit is None or frame_index < frame_limit):
            sample = nusc.get("sample", token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_sd = nusc.get("sample_data", lidar_token)
            lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            lidar2ego = rigid(lidar_cs["rotation"], lidar_cs["translation"])
            ego2global = rigid(lidar_pose["rotation"], lidar_pose["translation"])
            lidar2global = ego2global @ lidar2ego
            global2lidar = np.linalg.inv(lidar2global)

            images: dict[str, dict[str, Any]] = {}
            for camera in CAMERA_NAMES:
                cam_sd = nusc.get("sample_data", sample["data"][camera])
                cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
                cam2global = sensor_to_global(nusc, cam_sd["token"])
                lidar2cam = np.linalg.inv(cam2global) @ lidar2global
                images[camera] = {
                    "img_path": str(data_root / cam_sd["filename"]),
                    "cam2img": np.asarray(cam_cs["camera_intrinsic"], dtype=np.float32),
                    "lidar2cam": lidar2cam.astype(np.float32),
                    "timestamp": cam_sd["timestamp"] / 1e6,
                    "width": int(cam_sd["width"]),
                    "height": int(cam_sd["height"]),
                }

            gt = _ground_truth(nusc, sample, lidar_token, global2lidar)
            records.append(
                {
                    "token": sample["token"],
                    "scene_token": scene["token"],
                    "scene_name": scene_name,
                    "frame_index": frame_index,
                    "timestamp": sample["timestamp"] / 1e6,
                    "timestamp_us": int(sample["timestamp"]),
                    "lidar_timestamp": lidar_sd["timestamp"] / 1e6,
                    "lidar_path": str(data_root / lidar_sd["filename"]),
                    "lidar2ego": lidar2ego,
                    "ego2global": ego2global,
                    "lidar2global": lidar2global,
                    "images": images,
                    "lidar_sweeps": _lidar_sweeps(nusc, data_root, lidar_token, global2lidar, max_sweeps),
                    "gt": gt,
                    "prev_token": sample["prev"] or None,
                    "next_token": sample["next"] or None,
                }
            )
            token = sample["next"]
            frame_index += 1
    return records


def _ground_truth(nusc: Any, sample: dict[str, Any], lidar_token: str, global2lidar: np.ndarray) -> dict[str, np.ndarray]:
    """GT boxes in the keyframe LiDAR frame, mmdet3d convention.

    Box layout: ``x, y, z_bottom, dx(length), dy(width), dz(height), yaw, vx, vy``
    (velocity rotated from global into the LiDAR frame, NaN -> 0 with a mask).
    """

    rows, names, instance_tokens, ann_tokens, num_pts, visibility, vel_valid = [], [], [], [], [], [], []
    _, boxes, _ = nusc.get_sample_data(lidar_token)  # devkit boxes already in LiDAR frame
    box_by_token = {box.token: box for box in boxes}
    rot = global2lidar[:3, :3]
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        name = detection_name(ann["category_name"])
        if name is None:
            continue
        box = box_by_token[ann_token]
        velocity_global = nusc.box_velocity(ann_token)
        valid = bool(np.all(np.isfinite(velocity_global[:2])))
        velocity_lidar = rot @ np.nan_to_num(velocity_global, nan=0.0)
        width, length, height = box.wlh
        rows.append(
            [
                box.center[0],
                box.center[1],
                box.center[2] - height / 2.0,
                length,
                width,
                height,
                box.orientation.yaw_pitch_roll[0],
                velocity_lidar[0],
                velocity_lidar[1],
            ]
        )
        names.append(name)
        instance_tokens.append(ann["instance_token"])
        ann_tokens.append(ann_token)
        num_pts.append(int(ann["num_lidar_pts"]) + int(ann["num_radar_pts"]))
        visibility.append(int(ann["visibility_token"]) if ann["visibility_token"] else 0)
        vel_valid.append(valid)
    return {
        "boxes": np.asarray(rows, dtype=np.float64).reshape(-1, 9),
        "names": np.asarray(names, dtype=object),
        "labels": np.asarray([DETECTION_CLASSES.index(n) for n in names], dtype=np.int64),
        "instance_tokens": np.asarray(instance_tokens, dtype=object),
        "ann_tokens": np.asarray(ann_tokens, dtype=object),
        "num_pts": np.asarray(num_pts, dtype=np.int64),
        "visibility": np.asarray(visibility, dtype=np.int64),
        "velocity_valid": np.asarray(vel_valid, dtype=bool),
    }


def pipeline_input(record: dict[str, Any], *, use_sweeps: bool = True, image_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """mmdet3d test-pipeline input dict for one keyframe.

    ``image_record`` exists only to reproduce the checkpoint-1 stale-image bug
    in a controlled ablation; normal callers must leave it ``None``.
    """

    from mmdet3d.structures import get_box_type

    box_type_3d, box_mode_3d = get_box_type("LiDAR")
    source = image_record if image_record is not None else record
    images = {
        camera: {"img_path": item["img_path"], "cam2img": item["cam2img"], "lidar2cam": item["lidar2cam"]}
        for camera, item in source["images"].items()
    }
    data = {
        "sample_idx": record["token"],
        "lidar_points": {"lidar_path": record["lidar_path"]},
        "images": images,
        "timestamp": record["lidar_timestamp"],
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
    }
    if use_sweeps:
        data["lidar_sweeps"] = [
            {"lidar_points": dict(s["lidar_points"]), "timestamp": s["timestamp"]} for s in record["lidar_sweeps"]
        ]
    return data
