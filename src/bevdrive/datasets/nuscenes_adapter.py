"""Scene-safe nuScenes adapter for calibrated camera/LiDAR samples.

The adapter keeps the nuScenes SDK at the integration boundary.  Importing
this module does not require the SDK, which allows geometry and contract tests
to run on CPU in a minimal environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from bevdrive.geometry.transforms import invert_transform


CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Calibrated sensor references for one nuScenes sample."""

    scene_id: str
    sample_token: str
    timestamp_us: int
    camera_images: dict[str, str]
    lidar_path: str | None
    intrinsics: dict[str, tuple[tuple[float, ...], ...]]
    t_cam_from_lidar: dict[str, tuple[tuple[float, ...], ...]]
    t_world_from_ego: tuple[tuple[float, ...], ...]
    t_ego_from_sensor: dict[str, tuple[tuple[float, ...], ...]]
    valid_sensors: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "sample_token": self.sample_token,
            "timestamp_us": self.timestamp_us,
            "camera_images": self.camera_images,
            "lidar_path": self.lidar_path,
            "intrinsics": self.intrinsics,
            "T_cam_from_lidar": self.t_cam_from_lidar,
            "T_world_from_ego": self.t_world_from_ego,
            "T_ego_from_sensor": self.t_ego_from_sensor,
            "valid_sensors": self.valid_sensors,
        }


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=np.float64))


def _rigid_transform(rotation: np.ndarray, translation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


class NuScenesAdapter:
    """Read calibrated nuScenes samples without loading the whole dataset."""

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        version: str = "v1.0-mini",
        split: str | None = "mini_val",
        max_timestamp_delta_ms: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.version = version
        self.split = split
        self.max_timestamp_delta_us = int(max_timestamp_delta_ms * 1000.0)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"nuScenes root does not exist: {self.data_root}. Set NUSCENES_ROOT "
                "to a directory containing the requested nuScenes version."
            )
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError(
                "nuscenes-devkit is not installed in this environment; install the locked Phase 1 dependencies."
            ) from exc
        self.nusc = NuScenes(version=version, dataroot=str(self.data_root), verbose=False)
        self._scene_names: set[str] | None = None
        if split is not None:
            try:
                from nuscenes.utils.splits import create_splits_scenes

                splits = create_splits_scenes()
                if split not in splits:
                    raise ValueError(f"Unknown nuScenes split {split!r}; available: {sorted(splits)}")
                self._scene_names = set(splits[split])
            except ImportError as exc:
                raise RuntimeError("nuscenes-devkit split utilities are unavailable") from exc

    def sample_tokens(self, limit: int | None = None) -> list[str]:
        """Return deterministic sample tokens, optionally capped."""

        tokens = []
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            if self._scene_names is None or scene["name"] in self._scene_names:
                tokens.append(sample["token"])
        return tokens[:limit] if limit is not None else tokens

    def iter_frames(self, limit: int | None = None) -> Iterator[SensorFrame]:
        for token in self.sample_tokens(limit):
            yield self.get_frame(token)

    def get_frame(self, sample_token: str) -> SensorFrame:
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        camera_images: dict[str, str] = {}
        intrinsics: dict[str, tuple[tuple[float, ...], ...]] = {}
        t_cam_from_lidar: dict[str, tuple[tuple[float, ...], ...]] = {}
        t_ego_from_sensor: dict[str, tuple[tuple[float, ...], ...]] = {}
        valid_sensors: dict[str, bool] = {}

        lidar_token = sample["data"].get("LIDAR_TOP")
        lidar_path: str | None = None
        lidar_ego_transform: np.ndarray | None = None
        lidar_timestamp = None
        sensor_timestamps: dict[str, int] = {}
        if lidar_token:
            lidar_data = self.nusc.get("sample_data", lidar_token)
            lidar_calibration = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            lidar_path = str(self.data_root / lidar_data["filename"])
            lidar_ego_transform = _rigid_transform(
                _quaternion_rotation(lidar_calibration["rotation"]), lidar_calibration["translation"]
            )
            lidar_timestamp = int(lidar_data["timestamp"])
            sensor_timestamps["LIDAR_TOP"] = lidar_timestamp
            valid_sensors["LIDAR_TOP"] = Path(lidar_path).is_file()
            t_ego_from_sensor["LIDAR_TOP"] = _matrix_tuple(lidar_ego_transform)
        else:
            valid_sensors["LIDAR_TOP"] = False

        for camera_name in CAMERA_NAMES:
            data_token = sample["data"].get(camera_name)
            if not data_token:
                valid_sensors[camera_name] = False
                continue
            camera_data = self.nusc.get("sample_data", data_token)
            sensor_timestamps[camera_name] = int(camera_data["timestamp"])
            calibration = self.nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
            camera_path = self.data_root / camera_data["filename"]
            camera_ego_transform = _rigid_transform(
                _quaternion_rotation(calibration["rotation"]), calibration["translation"]
            )
            camera_images[camera_name] = str(camera_path)
            intrinsics[camera_name] = _matrix_tuple(np.asarray(calibration["camera_intrinsic"], dtype=np.float64))
            t_ego_from_sensor[camera_name] = _matrix_tuple(camera_ego_transform)
            if lidar_ego_transform is not None:
                t_cam_from_lidar[camera_name] = _matrix_tuple(invert_transform(camera_ego_transform) @ lidar_ego_transform)
            valid_sensors[camera_name] = camera_path.is_file()

        ego_pose = self.nusc.get("ego_pose", self.nusc.get("sample_data", lidar_token or sample["data"][CAMERA_NAMES[0]])["ego_pose_token"])
        t_world_from_ego = _rigid_transform(_quaternion_rotation(ego_pose["rotation"]), ego_pose["translation"])
        if lidar_timestamp is not None and abs(lidar_timestamp - int(sample["timestamp"])) > self.max_timestamp_delta_us:
            raise ValueError(
                f"sample {sample_token} has a LiDAR/sample timestamp mismatch greater than "
                f"{self.max_timestamp_delta_us / 1000.0:g} ms"
            )
        if sensor_timestamps:
            latest = max(sensor_timestamps.values())
            earliest = min(sensor_timestamps.values())
            if latest - earliest > self.max_timestamp_delta_us:
                raise ValueError(
                    f"sample {sample_token} has sensor timestamps spanning more than "
                    f"{self.max_timestamp_delta_us / 1000.0:g} ms: {sensor_timestamps}"
                )
        return SensorFrame(
            scene_id=scene["name"],
            sample_token=sample_token,
            timestamp_us=int(sample["timestamp"]),
            camera_images=camera_images,
            lidar_path=lidar_path,
            intrinsics=intrinsics,
            t_cam_from_lidar=t_cam_from_lidar,
            t_world_from_ego=_matrix_tuple(t_world_from_ego),
            t_ego_from_sensor=t_ego_from_sensor,
            valid_sensors=valid_sensors,
        )

    def annotation_boxes(self, sample_token: str) -> list[dict[str, Any]]:
        """Return raw nuScenes boxes with explicit ego-at-sample coordinates."""

        sample = self.nusc.get("sample", sample_token)
        boxes = []
        for annotation_token in sample["anns"]:
            box = self.nusc.get_box(annotation_token)
            boxes.append(
                {
                    "token": annotation_token,
                    "class_name": box.name,
                    "center_xyz_m": [float(value) for value in box.center],
                    "size_lwh_m": [float(value) for value in box.wlh],
                    "yaw_rad": float(box.orientation.yaw_pitch_roll[0]),
                    "coordinate_frame": "global",
                }
            )
        return boxes


def _quaternion_rotation(values: Any) -> np.ndarray:
    try:
        from pyquaternion import Quaternion
    except ImportError as exc:
        raise RuntimeError("pyquaternion is required for nuScenes calibration parsing") from exc
    return np.asarray(Quaternion(values).rotation_matrix, dtype=np.float64)
