"""Typed 3D detection contract used at the repository boundary.

The contract deliberately keeps units and the coordinate frame explicit.  It
is independent of MMDetection3D so downstream tracking and evaluation code can
be tested without importing a GPU model stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _vector(values: Sequence[float] | np.ndarray, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class Detection3D:
    """A single model-agnostic 3D detection.

    Coordinates use metres, velocity uses metres/second, and yaw uses radians.
    ``coordinate_frame`` must identify the frame and timestamp represented by
    ``center_xyz_m``.
    """

    scene_id: str
    sample_token: str
    timestamp_us: int
    class_name: str
    score: float
    center_xyz_m: tuple[float, float, float]
    size_lwh_m: tuple[float, float, float]
    yaw_rad: float
    coordinate_frame: str = "ego_at_timestamp"
    velocity_xy_mps: tuple[float, float] | None = None
    covariance: tuple[tuple[float, ...], ...] | None = None
    source: str = "bevfusion"

    def __post_init__(self) -> None:
        if not self.scene_id or not self.sample_token:
            raise ValueError("scene_id and sample_token are required")
        if self.timestamp_us < 0:
            raise ValueError("timestamp_us must be non-negative")
        if not self.class_name or not self.coordinate_frame or not self.source:
            raise ValueError("class_name, coordinate_frame and source are required")
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        object.__setattr__(self, "center_xyz_m", _vector(self.center_xyz_m, 3, "center_xyz_m"))
        size = _vector(self.size_lwh_m, 3, "size_lwh_m")
        if any(value <= 0.0 for value in size):
            raise ValueError("size_lwh_m must be strictly positive")
        object.__setattr__(self, "size_lwh_m", size)
        if not np.isfinite(self.yaw_rad):
            raise ValueError("yaw_rad must be finite")
        if self.velocity_xy_mps is not None:
            object.__setattr__(self, "velocity_xy_mps", _vector(self.velocity_xy_mps, 2, "velocity_xy_mps"))
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=np.float64)
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("covariance must be a square matrix")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("covariance must contain only finite values")
            object.__setattr__(
                self,
                "covariance",
                tuple(tuple(float(value) for value in row) for row in covariance),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Detection3D":
        """Construct a detection from a JSON/YAML-compatible mapping."""

        return cls(**dict(value))
