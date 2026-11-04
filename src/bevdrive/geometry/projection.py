"""Pinhole camera projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Projected pixels and a mask for points in front of the camera."""

    uv: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray


def project_points(
    points_camera_xyz: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int] | None = None,
    min_depth_m: float = 1e-6,
) -> ProjectionResult:
    """Project camera-frame points using a 3x3 pinhole intrinsic matrix.

    Args:
        points_camera_xyz: Points in the camera frame, with +Z forward.
        intrinsic: Camera matrix ``K``.
        image_size: Optional ``(width, height)`` used to include image bounds
            in the validity mask.
        min_depth_m: Strict minimum positive depth.
    """

    points = np.asarray(points_camera_xyz, dtype=np.float64)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera_xyz must have shape (N, 3), got {points.shape}")
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape (3, 3), got {matrix.shape}")
    if min_depth_m <= 0 or not np.all(np.isfinite(points)) or not np.all(np.isfinite(matrix)):
        raise ValueError("points, intrinsic and min_depth_m must be finite; min_depth_m must be > 0")
    homogeneous = (matrix @ points.T).T
    depth = points[:, 2]
    valid = depth > min_depth_m
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid] = homogeneous[valid, :2] / depth[valid, None]
    if image_size is not None:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive width and height")
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return ProjectionResult(uv=uv, depth_m=depth, valid=valid)
