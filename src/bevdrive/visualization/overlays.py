"""Lightweight visual sanity-check overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bevdrive.geometry.projection import project_points
from bevdrive.geometry.transforms import transform_points


def draw_projected_points(
    image_path: str | Path,
    points_lidar_xyz: np.ndarray,
    t_cam_from_lidar: np.ndarray,
    intrinsic: np.ndarray,
    output_path: str | Path,
) -> int:
    """Draw valid LiDAR projections onto an image and return point count."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to render calibration overlays") from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    camera_points = transform_points(points_lidar_xyz, t_cam_from_lidar)
    projection = project_points(camera_points, intrinsic, image_size=(width, height))
    for x, y in projection.uv[projection.valid].astype(int):
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Failed to write overlay: {target}")
    return int(projection.valid.sum())
