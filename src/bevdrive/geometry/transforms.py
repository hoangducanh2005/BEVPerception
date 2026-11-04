"""Homogeneous-coordinate transforms with strict shape validation."""

from __future__ import annotations

import numpy as np


def _transform_matrix(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("transform must be a homogeneous rigid transform")
    return matrix


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` point array."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xyz must contain only finite values")
    matrix = _transform_matrix(transform)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform without hiding malformed input."""

    matrix = _transform_matrix(transform)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("transform rotation must be orthonormal")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse
