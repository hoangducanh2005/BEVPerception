"""Inference-only temporal BEV/query utilities for the UniAD-inspired base.

This module deliberately does not train or alter the frozen BEVFusion detector.
It exposes the feature/memory/query interfaces needed by a future learned
TrackFormer head and provides measurable diagnostics for the base run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalBEVStats:
    """Frame-local diagnostics for aligned previous/current BEV features."""

    has_previous: bool
    feature_shape: tuple[int, ...]
    mean_abs_delta: float | None
    cosine_similarity: float | None
    valid_fraction: float | None


class TemporalBEVMemory:
    """Keep one frozen BEV feature and ego-pose-align it to the next frame."""

    def __init__(self, *, point_cloud_range: tuple[float, float, float, float] = (-54.0, -54.0, 54.0, 54.0)) -> None:
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.previous_feature: Any | None = None
        self.previous_lidar2global: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_feature = None
        self.previous_lidar2global = None

    def update(self, feature: Any, lidar2global: np.ndarray) -> tuple[Any | None, TemporalBEVStats]:
        """Align the previous feature into the current LiDAR frame and update memory."""

        current = _as_feature_tensor(feature)
        current_pose = np.asarray(lidar2global, dtype=np.float64)
        previous_aligned = None
        if self.previous_feature is not None and self.previous_lidar2global is not None:
            previous_aligned, valid_fraction = warp_bev_feature(
                self.previous_feature,
                self.previous_lidar2global,
                current_pose,
                self.point_cloud_range,
            )
            current_flat = current.detach().float().reshape(-1)
            previous_flat = previous_aligned.detach().float().reshape(-1)
            cosine = float(
                (current_flat @ previous_flat)
                / (current_flat.norm().clamp_min(1e-8) * previous_flat.norm().clamp_min(1e-8))
            )
            delta = float((current - previous_aligned).detach().float().abs().mean())
            stats = TemporalBEVStats(True, tuple(int(v) for v in current.shape), delta, cosine, valid_fraction)
        else:
            stats = TemporalBEVStats(False, tuple(int(v) for v in current.shape), None, None, None)
        self.previous_feature = current.detach().clone()
        self.previous_lidar2global = current_pose.copy()
        return previous_aligned, stats


class TemporalQueryBank:
    """Maintain BEV feature embeddings for persistent detector track IDs."""

    def __init__(self, *, ema: float = 0.7) -> None:
        self.ema = float(np.clip(ema, 0.0, 1.0))
        self._states: dict[int, Any] = {}
        self._ages: dict[int, int] = {}

    def reset(self) -> None:
        self._states.clear()
        self._ages.clear()

    def update(self, assignments: Sequence[Any], feature: Any, lidar2global: np.ndarray) -> dict[str, Any]:
        """Sample current BEV at track centers and update persistent query states."""

        import torch

        if not assignments:
            return {"num_queries": 0, "mean_query_cosine": None, "query_norms": {}}
        current = _as_feature_tensor(feature)
        world_centers = np.asarray([item.center_world_xyz for item in assignments], dtype=np.float64)
        inverse = np.linalg.inv(np.asarray(lidar2global, dtype=np.float64))
        local_centers = (inverse @ np.c_[world_centers, np.ones(len(world_centers))].T).T[:, :2]
        embeddings, valid = sample_bev_feature(current, local_centers)
        cosines: list[float] = []
        norms: dict[str, float] = {}
        for item, embedding, is_valid in zip(assignments, embeddings, valid):
            track_id = int(item.track_id)
            embedding = embedding.detach().float()
            if not bool(is_valid):
                continue
            embedding = torch.nn.functional.normalize(embedding, dim=0)
            old = self._states.get(track_id)
            if old is not None:
                cosines.append(float(torch.dot(old, embedding).clamp(-1.0, 1.0).item()))
                embedding = self.ema * old + (1.0 - self.ema) * embedding
                embedding = torch.nn.functional.normalize(embedding, dim=0)
            self._states[track_id] = embedding.detach().clone()
            self._ages[track_id] = self._ages.get(track_id, 0) + 1
            norms[str(track_id)] = float(embedding.norm().item())
        return {
            "num_queries": len(self._states),
            "num_queries_updated": len(norms),
            "mean_query_cosine": float(np.mean(cosines)) if cosines else None,
            "query_norms": norms,
            "query_ages": {str(track_id): int(age) for track_id, age in self._ages.items()},
        }


def attach_bev_feature_capture(model: Any) -> tuple[dict[str, Any], Any]:
    """Capture the final BEV feature passed from ``pts_neck`` to the head."""

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["feature"] = _as_feature_tensor(output).detach()

    if not hasattr(model, "pts_neck"):
        raise AttributeError("BEVFusion model has no pts_neck module to capture")
    handle = model.pts_neck.register_forward_hook(hook)
    return captured, handle


def _as_feature_tensor(feature: Any) -> Any:
    import torch

    if isinstance(feature, (list, tuple)):
        tensors = [item for item in feature if isinstance(item, torch.Tensor)]
        if not tensors:
            raise TypeError("BEV feature output contains no tensors")
        height = min(int(item.shape[-2]) for item in tensors)
        width = min(int(item.shape[-1]) for item in tensors)
        tensors = [item[..., :height, :width] for item in tensors]
        return torch.cat(tensors, dim=1)
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"Unsupported BEV feature type: {type(feature)!r}")
    return feature


def _grid_for_transform(
    height: int,
    width: int,
    previous_lidar2global: np.ndarray,
    current_lidar2global: np.ndarray,
    point_cloud_range: tuple[float, float, float, float],
    *,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    import torch

    min_x, min_y, max_x, max_y = point_cloud_range
    # ``grid_sample(..., align_corners=False)`` addresses cell centers. Use
    # center coordinates here; sampling the range endpoints introduces a
    # half-cell shift even for an identity ego pose.
    xs = min_x + (torch.arange(width, device=device, dtype=dtype) + 0.5) * ((max_x - min_x) / width)
    # BEV rows are rendered from +y at the top to -y at the bottom.
    ys = max_y - (torch.arange(height, device=device, dtype=dtype) + 0.5) * ((max_y - min_y) / height)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    current_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(height * width, device=device, dtype=dtype), torch.ones(height * width, device=device, dtype=dtype)], dim=1)
    transform = np.linalg.inv(previous_lidar2global) @ current_lidar2global
    transform_tensor = torch.as_tensor(transform, device=device, dtype=dtype)
    previous_xy = (transform_tensor @ current_xy.T).T[:, :2]
    normalized_x = 2.0 * (previous_xy[:, 0] - min_x) / (max_x - min_x) - 1.0
    normalized_y = 1.0 - 2.0 * (previous_xy[:, 1] - min_y) / (max_y - min_y)
    sampling_grid = torch.stack([normalized_x, normalized_y], dim=1).reshape(1, height, width, 2)
    valid = ((normalized_x >= -1.0) & (normalized_x <= 1.0) & (normalized_y >= -1.0) & (normalized_y <= 1.0)).reshape(height, width)
    return sampling_grid, valid


def warp_bev_feature(
    previous_feature: Any,
    previous_lidar2global: np.ndarray,
    current_lidar2global: np.ndarray,
    point_cloud_range: tuple[float, float, float, float],
) -> tuple[Any, float]:
    """Warp a previous BEV tensor into current ego/LiDAR coordinates."""

    import torch.nn.functional as F

    feature = _as_feature_tensor(previous_feature)
    _, _, height, width = feature.shape
    grid, valid = _grid_for_transform(height, width, previous_lidar2global, current_lidar2global, point_cloud_range, device=feature.device, dtype=feature.dtype)
    aligned = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return aligned, float(valid.float().mean().item())


def sample_bev_feature(feature: Any, centers_xy: np.ndarray) -> tuple[Any, np.ndarray]:
    """Bilinearly sample feature embeddings at current-frame BEV centers."""

    import torch
    import torch.nn.functional as F

    feature = _as_feature_tensor(feature)
    _, _, height, width = feature.shape
    min_x, min_y, max_x, max_y = (-54.0, -54.0, 54.0, 54.0)
    centers = np.asarray(centers_xy, dtype=np.float32).reshape(-1, 2)
    normalized_x = 2.0 * (centers[:, 0] - min_x) / (max_x - min_x) - 1.0
    normalized_y = 1.0 - 2.0 * (centers[:, 1] - min_y) / (max_y - min_y)
    valid = (np.abs(normalized_x) <= 1.0) & (np.abs(normalized_y) <= 1.0)
    grid = torch.as_tensor(np.stack([normalized_x, normalized_y], axis=1), device=feature.device, dtype=feature.dtype).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[0, :, :, 0].T, valid
