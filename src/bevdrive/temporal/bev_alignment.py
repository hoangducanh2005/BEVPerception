"""Temporal BEV alignment: ego-motion warping, history buffer, aggregation.

Coordinate conventions are explicit because BEVFusion and UniAD disagree:

* ``layout="xy"`` (BEVFusion / TransFusion): ``feature[..., ix, iy]`` - tensor
  dim -2 walks LiDAR **x**, dim -1 walks LiDAR **y**.
* ``layout="yx"`` (BEVFormer / UniAD): ``feature[..., iy, ix]``.

Cell ``i`` along an axis covers ``[min + i*res, min + (i+1)*res)`` and its
centre is ``min + (i + 0.5) * res``; ``grid_sample(align_corners=False)``
addresses exactly these centres.

Warping a previous BEV into the current frame: for every current cell centre
``p_cur`` (z = 0 plane) compute ``p_prev = T_prev^-1 T_cur p_cur`` with
``T = lidar2global`` and bilinearly sample the previous map there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BEVGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    size_x: int
    size_y: int
    layout: str = "xy"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size_x, self.size_y) if self.layout == "xy" else (self.size_y, self.size_x)

    @property
    def resolution(self) -> tuple[float, float]:
        return (self.x_max - self.x_min) / self.size_x, (self.y_max - self.y_min) / self.size_y

    def cell_centers(self, device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Metric (x, y) of every tensor cell, shape ``(*self.shape, 2)``."""

        rx, ry = self.resolution
        xs = self.x_min + (torch.arange(self.size_x, device=device, dtype=dtype) + 0.5) * rx
        ys = self.y_min + (torch.arange(self.size_y, device=device, dtype=dtype) + 0.5) * ry
        if self.layout == "xy":
            gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        else:
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1)

    def normalized(self, xy: torch.Tensor) -> torch.Tensor:
        """Metric (x, y) -> grid_sample coords (last dim = (W-coord, H-coord)) in [-1, 1]."""

        u = 2.0 * (xy[..., 0] - self.x_min) / (self.x_max - self.x_min) - 1.0
        v = 2.0 * (xy[..., 1] - self.y_min) / (self.y_max - self.y_min) - 1.0
        # grid[..., 0] indexes the last tensor dim (W), grid[..., 1] dim -2 (H)
        return torch.stack([v, u], dim=-1) if self.layout == "xy" else torch.stack([u, v], dim=-1)

    def index_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer tensor indices (dim -2, dim -1) of metric points."""

        rx, ry = self.resolution
        ix = np.floor((np.asarray(xy)[:, 0] - self.x_min) / rx).astype(int)
        iy = np.floor((np.asarray(xy)[:, 1] - self.y_min) / ry).astype(int)
        return (ix, iy) if self.layout == "xy" else (iy, ix)


BEVFUSION_GRID = BEVGrid(-54.0, 54.0, -54.0, 54.0, 180, 180, "xy")
UNIAD_GRID = BEVGrid(-51.2, 51.2, -51.2, 51.2, 200, 200, "yx")


def _apply_se3_xy(xy: torch.Tensor, transform: np.ndarray | torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(transform, dtype=np.float64), device=xy.device, dtype=torch.float64)
    flat = xy.reshape(-1, 2).to(torch.float64)
    pts = torch.cat([flat, torch.zeros_like(flat[:, :1]), torch.ones_like(flat[:, :1])], dim=1)
    out = (pts @ t.T)[:, :2]
    return out.reshape(xy.shape).to(xy.dtype)


def sample_at(feature: torch.Tensor, grid: BEVGrid, xy: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample ``feature [B, C, H, W]`` at metric points ``xy [..., 2]`` -> ``[B, C, *xy.shape[:-1]]``."""

    shape = xy.shape[:-1]
    coords = grid.normalized(xy.reshape(1, -1, 1, 2).to(feature.dtype)).expand(feature.shape[0], -1, -1, -1)
    out = F.grid_sample(feature, coords, mode=mode, padding_mode="zeros", align_corners=False)
    return out[..., 0].reshape(feature.shape[0], feature.shape[1], *shape)


def resample(
    feature: torch.Tensor,
    src_grid: BEVGrid,
    dst_grid: BEVGrid,
    dst_to_src: np.ndarray | None = None,
    mode: str = "bilinear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample ``feature`` (on ``src_grid``) onto ``dst_grid``; optional rigid dst->src transform.

    Returns the resampled feature ``[B, C, *dst_grid.shape]`` and a validity mask
    ``[*dst_grid.shape]`` (True where the sample lies inside the source map).
    """

    centers = dst_grid.cell_centers(feature.device, torch.float32)
    src_xy = _apply_se3_xy(centers, dst_to_src) if dst_to_src is not None else centers
    coords = src_grid.normalized(src_xy)
    valid = (coords.abs() <= 1.0).all(dim=-1)
    out = F.grid_sample(
        feature,
        coords.unsqueeze(0).to(feature.dtype).expand(feature.shape[0], -1, -1, -1),
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    return out, valid


def warp_to_current(
    prev_feature: torch.Tensor,
    prev_lidar2global: np.ndarray,
    cur_lidar2global: np.ndarray,
    grid: BEVGrid = BEVFUSION_GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ego-motion compensation of a previous BEV map into the current LiDAR frame."""

    cur_to_prev = np.linalg.inv(np.asarray(prev_lidar2global, dtype=np.float64)) @ np.asarray(cur_lidar2global, dtype=np.float64)
    return resample(prev_feature, grid, grid, cur_to_prev)


@dataclass
class _Entry:
    feature: torch.Tensor
    lidar2global: np.ndarray
    timestamp: float


@dataclass
class TemporalBEVBuffer:
    """Fixed-length history of BEV maps of the *current scene* (reset on scene change)."""

    history: int = 2
    grid: BEVGrid = BEVFUSION_GRID
    max_gap_s: float = 1.5
    scene: str | None = None
    entries: list[_Entry] = field(default_factory=list)

    def reset(self, scene: str | None = None) -> None:
        self.entries.clear()
        self.scene = scene

    def aligned_history(self, cur_lidar2global: np.ndarray, timestamp: float) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Previous maps warped into the current frame, newest first: (feature, valid, dt)."""

        out = []
        for entry in reversed(self.entries):
            dt = float(timestamp - entry.timestamp)
            if dt > self.max_gap_s * (len(out) + 1):
                break
            feat, valid = warp_to_current(entry.feature, entry.lidar2global, cur_lidar2global, self.grid)
            out.append((feat, valid, dt))
        return out

    def push(self, feature: torch.Tensor, lidar2global: np.ndarray, timestamp: float, scene: str) -> None:
        if scene != self.scene:
            self.reset(scene)
        self.entries.append(_Entry(feature.detach(), np.asarray(lidar2global, dtype=np.float64), float(timestamp)))
        if len(self.entries) > self.history:
            self.entries.pop(0)


def cosine_map(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-cell cosine similarity over channels: [B, C, H, W] x2 -> [B, H, W]."""

    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)


def aggregate(
    current: torch.Tensor,
    history: list[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    mode: str = "gated",
    decay: float = 0.5,
    gate_tau: float = 0.6,
    gate_sharpness: float = 10.0,
    include_current: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fuse the current map with aligned history.

    modes
      ``none``  current only (per-frame detection path)
      ``mean``  uniform average over valid cells (prone to motion smearing)
      ``decay`` exponential age decay ``decay**k``
      ``gated`` age decay x validity x feature-agreement gate
                ``sigmoid(s * (cos(current, prev_k) - tau))``. Cells whose
                content changed (moving objects, disocclusions) get ~0 weight,
                so static structure is integrated while movers are not smeared.

    With ``include_current=False`` the result is a pure history context map
    (used as the second input of the query tracker).
    """

    if mode == "none" or not history:
        return (current if include_current else torch.zeros_like(current)), {}
    weights_sum = torch.ones_like(current[:, :1]) if include_current else torch.zeros_like(current[:, :1])
    accum = current.clone() if include_current else torch.zeros_like(current)
    gates = []
    for k, (feat, valid, _dt) in enumerate(history, start=1):
        w = valid[None, None].to(current.dtype)
        if mode in ("decay", "gated"):
            w = w * (decay ** k)
        if mode == "gated":
            gate = torch.sigmoid(gate_sharpness * (cosine_map(current, feat) - gate_tau))[:, None]
            gates.append(gate)
            w = w * gate
        accum = accum + w * feat
        weights_sum = weights_sum + w
    out = accum / weights_sum.clamp_min(1e-6)
    diag = {"weight_sum": weights_sum}
    if gates:
        diag["gate_newest"] = gates[0]
    return out, diag
