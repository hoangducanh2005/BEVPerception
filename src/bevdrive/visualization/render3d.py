"""Six-camera projection and BEV rendering for 3D detections/tracks.

Boxes are in the keyframe LiDAR frame, layout
``x, y, z_bottom, dx, dy, dz, yaw[, vx, vy]``. BEV images are drawn in the
ego frame (+x forward = up, +y left = left) so they read like a map.
"""

from __future__ import annotations

import colorsys
from typing import Any, Sequence

import cv2
import numpy as np

CAMERA_GRID = (("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"), ("CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"))
CLASS_COLORS_BGR = {
    "pedestrian": (40, 40, 230),
    "bicycle": (230, 120, 20),
    "motorcycle": (200, 40, 200),
    "car": (90, 160, 90),
    "truck": (60, 130, 60),
    "bus": (60, 110, 60),
    "trailer": (60, 110, 60),
    "construction_vehicle": (60, 110, 60),
    "barrier": (128, 128, 128),
    "traffic_cone": (0, 170, 230),
}
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per track id (golden-ratio hue walk)."""

    hue = (int(track_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def box_corners(box: Sequence[float]) -> np.ndarray:
    """(8, 3) corners; 0-3 bottom, 4-7 top; 0-1 / 4-5 is the front (+heading) face."""

    x, y, z, dx, dy, dz, yaw = (float(v) for v in box[:7])
    local = np.array(
        [[dx / 2, dy / 2, 0], [dx / 2, -dy / 2, 0], [-dx / 2, -dy / 2, 0], [-dx / 2, dy / 2, 0]], dtype=np.float64
    )
    local = np.vstack([local, local + [0, 0, dz]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.array([x, y, z])


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def render_camera_grid(
    record: dict[str, Any],
    boxes: np.ndarray,
    colors: Sequence[tuple[int, int, int]],
    texts: Sequence[str],
    *,
    tile_width: int = 640,
    title: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """2x3 montage of the six cameras with projected 3D boxes."""

    tiles = []
    corners_all = [box_corners(b) for b in np.asarray(boxes, dtype=np.float64)] if len(boxes) else []
    for row in CAMERA_GRID:
        row_tiles = []
        for camera in row:
            info = record["images"][camera]
            image = cv2.imread(info["img_path"])
            if image is None:
                image = np.zeros((900, 1600, 3), np.uint8)
            height, width = image.shape[:2]
            lidar2cam = np.asarray(info["lidar2cam"], dtype=np.float64)
            intrinsic = np.asarray(info["cam2img"], dtype=np.float64)[:3, :3]
            for corners, color, text in zip(corners_all, colors, texts):
                cam = _transform(corners, lidar2cam)
                if np.any(cam[:, 2] < 0.5):
                    continue
                uv = (intrinsic @ cam.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                if uv[:, 0].max() < 0 or uv[:, 0].min() >= width or uv[:, 1].max() < 0 or uv[:, 1].min() >= height:
                    continue
                pts = uv.round().astype(np.int32)
                for a, b in EDGES:
                    cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, thickness * 2, cv2.LINE_AA)
                cv2.line(image, tuple(pts[0]), tuple(pts[5]), color, thickness, cv2.LINE_AA)
                cv2.line(image, tuple(pts[1]), tuple(pts[4]), color, thickness, cv2.LINE_AA)
                if text:
                    org = (int(np.clip(pts[:, 0].min(), 0, width - 10)), int(np.clip(pts[:, 1].min() - 8, 30, height - 5)))
                    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            cv2.putText(image, camera, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5, cv2.LINE_AA)
            scale = tile_width / width
            row_tiles.append(cv2.resize(image, (tile_width, int(height * scale)), interpolation=cv2.INTER_AREA))
        tiles.append(np.hstack(row_tiles))
    montage = np.vstack(tiles)
    if title:
        bar = np.full((40, montage.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        montage = np.vstack([bar, montage])
    return montage


class BEVCanvas:
    """Ego-frame BEV canvas: +x forward is up, +y left is left."""

    def __init__(self, lidar2ego: np.ndarray, *, range_m: float = 50.0, px_per_m: float = 8.0, background: int = 255) -> None:
        self.lidar2ego = np.asarray(lidar2ego, dtype=np.float64)
        self.range_m = range_m
        self.scale = px_per_m
        self.size = int(2 * range_m * px_per_m)
        self.image = np.full((self.size, self.size, 3), background, np.uint8)
        for radius in range(10, int(range_m) + 1, 10):
            cv2.circle(self.image, self.to_px(np.zeros((1, 2)))[0], int(radius * px_per_m), (225, 225, 225), 1, cv2.LINE_AA)

    def to_px(self, ego_xy: np.ndarray) -> np.ndarray:
        ego_xy = np.asarray(ego_xy, dtype=np.float64).reshape(-1, 2)
        u = self.size / 2 - ego_xy[:, 1] * self.scale
        v = self.size / 2 - ego_xy[:, 0] * self.scale
        return np.stack([u, v], axis=1).round().astype(np.int32)

    def lidar_to_px(self, lidar_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(lidar_xyz, dtype=np.float64).reshape(-1, 3)
        return self.to_px(_transform(pts, self.lidar2ego)[:, :2])

    def draw_points(self, lidar_points: np.ndarray, color: tuple[int, int, int] = (170, 170, 170)) -> None:
        if lidar_points is None or len(lidar_points) == 0:
            return
        px = self.lidar_to_px(lidar_points[:, :3])
        keep = (px[:, 0] >= 0) & (px[:, 0] < self.size) & (px[:, 1] >= 0) & (px[:, 1] < self.size)
        self.image[px[keep, 1], px[keep, 0]] = color

    def draw_ego(self) -> None:
        corners = np.array([[2.4, 0.95], [2.4, -0.95], [-2.4, -0.95], [-2.4, 0.95]])
        cv2.fillPoly(self.image, [self.to_px(corners)], (40, 40, 40))
        cv2.arrowedLine(self.image, tuple(self.to_px(np.array([[0, 0]]))[0]), tuple(self.to_px(np.array([[4.5, 0]]))[0]), (40, 40, 40), 2, tipLength=0.3)

    def draw_box(self, box: Sequence[float], color: tuple[int, int, int], *, thickness: int = 2, text: str | None = None, dashed: bool = False) -> None:
        corners = box_corners(box)[:4]
        px = self.lidar_to_px(corners)
        if dashed:
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                p, q = px[a].astype(float), px[b].astype(float)
                for t in np.linspace(0, 1, 6)[:-1:2]:
                    cv2.line(self.image, tuple((p + (q - p) * t).astype(int)), tuple((p + (q - p) * (t + 0.2)).astype(int)), color, 1, cv2.LINE_AA)
        else:
            cv2.polylines(self.image, [px], True, color, thickness, cv2.LINE_AA)
            cv2.line(self.image, tuple(px[0]), tuple(px[1]), color, thickness + 2, cv2.LINE_AA)  # heading face
        if text:
            org = tuple(px.max(axis=0) + np.array([2, 0]))
            cv2.putText(self.image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(self.image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def draw_velocity(self, box: Sequence[float], velocity_xy_lidar: Sequence[float], color: tuple[int, int, int], seconds: float = 1.0) -> None:
        start = np.array([box[0], box[1], 0.0])
        end = start + np.array([velocity_xy_lidar[0], velocity_xy_lidar[1], 0.0]) * seconds
        p, q = self.lidar_to_px(np.vstack([start, end]))
        if np.linalg.norm(q - p) >= 2:
            cv2.arrowedLine(self.image, tuple(p), tuple(q), color, 2, cv2.LINE_AA, tipLength=0.25)

    def draw_polyline(self, lidar_xy: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
        if lidar_xy is None or len(lidar_xy) < 2:
            return
        pts = np.c_[np.asarray(lidar_xy)[:, :2], np.zeros(len(lidar_xy))]
        cv2.polylines(self.image, [self.lidar_to_px(pts)], False, color, thickness, cv2.LINE_AA)
        for p in self.lidar_to_px(pts):
            cv2.circle(self.image, tuple(p), 2, color, -1, cv2.LINE_AA)

    def title(self, text: str, line: int = 0) -> None:
        cv2.putText(self.image, text, (10, 24 + 22 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def load_lidar_points(path: str, max_points: int = 60000) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    if len(points) > max_points:
        points = points[:: int(np.ceil(len(points) / max_points))]
    return points


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == height:
            return img
        return np.vstack([img, np.full((height - img.shape[0], img.shape[1], 3), 255, np.uint8)])
    return np.hstack([pad(left), pad(right)])
