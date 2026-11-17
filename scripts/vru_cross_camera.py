#!/usr/bin/env python3
"""Mine bounded VRU camera-transition candidates with explicit visibility evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d")
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d/projects/BEVFusion")

from bevdrive.tracking import DetectionMeasurement, TrackManager  # noqa: E402
from phase12_nuscenes_infer import build_records  # noqa: E402
from vru_detection_experiment import VRU_CLASSES  # noqa: E402
from vru_tracking_oracle import gated_hungarian  # noqa: E402


CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
VISIBILITY_NAMES = {"1": "v1-40", "2": "v2-60", "3": "v3-80", "4": "v4-100"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-name", action="append", required=True)
    parser.add_argument("--frame-limit", type=int, default=100)
    parser.add_argument("--cached-detections", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--distance-threshold-m", type=float, default=2.0)
    return parser.parse_args()


def box_corners(box: np.ndarray) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array(
        [[-dx / 2, -dy / 2, -dz / 2], [-dx / 2, -dy / 2, dz / 2], [-dx / 2, dy / 2, -dz / 2], [-dx / 2, dy / 2, dz / 2], [dx / 2, -dy / 2, -dz / 2], [dx / 2, -dy / 2, dz / 2], [dx / 2, dy / 2, -dz / 2], [dx / 2, dy / 2, dz / 2]],
        dtype=np.float32,
    )
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return local @ rotation.T + np.asarray([x, y, z], dtype=np.float32)


def project_box(box: np.ndarray, camera: dict[str, Any], image_size: tuple[int, int]) -> tuple[bool, np.ndarray]:
    corners = box_corners(box)
    transform = np.asarray(camera["lidar2cam"], dtype=np.float64)
    intrinsic = np.asarray(camera["cam2img"], dtype=np.float64)
    camera_points = (transform @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
    valid_depth = camera_points[:, 2] > 0.1
    if not valid_depth.any():
        return False, np.full((8, 2), np.nan)
    pixels = np.full((len(corners), 2), np.nan, dtype=np.float64)
    projected = (intrinsic @ camera_points[valid_depth].T).T
    pixels[valid_depth] = projected[:, :2] / projected[:, 2:3]
    width, height = image_size
    inside = valid_depth & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    # This is geometric visibility only: at least two projected corners inside.
    return bool(np.count_nonzero(inside) >= 2), pixels


def make_measurements(frame: dict[str, Any], pose: np.ndarray) -> list[DetectionMeasurement]:
    boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
    scores = np.asarray(frame["scores"], dtype=np.float64)
    names = [str(value) for value in frame["names"]]
    centers = (pose @ np.c_[boxes[:, :3], np.ones(len(boxes))].T).T[:, :3]
    return [DetectionMeasurement(name, float(score), tuple(float(value) for value in center), tuple(float(value) for value in box[3:6]), float(box[6])) for box, score, name, center in zip(boxes, scores, names, centers)]


def draw_projected_box(image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int]) -> None:
    edges = ((0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    for start, end in edges:
        if np.isfinite(pixels[[start, end]]).all():
            cv2.line(image, tuple(np.round(pixels[start]).astype(int)), tuple(np.round(pixels[end]).astype(int)), color, 3, cv2.LINE_AA)


def render_event(event: dict[str, Any], by_token: dict[str, dict[str, Any]], by_record: dict[str, dict[str, Any]], root: Path, output_dir: Path) -> list[str]:
    paths: list[str] = []
    for side, frame_token in (("before", event["before_token"]), ("after", event["after_token"])):
        record = by_record[frame_token]
        panels = []
        for camera_name in CAMERAS:
            camera = record["cams"][camera_name]
            image = cv2.imread(str(root / camera["img_path"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            visible = camera_name in event[f"{side}_views"]
            _, pixels = project_box(np.asarray(event[f"{side}_gt_box"], dtype=np.float32), camera, (image.shape[1], image.shape[0]))
            draw_projected_box(image, pixels, (0, 0, 255) if visible else (90, 90, 90))
            label = f"{camera_name} {'VISIBLE' if visible else 'not-visible'}"
            cv2.putText(image, label, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
        if len(panels) == 6:
            montage = cv2.vconcat([cv2.hconcat(panels[:3]), cv2.hconcat(panels[3:])])
            cv2.putText(montage, f"{side} frame={event[f'{side}_frame_index']} gt={event['instance_token'][:8]} track={event[f'{side}_track_id']}", (18, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            path = output_dir / f"{event['event_id']}_{side}_camera.jpg"
            cv2.imwrite(str(path), montage)
            paths.append(str(path))
    return paths


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = json.loads(args.cached_detections.read_text(encoding="utf-8"))
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    records = build_records(nusc, args.data_root.resolve(), args.scene_name, args.frame_limit)
    if [record["token"] for record in records] != [frame["token"] for frame in frames]:
        raise ValueError("cached detections do not match selected records")
    by_record = {record["token"]: record for record in records}
    tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=args.score_threshold)
    track_maps: dict[str, dict[str, int]] = {}
    matched_maps: dict[str, dict[str, int]] = {}
    visibility_maps: dict[str, dict[str, list[str]]] = {}
    box_maps: dict[str, dict[str, list[float]]] = {}
    for frame in frames:
        record = by_record[frame["token"]]
        pose = np.asarray(record["lidar2global"], dtype=np.float64)
        assignments = tracker.update(make_measurements(frame, pose), timestamp_us=int(frame["timestamp_us"]), scene_id=str(frame["scene_name"]))
        track_by_detection = {int(item.detection_index): int(item.track_id) for item in assignments}
        boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(frame["scores"], dtype=np.float64)
        names = [str(value) for value in frame["names"]]
        gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        gt_names = [str(value) for value in frame["gt_native_names"]]
        pairs = gated_hungarian(boxes, scores, names, gt_boxes, gt_names, args.score_threshold, args.distance_threshold_m)
        matched_maps[frame["token"]] = {str(frame["gt_instance_tokens"][gt_index]): track_by_detection[pred_index] for pred_index, gt_index in pairs if pred_index in track_by_detection}
        track_maps[frame["token"]] = track_by_detection
        frame_visibility: dict[str, list[str]] = {}
        frame_boxes: dict[str, list[float]] = {}
        for gt_index, (name, token) in enumerate(zip(gt_names, frame["gt_instance_tokens"])):
            if name not in VRU_CLASSES:
                continue
            views: list[str] = []
            for camera_name in CAMERAS:
                camera = record["cams"][camera_name]
                image = cv2.imread(str(args.data_root / camera["img_path"]), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                visible, _ = project_box(gt_boxes[gt_index], camera, (image.shape[1], image.shape[0]))
                if visible:
                    views.append(camera_name)
            frame_visibility[str(token)] = views
            frame_boxes[str(token)] = gt_boxes[gt_index].astype(float).tolist()
        visibility_maps[frame["token"]] = frame_visibility
        box_maps[frame["token"]] = frame_boxes

    by_instance: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, frame in enumerate(frames):
        for token, views in visibility_maps[frame["token"]].items():
            if views:
                by_instance[token].append((index, {"token": frame["token"], "scene": frame["scene_name"], "views": views, "box": box_maps[frame["token"]][token], "track_id": matched_maps[frame["token"]].get(token)}))
    events: list[dict[str, Any]] = []
    for instance_token, observations in by_instance.items():
        for (before_index, before), (after_index, after) in zip(observations[:-1], observations[1:]):
            if before["scene"] != after["scene"] or after_index != before_index + 1:
                continue
            before_views, after_views = set(before["views"]), set(after["views"])
            if not before_views or not after_views or before_views.intersection(after_views):
                continue
            annotation = nusc.get("sample_annotation", next(token for token in nusc.get("sample", after["token"])["anns"] if nusc.get("sample_annotation", token)["instance_token"] == instance_token))
            event = {
                "event_id": f"{after['scene']}_{after_index}_{instance_token[:8]}",
                "instance_token": instance_token,
                "scene_name": after["scene"],
                "before_frame_index": before_index,
                "after_frame_index": after_index,
                "before_token": before["token"],
                "after_token": after["token"],
                "before_views": sorted(before_views),
                "after_views": sorted(after_views),
                "before_track_id": before["track_id"],
                "after_track_id": after["track_id"],
                "track_id_preserved_geometric_candidate": before["track_id"] is not None and before["track_id"] == after["track_id"],
                "before_gt_box": before["box"],
                "after_gt_box": after["box"],
                "visibility_token": annotation.get("visibility_token"),
                "visibility_name": VISIBILITY_NAMES.get(str(annotation.get("visibility_token")), "unknown"),
                "validation_status": "geometric_candidate_not_manually_validated",
            }
            events.append(event)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for event in events[:10]:
        event["visualizations"] = render_event(event, {frame["token"]: frame for frame in frames}, by_record, args.data_root.resolve(), visual_dir)
    result = {
        "status": "COMPLETED",
        "frames": len(frames),
        "scenes": args.scene_name,
        "geometric_handoff_candidates": len(events),
        "manually_validated_handoffs": 0,
        "candidate_track_id_preserved": sum(bool(event["track_id_preserved_geometric_candidate"]) for event in events),
        "events": events,
        "note": "Camera visibility is geometry-based (projected GT box corners); no manual visibility validation was performed by this runner.",
    }
    (args.output_dir / "cross_camera_events.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "frames", "scenes", "geometric_handoff_candidates", "manually_validated_handoffs", "candidate_track_id_preserved")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
