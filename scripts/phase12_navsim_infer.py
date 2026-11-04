#!/usr/bin/env python3
"""Run the two bounded tracking phases on a small NavSim shard-5 slice.

Phase 1 is the real-time baseline: BEVFusion detections followed by a
constant-velocity Kalman/Hungarian tracker. Phase 2 uses the same detector and
association, but maintains an explicit persistent query state as a temporal
TrackFormer-inspired interface. It is intentionally labelled a prototype: no
learned TrackFormer weights are claimed by this runner.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_navsim_infer import (  # noqa: E402
    CAMERA_MAP,
    MODEL_CLASSES,
    box_corners,
    load_pcd,
    make_info,
)
from bevdrive.tracking import DetectionMeasurement, TrackManager  # noqa: E402
from evidence_metrics import aggregate_frame_metrics, match_detection_and_tracking  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-pkl", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[np.asarray(points_xyz, dtype=np.float64), np.ones(len(points_xyz))]
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def id_color(track_id: int) -> tuple[int, int, int]:
    # Deterministic BGR palette; stable across both phases and all frames.
    palette = [
        (255, 160, 0),
        (0, 220, 220),
        (255, 80, 180),
        (80, 220, 80),
        (180, 100, 255),
        (80, 170, 255),
        (220, 220, 80),
        (150, 150, 255),
    ]
    return palette[(int(track_id) - 1) % len(palette)]


def assignment_by_detection(assignments: list[Any]) -> dict[int, Any]:
    return {int(item.detection_index): item for item in assignments}


def draw_camera_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    assignments: list[Any],
    camera: dict[str, Any],
    score_thr: float,
) -> np.ndarray:
    intrinsics = np.asarray(camera["cam2img"], dtype=np.float32)
    lidar_to_camera = np.asarray(camera["lidar2cam"], dtype=np.float32)
    edges = ((0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    assignment_map = assignment_by_detection(assignments)
    drawn = 0
    for index, (box, score, name) in enumerate(zip(boxes, scores, names)):
        if float(score) < score_thr:
            continue
        corners = box_corners(box)
        camera_points = (lidar_to_camera @ np.c_[corners, np.ones(len(corners), dtype=np.float32)].T).T[:, :3]
        if np.count_nonzero(camera_points[:, 2] > 0.1) < 2:
            continue
        pixels = (intrinsics @ camera_points.T).T
        pixels[:, :2] /= np.maximum(pixels[:, 2:3], 1e-6)
        item = assignment_map.get(index)
        track_id = int(item.track_id) if item is not None else -1
        color = id_color(track_id) if track_id > 0 else (120, 120, 120)
        for start, end in edges:
            if camera_points[start, 2] > 0.1 and camera_points[end, 2] > 0.1:
                p1 = tuple(np.round(pixels[start, :2]).astype(int))
                p2 = tuple(np.round(pixels[end, :2]).astype(int))
                cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)
        visible = pixels[camera_points[:, 2] > 0.1, :2]
        label_xy = tuple(np.round(np.min(visible, axis=0)).astype(int))
        suffix = f" id={track_id}" if track_id > 0 else ""
        cv2.putText(image, f"{name}:{float(score):.2f}{suffix}", label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        drawn += 1
    cv2.putText(image, f"boxes >= {score_thr:.2f}: {drawn}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def render_camera_montage(
    record: dict[str, Any],
    data: dict[str, Any],
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    assignments: list[Any],
    sensor_root: Path,
    output_path: Path,
    score_thr: float,
    phase_label: str,
) -> None:
    panels = []
    for camera_name, camera_info in data["images"].items():
        image = cv2.imread(str(sensor_root / camera_info["img_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(camera_info["img_path"])
        image = draw_camera_boxes(image, boxes, scores, names, assignments, camera_info, score_thr)
        cv2.putText(image, camera_name, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
    montage = cv2.vconcat([cv2.hconcat(panels[:3]), cv2.hconcat(panels[3:6])])
    cv2.putText(montage, f"token={record['token']} timestamp={record['timestamp']}", (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(montage, phase_label, (20, 745), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Could not write {output_path}")


def render_bev(
    record: dict[str, Any],
    raw_path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    assignments: list[Any],
    output_path: Path,
    score_thr: float,
    phase_label: str,
) -> None:
    canvas_size = 800
    extent = 54.0
    points = np.fromfile(raw_path, dtype=np.float32).reshape(-1, 5)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    visible = points[(np.abs(points[:, 0]) <= extent) & (np.abs(points[:, 1]) <= extent)]
    pixels_x = np.round((visible[:, 0] + extent) / (2 * extent) * (canvas_size - 1)).astype(int)
    pixels_y = np.round((extent - visible[:, 1]) / (2 * extent) * (canvas_size - 1)).astype(int)
    intensity = np.clip(visible[:, 3], 0.0, 1.0) * 180.0 + 45.0
    canvas[pixels_y, pixels_x] = np.stack([intensity, intensity, intensity], axis=1).astype(np.uint8)

    def to_pixel(xy: np.ndarray) -> tuple[int, int]:
        return (int((float(xy[0]) + extent) / (2 * extent) * (canvas_size - 1)), int((extent - float(xy[1])) / (2 * extent) * (canvas_size - 1)))

    def draw_box(box: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        corners = box_corners(box)[[0, 2, 6, 4], :2]
        polygon = np.asarray([to_pixel(point) for point in corners], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, color, thickness, cv2.LINE_AA)

    for box in np.asarray(record["anns"]["gt_boxes"], dtype=np.float32):
        draw_box(box, (0, 0, 220), 1)
    assignment_map = assignment_by_detection(assignments)
    visible_predictions = 0
    for index, (box, score, name) in enumerate(zip(boxes, scores, names)):
        if float(score) < score_thr:
            continue
        item = assignment_map.get(index)
        track_id = int(item.track_id) if item is not None else -1
        color = id_color(track_id) if track_id > 0 else (120, 120, 120)
        # Draw the historical world trajectory transformed back into this ego frame.
        if item is not None and len(item.history_world_xy) > 1:
            inverse = np.linalg.inv(np.asarray(record["lidar2global"], dtype=np.float64))
            history_world = np.c_[np.asarray(item.history_world_xy, dtype=np.float64), np.zeros(len(item.history_world_xy)), np.ones(len(item.history_world_xy))]
            history_local = (inverse @ history_world.T).T[:, :2]
            for start, end in zip(history_local[:-1], history_local[1:]):
                cv2.line(canvas, to_pixel(start), to_pixel(end), color, 2, cv2.LINE_AA)
        draw_box(box, color, 3)
        center = np.asarray(box[:2], dtype=np.float64)
        label_x, label_y = to_pixel(center)
        velocity = item.velocity_world_xy_mps if item is not None else (0.0, 0.0)
        cv2.putText(canvas, f"#{track_id} {name} {float(score):.2f}", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
        if item is not None:
            # Velocity is represented in world coordinates; show a short local visual cue only.
            cv2.arrowedLine(canvas, (label_x, label_y), (label_x + int(velocity[0] * 8), label_y - int(velocity[1] * 8)), color, 1, cv2.LINE_AA, tipLength=0.25)
        visible_predictions += 1
    center = canvas_size // 2
    cv2.drawMarker(canvas, (center, center), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(canvas, f"{phase_label} | red=GT | color=track_id | trail=history", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"token={record['token']} boxes={visible_predictions} tracks={len({a.track_id for a in assignments})}", (12, 785), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write {output_path}")


def write_overview(image_paths: list[Path], output_path: Path, tile_size: tuple[int, int]) -> None:
    tiles = [cv2.resize(cv2.imread(str(path), cv2.IMREAD_COLOR), tile_size, interpolation=cv2.INTER_AREA) for path in image_paths]
    tiles = [tile for tile in tiles if tile is not None]
    if not tiles:
        return
    rows = []
    columns = 5
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(output_path), cv2.vconcat(rows))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.record_pkl.open("rb") as handle:
        records = pickle.load(handle)[: args.limit]
    if not records:
        raise RuntimeError("No records selected")
    sensor_root = args.sensor_root.resolve()
    raw_dir = args.output_dir / "raw_points"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_list = []
    for record in records:
        raw_path = raw_dir / f"{record['token']}.bin"
        np.asarray(load_pcd(sensor_root / record["lidar_path"]), dtype=np.float32).tofile(raw_path)
        data_list.append(make_info(record, sensor_root, raw_path))
    ann_file = args.output_dir / "mmdet3d_info.pkl"
    with ann_file.open("wb") as handle:
        pickle.dump({"data_list": data_list}, handle)

    sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d")
    sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d/projects/BEVFusion")
    import torch
    from mmengine.config import Config
    from mmdet3d.apis import inference_multi_modality_detector, init_model

    config = Config.fromfile(args.config)
    config.model.img_backbone.init_cfg = None
    if config.test_dataloader.dataset.pipeline[0].get("num_views", None) is not None:
        config.test_dataloader.dataset.pipeline[0].num_views = len(CAMERA_MAP)
    model = init_model(config, str(args.checkpoint), device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    phase1_tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=args.score_thr, history_size=12)
    phase2_tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=args.score_thr, history_size=12, query_ema=0.7)
    phase_outputs = {
        "phase1": args.output_dir / "phase1",
        "phase2": args.output_dir / "phase2",
    }
    for directory in phase_outputs.values():
        (directory / "camera").mkdir(parents=True, exist_ok=True)
        (directory / "bev").mkdir(parents=True, exist_ok=True)

    all_evidence: dict[str, list[dict[str, Any]]] = {"phase1": [], "phase2": []}
    previous_gt_tracks: dict[str, dict[str, int]] = {"phase1": {}, "phase2": {}}
    for frame_index, (record, data) in enumerate(zip(records, data_list)):
        start = time.perf_counter()
        result, _ = inference_multi_modality_detector(model, data["lidar_points"]["lidar_path"], str(sensor_root), str(ann_file), cam_type="all")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        instances = result.pred_instances_3d
        boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy().tolist()
        names = [MODEL_CLASSES[int(label)] for label in labels]
        transform = np.asarray(record["lidar2global"], dtype=np.float64)
        world_centers = transform_points(boxes[:, :3], transform) if len(boxes) else np.empty((0, 3), dtype=np.float64)
        measurements = [
            DetectionMeasurement(
                class_name=name,
                score=float(score),
                center_world_xyz=tuple(float(v) for v in center),
                size_lwh=tuple(float(v) for v in box[3:6]),
                yaw_rad=float(box[6]),
            )
            for box, score, name, center in zip(boxes, scores, names, world_centers)
        ]
        for phase_name, tracker in (("phase1", phase1_tracker), ("phase2", phase2_tracker)):
            assignments = tracker.update(measurements, timestamp_us=int(record["timestamp"]), scene_id=str(record["scene_token"]))
            phase_dir = phase_outputs[phase_name]
            camera_path = phase_dir / "camera" / f"{frame_index:04d}_{record['token']}.jpg"
            bev_path = phase_dir / "bev" / f"{frame_index:04d}_{record['token']}.jpg"
            phase_label = "PHASE 1 | Kalman/Hungarian" if phase_name == "phase1" else "PHASE 2 | Temporal Query Prototype"
            render_camera_montage(record, data, boxes, scores, names, assignments, sensor_root, camera_path, args.score_thr, phase_label)
            render_bev(record, raw_dir / f"{record['token']}.bin", boxes, scores, names, assignments, bev_path, args.score_thr, phase_label)
            instance_tokens = record["anns"].get("instance_tokens")
            if instance_tokens is None:
                instance_tokens = record["anns"].get("track_tokens")
            detection_and_tracking, previous_gt_tracks[phase_name] = match_detection_and_tracking(
                boxes,
                scores,
                names,
                assignments,
                record["anns"]["gt_boxes"],
                record["anns"]["gt_names"],
                instance_tokens,
                previous_gt_tracks[phase_name],
                score_thr=args.score_thr,
            )
            all_evidence[phase_name].append(
                {
                    "frame_index": frame_index,
                    "token": record["token"],
                    "scene_token": record["scene_token"],
                    "timestamp_us": int(record["timestamp"]),
                    "latency_ms": elapsed_ms,
                    "num_predictions_raw": int(len(boxes)),
                    "num_tracks_observed": len(assignments),
                    "track_ids": [int(item.track_id) for item in assignments],
                    "track_status": {str(item.track_id): item.status for item in assignments},
                    "detection_metrics": detection_and_tracking["per_class"],
                    "tracking_metrics": detection_and_tracking,
                    "visuals": {"camera": str(camera_path), "bev": str(bev_path)},
                    "detections": [
                        {
                            "class_name": name,
                            "score": float(score),
                            "box_lidar": [float(value) for value in box[:7]],
                            "track_id": int(next((item.track_id for item in assignments if item.detection_index == index), -1)),
                        }
                        for index, (box, score, name) in enumerate(zip(boxes, scores, names))
                        if float(score) >= args.score_thr
                    ],
                }
            )

    for phase_name, phase_dir in phase_outputs.items():
        evidence = all_evidence[phase_name]
        (phase_dir / "tracks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        write_overview([Path(item["visuals"]["camera"]) for item in evidence], phase_dir / "camera_overview.jpg", (384, 216))
        write_overview([Path(item["visuals"]["bev"]) for item in evidence], phase_dir / "bev_overview.jpg", (400, 400))
        metrics = {
            "status": "COMPLETED",
            "phase": phase_name,
            "dataset": "NavSim/OpenScene shard 5 bounded smoke slice",
            "frames": len(evidence),
            "score_threshold": args.score_thr,
            "tracking_mode": "kalman_hungarian" if phase_name == "phase1" else "persistent_temporal_query_state_prototype",
            "learned_trackformer_weights": False,
            "official_nuscenes_metrics": False,
            "latency_ms": {
                "mean": float(np.mean([item["latency_ms"] for item in evidence])),
                "p50": float(np.percentile([item["latency_ms"] for item in evidence], 50)),
                "p95": float(np.percentile([item["latency_ms"] for item in evidence], 95)),
            },
            "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else None,
            "metrics": aggregate_frame_metrics(evidence),
            "visuals": {
                "camera_overview": str(phase_dir / "camera_overview.jpg"),
                "bev_overview": str(phase_dir / "bev_overview.jpg"),
            },
            "frames_evidence": evidence,
        }
        (phase_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        aggregate = metrics["metrics"]
        det = aggregate["detection"]["per_class"]
        track = aggregate["tracking_proxy"]
        (phase_dir / "evidence.md").write_text(
            "\n".join(
                [
                    f"# {phase_name} evidence",
                    "",
                    "- Dataset: NavSim/OpenScene shard 5 bounded slice.",
                    f"- Frames: {len(evidence)} consecutive frames; no full-shard scan.",
                    f"- Tracking mode: {metrics['tracking_mode']}.",
                    "- BEVFusion checkpoint provides frame detections; tracking is applied online in metric world coordinates.",
                    "- Phase 2 query state is an explicit non-neural temporal-query prototype; it is not claimed as a trained UniAD TrackFormer checkpoint." if phase_name == "phase2" else "- Phase 1 is the online Kalman/Hungarian tracking baseline.",
                    "- Metrics are smoke evidence only and are not official nuScenes tracking scores.",
                    "",
                    "## Detection metrics",
                    "",
                    "| Class | TP | FP | FN | Precision | Recall | F1 |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                    *[f"| {name} | {det[name]['tp']} | {det[name]['fp']} | {det[name]['fn']} | {det[name]['precision']:.3f} | {det[name]['recall']:.3f} | {det[name]['f1']:.3f} |" for name in det],
                    "",
                    "## Tracking proxy metrics",
                    "",
                    f"- Precision / Recall / F1: {track['precision']:.3f} / {track['recall']:.3f} / {track['f1']:.3f}",
                    f"- Matched / GT / predicted: {track['matched']} / {track['ground_truth']} / {track['predicted']}",
                    f"- ID switches: {track['id_switches']}",
                    f"- Unique track IDs: {track['unique_track_ids']}",
                    "- This is a bounded center-distance + instance-token diagnostic, not official nuScenes AMOTA/AMOTP.",
                    "",
                    f"Visuals: `{phase_dir / 'camera_overview.jpg'}` and `{phase_dir / 'bev_overview.jpg'}`.",
                    f"Machine-readable evidence: `{phase_dir / 'metrics.json'}`.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "COMPLETED", "frames": len(records), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
