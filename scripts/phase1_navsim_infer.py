#!/usr/bin/env python3
"""Run BEVFusion on an OpenScene/NavSim sensor-blob sequence.

This is an evidence runner, not an official nuScenes evaluator.  The input
sequence has eight NavSim cameras and PCD files, while the released BEVFusion
checkpoint/config expects the six-camera nuScenes layout.  The mapping and
the coarse evaluation protocol are recorded in the output JSON.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_MAP = {
    "CAM_FRONT": "CAM_F0",
    "CAM_FRONT_LEFT": "CAM_L0",
    "CAM_FRONT_RIGHT": "CAM_R0",
    "CAM_BACK": "CAM_B0",
    "CAM_BACK_RIGHT": "CAM_R2",
    "CAM_BACK_LEFT": "CAM_L2",
}
MODEL_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
PRED_TO_COARSE = {
    **{name: "vehicle" for name in MODEL_CLASSES[:5]},
    "barrier": "generic_object",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone",
}
GT_CLASSES = ("vehicle", "pedestrian", "traffic_cone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-pkl", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def load_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    marker = raw.find(b"DATA binary\n")
    if marker < 0:
        raise ValueError(f"Only binary PCD is supported: {path}")
    header = raw[: marker + len(b"DATA binary\n")].decode("ascii")
    fields = next(line for line in header.splitlines() if line.startswith("FIELDS ")).split()[1:]
    sizes = [int(v) for v in next(line for line in header.splitlines() if line.startswith("SIZE ")).split()[1:]]
    types = next(line for line in header.splitlines() if line.startswith("TYPE ")).split()[1:]
    counts = [int(v) for v in next(line for line in header.splitlines() if line.startswith("COUNT ")).split()[1:]]
    dtype_fields: list[tuple[str, str]] = []
    for name, size, kind, count in zip(fields, sizes, types, counts):
        if count != 1:
            raise ValueError(f"Unsupported PCD COUNT in {path}")
        if kind == "F" and size == 4:
            dtype_fields.append((name, "<f4"))
        elif kind == "U" and size == 1:
            dtype_fields.append((name, "u1"))
        else:
            raise ValueError(f"Unsupported PCD field {name}/{kind}{size} in {path}")
    points = np.frombuffer(raw[marker + len(b"DATA binary\n") :], dtype=np.dtype(dtype_fields))
    return np.stack(
        [
            points["x"].astype(np.float32),
            points["y"].astype(np.float32),
            points["z"].astype(np.float32),
            points["intensity"].astype(np.float32),
            points["ring"].astype(np.float32),
        ],
        axis=1,
    )


def lidar2cam(sensor2lidar_rotation: Any, sensor2lidar_translation: Any) -> np.ndarray:
    sensor_from_lidar = np.eye(4, dtype=np.float32)
    rotation = np.asarray(sensor2lidar_rotation, dtype=np.float32)
    translation = np.asarray(sensor2lidar_translation, dtype=np.float32)
    sensor_from_lidar[:3, :3] = rotation.T
    sensor_from_lidar[:3, 3] = -rotation.T @ translation
    return sensor_from_lidar


def make_info(record: dict[str, Any], sensor_root: Path, raw_path: Path) -> dict[str, Any]:
    images: dict[str, dict[str, Any]] = {}
    for target_name, source_name in CAMERA_MAP.items():
        camera = record["cams"][source_name]
        images[target_name] = {
            "img_path": camera["data_path"],
            "cam2img": np.asarray(camera["cam_intrinsic"], dtype=np.float32),
            "lidar2cam": lidar2cam(camera["sensor2lidar_rotation"], camera["sensor2lidar_translation"]),
        }
    return {
        "sample_idx": record["token"],
        "timestamp": float(record["timestamp"]) / 1_000_000.0,
        "lidar_points": {"lidar_path": str(raw_path)},
        "images": images,
        "ann_info": {},
    }


def match_frame(pred_boxes: np.ndarray, pred_scores: np.ndarray, pred_names: list[str], record: dict[str, Any], score_thr: float) -> dict[str, Any]:
    gt_boxes = np.asarray(record["anns"]["gt_boxes"], dtype=np.float32)
    gt_names = np.asarray(record["anns"]["gt_names"])
    pred_mask = pred_scores >= score_thr
    pred_boxes = pred_boxes[pred_mask]
    pred_scores = pred_scores[pred_mask]
    pred_names = [name for name, keep in zip(pred_names, pred_mask) if keep]
    result: dict[str, Any] = {"predictions": int(len(pred_boxes)), "ground_truth": int(len(gt_boxes))}
    for coarse in GT_CLASSES:
        gt_idx = [i for i, name in enumerate(gt_names) if name == coarse]
        pred_idx = [i for i, name in enumerate(pred_names) if PRED_TO_COARSE.get(name) == coarse]
        used: set[int] = set()
        tp = 0
        for pi in sorted(pred_idx, key=lambda i: float(pred_scores[i]), reverse=True):
            candidates = [gi for gi in gt_idx if gi not in used]
            if not candidates:
                continue
            distances = np.linalg.norm(gt_boxes[candidates, :2] - pred_boxes[pi, :2], axis=1)
            best = int(np.argmin(distances))
            if float(distances[best]) <= 2.0:
                used.add(candidates[best])
                tp += 1
        fp = len(pred_idx) - tp
        fn = len(gt_idx) - tp
        result[coarse] = {"tp": tp, "fp": fp, "fn": fn}
    return result


def box_corners(box: np.ndarray) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array(
        [
            [-dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [-dx / 2, dy / 2, dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [dx / 2, dy / 2, -dz / 2],
            [dx / 2, dy / 2, dz / 2],
        ],
        dtype=np.float32,
    )
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return local @ rotation.T + np.array([x, y, z], dtype=np.float32)


def draw_projected_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    camera: dict[str, Any],
    score_thr: float,
) -> np.ndarray:
    intrinsics = np.asarray(camera["cam2img"], dtype=np.float32)
    lidar_to_camera = np.asarray(camera["lidar2cam"], dtype=np.float32)
    edges = ((0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    colors = {"vehicle": (0, 220, 0), "pedestrian": (0, 165, 255), "traffic_cone": (0, 0, 255)}
    drawn = 0
    for box, score, name in zip(boxes, scores, names):
        if float(score) < score_thr:
            continue
        corners = box_corners(box)
        camera_points = (lidar_to_camera @ np.c_[corners, np.ones(len(corners), dtype=np.float32)].T).T[:, :3]
        if np.count_nonzero(camera_points[:, 2] > 0.1) < 2:
            continue
        pixels = (intrinsics @ camera_points.T).T
        pixels[:, :2] /= np.maximum(pixels[:, 2:3], 1e-6)
        coarse = PRED_TO_COARSE.get(name, "generic_object")
        color = colors.get(coarse, (255, 0, 255))
        for start, end in edges:
            if camera_points[start, 2] > 0.1 and camera_points[end, 2] > 0.1:
                p1 = tuple(np.round(pixels[start, :2]).astype(int))
                p2 = tuple(np.round(pixels[end, :2]).astype(int))
                cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)
        visible = pixels[camera_points[:, 2] > 0.1, :2]
        label_xy = tuple(np.round(np.min(visible, axis=0)).astype(int))
        cv2.putText(
            image,
            f"{name}:{float(score):.2f}",
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        drawn += 1
    cv2.putText(image, f"predictions >= {score_thr:.2f}: {drawn}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def render_frame(
    record: dict[str, Any],
    data: dict[str, Any],
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    sensor_root: Path,
    output_path: Path,
    score_thr: float,
) -> None:
    panels = []
    for camera_name, camera_info in data["images"].items():
        image_path = sensor_root / camera_info["img_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read camera image: {image_path}")
        image = draw_projected_boxes(image, boxes, scores, names, camera_info, score_thr)
        cv2.putText(image, camera_name, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
    rows = [cv2.hconcat(panels[index : index + 3]) for index in (0, 3)]
    montage = cv2.vconcat(rows)
    cv2.putText(montage, f"token={record['token']}", (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Could not write visualization: {output_path}")


def render_bev(
    record: dict[str, Any],
    raw_path: Path,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_names: list[str],
    output_path: Path,
    score_thr: float,
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

    def draw_box(box: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        corners = box_corners(box)[[0, 2, 6, 4], :2]
        px = np.round((corners[:, 0] + extent) / (2 * extent) * (canvas_size - 1)).astype(int)
        py = np.round((extent - corners[:, 1]) / (2 * extent) * (canvas_size - 1)).astype(int)
        polygon = np.stack([px, py], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, color, thickness, cv2.LINE_AA)

    for box in np.asarray(record["anns"]["gt_boxes"], dtype=np.float32):
        draw_box(box, (0, 0, 220), 1)
    colors = {"vehicle": (0, 220, 0), "pedestrian": (0, 165, 255), "traffic_cone": (255, 0, 255)}
    visible_predictions = 0
    for box, score, name in zip(pred_boxes, pred_scores, pred_names):
        if float(score) < score_thr:
            continue
        coarse = PRED_TO_COARSE.get(name, "generic_object")
        draw_box(box, colors.get(coarse, (0, 220, 220)), 3)
        center = box_corners(box).mean(axis=0)
        label_x = int((center[0] + extent) / (2 * extent) * (canvas_size - 1))
        label_y = int((extent - center[1]) / (2 * extent) * (canvas_size - 1))
        cv2.putText(canvas, f"{name}:{float(score):.2f}", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(coarse, (0, 220, 220)), 1, cv2.LINE_AA)
        visible_predictions += 1
    center = canvas_size // 2
    cv2.drawMarker(canvas, (center, center), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(canvas, "BEV: x forward, y left | red GT | green/orange/magenta predictions", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"token={record['token']}  predictions >= {score_thr:.2f}: {visible_predictions}", (12, 785), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write BEV visualization: {output_path}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.record_pkl.open("rb") as handle:
        records = pickle.load(handle)
    records = records[: args.limit]
    if not records:
        raise RuntimeError("record pkl contains no frames")
    sensor_root = args.sensor_root.resolve()
    raw_dir = args.output_dir / "raw_points"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_list = []
    for record in records:
        pcd_path = sensor_root / record["lidar_path"]
        raw_path = raw_dir / f"{record['token']}.bin"
        np.asarray(load_pcd(pcd_path), dtype=np.float32).tofile(raw_path)
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
    timings_ms: list[float] = []
    frame_evidence: list[dict[str, Any]] = []
    visualization_paths: list[str] = []
    bev_visualization_paths: list[str] = []
    for record, data in zip(records, data_list):
        start = time.perf_counter()
        result, _ = inference_multi_modality_detector(
            model,
            data["lidar_points"]["lidar_path"],
            str(sensor_root),
            str(ann_file),
            cam_type="all",
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed)
        instances = result.pred_instances_3d
        boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy().tolist()
        names = [MODEL_CLASSES[int(label)] for label in labels]
        if args.visualize:
            visualization_path = args.output_dir / f"{record['token']}.jpg"
            render_frame(record, data, boxes, scores, names, sensor_root, visualization_path, args.score_thr)
            visualization_paths.append(str(visualization_path))
            bev_path = args.output_dir / f"bev_{record['token']}.jpg"
            render_bev(record, Path(data["lidar_points"]["lidar_path"]), boxes, scores, names, bev_path, args.score_thr)
            bev_visualization_paths.append(str(bev_path))
        frame_evidence.append(
            {
                "token": record["token"],
                "latency_ms": elapsed,
                "num_predictions_raw": int(len(boxes)),
                "metrics": match_frame(boxes, scores, names, record, args.score_thr),
            }
        )
    summary: dict[str, Any] = {
        "status": "COMPLETED",
        "dataset": "OpenScene/NavSim sensor_blobs (not native nuScenes)",
        "frames": len(records),
        "camera_mapping": CAMERA_MAP,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "score_threshold": args.score_thr,
        "metric_protocol": "coarse class mapping; greedy BEV center-distance <= 2m; no official nuScenes score",
        "timing_ms": {
            "mean": float(np.mean(timings_ms)),
            "p50": float(np.percentile(timings_ms, 50)),
            "p95": float(np.percentile(timings_ms, 95)),
        },
        "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else None,
        "frames_evidence": frame_evidence,
    }
    aggregate: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for coarse in GT_CLASSES:
        tp = sum(frame["metrics"][coarse]["tp"] for frame in frame_evidence)
        fp = sum(frame["metrics"][coarse]["fp"] for frame in frame_evidence)
        fn = sum(frame["metrics"][coarse]["fn"] for frame in frame_evidence)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate[coarse] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    aggregate["micro_average"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    summary["aggregate_metrics"] = aggregate
    if args.visualize:
        overview_paths = []
        for path in visualization_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                overview_paths.append(cv2.resize(image, (384, 216), interpolation=cv2.INTER_AREA))
        if overview_paths:
            rows = []
            for index in range(0, len(overview_paths), 5):
                row = overview_paths[index : index + 5]
                while len(row) < 5:
                    row.append(np.zeros_like(overview_paths[0]))
                rows.append(cv2.hconcat(row))
            overview_path = args.output_dir / "overview.jpg"
            cv2.imwrite(str(overview_path), cv2.vconcat(rows))
            bev_overview_images = [cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (400, 400), interpolation=cv2.INTER_AREA) for path in bev_visualization_paths]
            bev_rows = []
            for index in range(0, len(bev_overview_images), 5):
                row = bev_overview_images[index : index + 5]
                while len(row) < 5:
                    row.append(np.zeros_like(bev_overview_images[0]))
                bev_rows.append(cv2.hconcat(row))
            bev_overview_path = args.output_dir / "bev_overview.jpg"
            cv2.imwrite(str(bev_overview_path), cv2.vconcat(bev_rows))
            summary["visualizations"] = {
                "overview": str(overview_path),
                "frames": visualization_paths,
                "bev_overview": str(bev_overview_path),
                "bev_frames": bev_visualization_paths,
            }
    (args.output_dir / "metrics.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(__import__("json").dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
