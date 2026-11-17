#!/usr/bin/env python3
"""Cache frozen BEVFusion detections and run bounded VRU recall diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d")
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d/projects/BEVFusion")

from phase12_navsim_infer import render_bev, render_camera_montage, write_overview  # noqa: E402
from phase12_nuscenes_infer import CAMERA_NAMES, build_records, make_info  # noqa: E402
from phase1_navsim_infer import MODEL_CLASSES  # noqa: E402


THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)
VEHICLE_CLASSES = {"car", "truck", "construction_vehicle", "bus", "trailer"}
VRU_CLASSES = {"pedestrian", "bicycle", "motorcycle"}
REPORT_CLASSES = ("pedestrian", "bicycle", "motorcycle", "vru", "vehicle", "traffic_cone")
VALID_NATIVE_CLASSES = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-name", action="append", required=True)
    parser.add_argument("--frame-limit", type=int, default=100)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-thresholds", default=",".join(str(value) for value in THRESHOLDS))
    parser.add_argument("--distance-threshold-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("score thresholds must be non-empty values in [0, 1]")
    return thresholds


def class_matches(name: str, target: str) -> bool:
    if target == "vehicle":
        return name in VEHICLE_CLASSES
    if target == "vru":
        return name in VRU_CLASSES
    return name == target


def native_gt_names(record: dict[str, Any]) -> list[str]:
    values = record["anns"].get("gt_native_names")
    if values is None:
        raise RuntimeError("Native GT names are missing; phase12_nuscenes_infer.py must be updated")
    return [str(value) for value in values]


def match_group(
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    gt_boxes: np.ndarray,
    gt_names: list[str],
    target: str,
    threshold: float,
    distance_threshold_m: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pred_indices = [index for index, score in enumerate(scores) if float(score) >= threshold and class_matches(names[index], target)]
    gt_indices = [index for index, name in enumerate(gt_names) if class_matches(name, target)]
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    for pred_index in sorted(pred_indices, key=lambda index: float(scores[index]), reverse=True):
        candidates = [index for index in gt_indices if index not in used_gt]
        if not candidates:
            continue
        distances = [float(np.linalg.norm(boxes[pred_index, :2] - gt_boxes[index, :2])) for index in candidates]
        best = int(np.argmin(distances))
        if distances[best] <= distance_threshold_m:
            gt_index = candidates[best]
            used_gt.add(gt_index)
            matches.append((pred_index, gt_index))
    unmatched_predictions = [index for index in pred_indices if index not in {pair[0] for pair in matches}]
    unmatched_gt = [index for index in gt_indices if index not in used_gt]
    return matches, unmatched_predictions, unmatched_gt


def accumulate_counts(rows: Iterable[tuple[int, int, int, int]]) -> dict[str, int | float]:
    tp = sum(row[0] for row in rows)
    fp = sum(row[1] for row in rows)
    fn = sum(row[2] for row in rows)
    gt = sum(row[3] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def frame_payload(record: dict[str, Any], boxes: np.ndarray, scores: np.ndarray, names: list[str]) -> dict[str, Any]:
    return {
        "frame_index": int(record["frame_index"]),
        "token": record["token"],
        "scene_name": record["scene_name"],
        "scene_token": record["scene_token"],
        "timestamp_us": int(record["timestamp"]),
        "lidar_path": record["lidar_path"],
        "boxes_lidar": boxes[:, :7].astype(float).tolist(),
        "scores": scores.astype(float).tolist(),
        "names": names,
        "gt_boxes_lidar": np.asarray(record["anns"]["gt_boxes"], dtype=np.float32).astype(float).tolist(),
        "gt_native_names": native_gt_names(record),
        "gt_instance_tokens": list(record["anns"].get("instance_tokens", [])),
    }


def load_cached_frames(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty frame list in {path}")
    return value


def evaluate_thresholds(frames: list[dict[str, Any]], thresholds: list[float], distance_threshold_m: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for target in REPORT_CLASSES:
            frame_counts = []
            for frame in frames:
                boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
                scores = np.asarray(frame["scores"], dtype=np.float64)
                names = [str(value) for value in frame["names"]]
                gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
                gt_names = [str(value) for value in frame["gt_native_names"]]
                matches, unmatched_predictions, unmatched_gt = match_group(
                    boxes, scores, names, gt_boxes, gt_names, target, threshold, distance_threshold_m
                )
                frame_counts.append((len(matches), len(unmatched_predictions), len(unmatched_gt), sum(class_matches(name, target) for name in gt_names)))
            summary = accumulate_counts(frame_counts)
            rows.append({"threshold": threshold, "class": target, "detections": sum(row[0] + row[1] for row in frame_counts), **summary})
    return rows


def evaluate_distance_recall(frames: list[dict[str, Any]], threshold: float, distance_threshold_m: float) -> list[dict[str, Any]]:
    bins = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0))
    accum: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for frame in frames:
        boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(frame["scores"], dtype=np.float64)
        names = [str(value) for value in frame["names"]]
        gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        gt_names = [str(value) for value in frame["gt_native_names"]]
        matches, _, _ = match_group(boxes, scores, names, gt_boxes, gt_names, "vru", threshold, distance_threshold_m)
        matched_gt = {gt_index for _, gt_index in matches}
        for gt_index, name in enumerate(gt_names):
            if name not in VRU_CLASSES:
                continue
            distance = float(np.linalg.norm(gt_boxes[gt_index, :2]))
            bin_name = next((f"{low:g}-{high:g}m" for low, high in bins if low <= distance < high), None)
            if bin_name is None:
                continue
            key = (name, bin_name)
            accum[key][1] += 1
            if gt_index in matched_gt:
                accum[key][0] += 1
    rows = []
    for (class_name, bin_name), (tp, gt_count) in sorted(accum.items()):
        rows.append({"class": class_name, "distance_bin": bin_name, "tp": tp, "gt_count": gt_count, "recall": tp / gt_count if gt_count else 0.0})
    return rows


def false_negative_analysis(frames: list[dict[str, Any]], threshold: float, distance_threshold_m: float) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(frame["scores"], dtype=np.float64)
        names = [str(value) for value in frame["names"]]
        gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        gt_names = [str(value) for value in frame["gt_native_names"]]
        _, _, unmatched_gt = match_group(boxes, scores, names, gt_boxes, gt_names, "vru", threshold, distance_threshold_m)
        for gt_index in unmatched_gt:
            gt_name = gt_names[gt_index]
            distances = np.linalg.norm(boxes[:, :2] - gt_boxes[gt_index, :2], axis=1) if len(boxes) else np.empty(0)
            same_class = [index for index, (name, distance) in enumerate(zip(names, distances)) if name == gt_name and distance <= distance_threshold_m]
            any_class = [index for index, distance in enumerate(distances) if distance <= distance_threshold_m]
            if any(index for index in same_class if float(scores[index]) >= threshold):
                category = "candidate_competed_or_matching_conflict"
            elif same_class:
                category = "candidate_below_threshold"
            elif any_class:
                category = "wrong_class_raw_candidate"
            else:
                category = "no_same_class_candidate_within_2m"
            counts[category] += 1
            if len(examples[category]) < 10:
                candidates = sorted(
                    ({"name": names[index], "score": float(scores[index]), "distance_m": float(distances[index])} for index in any_class),
                    key=lambda item: item["distance_m"],
                )[:5]
                examples[category].append(
                    {
                        "frame_index": frame["frame_index"],
                        "token": frame["token"],
                        "scene_name": frame["scene_name"],
                        "gt_class": gt_name,
                        "gt_distance_m": float(np.linalg.norm(gt_boxes[gt_index, :2])),
                        "gt_center_lidar": gt_boxes[gt_index, :3].astype(float).tolist(),
                        "nearby_raw_predictions": candidates,
                    }
                )
    return {"score_threshold": threshold, "distance_threshold_m": distance_threshold_m, "counts": dict(counts), "examples": dict(examples)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_diagnostics(frames: list[dict[str, Any]], analysis: dict[str, Any], root: Path, output_dir: Path, threshold: float) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict[str, Any]] = {}
    for category, values in analysis.get("examples", {}).items():
        if values:
            selected[category] = values[0]
    paths: list[str] = []
    by_token = {frame["token"]: frame for frame in frames}
    for category, example in selected.items():
        frame = by_token[example["token"]]
        record = {
            "token": frame["token"],
            "timestamp": frame["timestamp_us"],
            "lidar_path": frame["lidar_path"],
            "lidar2global": np.eye(4, dtype=np.float64),
            "anns": {"gt_boxes": np.asarray(frame["gt_boxes_lidar"], dtype=np.float32)},
        }
        data = {"images": {name: {"img_path": "samples/" + name + "/missing.jpg"} for name in CAMERA_NAMES}}
        # Reconstruct camera paths from the nuScenes record through the cached info pkl.
        info_path = output_dir.parent / "mmdet3d_info.pkl"
        with info_path.open("rb") as handle:
            info = pickle.load(handle)["data_list"][int(frame["frame_index"])]
        data = {"images": info["images"]}
        boxes = np.asarray(frame["boxes_lidar"], dtype=np.float32).reshape(-1, 7)
        scores = np.asarray(frame["scores"], dtype=np.float32)
        names = [str(value) for value in frame["names"]]
        bev_path = output_dir / f"{category}_bev.jpg"
        camera_path = output_dir / f"{category}_camera.jpg"
        render_bev(record, Path(frame["lidar_path"]), boxes, scores, names, [], bev_path, threshold, f"PHASE 1 FN | {category}")
        render_camera_montage(record, data, boxes, scores, names, [], root, camera_path, threshold, f"PHASE 1 FN | {category}")
        paths.extend([str(bev_path), str(camera_path)])
    if paths:
        write_overview([Path(path) for path in paths], output_dir / "diagnostic_overview.jpg", (400, 300))
    return paths


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.score_thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.data_root.is_dir() or not args.checkpoint.is_file() or not args.config.is_file():
        raise FileNotFoundError("data root, checkpoint and config must exist")
    np.random.seed(args.seed)

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    records = build_records(nusc, args.data_root.resolve(), args.scene_name, args.frame_limit)
    if not records:
        raise RuntimeError("No complete nuScenes frames selected")
    for index, record in enumerate(records):
        record["frame_index"] = index
    data_list = [make_info(record) for record in records]
    ann_file = args.output_dir / "mmdet3d_info.pkl"
    with ann_file.open("wb") as handle:
        pickle.dump({"data_list": data_list}, handle)

    import torch
    from mmengine.config import Config
    from mmdet3d.apis import inference_multi_modality_detector, init_model

    torch.manual_seed(args.seed)
    config = Config.fromfile(args.config)
    config.model.img_backbone.init_cfg = None
    if config.test_dataloader.dataset.pipeline[0].get("num_views", None) is not None:
        config.test_dataloader.dataset.pipeline[0].num_views = len(CAMERA_NAMES)
    model = init_model(config, str(args.checkpoint), device=args.device)
    frames: list[dict[str, Any]] = []
    for record, data in zip(records, data_list):
        result, _ = inference_multi_modality_detector(model, record["lidar_path"], str(args.data_root), str(ann_file), cam_type="all")
        instances = result.pred_instances_3d
        boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy().tolist()
        names = [MODEL_CLASSES[int(label)] for label in labels]
        frames.append(frame_payload(record, boxes, scores, names))

    raw_path = args.output_dir / "raw_detections.json"
    raw_path.write_text(json.dumps(frames, indent=2), encoding="utf-8")
    manifest = {
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "data_root": str(args.data_root.resolve()),
        "version": args.version,
        "scenes": args.scene_name,
        "frames": len(frames),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "raw_detections_note": "model outputs after BEVFusion inference/NMS/top-K, before experiment score filtering",
        "distance_threshold_m": args.distance_threshold_m,
        "thresholds": thresholds,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sweep = evaluate_thresholds(frames, thresholds, args.distance_threshold_m)
    write_csv(args.output_dir / "threshold_sweep.csv", sweep)
    distance = evaluate_distance_recall(frames, 0.20, args.distance_threshold_m)
    write_csv(args.output_dir / "distance_recall.csv", distance)
    fn_analysis = false_negative_analysis(frames, 0.20, args.distance_threshold_m)
    diagnostic_paths = render_diagnostics(frames, fn_analysis, args.data_root.resolve(), args.output_dir / "diagnostics", 0.20)
    fn_analysis["diagnostic_visualizations"] = diagnostic_paths
    (args.output_dir / "false_negative_analysis.json").write_text(json.dumps(fn_analysis, indent=2), encoding="utf-8")
    summary = {
        "status": "COMPLETED",
        "frames": len(frames),
        "scenes": args.scene_name,
        "raw_detections": str(raw_path),
        "threshold_sweep": str(args.output_dir / "threshold_sweep.csv"),
        "distance_recall": str(args.output_dir / "distance_recall.csv"),
        "false_negative_analysis": str(args.output_dir / "false_negative_analysis.json"),
        "manifest": str(args.output_dir / "manifest.json"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
