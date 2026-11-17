#!/usr/bin/env python3
"""Evaluate the classical tracker and a GT-box oracle on cached detections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d")
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d/projects/BEVFusion")

from bevdrive.tracking import DetectionMeasurement, TrackManager  # noqa: E402
from phase12_nuscenes_infer import build_records  # noqa: E402
from phase1_navsim_infer import MODEL_CLASSES  # noqa: E402
from vru_detection_experiment import VEHICLE_CLASSES, VRU_CLASSES  # noqa: E402


CLASS_NAMES = tuple(MODEL_CLASSES)


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
    parser.add_argument("--tracker-distance-m", type=float, default=3.0)
    parser.add_argument("--max-missed-frames", type=int, default=3)
    return parser.parse_args()


def class_group(name: str) -> str:
    if name in VEHICLE_CLASSES:
        return "vehicle"
    if name in VRU_CLASSES:
        return "vru"
    return name


def gated_hungarian(
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    gt_boxes: np.ndarray,
    gt_names: list[str],
    threshold: float,
    distance_threshold_m: float,
) -> list[tuple[int, int]]:
    pred_indices = [index for index, score in enumerate(scores) if float(score) >= threshold]
    if not pred_indices or not len(gt_boxes):
        return []
    cost = np.full((len(pred_indices), len(gt_boxes)), 1e6, dtype=np.float64)
    for row, pred_index in enumerate(pred_indices):
        for gt_index, gt_name in enumerate(gt_names):
            if names[pred_index] != gt_name:
                continue
            distance = float(np.linalg.norm(boxes[pred_index, :2] - gt_boxes[gt_index, :2]))
            if distance <= distance_threshold_m:
                cost[row, gt_index] = distance
    rows, columns = linear_sum_assignment(cost)
    return [(pred_indices[int(row)], int(column)) for row, column in zip(rows, columns) if cost[row, column] < 1e6]


def measurements(boxes: np.ndarray, scores: np.ndarray, names: list[str], lidar2global: np.ndarray) -> list[DetectionMeasurement]:
    centers = np.c_[boxes[:, :3], np.ones(len(boxes))]
    world_centers = (lidar2global @ centers.T).T[:, :3]
    return [
        DetectionMeasurement(
            class_name=name,
            score=float(score),
            center_world_xyz=tuple(float(value) for value in center),
            size_lwh=tuple(float(value) for value in box[3:6]),
            yaw_rad=float(box[6]),
        )
        for box, score, name, center in zip(boxes, scores, names, world_centers)
    ]


def track_identity_metrics(
    frames: list[dict[str, Any]],
    transforms: dict[str, np.ndarray],
    *,
    oracle: bool,
    score_threshold: float,
    tracker_distance_m: float,
    max_missed_frames: int,
    distance_threshold_m: float,
) -> dict[str, Any]:
    tracker = TrackManager(max_distance_m=tracker_distance_m, max_missed_frames=max_missed_frames, min_score=score_threshold)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    states: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    scene_metrics: dict[str, dict[str, Any]] = {}
    for frame_index, frame in enumerate(frames):
        scene = str(frame["scene_name"])
        if scene not in scene_metrics:
            scene_metrics[scene] = {"frames": 0, "id_switches": 0, "fragmentations": 0, "gap_reappearances": 0, "lost_frames": 0}
        scene_metrics[scene]["frames"] += 1
        for token in list(states):
            if states[token].get("scene") != scene:
                states.pop(token, None)
        boxes = np.asarray(frame["gt_boxes_lidar"] if oracle else frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        scores = np.ones(len(boxes), dtype=np.float64) if oracle else np.asarray(frame["scores"], dtype=np.float64)
        names = [str(value) for value in (frame["gt_native_names"] if oracle else frame["names"])]
        gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        gt_names = [str(value) for value in frame["gt_native_names"]]
        assignment = tracker.update(measurements(boxes, scores, names, transforms[frame["token"]]), timestamp_us=int(frame["timestamp_us"]), scene_id=scene)
        track_by_detection = {int(item.detection_index): int(item.track_id) for item in assignment}
        matches = gated_hungarian(boxes, scores, names, gt_boxes, gt_names, score_threshold, distance_threshold_m)
        per_class_frame: dict[str, dict[str, int]] = {}
        for class_name in CLASS_NAMES:
            gt_count = sum(name == class_name for name in gt_names)
            pred_count = sum(name == class_name and index in track_by_detection for index, name in enumerate(names))
            tp = sum(gt_names[gt_index] == class_name for pred_index, gt_index in matches)
            per_class_frame[class_name] = {"tp": tp, "fp": pred_count - tp, "fn": gt_count - tp}
            current = counts[class_name]
            current[0] += tp
            current[1] += pred_count - tp
            current[2] += gt_count - tp
        frame_events: list[dict[str, Any]] = []
        for pred_index, gt_index in matches:
            token = str(frame["gt_instance_tokens"][gt_index])
            track_id = int(track_by_detection.get(pred_index, -1))
            if track_id < 0:
                continue
            previous = states.get(token)
            if previous is not None:
                gap = frame_index - int(previous["frame_index"]) - 1
                if gap == 0 and track_id != int(previous["track_id"]):
                    event = {"type": "id_switch", "scene_name": scene, "frame_index": frame_index, "token": token, "class_name": gt_names[gt_index], "old_track_id": int(previous["track_id"]), "new_track_id": track_id, "gap_frames": 0}
                    events.append(event)
                    frame_events.append(event)
                    scene_metrics[scene]["id_switches"] += 1
                elif gap > 0:
                    scene_metrics[scene]["gap_reappearances"] += 1
                    scene_metrics[scene]["lost_frames"] += gap
                    if track_id != int(previous["track_id"]):
                        event = {"type": "fragmentation", "scene_name": scene, "frame_index": frame_index, "token": token, "class_name": gt_names[gt_index], "old_track_id": int(previous["track_id"]), "new_track_id": track_id, "gap_frames": gap}
                        events.append(event)
                        frame_events.append(event)
                        scene_metrics[scene]["fragmentations"] += 1
            states[token] = {"scene": scene, "frame_index": frame_index, "track_id": track_id}
        all_frames.append({"frame_index": frame_index, "scene_name": scene, "matched": len(matches), "predicted": len(track_by_detection), "ground_truth": len(gt_boxes), "per_class": per_class_frame, "events": frame_events})

    aggregate_classes: dict[str, dict[str, float | int]] = {}
    for class_name, (tp, fp, fn) in counts.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate_classes[class_name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    total = {key: sum(int(frame[key]) for frame in all_frames) for key in ("matched", "predicted", "ground_truth")}
    total["id_switches"] = sum(int(value["id_switches"]) for value in scene_metrics.values())
    total["fragmentations"] = sum(int(value["fragmentations"]) for value in scene_metrics.values())
    total["gap_reappearances"] = sum(int(value["gap_reappearances"]) for value in scene_metrics.values())
    total["lost_frames"] = sum(int(value["lost_frames"]) for value in scene_metrics.values())
    total["precision"] = total["matched"] / total["predicted"] if total["predicted"] else 0.0
    total["recall"] = total["matched"] / total["ground_truth"] if total["ground_truth"] else 0.0
    total["f1"] = 2.0 * total["precision"] * total["recall"] / (total["precision"] + total["recall"]) if total["precision"] + total["recall"] else 0.0
    return {"oracle_boxes": oracle, "score_threshold": score_threshold, "distance_threshold_m": distance_threshold_m, "tracking": total, "per_class": aggregate_classes, "per_scene": scene_metrics, "id_switch_events": events, "frames": all_frames}


def write_per_scene_class_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = []
    for class_name, values in metrics["per_class"].items():
        rows.append({"scope": "aggregate", "scene": "all", "class": class_name, **values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = json.loads(args.cached_detections.read_text(encoding="utf-8"))
    if not frames:
        raise ValueError("cached detections are empty")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    records = build_records(nusc, args.data_root.resolve(), args.scene_name, args.frame_limit)
    record_by_token = {record["token"]: record for record in records}
    if set(record_by_token) != {frame["token"] for frame in frames}:
        raise ValueError("cached detections do not match the selected scene/frame records")
    transforms = {token: np.asarray(record["lidar2global"], dtype=np.float64) for token, record in record_by_token.items()}
    predicted = track_identity_metrics(frames, transforms, oracle=False, score_threshold=args.score_threshold, tracker_distance_m=args.tracker_distance_m, max_missed_frames=args.max_missed_frames, distance_threshold_m=args.distance_threshold_m)
    oracle = track_identity_metrics(frames, transforms, oracle=True, score_threshold=args.score_threshold, tracker_distance_m=args.tracker_distance_m, max_missed_frames=args.max_missed_frames, distance_threshold_m=args.distance_threshold_m)
    result = {"status": "COMPLETED", "config": {"score_threshold": args.score_threshold, "tracker_distance_m": args.tracker_distance_m, "max_missed_frames": args.max_missed_frames, "distance_threshold_m": args.distance_threshold_m, "scenes": args.scene_name, "frames": len(frames)}, "predicted_boxes": predicted, "gt_box_oracle": oracle}
    (args.output_dir / "oracle_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "id_switch_events.json").write_text(json.dumps({"predicted_boxes": predicted["id_switch_events"], "gt_box_oracle": oracle["id_switch_events"]}, indent=2), encoding="utf-8")
    write_per_scene_class_csv(args.output_dir / "predicted_per_class.csv", predicted)
    write_per_scene_class_csv(args.output_dir / "oracle_per_class.csv", oracle)
    print(json.dumps({"status": "COMPLETED", "frames": len(frames), "output_dir": str(args.output_dir), "predicted": predicted["tracking"], "oracle": oracle["tracking"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
