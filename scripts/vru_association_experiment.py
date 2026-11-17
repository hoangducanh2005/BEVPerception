#!/usr/bin/env python3
"""Run controlled classical versus optional BEV-query association ablations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bevdrive.tracking import DetectionMeasurement, QueryAssociationTrackManager, TrackManager  # noqa: E402
from vru_detection_experiment import VRU_CLASSES  # noqa: E402
from vru_tracking_oracle import gated_hungarian  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transforms-json", required=True, type=Path)
    parser.add_argument("--cached-detections", required=True, type=Path)
    parser.add_argument("--detection-embedding-meta", required=True, type=Path)
    parser.add_argument("--detection-embeddings", required=True, type=Path)
    parser.add_argument("--cross-camera-events", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--distance-threshold-m", type=float, default=2.0)
    return parser.parse_args()


def measurements(frame: dict[str, Any], lidar2global: np.ndarray) -> list[DetectionMeasurement]:
    boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
    scores = np.asarray(frame["scores"], dtype=np.float64)
    names = [str(value) for value in frame["names"]]
    centers = (lidar2global @ np.c_[boxes[:, :3], np.ones(len(boxes))].T).T[:, :3]
    return [DetectionMeasurement(name, float(score), tuple(float(value) for value in center), tuple(float(value) for value in box[3:6]), float(box[6])) for box, score, name, center in zip(boxes, scores, names, centers)]


def run_assignments(
    frames: list[dict[str, Any]],
    method: str,
    embedding_by_frame: dict[int, dict[int, np.ndarray | None]],
    score_threshold: float,
    transforms: dict[str, np.ndarray],
) -> tuple[dict[str, dict[int, int]], dict[str, float]]:
    if method == "classical":
        tracker: Any = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=score_threshold)
    else:
        tracker = QueryAssociationTrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=score_threshold, motion_weight=0.7, appearance_weight=0.3)
    track_maps: dict[str, dict[int, int]] = {}
    timings: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        values = measurements(frame, transforms[frame["token"]])
        if method == "classical":
            assignments = tracker.update(values, timestamp_us=int(frame["timestamp_us"]), scene_id=str(frame["scene_name"]))
        else:
            embeddings = [embedding_by_frame.get(int(frame["frame_index"]), {}).get(index) for index in range(len(values))]
            assignments = tracker.update_with_embeddings(values, embeddings, timestamp_us=int(frame["timestamp_us"]), scene_id=str(frame["scene_name"]))
        timings.append((time.perf_counter() - start) * 1000.0)
        track_maps[frame["token"]] = {int(item.detection_index): int(item.track_id) for item in assignments}
    return track_maps, {"mean_ms": float(np.mean(timings)), "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95))}


def evaluate(
    frames: list[dict[str, Any]],
    track_maps: dict[str, dict[int, int]],
    score_threshold: float,
    distance_threshold_m: float,
) -> dict[str, Any]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    states: dict[str, dict[str, Any]] = {}
    scene_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"id_switches": 0, "fragmentations": 0, "gap_reappearances": 0, "lost_frames": 0})
    events: list[dict[str, Any]] = []
    total_matched = total_predicted = total_gt = 0
    for frame_index, frame in enumerate(frames):
        scene = str(frame["scene_name"])
        for token in list(states):
            if states[token]["scene"] != scene:
                states.pop(token, None)
        boxes = np.asarray(frame["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(frame["scores"], dtype=np.float64)
        names = [str(value) for value in frame["names"]]
        gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        gt_names = [str(value) for value in frame["gt_native_names"]]
        matches = gated_hungarian(boxes, scores, names, gt_boxes, gt_names, score_threshold, distance_threshold_m)
        track_by_detection = track_maps[frame["token"]]
        predicted_count = len(track_by_detection)
        total_matched += len(matches)
        total_predicted += predicted_count
        total_gt += len(gt_boxes)
        for class_name in sorted(set(gt_names) | set(names)):
            gt_count = sum(name == class_name for name in gt_names)
            pred_count = sum(name == class_name and index in track_by_detection for index, name in enumerate(names))
            tp = sum(gt_names[gt_index] == class_name for pred_index, gt_index in matches)
            current = counts[class_name]
            current[0] += tp
            current[1] += pred_count - tp
            current[2] += gt_count - tp
        for pred_index, gt_index in matches:
            if pred_index not in track_by_detection:
                continue
            token = str(frame["gt_instance_tokens"][gt_index])
            track_id = track_by_detection[pred_index]
            previous = states.get(token)
            if previous is not None:
                gap = frame_index - previous["frame_index"] - 1
                if gap == 0 and track_id != previous["track_id"]:
                    scene_metrics[scene]["id_switches"] += 1
                    events.append({"type": "id_switch", "scene_name": scene, "frame_index": frame_index, "instance_token": token, "old_track_id": previous["track_id"], "new_track_id": track_id, "gap_frames": 0})
                elif gap > 0:
                    scene_metrics[scene]["gap_reappearances"] += 1
                    scene_metrics[scene]["lost_frames"] += gap
                    if track_id != previous["track_id"]:
                        scene_metrics[scene]["fragmentations"] += 1
                        events.append({"type": "fragmentation", "scene_name": scene, "frame_index": frame_index, "instance_token": token, "old_track_id": previous["track_id"], "new_track_id": track_id, "gap_frames": gap})
            states[token] = {"scene": scene, "frame_index": frame_index, "track_id": track_id}
    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class = {}
    for class_name, (tp, fp, fn) in counts.items():
        precision_class = tp / (tp + fp) if tp + fp else 0.0
        recall_class = tp / (tp + fn) if tp + fn else 0.0
        f1_class = 2.0 * precision_class * recall_class / (precision_class + recall_class) if precision_class + recall_class else 0.0
        per_class[class_name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision_class, "recall": recall_class, "f1": f1_class}
    tracking = {"matched": total_matched, "predicted": total_predicted, "ground_truth": total_gt, "precision": precision, "recall": recall, "f1": f1, "id_switches": sum(value["id_switches"] for value in scene_metrics.values()), "fragmentations": sum(value["fragmentations"] for value in scene_metrics.values()), "gap_reappearances": sum(value["gap_reappearances"] for value in scene_metrics.values()), "lost_frames": sum(value["lost_frames"] for value in scene_metrics.values()), "unique_track_ids": len({track_id for values in track_maps.values() for track_id in values.values()})}
    vru_tp = sum(per_class.get(name, {}).get("tp", 0) for name in VRU_CLASSES)
    vru_fp = sum(per_class.get(name, {}).get("fp", 0) for name in VRU_CLASSES)
    vru_fn = sum(per_class.get(name, {}).get("fn", 0) for name in VRU_CLASSES)
    vru_precision = vru_tp / (vru_tp + vru_fp) if vru_tp + vru_fp else 0.0
    vru_recall = vru_tp / (vru_tp + vru_fn) if vru_tp + vru_fn else 0.0
    tracking["vru_f1"] = 2.0 * vru_precision * vru_recall / (vru_precision + vru_recall) if vru_precision + vru_recall else 0.0
    tracking["vru_precision"] = vru_precision
    tracking["vru_recall"] = vru_recall
    return {"tracking": tracking, "per_class": per_class, "per_scene": dict(scene_metrics), "events": events}


def handoff_continuity(events: list[dict[str, Any]], track_maps: dict[str, dict[int, int]], by_token: dict[str, dict[str, Any]], threshold: float, distance_threshold_m: float) -> dict[str, int | float]:
    total = preserved = missing = 0
    for event in events:
        before = by_token[event["before_token"]]
        after = by_token[event["after_token"]]
        before_boxes = np.asarray(before["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        before_scores = np.asarray(before["scores"], dtype=np.float64)
        before_names = [str(value) for value in before["names"]]
        before_gt_boxes = np.asarray(before["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        before_gt_names = [str(value) for value in before["gt_native_names"]]
        after_boxes = np.asarray(after["boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        after_scores = np.asarray(after["scores"], dtype=np.float64)
        after_names = [str(value) for value in after["names"]]
        after_gt_boxes = np.asarray(after["gt_boxes_lidar"], dtype=np.float64).reshape(-1, 7)
        after_gt_names = [str(value) for value in after["gt_native_names"]]
        before_matches = gated_hungarian(before_boxes, before_scores, before_names, before_gt_boxes, before_gt_names, threshold, distance_threshold_m)
        after_matches = gated_hungarian(after_boxes, after_scores, after_names, after_gt_boxes, after_gt_names, threshold, distance_threshold_m)
        before_lookup = {str(token): track_maps[event["before_token"]].get(pred_index) for pred_index, gt_index in before_matches for token in [before["gt_instance_tokens"][gt_index]]}
        after_lookup = {str(token): track_maps[event["after_token"]].get(pred_index) for pred_index, gt_index in after_matches for token in [after["gt_instance_tokens"][gt_index]]}
        before_id, after_id = before_lookup.get(event["instance_token"]), after_lookup.get(event["instance_token"])
        total += 1
        if before_id is None or after_id is None:
            missing += 1
        elif before_id == after_id:
            preserved += 1
    return {"candidates": total, "preserved": preserved, "missing": missing, "continuity": preserved / total if total else None}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = json.loads(args.cached_detections.read_text(encoding="utf-8"))
    metadata = json.loads(args.detection_embedding_meta.read_text(encoding="utf-8"))
    arrays = np.load(args.detection_embeddings)
    if len(metadata) != len(arrays["roi"]):
        raise ValueError("embedding metadata and vector count differ")
    embedding_by_method: dict[str, dict[int, dict[int, np.ndarray | None]]] = {"single": defaultdict(dict), "roi": defaultdict(dict)}
    for index, item in enumerate(metadata):
        for method in embedding_by_method:
            embedding_by_method[method][int(item["frame_index"])][int(item["detection_index"])] = arrays[method][index] if bool(arrays["valid"][index]) else None
    all_events = json.loads(args.cross_camera_events.read_text(encoding="utf-8"))["events"]
    transform_payload = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    transforms = {token: np.asarray(value["lidar2global"], dtype=np.float64) for token, value in transform_payload.items()}
    if {frame["token"] for frame in frames} != set(transforms):
        raise ValueError("cached detections and transform cache do not match")
    results: dict[str, Any] = {
        "config": {"score_threshold": args.score_threshold, "distance_threshold_m": args.distance_threshold_m, "same_detections": True, "training": False},
        "experiments": {},
        "notes": ["E1 legacy numbers are from the fresh Phase 0 baseline protocol.", "E2/E4/E5 use the corrected Hungarian/native-class/gap-aware protocol on the same cached detections.", "E3 low-confidence association was not implemented in this run."],
    }
    baseline_legacy = json.loads(Path("artifacts/experiments/vru_tracking/phase_00/nuscenes_baseline/phase1/metrics.json").read_text(encoding="utf-8"))
    legacy = baseline_legacy["metrics"]["tracking_proxy"]
    results["experiments"]["E1_legacy_kalman_hungarian"] = {"tracking": legacy, "protocol": "legacy bounded greedy matcher; ID changes include gaps", "tracking_latency_ms": None}
    methods = (("E2_corrected_classical", "classical", None), ("E4_query_roi", "query", "roi"), ("E5_query_single", "query", "single"))
    for label, method, embedding_method in methods:
        embedding_by_frame = embedding_by_method[embedding_method] if embedding_method else {}
        track_maps, timing = run_assignments(frames, method, embedding_by_frame, args.score_threshold, transforms)
        evaluated = evaluate(frames, track_maps, args.score_threshold, args.distance_threshold_m)
        evaluated["tracking_latency_ms"] = timing
        evaluated["association"] = "motion-only" if label == "E2_corrected_classical" else f"0.7 motion + 0.3 {embedding_method} cosine, 3m gate"
        results["experiments"][label] = evaluated
        if label == "E2_corrected_classical":
            results["handoff_candidates"] = handoff_continuity(all_events, track_maps, {frame["token"]: frame for frame in frames}, args.score_threshold, args.distance_threshold_m)
    (args.output_dir / "association_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for label, value in results["experiments"].items():
        tracking = value["tracking"]
        rows.append({"experiment": label, "tracking_f1": tracking.get("f1"), "vru_f1": tracking.get("vru_f1"), "id_switches": tracking.get("id_switches"), "fragmentations": tracking.get("fragmentations"), "unique_track_ids": tracking.get("unique_track_ids"), "latency_mean_ms": (value.get("tracking_latency_ms") or {}).get("mean_ms")})
    (args.output_dir / "association_ablation.csv").write_text("experiment,tracking_f1,vru_f1,id_switches,fragmentations,unique_track_ids,latency_mean_ms\n" + "\n".join(",".join(str(row[key]) for key in row) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "output_dir": str(args.output_dir), "experiments": {key: value["tracking"] for key, value in results["experiments"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
