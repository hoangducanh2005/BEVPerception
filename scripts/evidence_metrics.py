"""Detection and bounded tracking metrics for phase evidence runs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from phase1_navsim_infer import PRED_TO_COARSE


METRIC_CLASSES = ("vehicle", "pedestrian", "traffic_cone", "generic_object")


def _coarse(name: str) -> str:
    return PRED_TO_COARSE.get(str(name), str(name) if str(name) in METRIC_CLASSES else "generic_object")


def match_detection_and_tracking(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_names: Sequence[str],
    assignments: Sequence[Any],
    gt_boxes: np.ndarray,
    gt_names: Sequence[str],
    gt_instance_tokens: Sequence[str] | None,
    previous_gt_tracks: dict[str, int],
    *,
    score_thr: float,
    distance_thr_m: float = 2.0,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Greedily match predictions and return detection + identity diagnostics."""

    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
    assignment_by_index = {int(item.detection_index): item for item in assignments}
    pred_indices = [i for i, score in enumerate(pred_scores) if float(score) >= score_thr]
    gt_coarse = [_coarse(name) for name in gt_names]
    used_gt: set[int] = set()
    matched: list[tuple[int, int]] = []
    for pred_index in sorted(pred_indices, key=lambda i: float(pred_scores[i]), reverse=True):
        candidates = [
            gt_index
            for gt_index, name in enumerate(gt_coarse)
            if gt_index not in used_gt and _coarse(pred_names[pred_index]) == name
        ]
        if not candidates:
            continue
        distances = [float(np.linalg.norm(pred_boxes[pred_index, :2] - gt_boxes[gt_index, :2])) for gt_index in candidates]
        best_position = int(np.argmin(distances))
        if distances[best_position] <= distance_thr_m:
            gt_index = candidates[best_position]
            used_gt.add(gt_index)
            matched.append((pred_index, gt_index))

    per_class: dict[str, dict[str, int]] = {}
    for class_name in METRIC_CLASSES:
        gt_count = sum(name == class_name for name in gt_coarse)
        pred_count = sum(_coarse(pred_names[i]) == class_name for i in pred_indices)
        tp = sum(_coarse(pred_names[pred_index]) == class_name for pred_index, _ in matched)
        per_class[class_name] = {"tp": tp, "fp": pred_count - tp, "fn": gt_count - tp}

    instance_tokens = list(gt_instance_tokens or [f"gt_{index}" for index in range(len(gt_boxes))])
    id_switches = 0
    current_gt_tracks = dict(previous_gt_tracks)
    matched_track_ids: dict[str, int] = {}
    for pred_index, gt_index in matched:
        if gt_index >= len(instance_tokens):
            continue
        track_id = int(getattr(assignment_by_index.get(pred_index), "track_id", -1))
        if track_id < 0:
            continue
        token = str(instance_tokens[gt_index])
        old_track_id = current_gt_tracks.get(token)
        if old_track_id is not None and old_track_id != track_id:
            id_switches += 1
        current_gt_tracks[token] = track_id
        matched_track_ids[token] = track_id
    result = {
        "per_class": per_class,
        "matched": len(matched),
        "predicted": len(pred_indices),
        "ground_truth": len(gt_boxes),
        "true_positive": len(matched),
        "false_positive": len(pred_indices) - len(matched),
        "false_negative": len(gt_boxes) - len(matched),
        "id_switches": id_switches,
        "matched_track_ids": matched_track_ids,
        "match_distance_threshold_m": distance_thr_m,
    }
    return result, current_gt_tracks


def aggregate_frame_metrics(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate frame-level detection and bounded tracking diagnostics."""

    per_class: dict[str, dict[str, int]] = {}
    for class_name in METRIC_CLASSES:
        frame_per_class = [frame["detection_metrics"].get("per_class", frame["detection_metrics"]) for frame in frames]
        tp = sum(int(values[class_name]["tp"]) for values in frame_per_class)
        fp = sum(int(values[class_name]["fp"]) for values in frame_per_class)
        fn = sum(int(values[class_name]["fn"]) for values in frame_per_class)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    tp = sum(int(frame["tracking_metrics"]["true_positive"]) for frame in frames)
    fp = sum(int(frame["tracking_metrics"]["false_positive"]) for frame in frames)
    fn = sum(int(frame["tracking_metrics"]["false_negative"]) for frame in frames)
    gt = sum(int(frame["tracking_metrics"]["ground_truth"]) for frame in frames)
    matched = sum(int(frame["tracking_metrics"]["matched"]) for frame in frames)
    predicted = sum(int(frame["tracking_metrics"]["predicted"]) for frame in frames)
    ids = sum(int(frame["tracking_metrics"]["id_switches"]) for frame in frames)
    track_ids = {track_id for frame in frames for track_id in frame.get("track_ids", [])}
    tracking_precision = tp / (tp + fp) if tp + fp else 0.0
    tracking_recall = tp / (tp + fn) if tp + fn else 0.0
    tracking_f1 = 2.0 * tracking_precision * tracking_recall / (tracking_precision + tracking_recall) if tracking_precision + tracking_recall else 0.0
    return {
        "detection": {"per_class": per_class},
        "tracking_proxy": {
            "matched": matched,
            "predicted": predicted,
            "ground_truth": gt,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tracking_precision,
            "recall": tracking_recall,
            "f1": tracking_f1,
            "id_switches": ids,
            "unique_track_ids": len(track_ids),
            "note": "bounded center-distance/instance-token diagnostic; not official nuScenes AMOTA/AMOTP",
        },
    }
