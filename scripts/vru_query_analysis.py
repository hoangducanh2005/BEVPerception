#!/usr/bin/env python3
"""Analyze frozen BEV embeddings, temporal warp, and query lifecycle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d")
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d/projects/BEVFusion")

from bevdrive.tracking import DetectionMeasurement, TemporalBEVMemory, TemporalQueryBank, TrackManager  # noqa: E402
from bevdrive.tracking.temporal_query import attach_bev_feature_capture, sample_bev_feature  # noqa: E402
from phase12_nuscenes_infer import build_records, make_info  # noqa: E402
from phase1_navsim_infer import MODEL_CLASSES  # noqa: E402
from vru_detection_experiment import VRU_CLASSES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-name", action="append", required=True)
    parser.add_argument("--frame-limit", type=int, default=100)
    parser.add_argument("--cached-detections", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def normalize(vector: Any) -> Any:
    import torch

    return torch.nn.functional.normalize(vector.float(), dim=-1)


def roi_pool(feature: Any, boxes: np.ndarray) -> tuple[Any, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    feature = feature.float()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    if not len(boxes):
        return torch.empty((0, feature.shape[1]), device=feature.device), np.empty(0, dtype=bool)
    offsets = np.asarray((-0.5, 0.0, 0.5), dtype=np.float32)
    embeddings = []
    valid_values = []
    min_x, min_y, max_x, max_y = -54.0, -54.0, 54.0, 54.0
    for box in boxes:
        dx, dy = max(float(box[3]), 0.1), max(float(box[4]), 0.1)
        grid_x, grid_y = np.meshgrid(box[0] + offsets * dx, box[1] + offsets * dy, indexing="xy")
        centers = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        normalized_x = 2.0 * (centers[:, 0] - min_x) / (max_x - min_x) - 1.0
        normalized_y = 1.0 - 2.0 * (centers[:, 1] - min_y) / (max_y - min_y)
        valid = (np.abs(normalized_x) <= 1.0) & (np.abs(normalized_y) <= 1.0)
        grid = torch.as_tensor(np.stack([normalized_x, normalized_y], axis=1), device=feature.device, dtype=feature.dtype).reshape(1, 3, 3, 2)
        sampled = functional.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, :, :].reshape(feature.shape[1], -1).T
        embeddings.append(sampled[torch.as_tensor(valid, device=feature.device)].mean(dim=0) if valid.any() else sampled.mean(dim=0) * 0.0)
        valid_values.append(bool(valid.any()))
    return torch.stack(embeddings, dim=0), np.asarray(valid_values, dtype=bool)


def cosine(left: Any, right: Any) -> np.ndarray:
    values = (normalize(left) * normalize(right)).sum(dim=-1).detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def auc_score(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    values = [(score, 1) for score in positive] + [(score, 0) for score in negative]
    values.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        mean_rank = (index + 1 + end) / 2.0
        rank_sum += mean_rank * sum(label for _, label in values[index:end])
        index = end
    n_pos, n_neg = len(positive), len(negative)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measurements(boxes: np.ndarray, scores: np.ndarray, names: list[str], pose: np.ndarray) -> list[DetectionMeasurement]:
    centers = np.c_[boxes[:, :3], np.ones(len(boxes))]
    world = (pose @ centers.T).T[:, :3]
    return [DetectionMeasurement(name, float(score), tuple(float(v) for v in center), tuple(float(v) for v in box[3:6]), float(box[6])) for box, score, name, center in zip(boxes, scores, names, world)]


def add_pair(rows: list[dict[str, Any]], feature_type: str, pair_type: str, class_name: str, gap: int, value: float) -> None:
    rows.append({"feature_type": feature_type, "pair_type": pair_type, "class_name": class_name, "time_gap_frames": gap, "cosine": value})


def build_similarity_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_token[sample["instance_token"]].append(sample)
        by_frame[(sample["scene_name"], sample["frame_index"])].append(sample)
    for values in by_token.values():
        values.sort(key=lambda item: item["frame_index"])
        for left, right in zip(values[:-1], values[1:]):
            gap = int(right["frame_index"] - left["frame_index"])
            for feature_type in ("single", "roi"):
                add_pair(rows, feature_type, "positive", left["class_name"], gap, float(np.dot(left[f"{feature_type}_normalized"], right[f"{feature_type}_normalized"])))
    for values in by_frame.values():
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                if left["instance_token"] == right["instance_token"]:
                    continue
                for feature_type in ("single", "roi"):
                    add_pair(rows, feature_type, "negative_same_frame", f"{left['class_name']}|{right['class_name']}", 0, float(np.dot(left[f"{feature_type}_normalized"], right[f"{feature_type}_normalized"])))
    return rows


def summarize_similarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for feature_type in ("single", "roi"):
        positive = [float(row["cosine"]) for row in rows if row["feature_type"] == feature_type and row["pair_type"] == "positive"]
        negative = [float(row["cosine"]) for row in rows if row["feature_type"] == feature_type and row["pair_type"] == "negative_same_frame"]
        summary[feature_type] = {
            "positive_count": len(positive),
            "negative_count": len(negative),
            "positive_mean": float(np.mean(positive)) if positive else None,
            "positive_median": float(np.median(positive)) if positive else None,
            "negative_mean": float(np.mean(negative)) if negative else None,
            "negative_median": float(np.median(negative)) if negative else None,
            "roc_auc": auc_score(positive, negative),
        }
    return summary


def plot_similarity(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, feature_type in zip(axes, ("single", "roi")):
        for pair_type, color in (("positive", "tab:blue"), ("negative_same_frame", "tab:orange")):
            values = [float(row["cosine"]) for row in rows if row["feature_type"] == feature_type and row["pair_type"] == pair_type]
            axis.hist(values, bins=30, alpha=0.55, label=pair_type, color=color)
        axis.set_title(feature_type)
        axis.set_xlabel("cosine similarity")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("pair count")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached = json.loads(args.cached_detections.read_text(encoding="utf-8"))
    from nuscenes.nuscenes import NuScenes
    import torch
    from mmengine.config import Config
    from mmdet3d.apis import inference_multi_modality_detector, init_model

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    records = build_records(nusc, args.data_root.resolve(), args.scene_name, args.frame_limit)
    if [record["token"] for record in records] != [frame["token"] for frame in cached]:
        raise ValueError("cached detections and selected records are not aligned")
    data_list = [make_info(record) for record in records]
    ann_file = args.output_dir / "mmdet3d_info.pkl"
    with ann_file.open("wb") as handle:
        import pickle

        pickle.dump({"data_list": data_list}, handle)
    torch.manual_seed(args.seed)
    config = Config.fromfile(args.config)
    config.model.img_backbone.init_cfg = None
    if config.test_dataloader.dataset.pipeline[0].get("num_views", None) is not None:
        config.test_dataloader.dataset.pipeline[0].num_views = 6
    model = init_model(config, str(args.checkpoint), device=args.device)
    captured, hook = attach_bev_feature_capture(model)
    temporal = TemporalBEVMemory()
    tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=args.score_threshold)
    query_bank = TemporalQueryBank(ema=0.7)
    samples: list[dict[str, Any]] = []
    detection_embedding_meta: list[dict[str, Any]] = []
    detection_single_vectors: list[np.ndarray] = []
    detection_roi_vectors: list[np.ndarray] = []
    detection_embedding_valid: list[bool] = []
    temporal_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    last_scene: str | None = None
    for frame, record, data in zip(cached, records, data_list):
        scene = str(frame["scene_name"])
        if scene != last_scene:
            temporal.reset()
            query_bank.reset()
            last_scene = scene
        result, _ = inference_multi_modality_detector(model, record["lidar_path"], str(args.data_root), str(ann_file), cam_type="all")
        feature = captured.get("feature")
        if feature is None:
            raise RuntimeError("BEV feature hook captured no feature")
        pose = np.asarray(record["lidar2global"], dtype=np.float64)
        previous_aligned, stats = temporal.update(feature, pose)
        gt_boxes = np.asarray(frame["gt_boxes_lidar"], dtype=np.float32).reshape(-1, 7)
        gt_names = [str(value) for value in frame["gt_native_names"]]
        gt_vru_indices = [index for index, name in enumerate(gt_names) if name in VRU_CLASSES]
        if gt_vru_indices:
            centers = gt_boxes[gt_vru_indices, :2]
            single, single_valid = sample_bev_feature(feature, centers)
            roi, roi_valid = roi_pool(feature, gt_boxes[gt_vru_indices])
            for local_index, gt_index in enumerate(gt_vru_indices):
                if not single_valid[local_index] or not roi_valid[local_index]:
                    continue
                samples.append({
                    "instance_token": str(frame["gt_instance_tokens"][gt_index]),
                    "scene_name": scene,
                    "frame_index": int(frame["frame_index"]),
                    "class_name": gt_names[gt_index],
                    "timestamp_us": int(frame["timestamp_us"]),
                    "single_normalized": normalize(single[local_index]).detach().cpu().numpy().astype(float).tolist(),
                    "roi_normalized": normalize(roi[local_index]).detach().cpu().numpy().astype(float).tolist(),
                })
            object_cosine = None
            if previous_aligned is not None:
                previous_single, previous_valid = sample_bev_feature(previous_aligned, centers)
                valid = single_valid & previous_valid
                if valid.any():
                    object_cosine = float(np.mean(cosine(single[valid], previous_single[valid])))
        else:
            object_cosine = None
        pred_instances = result.pred_instances_3d
        pred_boxes = pred_instances.bboxes_3d.tensor.detach().cpu().numpy()
        pred_scores = pred_instances.scores_3d.detach().cpu().numpy()
        pred_labels = pred_instances.labels_3d.detach().cpu().numpy().tolist()
        pred_names = [MODEL_CLASSES[int(label)] for label in pred_labels]
        pred_single, pred_single_valid = sample_bev_feature(feature, pred_boxes[:, :2])
        pred_roi, pred_roi_valid = roi_pool(feature, pred_boxes[:, :7])
        for detection_index, (box, score, name) in enumerate(zip(pred_boxes, pred_scores, pred_names)):
            valid = bool(pred_single_valid[detection_index] and pred_roi_valid[detection_index])
            detection_embedding_meta.append({"frame_index": int(frame["frame_index"]), "detection_index": detection_index, "class_name": name, "score": float(score), "box_lidar": box[:7].astype(float).tolist()})
            detection_single_vectors.append(normalize(pred_single[detection_index]).detach().cpu().numpy().astype(np.float32))
            detection_roi_vectors.append(normalize(pred_roi[detection_index]).detach().cpu().numpy().astype(np.float32))
            detection_embedding_valid.append(valid)
        assignments = tracker.update(measurements(pred_boxes, pred_scores, pred_names, pose), timestamp_us=int(frame["timestamp_us"]), scene_id=scene)
        query_state = query_bank.update(assignments, feature, pose)
        lifecycle_rows.append({"frame_index": int(frame["frame_index"]), "scene_name": scene, "active_tracker_tracks": tracker.active_track_count, "query_states": int(query_state["num_queries"]), "query_updates": int(query_state["num_queries_updated"]), "query_state_minus_active": int(query_state["num_queries"] - tracker.active_track_count)})
        temporal_rows.append({"frame_index": int(frame["frame_index"]), "scene_name": scene, "has_previous": stats.has_previous, "feature_shape": list(stats.feature_shape), "global_cosine": stats.cosine_similarity, "global_mean_abs_delta": stats.mean_abs_delta, "valid_fraction": stats.valid_fraction, "object_region_cosine": object_cosine})
    hook.remove()
    rows = build_similarity_rows(samples)
    write_csv(args.output_dir / "positive_negative_similarity.csv", rows)
    summary = summarize_similarity(rows)
    (args.output_dir / "feature_similarity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "temporal_warp.json").write_text(json.dumps(temporal_rows, indent=2), encoding="utf-8")
    lifecycle = {"frames": lifecycle_rows, "max_active_tracker_tracks": max(row["active_tracker_tracks"] for row in lifecycle_rows), "max_query_states": max(row["query_states"] for row in lifecycle_rows), "max_query_state_minus_active": max(row["query_state_minus_active"] for row in lifecycle_rows), "query_expiration_implemented": False}
    (args.output_dir / "query_lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
    plot_similarity(rows, args.output_dir / "similarity_histogram.png")
    np.savez_compressed(
        args.output_dir / "detection_embeddings.npz",
        single=np.stack(detection_single_vectors),
        roi=np.stack(detection_roi_vectors),
        valid=np.asarray(detection_embedding_valid, dtype=bool),
    )
    (args.output_dir / "detection_embedding_meta.json").write_text(json.dumps(detection_embedding_meta, indent=2), encoding="utf-8")
    manifest = {"scenes": args.scene_name, "frames": len(cached), "score_threshold": args.score_threshold, "seed": args.seed, "cached_detections": str(args.cached_detections), "checkpoint": str(args.checkpoint), "config": str(args.config), "samples": len(samples)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "frames": len(cached), "embedding_samples": len(samples), "similarity": summary, "max_query_states": lifecycle["max_query_states"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
