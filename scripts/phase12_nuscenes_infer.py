#!/usr/bin/env python3
"""Run bounded Phase 1/2 BEVFusion tracking evidence on native nuScenes."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d")
sys.path.insert(0, "/home/anhhd/GIT/mmdetection3d/projects/BEVFusion")

from bevdrive.tracking import (  # noqa: E402
    DetectionMeasurement,
    TemporalBEVMemory,
    TemporalQueryBank,
    TrackManager,
    attach_bev_feature_capture,
)
from evidence_metrics import aggregate_frame_metrics, aggregate_temporal_metrics, match_detection_and_tracking  # noqa: E402
from phase12_navsim_infer import (  # noqa: E402
    render_bev,
    render_camera_montage,
    transform_points,
    write_overview,
)
from phase1_navsim_infer import MODEL_CLASSES, PRED_TO_COARSE  # noqa: E402


CAMERA_NAMES = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
VRU_NAMES = {"pedestrian", "bicycle", "motorcycle"}
NUSCENE_NAME_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
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
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def quaternion_matrix(values: Any) -> np.ndarray:
    from pyquaternion import Quaternion

    return np.asarray(Quaternion(values).transformation_matrix, dtype=np.float64)


def sensor_transform(rotation: Any, translation: Any) -> np.ndarray:
    matrix = quaternion_matrix(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def category_name(box_name: str) -> str:
    return NUSCENE_NAME_MAP.get(box_name, box_name)


def build_records(nusc: Any, root: Path, scene_names: list[str], frame_limit: int) -> list[dict[str, Any]]:
    scene_by_name = {scene["name"]: scene for scene in nusc.scene}
    missing = [name for name in scene_names if name not in scene_by_name]
    if missing:
        raise ValueError(f"Unknown scene names: {missing}")
    records: list[dict[str, Any]] = []
    for scene_name in scene_names:
        scene = scene_by_name[scene_name]
        sample_token = scene["first_sample_token"]
        while sample_token and len(records) < frame_limit:
            sample = nusc.get("sample", sample_token)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lidar_cal = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            lidar_to_ego = sensor_transform(lidar_cal["rotation"], lidar_cal["translation"])
            ego_to_global = sensor_transform(lidar_pose["rotation"], lidar_pose["translation"])
            lidar_to_global = ego_to_global @ lidar_to_ego
            lidar_relative = lidar_sd["filename"]
            lidar_path = root / lidar_relative
            image_info: dict[str, dict[str, Any]] = {}
            complete = lidar_path.is_file()
            for camera_name in CAMERA_NAMES:
                camera_sd = nusc.get("sample_data", sample["data"][camera_name])
                camera_cal = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
                camera_path = root / camera_sd["filename"]
                complete = complete and camera_path.is_file()
                camera_to_ego = sensor_transform(camera_cal["rotation"], camera_cal["translation"])
                lidar_to_camera = np.linalg.inv(camera_to_ego) @ lidar_to_ego
                image_info[camera_name] = {
                    "img_path": camera_sd["filename"],
                    "cam2img": np.asarray(camera_cal["camera_intrinsic"], dtype=np.float32),
                    "lidar2cam": lidar_to_camera.astype(np.float32),
                }
            if complete:
                _, boxes, _ = nusc.get_sample_data(sample["data"]["LIDAR_TOP"])
                gt_boxes = []
                gt_names = []
                native_gt_names = []
                instance_tokens = []
                for annotation_token, box in zip(sample["anns"], boxes):
                    name = category_name(box.name)
                    gt_boxes.append([float(box.center[0]), float(box.center[1]), float(box.center[2]), float(box.wlh[1]), float(box.wlh[0]), float(box.wlh[2]), float(box.orientation.yaw_pitch_roll[0])])
                    gt_names.append(name if name in {"car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"} else "generic_object")
                    native_gt_names.append(name if name in {"car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"} else "generic_object")
                    instance_tokens.append(nusc.get("sample_annotation", annotation_token)["instance_token"])
                records.append(
                    {
                        "token": sample["token"],
                        "timestamp": int(sample["timestamp"]),
                        "scene_token": scene["token"],
                        "scene_name": scene_name,
                        "lidar_path": str(lidar_path),
                        "lidar2global": lidar_to_global,
                        "cams": image_info,
                        "images": image_info,
                        "anns": {
                            "gt_boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7),
                            "gt_names": np.asarray([PRED_TO_COARSE.get(name, "generic_object") for name in gt_names]),
                            "gt_native_names": np.asarray(native_gt_names),
                            "instance_tokens": instance_tokens,
                        },
                    }
                )
            sample_token = sample["next"] or None
        if len(records) >= frame_limit:
            break
    return records


def make_info(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_idx": record["token"],
        "timestamp": record["timestamp"],
        "lidar_points": {"lidar_path": record["lidar_path"]},
        "images": record["images"],
        "ann_info": {},
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    records = build_records(nusc, args.data_root.resolve(), args.scene_name, args.frame_limit)
    if not records:
        raise RuntimeError("No complete nuScenes frames selected")
    data_list = [make_info(record) for record in records]
    ann_file = args.output_dir / "mmdet3d_info.pkl"
    with ann_file.open("wb") as handle:
        pickle.dump({"data_list": data_list}, handle)

    import torch
    from mmengine.config import Config
    from mmdet3d.apis import inference_multi_modality_detector, init_model

    config = Config.fromfile(args.config)
    config.model.img_backbone.init_cfg = None
    if config.test_dataloader.dataset.pipeline[0].get("num_views", None) is not None:
        config.test_dataloader.dataset.pipeline[0].num_views = len(CAMERA_NAMES)
    model = init_model(config, str(args.checkpoint), device=args.device)
    bev_capture, bev_hook = attach_bev_feature_capture(model)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    phase1_tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=args.score_thr, history_size=12)
    phase2_tracker = TrackManager(max_distance_m=3.0, max_missed_frames=3, min_score=args.score_thr, history_size=12, query_ema=0.7)
    phase_outputs = {"phase1": args.output_dir / "phase1", "phase2": args.output_dir / "phase2"}
    for directory in phase_outputs.values():
        (directory / "camera").mkdir(parents=True, exist_ok=True)
        (directory / "bev").mkdir(parents=True, exist_ok=True)
    all_evidence: dict[str, list[dict[str, Any]]] = {"phase1": [], "phase2": []}
    previous_gt_tracks: dict[str, dict[str, int]] = {"phase1": {}, "phase2": {}}
    temporal_memory = TemporalBEVMemory()
    query_bank = TemporalQueryBank(ema=0.7)
    last_scene_id: str | None = None

    for frame_index, (record, data) in enumerate(zip(records, data_list)):
        scene_id = str(record["scene_token"])
        if scene_id != last_scene_id:
            temporal_memory.reset()
            query_bank.reset()
            last_scene_id = scene_id
        start = time.perf_counter()
        result, _ = inference_multi_modality_detector(model, record["lidar_path"], str(args.data_root), str(ann_file), cam_type="all")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        instances = result.pred_instances_3d
        boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy().tolist()
        names = [MODEL_CLASSES[int(label)] for label in labels]
        transform = np.asarray(record["lidar2global"], dtype=np.float64)
        bev_feature = bev_capture.get("feature")
        if bev_feature is None:
            raise RuntimeError("BEVFusion pts_neck hook did not capture a BEV feature")
        _, temporal_stats = temporal_memory.update(bev_feature, transform)
        world_centers = transform_points(boxes[:, :3], transform) if len(boxes) else np.empty((0, 3), dtype=np.float64)
        measurements = [
            DetectionMeasurement(name, float(score), tuple(float(v) for v in center), tuple(float(v) for v in box[3:6]), float(box[6]))
            for box, score, name, center in zip(boxes, scores, names, world_centers)
        ]
        for phase_name, tracker in (("phase1", phase1_tracker), ("phase2", phase2_tracker)):
            assignments = tracker.update(measurements, timestamp_us=record["timestamp"], scene_id=record["scene_token"])
            query_state = query_bank.update(assignments, bev_feature, transform) if phase_name == "phase2" else {"num_queries": 0, "num_queries_updated": 0, "mean_query_cosine": None, "query_norms": {}, "query_ages": {}}
            phase_dir = phase_outputs[phase_name]
            camera_path = phase_dir / "camera" / f"{frame_index:04d}_{record['token']}.jpg"
            bev_path = phase_dir / "bev" / f"{frame_index:04d}_{record['token']}.jpg"
            phase_label = "PHASE 1 | Kalman/Hungarian" if phase_name == "phase1" else "PHASE 2 | Temporal Query Prototype"
            render_camera_montage(record, data, boxes, scores, names, assignments, args.data_root, camera_path, args.score_thr, phase_label)
            render_bev(record, Path(record["lidar_path"]), boxes, scores, names, assignments, bev_path, args.score_thr, phase_label)
            detection_and_tracking, previous_gt_tracks[phase_name] = match_detection_and_tracking(
                boxes,
                scores,
                names,
                assignments,
                record["anns"]["gt_boxes"],
                record["anns"]["gt_names"],
                record["anns"].get("instance_tokens"),
                previous_gt_tracks[phase_name],
                score_thr=args.score_thr,
            )
            all_evidence[phase_name].append(
                {
                    "frame_index": frame_index,
                    "token": record["token"],
                    "scene_name": record["scene_name"],
                    "scene_token": record["scene_token"],
                    "timestamp_us": record["timestamp"],
                    "latency_ms": elapsed_ms,
                    "num_predictions_raw": int(len(boxes)),
                    "num_tracks_observed": len(assignments),
                    "track_ids": [int(item.track_id) for item in assignments],
                    "track_status": {str(item.track_id): item.status for item in assignments},
                    "detection_metrics": detection_and_tracking["per_class"],
                    "tracking_metrics": detection_and_tracking,
                    "temporal_bev": {
                        "has_previous": temporal_stats.has_previous,
                        "feature_shape": list(temporal_stats.feature_shape),
                        "mean_abs_delta": temporal_stats.mean_abs_delta,
                        "cosine_similarity": temporal_stats.cosine_similarity,
                        "valid_fraction": temporal_stats.valid_fraction,
                    },
                    "query_state": query_state,
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

    bev_hook.remove()

    for phase_name, phase_dir in phase_outputs.items():
        evidence = all_evidence[phase_name]
        write_overview([Path(item["visuals"]["camera"]) for item in evidence], phase_dir / "camera_overview.jpg", (384, 216))
        write_overview([Path(item["visuals"]["bev"]) for item in evidence], phase_dir / "bev_overview.jpg", (400, 400))
        metrics = {
            "status": "COMPLETED",
            "phase": phase_name,
            "dataset": "native nuScenes v1.0-trainval bounded subset",
            "scenes": args.scene_name,
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
            "temporal_metrics": aggregate_temporal_metrics(evidence),
            "visuals": {"camera_overview": str(phase_dir / "camera_overview.jpg"), "bev_overview": str(phase_dir / "bev_overview.jpg")},
            "frames_evidence": evidence,
        }
        (phase_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        aggregate = metrics["metrics"]
        det = aggregate["detection"]["per_class"]
        track = aggregate["tracking_proxy"]
        temporal = metrics["temporal_metrics"]
        (phase_dir / "tracks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        (phase_dir / "evidence.md").write_text(
            "\n".join(
                [
                    f"# {phase_name} evidence — native nuScenes",
                    "",
                    f"- Scenes: {', '.join(args.scene_name)}.",
                    f"- Frames: {len(evidence)} complete keyframes; no full-dataset inference.",
                    f"- Tracking mode: {metrics['tracking_mode']}.",
                    "- Phase 2 is a persistent non-neural temporal-query prototype, not a trained UniAD TrackFormer checkpoint." if phase_name == "phase2" else "- Phase 1 is the online Kalman/Hungarian tracking baseline.",
                    "- Metrics are bounded smoke evidence, not official full-split nuScenes tracking scores.",
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
                    "## Inference-only temporal base diagnostics",
                    "",
                    f"- BEV frames with aligned previous feature: {temporal['temporal_bev']['frames_with_previous']}",
                    f"- Mean temporal BEV cosine: {temporal['temporal_bev']['mean_cosine_similarity']}",
                    f"- Mean temporal BEV absolute delta: {temporal['temporal_bev']['mean_abs_delta']}",
                    f"- Mean valid warp fraction: {temporal['temporal_bev']['mean_valid_fraction']}",
                    f"- Mean persistent query cosine: {temporal['query_state']['mean_query_cosine']}",
                    f"- Max active query states: {temporal['query_state']['max_active_queries']}",
                    "- No training was performed; these diagnostics validate feature alignment and query-state plumbing only.",
                    "",
                    f"Visuals: `{phase_dir / 'camera_overview.jpg'}` and `{phase_dir / 'bev_overview.jpg'}`.",
                    f"Machine-readable evidence: `{phase_dir / 'metrics.json'}`.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "COMPLETED", "frames": len(records), "scenes": args.scene_name, "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
