#!/usr/bin/env python3
"""End-to-end online VRU detection + tracking (no caches): one process, frame by frame.

  6 cameras + LiDAR (10 sweeps) -> frozen BEVFusion -> fused BEV + proposals
  -> temporal BEV (ego-aligned, motion-gated history)
  -> tracker: 'query' (trained BEVFusion-TrackFormer, Proposed B) or 'kf' (Baseline A)
  -> VRU 3D boxes + track ids + velocity (+ learned track embeddings for 'query')

Writes tracks.pkl, submission_track.json, runtime.json (per-stage latency, FPS,
peak GPU memory) and optionally renders frames + MP4.

Example:
  conda run -n ai_core python scripts/cp2_online_pipeline.py --tracker query \
    --checkpoint artifacts/checkpoint_2/phase_04/train_main_temporal/final.pth \
    --scenes scene-0103 --output-dir artifacts/checkpoint_2/phase_05/online_B --render outputs/phase_05/online_B
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bevdrive.datasets.nuscenes_infos import DETECTION_CLASSES, TRACKING_CLASSES, build_frame_records, pipeline_input, resolve_scenes  # noqa: E402
from bevdrive.evaluation.nuscenes_submission import tracking_entries, write_submission  # noqa: E402
from bevdrive.temporal.bev_alignment import TemporalBEVBuffer, aggregate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker", choices=("query", "kf"), default="query")
    p.add_argument("--checkpoint", type=Path, default=REPO / "artifacts/checkpoint_2/phase_04/train_main_temporal/final.pth")
    p.add_argument("--data-root", type=Path, default=Path("/home/anhhd/dataset/nuscenes-mini"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scenes", nargs="+", default=["mini_val"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--render", type=Path, default=None)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    from nuscenes.nuscenes import NuScenes

    from bevdrive.perception.bevfusion_runner import BEVFusionRunner

    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    records = build_frame_records(nusc, args.data_root, resolve_scenes(nusc, args.scenes))
    runner = BEVFusionRunner(device=args.device)
    if args.tracker == "query":
        from bevdrive.trackformer import BEVFusionTrackFormer, TrackFormerConfig

        ckpt = torch.load(str(args.checkpoint), map_location="cpu")
        tracker = BEVFusionTrackFormer(TrackFormerConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ckpt["config"].items()}))
        tracker.load_state_dict(ckpt["model"])
        tracker.to(args.device).eval()
    else:
        from bevdrive.tracking.mot_baseline import KFTracker

        tracker = KFTracker()
    buffer = TemporalBEVBuffer(history=args.history)
    torch.cuda.reset_peak_memory_stats()
    tracks, submission, timing, history_by_id = {}, {}, [], {}
    for record in records:
        t0 = time.perf_counter()
        det = runner.infer(pipeline_input(record))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if buffer.scene != record["scene_name"]:
            buffer.reset(record["scene_name"])
        hist = buffer.aligned_history(record["lidar2global"], record["timestamp"])
        context = aggregate(det.bev, hist, mode="gated", include_current=False)[0] if hist else torch.zeros_like(det.bev)
        buffer.push(det.bev, record["lidar2global"], record["timestamp"], record["scene_name"])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if args.tracker == "query":
            out = tracker.forward_inference(det.bev, context, {"boxes": det.boxes, "scores": det.scores, "labels": det.labels}, record["lidar2global"], record["timestamp"], record["scene_name"])
            boxes, scores, ids = out["boxes"], out["scores"], out["ids"]
            names = [DETECTION_CLASSES[int(label)] for label in out["labels"]]
        else:
            outs = tracker.step(det.boxes, det.scores, [DETECTION_CLASSES[int(label)] for label in det.labels], record["lidar2global"], record["timestamp"], record["scene_token"])
            boxes = np.asarray([o.box_lidar for o in outs]).reshape(-1, 9)
            scores, ids, names = np.asarray([o.score for o in outs]), np.asarray([o.track_id for o in outs]), [o.name for o in outs]
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        timing.append({"preprocess_ms": det.timings_ms["preprocess_ms"], "bevfusion_ms": det.timings_ms["model_ms"], "temporal_ms": (t2 - t1) * 1e3, "tracker_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3})
        centers = (np.c_[boxes[:, :3], np.ones(len(boxes))] @ record["lidar2global"].T)[:, :2] if len(boxes) else np.zeros((0, 2))
        hists = []
        for tid, c in zip(ids, centers):
            key = (record["scene_name"], int(tid))
            history_by_id.setdefault(key, []).append(tuple(c))
            hists.append(np.asarray(history_by_id[key][-12:]))
        tracks[record["token"]] = {"boxes": boxes, "scores": np.asarray(scores), "names": names, "ids": np.asarray(ids, dtype=np.int64), "status": [""] * len(ids), "history_global": hists}
        submission[record["token"]] = tracking_entries(record["token"], boxes, scores, names, ids, record["lidar2global"], keep_classes=TRACKING_CLASSES)
        print(f"{record['scene_name']} f{record['frame_index']:02d} total {timing[-1]['total_ms']:.0f} ms, VRU tracks {sum(n in ('pedestrian', 'bicycle', 'motorcycle') for n in names)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tracks.pkl").open("wb") as handle:
        pickle.dump({"method": f"online {args.tracker}", "tracks": tracks}, handle)
    with (args.output_dir / "records.pkl").open("wb") as handle:
        pickle.dump({"records": records}, handle)
    write_submission(args.output_dir / "submission_track.json", submission)
    steady = timing[3:] or timing
    runtime = {k: float(np.mean([t[k] for t in steady])) for k in steady[0]}
    runtime.update({"fps_end_to_end": 1000.0 / runtime["total_ms"], "fps_model_only": 1000.0 / (runtime["bevfusion_ms"] + runtime["temporal_ms"] + runtime["tracker_ms"]), "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20, "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / 2**20, "frames": len(records), "tracker": args.tracker, "note": "means exclude the first 3 frames; total includes disk I/O + pre-processing"})
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2))
    if args.render:
        import cv2

        from cp2_render_tracks import render

        args.render.mkdir(parents=True, exist_ok=True)
        writer = None
        for record in records:
            img = render(record, tracks[record["token"]], f"ONLINE {'B: BEVFusion+temporal BEV+TrackFormer' if args.tracker == 'query' else 'A: BEVFusion+KF'}", vru_only=True, score_thr=0.3)
            img = cv2.resize(img, (1920, int(img.shape[0] * 1920 / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.render / f"{record['scene_name']}_f{record['frame_index']:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if writer is None:
                writer = cv2.VideoWriter(str(args.render / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (img.shape[1], img.shape[0]))
            writer.write(img)
        writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
