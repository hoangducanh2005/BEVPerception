#!/usr/bin/env python3
"""Checkpoint-2 Phase 3: Baseline A = BEVFusion + Kalman filter + Hungarian.

Consumes Phase-1 outputs (records.pkl + detections.pkl), runs the online
tracker scene by scene in temporal order and writes
  <out>/tracks.pkl              per-frame tracks (LiDAR boxes, ids, scores, history)
  <out>/submission_track.json   official nuScenes tracking submission (global frame)
  <out>/runtime.json            tracker latency
Evaluate with scripts/cp2_eval_nuscenes.py --task track (uniad2.0 env).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.datasets.nuscenes_infos import DETECTION_CLASSES, TRACKING_CLASSES  # noqa: E402
from bevdrive.evaluation.nuscenes_submission import tracking_entries, write_submission  # noqa: E402
from bevdrive.tracking.mot_baseline import KFTracker, KFTrackerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True, help="scene names to track (e.g. the mini_val scenes)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--det-score-min", type=float, default=0.1)
    p.add_argument("--high-score", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=3)
    p.add_argument("--min-hits", type=int, default=1)
    p.add_argument("--no-detector-velocity", action="store_true")
    p.add_argument("--report-coasted", action="store_true")
    p.add_argument("--gate-scale", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = pickle.load(args.records.open("rb"))
    detections = pickle.load(args.detections.open("rb"))
    records = [r for r in data["records"] if r["scene_name"] in set(args.scenes)]
    cfg = KFTrackerConfig(
        det_score_min=args.det_score_min,
        high_score=args.high_score,
        max_age=args.max_age,
        min_hits=args.min_hits,
        use_detector_velocity=not args.no_detector_velocity,
        report_coasted=args.report_coasted,
    )
    cfg.gates_m = {k: v * args.gate_scale for k, v in cfg.gates_m.items()}
    tracker = KFTracker(cfg)
    tracks: dict[str, dict] = {}
    submission: dict[str, list] = {}
    latencies = []
    for record in records:
        det = detections[record["token"]]
        names = [DETECTION_CLASSES[int(label)] for label in det["labels"]]
        start = time.perf_counter()
        outs = tracker.step(det["boxes"], det["scores"], names, record["lidar2global"], record["timestamp"], record["scene_token"])
        latencies.append((time.perf_counter() - start) * 1e3)
        boxes = np.asarray([o.box_lidar for o in outs]).reshape(-1, 9)
        frame = {
            "boxes": boxes,
            "scores": np.asarray([o.score for o in outs]),
            "names": [o.name for o in outs],
            "ids": np.asarray([o.track_id for o in outs], dtype=np.int64),
            "status": [o.status for o in outs],
            "history_global": [o.history_global for o in outs],
        }
        tracks[record["token"]] = frame
        submission[record["token"]] = tracking_entries(
            record["token"], boxes, frame["scores"], frame["names"], frame["ids"], record["lidar2global"], keep_classes=TRACKING_CLASSES
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tracks.pkl").open("wb") as handle:
        pickle.dump({"method": "A: BEVFusion + KF/Hungarian", "config": asdict(cfg), "tracks": tracks}, handle)
    write_submission(args.output_dir / "submission_track.json", submission)
    runtime = {"frames": len(records), "tracker_ms_mean": float(np.mean(latencies)), "tracker_ms_max": float(np.max(latencies)), "config": asdict(cfg)}
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps({k: v for k, v in runtime.items() if k != "config"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
