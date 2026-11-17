#!/usr/bin/env python3
"""Checkpoint-2 Phase 4/5: Proposed B = BEVFusion + temporal BEV + query-based tracking.

Online inference of a *trained* BEVFusion-TrackFormer checkpoint over the
selected scenes using the Phase-1 BEV cache and proposals. Writes the same
artefacts as Baseline A (tracks.pkl, submission_track.json) plus the decoder's
detection output (submission_det.json), learned track embeddings and latency.
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

from bevdrive.datasets.nuscenes_infos import DETECTION_CLASSES, TRACKING_CLASSES  # noqa: E402
from bevdrive.evaluation.nuscenes_submission import detection_entries, tracking_entries, write_submission  # noqa: E402
from bevdrive.temporal.bev_alignment import TemporalBEVBuffer, aggregate  # noqa: E402
from bevdrive.trackformer import BEVFusionTrackFormer, TrackFormerConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--records", type=Path, default=REPO / "artifacts/checkpoint_2/phase_01/mini_all/records.pkl")
    p.add_argument("--detections", type=Path, default=REPO / "artifacts/checkpoint_2/phase_01/mini_all/detections.pkl")
    p.add_argument("--cache", type=Path, default=REPO / "artifacts/checkpoint_2/bev_cache/v1.0-mini")
    p.add_argument("--scenes", nargs="+", default=["scene-0103", "scene-0916"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--score-thresh", type=float, default=None, help="override birth threshold")
    p.add_argument("--filter-score-thresh", type=float, default=None, help="override output/sleep threshold")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    cfg = TrackFormerConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ckpt["config"].items()})
    if args.score_thresh is not None:
        cfg.score_thresh = args.score_thresh
    if args.filter_score_thresh is not None:
        cfg.filter_score_thresh = args.filter_score_thresh
    model = BEVFusionTrackFormer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    records = [r for r in pickle.load(args.records.open("rb"))["records"] if r["scene_name"] in set(args.scenes)]
    detections = pickle.load(args.detections.open("rb"))
    buffer = TemporalBEVBuffer(history=args.history)
    tracks, sub_track, sub_det, latency, temporal_ms, embeddings = {}, {}, {}, [], [], {}
    history_by_id: dict[tuple[str, int], list] = {}
    torch.cuda.reset_peak_memory_stats()
    for record in records:
        bev = torch.from_numpy(np.load(args.cache / f"{record['token']}.npy")).to(args.device).float()[None]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if buffer.scene != record["scene_name"]:
            buffer.reset(record["scene_name"])
        hist_ctx = None
        if cfg.use_temporal_bev:
            hist = buffer.aligned_history(record["lidar2global"], record["timestamp"])
            hist_ctx = aggregate(bev, hist, mode="gated", include_current=False)[0] if hist else torch.zeros_like(bev)
        buffer.push(bev, record["lidar2global"], record["timestamp"], record["scene_name"])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        det = detections[record["token"]]
        out = model.forward_inference(bev, hist_ctx, {"boxes": det["boxes"], "scores": det["scores"], "labels": det["labels"]}, record["lidar2global"], record["timestamp"], record["scene_name"])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        temporal_ms.append((t1 - t0) * 1e3)
        latency.append((t2 - t1) * 1e3)
        names = [DETECTION_CLASSES[int(label)] for label in out["labels"]]
        centers_global = (np.c_[out["boxes"][:, :3], np.ones(len(out["boxes"]))] @ record["lidar2global"].T)[:, :2] if len(out["boxes"]) else np.zeros((0, 2))
        hists = []
        for tid, c in zip(out["ids"], centers_global):
            key = (record["scene_name"], int(tid))
            history_by_id.setdefault(key, []).append(tuple(c))
            hists.append(np.asarray(history_by_id[key][-12:]))
        tracks[record["token"]] = {
            "boxes": out["boxes"],
            "scores": out["scores"],
            "names": names,
            "ids": out["ids"].astype(np.int64),
            "status": ["track_query" if f else "det_query" for f in out["from_track_query"]],
            "history_global": hists,
        }
        embeddings[record["token"]] = out["embeddings"].astype(np.float16)
        sub_track[record["token"]] = tracking_entries(record["token"], out["boxes"], out["scores"], names, out["ids"], record["lidar2global"], keep_classes=TRACKING_CLASSES)
        det_names = [DETECTION_CLASSES[int(label)] for label in out["det_labels"]]
        sub_det[record["token"]] = detection_entries(record["token"], out["det_boxes"], out["det_scores"], det_names, record["lidar2global"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tracks.pkl").open("wb") as handle:
        pickle.dump({"method": "B: BEVFusion + temporal BEV + query tracking", "checkpoint": str(args.checkpoint), "config": ckpt["config"], "tracks": tracks}, handle)
    with (args.output_dir / "track_embeddings.pkl").open("wb") as handle:
        pickle.dump(embeddings, handle)
    write_submission(args.output_dir / "submission_track.json", sub_track)
    write_submission(args.output_dir / "submission_det.json", sub_det)
    lat = np.asarray(latency[3:] or latency)
    runtime = {
        "frames": len(records),
        "tracker_ms_mean": float(lat.mean()),
        "tracker_ms_median": float(np.median(lat)),
        "temporal_bev_ms_mean": float(np.mean(temporal_ms[3:] or temporal_ms)),
        "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated() / 2**20),
        "score_thresh": cfg.score_thresh,
        "filter_score_thresh": cfg.filter_score_thresh,
    }
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
