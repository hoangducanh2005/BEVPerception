#!/usr/bin/env python3
"""Phase 4 port check: does the BEVPerception TrackFormer reproduce UniAD's tracker?

Feeds UniAD's own BEVFormer ``bev_embed`` and ego poses (dumped by
``uniad_dump_track_io.py``) through ``BEVFusionTrackFormer`` in
``query_mode='learned'`` with the UniAD stage-1 weights, and compares per frame
with UniAD's own outputs: track-id sets, box centres, scores. Also writes a
nuScenes tracking submission of the port's output for an AMOTA check.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.datasets.nuscenes_infos import DETECTION_CLASSES, TRACKING_CLASSES  # noqa: E402
from bevdrive.evaluation.nuscenes_submission import tracking_entries, write_submission  # noqa: E402
from bevdrive.trackformer import BEVFusionTrackFormer, TrackFormerConfig, load_uniad_track_weights  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, default=REPO / "artifacts/checkpoint_2/phase_04/uniad_port_check")
    p.add_argument("--output-dir", type=Path, default=REPO / "results/phase_04/port_check")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    index = pickle.load((args.dump / "index.pkl").open("rb"))
    cfg = TrackFormerConfig(query_mode="learned", include_sdc_query=True, use_temporal_bev=False, reset_ids_per_scene=False)
    model = BEVFusionTrackFormer(cfg)
    report = load_uniad_track_weights(model)
    model.to(args.device).eval()
    per_frame, submission = [], {}
    for i, meta in enumerate(index):
        frame = np.load(args.dump / f"frame_{i:03d}.npz")
        l2g = np.eye(4)
        l2g[:3, :3] = frame["l2g_r_mat"].T  # UniAD row-vector convention: p_g = p @ R + t
        l2g[:3, 3] = frame["l2g_t"]
        value = torch.from_numpy(frame["bev_embed"].astype(np.float32)).to(args.device)[:, None]
        out = model.forward_inference(None, None, None, l2g, meta["timestamp"], meta["scene_token"], bev_value=value)
        ours = {int(t): k for k, t in enumerate(out["ids"])}
        theirs = {int(t): k for k, t in enumerate(frame["track_ids"])}
        common = sorted(set(ours) & set(theirs))
        center_err = [float(np.linalg.norm(out["boxes"][ours[t], :3] - frame["boxes"][theirs[t], :3])) for t in common]
        score_err = [abs(float(out["scores"][ours[t]] - frame["scores"][theirs[t]])) for t in common]
        per_frame.append({"n_uniad": len(theirs), "n_port": len(ours), "n_common_ids": len(common), "max_center_err_m": max(center_err, default=0.0), "max_score_err": max(score_err, default=0.0)})
        names = [DETECTION_CLASSES[int(label)] for label in out["labels"]]
        submission[meta["sample_token"]] = tracking_entries(meta["sample_token"], out["boxes"], out["scores"], names, out["ids"], l2g, keep_classes=TRACKING_CLASSES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_submission(args.output_dir / "submission_track_port.json", submission, use_lidar=False)
    total_u = sum(f["n_uniad"] for f in per_frame)
    total_c = sum(f["n_common_ids"] for f in per_frame)
    summary = {
        "frames": len(per_frame),
        "weights_loaded": len(report["loaded"]),
        "weights_not_initialised": report["not_initialised"],
        "uniad_track_outputs": total_u,
        "port_track_outputs": sum(f["n_port"] for f in per_frame),
        "same_track_id_fraction": total_c / max(total_u, 1),
        "frames_with_identical_id_sets": sum(f["n_uniad"] == f["n_port"] == f["n_common_ids"] for f in per_frame),
        "max_center_err_m": max(f["max_center_err_m"] for f in per_frame),
        "median_frame_max_center_err_m": float(np.median([f["max_center_err_m"] for f in per_frame])),
        "max_score_err": max(f["max_score_err"] for f in per_frame),
        "per_frame": per_frame,
    }
    (args.output_dir / "port_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
