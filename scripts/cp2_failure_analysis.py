#!/usr/bin/env python3
"""VRU failure-case analysis for a tracks.pkl (diagnostic, not the official metric).

Per frame, GT VRUs (>=1 LiDAR/radar point, within the nuScenes 40 m class
range) are Hungarian-matched to same-class tracks with score >= --score-thr at
<= 2 m centre distance. Events:
  * ID switch   : matched in consecutive frames but to a different track id
  * re-id fail  : matched again after a gap (>=1 unmatched frame) with a new id
  * fragmentation: a GT instance was matched, then lost, then matched again
  * miss         : unmatched GT, split into 'occluded' (visibility <= 40 % or <5 pts)
                   and 'visible'
  * false positive: unmatched track
The worst frames are rendered with highlights (red = missed GT, yellow = ID switch,
orange = false positive).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bevdrive.datasets.nuscenes_infos import VRU_CLASSES  # noqa: E402
from cp2_render_tracks import render  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--records", type=Path, default=REPO / "artifacts/checkpoint_2/phase_01/mini_all/records.pkl")
    p.add_argument("--scenes", nargs="+", default=["scene-0103", "scene-0916"])
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--vis-dir", type=Path, default=None)
    p.add_argument("--label", default="tracks")
    p.add_argument("--num-vis", type=int, default=6)
    args = p.parse_args()
    tracks = pickle.load(args.tracks.open("rb"))["tracks"]
    records = [r for r in pickle.load(args.records.open("rb"))["records"] if r["scene_name"] in set(args.scenes)]
    counts = {c: {"gt": 0, "tp": 0, "idsw": 0, "reid_fail_after_gap": 0, "frag": 0, "miss_occluded": 0, "miss_visible": 0, "fp": 0} for c in VRU_CLASSES}
    events, frame_scores = [], []
    last_id: dict[str, int] = {}
    last_frame: dict[str, int] = {}
    was_lost: dict[str, bool] = {}
    for record in records:
        gt, fr = record["gt"], tracks[record["token"]]
        dist = np.hypot(gt["boxes"][:, 0], gt["boxes"][:, 1]) if len(gt["boxes"]) else np.zeros(0)
        g_idx = [k for k in range(len(gt["names"])) if gt["names"][k] in VRU_CLASSES and gt["num_pts"][k] > 0 and dist[k] <= 40.0]
        t_idx = [k for k in range(len(fr["names"])) if fr["names"][k] in VRU_CLASSES and fr["scores"][k] >= args.score_thr]
        cost = np.full((len(g_idx), len(t_idx)), 1e6)
        for a, g in enumerate(g_idx):
            for b, t in enumerate(t_idx):
                if gt["names"][g] == fr["names"][t]:
                    d = np.linalg.norm(gt["boxes"][g, :2] - fr["boxes"][t, :2])
                    if d <= 2.0:
                        cost[a, b] = d
        rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
        matched = {g_idx[r]: t_idx[c] for r, c in zip(rows, cols) if cost[r, c] < 1e6}
        highlights, n_events = [], 0
        for g in g_idx:
            name, token = gt["names"][g], gt["instance_tokens"][g]
            counts[name]["gt"] += 1
            if g in matched:
                tid = int(fr["ids"][matched[g]])
                counts[name]["tp"] += 1
                consecutive = last_frame.get(token) == record["frame_index"] - 1
                if token in last_id and last_id[token] != tid and not consecutive:
                    counts[name]["reid_fail_after_gap"] += 1
                    events.append({"type": "reid_fail", "scene": record["scene_name"], "frame": record["frame_index"], "class": name, "from": last_id[token], "to": tid})
                if token in last_id and last_id[token] != tid and consecutive:
                    counts[name]["idsw"] += 1
                    n_events += 2
                    events.append({"type": "idsw", "scene": record["scene_name"], "frame": record["frame_index"], "class": name, "from": last_id[token], "to": tid})
                    highlights.append((gt["boxes"][g], f"IDSW {last_id[token]}->{tid}", (0, 220, 255)))
                if was_lost.get(token):
                    counts[name]["frag"] += 1
                    events.append({"type": "frag", "scene": record["scene_name"], "frame": record["frame_index"], "class": name, "id": tid})
                last_id[token], was_lost[token], last_frame[token] = tid, False, record["frame_index"]
            else:
                occluded = gt["visibility"][g] <= 2 or gt["num_pts"][g] < 5
                counts[name]["miss_occluded" if occluded else "miss_visible"] += 1
                n_events += 0 if occluded else 1
                highlights.append((gt["boxes"][g], "miss(occ)" if occluded else "MISS", (0, 0, 255)))
                if token in last_id:
                    was_lost[token] = True
        for t in t_idx:
            if t not in matched.values():
                counts[fr["names"][t]]["fp"] += 1
                highlights.append((fr["boxes"][t], "FP", (0, 140, 255)))
        frame_scores.append((n_events, record, highlights))
    summary = {"score_thr": args.score_thr, "counts": counts, "events": events}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(counts, indent=2))
    if args.vis_dir:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
        for n_events, record, highlights in sorted(frame_scores, key=lambda x: -x[0])[: args.num_vis]:
            img = render(record, tracks[record["token"]], f"{args.label} FAILURES (red=miss, yellow=IDSW, orange=FP)", vru_only=True, score_thr=args.score_thr, highlights={"boxes": highlights})
            img = cv2.resize(img, (1920, int(img.shape[0] * 1920 / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.vis_dir / f"failure_{record['scene_name']}_f{record['frame_index']:02d}_events{n_events}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
