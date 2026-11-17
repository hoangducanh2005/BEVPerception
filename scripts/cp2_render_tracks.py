#!/usr/bin/env python3
"""Render tracking results: six cameras + BEV per frame, coloured by track id, plus an MP4.

Works for any tracks.pkl produced by cp2_track_baseline.py (A) or
cp2_track_query.py (B). BEV shows ego, GT (thin green), tracked boxes with
ids, velocity arrows (1 s) and the last <=12 track positions.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.datasets.nuscenes_infos import VRU_CLASSES  # noqa: E402
from bevdrive.visualization.render3d import BEVCanvas, id_color, load_lidar_points, render_camera_grid, side_by_side  # noqa: E402


def render(record: dict, frame: dict, label: str, *, vru_only: bool, score_thr: float, highlights: dict | None = None) -> np.ndarray:
    keep = [k for k, (n, s) in enumerate(zip(frame["names"], frame["scores"])) if s >= score_thr and (n in VRU_CLASSES or not vru_only)]
    boxes = frame["boxes"][keep] if len(keep) else np.zeros((0, 9))
    colors = [id_color(int(frame["ids"][k])) if frame["names"][k] in VRU_CLASSES else (170, 170, 170) for k in keep]
    texts = [f"{frame['names'][k][:3]} #{int(frame['ids'][k])}" if frame["names"][k] in VRU_CLASSES else "" for k in keep]
    title = f"{label} | {record['scene_name']} frame {record['frame_index']:02d} | tracks score>={score_thr}"
    cams = render_camera_grid(record, boxes, colors, texts, title=title)
    canvas = BEVCanvas(record["lidar2ego"], range_m=40, px_per_m=10)
    canvas.draw_points(load_lidar_points(record["lidar_path"]))
    for box, name, npts in zip(record["gt"]["boxes"], record["gt"]["names"], record["gt"]["num_pts"]):
        if npts > 0 and (name in VRU_CLASSES or not vru_only):
            canvas.draw_box(box, (0, 170, 0), thickness=1)
    g2l = np.linalg.inv(record["lidar2global"])
    for k, box, color in zip(keep, boxes, colors):
        hist = frame["history_global"][k]
        if hist is not None and len(hist) > 1:
            pts = np.c_[np.asarray(hist)[:, :2], np.zeros(len(hist)), np.ones(len(hist))] @ g2l.T
            canvas.draw_polyline(pts[:, :2], color, 2)
        name = frame["names"][k]
        canvas.draw_box(box, color, text=f"#{int(frame['ids'][k])}" if name in VRU_CLASSES else None)
        if name in VRU_CLASSES:
            canvas.draw_velocity(box, box[7:9], color)
    if highlights:
        for box, text, color in highlights.get("boxes", []):
            canvas.draw_box(box, color, thickness=3, text=text)
    canvas.draw_ego()
    canvas.title("BEV ego frame (up=forward), 40 m. green=GT, colour=track id")
    canvas.title("trail = last 12 positions, arrow = 1 s velocity", line=1)
    bev = cv2.resize(canvas.image, (cams.shape[0], cams.shape[0]))
    return side_by_side(cams, bev)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--records", type=Path, default=REPO / "artifacts/checkpoint_2/phase_01/mini_all/records.pkl")
    p.add_argument("--scenes", nargs="+", default=["scene-0103", "scene-0916"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label", default="tracks")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--all-classes", action="store_true")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--video", action="store_true")
    p.add_argument("--width", type=int, default=1920)
    args = p.parse_args()
    tracks = pickle.load(args.tracks.open("rb"))["tracks"]
    records = [r for r in pickle.load(args.records.open("rb"))["records"] if r["scene_name"] in set(args.scenes)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    for i, record in enumerate(records):
        if i % args.every:
            continue
        img = render(record, tracks[record["token"]], args.label, vru_only=not args.all_classes, score_thr=args.score_thr)
        img = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out_dir / f"{record['scene_name']}_f{record['frame_index']:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if args.video:
            if writer is None:
                writer = cv2.VideoWriter(str(args.out_dir / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (img.shape[1], img.shape[0]))
            writer.write(img)
    if writer is not None:
        writer.release()
    print(f"rendered {len(records)} frames to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
