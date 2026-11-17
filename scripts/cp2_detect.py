#!/usr/bin/env python3
"""Checkpoint-2 Phase 1: frozen BEVFusion (6 cameras + LiDAR) 3D detection.

Writes, for every keyframe of the selected scenes:
  * raw detections (all TransFusion proposals, LiDAR frame) -> <run>/detections.pkl
  * official nuScenes detection submission (global frame)   -> <run>/submission_det.json
  * latency / peak GPU memory                              -> <run>/runtime.json
  * optional fp16 fused-BEV cache [512,180,180] per frame  -> --cache-bev DIR/<token>.npy
  * optional six-camera + BEV visualisations               -> --vis-dir

Ablation flags reproduce the checkpoint-1 input bugs on purpose:
  --no-sweeps     single LiDAR sweep (mmdet3d pads by duplicating the keyframe)
  --stale-images  every frame reuses the camera images of the first frame
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.datasets.nuscenes_infos import (  # noqa: E402
    DETECTION_CLASSES,
    VRU_CLASSES,
    build_frame_records,
    pipeline_input,
    resolve_scenes,
)
from bevdrive.evaluation.nuscenes_submission import detection_entries, write_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("/home/anhhd/dataset/nuscenes-mini"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scenes", nargs="+", default=["mini_val"], help="split names and/or scene names")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--records-out", type=Path, default=None, help="pickle of frame records (default <output-dir>/records.pkl)")
    p.add_argument("--frame-limit", type=int, default=None, help="max keyframes per scene")
    p.add_argument("--complete-only", action="store_true", help="keep only scenes whose files (incl. 9 sweeps) exist on disk")
    p.add_argument("--exclude-scenes", nargs="*", default=[])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-sweeps", action="store_true")
    p.add_argument("--stale-images", action="store_true")
    p.add_argument("--cache-bev", type=Path, default=None)
    p.add_argument("--vis-dir", type=Path, default=None)
    p.add_argument("--vis-every", type=int, default=10)
    p.add_argument("--vis-score", type=float, default=0.3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from nuscenes.nuscenes import NuScenes

    from bevdrive.perception.bevfusion_runner import BOX_CONVENTION_TAG, DEFAULT_CHECKPOINT, DEFAULT_CONFIG, BEVFusionRunner

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=str(args.data_root), verbose=False)
    scenes = [s for s in resolve_scenes(nusc, args.scenes) if s not in set(args.exclude_scenes)]
    if args.complete_only:
        from bevdrive.datasets.nuscenes_infos import scene_is_complete

        lidar_files = {f.name for f in (args.data_root / "samples/LIDAR_TOP").iterdir()}
        by_name = {s["name"]: s for s in nusc.scene}

        def first_lidar_present(name: str, nusc=nusc) -> bool:
            sample = nusc.get("sample", by_name[name]["first_sample_token"])
            return Path(nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["filename"]).name in lidar_files

        scenes = [s for s in scenes if first_lidar_present(s) and scene_is_complete(nusc, args.data_root, s)]
        print(f"[complete] {len(scenes)} complete scenes", flush=True)
    t0 = time.perf_counter()
    records = build_frame_records(nusc, args.data_root, scenes, frame_limit=args.frame_limit)
    print(f"[records] {len(scenes)} scenes, {len(records)} keyframes in {time.perf_counter() - t0:.1f}s", flush=True)
    records_out = args.records_out or args.output_dir / "records.pkl"
    records_out.parent.mkdir(parents=True, exist_ok=True)
    with records_out.open("wb") as handle:
        pickle.dump({"version": args.version, "data_root": str(args.data_root), "scenes": scenes, "records": records}, handle)

    import gc

    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    (args.output_dir / "scenes.json").write_text(json.dumps({"train": [s for s in scenes if s in splits["train"]], "val": [s for s in scenes if s in splits["val"]]}, indent=1))
    del nusc  # trainval metadata is ~8 GB in RAM; free it before loading the model
    gc.collect()
    runner = BEVFusionRunner(args.config or DEFAULT_CONFIG, args.checkpoint or DEFAULT_CHECKPOINT, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.cache_bev:
        args.cache_bev.mkdir(parents=True, exist_ok=True)

    detections: dict[str, dict[str, np.ndarray]] = {}
    submission: dict[str, list] = {}
    timings: list[dict[str, float]] = []
    first_of_scene: dict[str, dict] = {}
    for i, record in enumerate(records):
        stale_source = first_of_scene.setdefault("run", record) if args.stale_images else None
        det = runner.infer(pipeline_input(record, use_sweeps=not args.no_sweeps, image_record=stale_source))
        timings.append(det.timings_ms)
        names = [DETECTION_CLASSES[int(label)] for label in det.labels]
        detections[record["token"]] = {"boxes": det.boxes, "scores": det.scores, "labels": det.labels}
        submission[record["token"]] = detection_entries(record["token"], det.boxes, det.scores, names, record["lidar2global"])
        if args.cache_bev:
            np.save(args.cache_bev / f"{record['token']}.npy", det.bev[0].to(torch.float16).cpu().numpy())
        if args.vis_dir and (i % args.vis_every == 0):
            render_frame(args.vis_dir, i, record, det, names, args.vis_score)
        if i % 20 == 0:
            print(f"[{i:4d}/{len(records)}] {record['scene_name']} f{record['frame_index']:02d} "
                  f"pre {det.timings_ms['preprocess_ms']:.0f} ms model {det.timings_ms['model_ms']:.0f} ms "
                  f"VRU>0.3: {int(sum(1 for n, s in zip(names, det.scores) if n in VRU_CLASSES and s > 0.3))}", flush=True)

    with (args.output_dir / "detections.pkl").open("wb") as handle:
        pickle.dump(detections, handle)
    write_submission(args.output_dir / "submission_det.json", submission)
    model_ms = np.array([t["model_ms"] for t in timings][3:] or [t["model_ms"] for t in timings])
    pre_ms = np.array([t["preprocess_ms"] for t in timings][3:] or [t["preprocess_ms"] for t in timings])
    runtime = {
        "frames": len(records),
        "scenes": scenes,
        "use_sweeps": not args.no_sweeps,
        "stale_images": bool(args.stale_images),
        "box_convention": BOX_CONVENTION_TAG if runner.legacy_box_convention else "raw checkpoint output",
        "model_ms_mean": float(model_ms.mean()),
        "model_ms_median": float(np.median(model_ms)),
        "preprocess_ms_mean": float(pre_ms.mean()),
        "model_fps": float(1000.0 / model_ms.mean()),
        "end_to_end_fps": float(1000.0 / (model_ms.mean() + pre_ms.mean())),
        "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated() / 2**20) if args.device.startswith("cuda") else None,
        "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved() / 2**20) if args.device.startswith("cuda") else None,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "note": "first 3 frames excluded from timing statistics (warm-up)",
    }
    (args.output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime, indent=2))
    return 0


def render_frame(vis_dir: Path, index: int, record: dict, det, names: list[str], score_thr: float) -> None:
    import cv2

    from bevdrive.visualization.render3d import CLASS_COLORS_BGR, BEVCanvas, load_lidar_points, render_camera_grid, side_by_side

    keep = [k for k, (n, s) in enumerate(zip(names, det.scores)) if s >= score_thr]
    boxes = det.boxes[keep]
    colors = [CLASS_COLORS_BGR[names[k]] if names[k] in VRU_CLASSES else (150, 150, 150) for k in keep]
    texts = [f"{names[k][:3]} {det.scores[k]:.2f}" if names[k] in VRU_CLASSES else "" for k in keep]
    title = f"BEVFusion (frozen) | {record['scene_name']} frame {record['frame_index']} | score>={score_thr} | VRU coloured"
    cams = render_camera_grid(record, boxes, colors, texts, title=title)
    canvas = BEVCanvas(record["lidar2ego"], range_m=50, px_per_m=8)
    canvas.draw_points(load_lidar_points(record["lidar_path"]))
    gt = record["gt"]
    for box, name in zip(gt["boxes"], gt["names"]):
        canvas.draw_box(box, (0, 170, 0), thickness=1)
    for box, color, k in zip(boxes, colors, keep):
        canvas.draw_box(box, color, text=f"{names[k][:3]}" if names[k] in VRU_CLASSES else None)
        if names[k] in VRU_CLASSES:
            canvas.draw_velocity(box, box[7:9], color)
    canvas.draw_ego()
    canvas.title("BEV (ego frame, up=forward). green = GT, colour/grey = pred")
    bev = cv2.resize(canvas.image, (cams.shape[0], cams.shape[0]))
    vis_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_dir / f"{index:04d}_{record['scene_name']}_f{record['frame_index']:02d}.jpg"), side_by_side(cams, bev), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    raise SystemExit(main())
