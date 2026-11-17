#!/usr/bin/env python3
"""Checkpoint-2 Phase 2: temporal BEV alignment - validation and comparison.

Uses the fused-BEV cache written in Phase 1 (no backbone re-run) and the
frozen TransFusion head. Steps:

  1. BEV tensor-layout check: frozen heatmap sampled at GT centres under the
     candidate layouts ('xy' = ours, 'yx', checkpoint-1 convention).
  2. Ego-motion warp validation on real features: cosine(current, previous)
     raw vs warped (ours) vs warped with the checkpoint-1 function; static-GT
     pose-chain error and feature consistency at static objects.
  3. Single-frame vs temporal BEV: frozen-head detection on aggregated maps
     (mean / decay / motion-gated) evaluated with the nuScenes devkit, plus
     object-level identity separability (ROC-AUC) and ghost energy at the
     vacated positions of moving objects (motion smearing).
  4. Figures: aligned-BEV panels with GT overlays.
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

from bevdrive.datasets.nuscenes_infos import DETECTION_CLASSES  # noqa: E402
from bevdrive.evaluation.nuscenes_submission import detection_entries, write_submission  # noqa: E402
from bevdrive.temporal.bev_alignment import BEVFUSION_GRID, BEVGrid, TemporalBEVBuffer, aggregate, cosine_map, sample_at, warp_to_current  # noqa: E402
from bevdrive.tracking.temporal_query import warp_bev_feature as ckpt1_warp  # noqa: E402

GRID_YX = BEVGrid(-54.0, 54.0, -54.0, 54.0, 180, 180, "yx")


def load_bev(cache: Path, token: str, device: str) -> torch.Tensor:
    return torch.from_numpy(np.load(cache / f"{token}.npy")).to(device).float()[None]


def gt_valid(gt: dict, max_range: float = 50.0) -> np.ndarray:
    return (gt["num_pts"] > 0) & (np.abs(gt["boxes"][:, 0]) < max_range) & (np.abs(gt["boxes"][:, 1]) < max_range)


def ckpt1_sample(feature: torch.Tensor, xy: np.ndarray) -> torch.Tensor:
    """Checkpoint-1 sampling convention: rows +y (top) -> -y, columns x."""

    u = 2.0 * (xy[:, 0] + 54.0) / 108.0 - 1.0
    v = 1.0 - 2.0 * (xy[:, 1] + 54.0) / 108.0
    grid = torch.tensor(np.stack([u, v], 1), dtype=feature.dtype, device=feature.device).view(1, -1, 1, 2)
    return torch.nn.functional.grid_sample(feature, grid, align_corners=False)[0, :, :, 0]


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.r_[pos, neg]
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, default=REPO / "artifacts/checkpoint_2/phase_01/mini_all/records.pkl")
    p.add_argument("--cache", type=Path, default=REPO / "artifacts/checkpoint_2/bev_cache/v1.0-mini")
    p.add_argument("--scenes", nargs="+", default=["scene-0103", "scene-0916"])
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--results-dir", type=Path, default=REPO / "results/phase_02")
    p.add_argument("--artifacts-dir", type=Path, default=REPO / "artifacts/checkpoint_2/phase_02")
    p.add_argument("--vis-dir", type=Path, default=REPO / "outputs/phase_02")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    for d in (args.results_dir, args.artifacts_dir, args.vis_dir):
        d.mkdir(parents=True, exist_ok=True)
    from bevdrive.perception.bevfusion_runner import BEVFusionRunner

    runner = BEVFusionRunner(device=args.device)
    records = [r for r in pickle.load(args.records.open("rb"))["records"] if r["scene_name"] in set(args.scenes)]
    dev = args.device

    layout_hits = {"xy (ours)": [], "yx": [], "checkpoint-1 (rows=+y..-y, cols=x)": []}
    layout_rand = []
    warp_cos = {"no alignment": [], "aligned (ours)": [], "aligned (checkpoint-1 function)": []}
    warp_cos_occ = {k: [] for k in warp_cos}
    static_pose_err, static_feat = [], {"raw": [], "aligned": []}
    modes = ("none", "mean", "decay", "gated")
    submissions = {m: {} for m in modes}
    id_pairs = {m: {"pos": [], "neg": []} for m in modes}
    ghost = {m: [] for m in modes}
    gate_means = []
    buffers = {m: TemporalBEVBuffer(history=args.history) for m in modes}
    prev = None
    # visualise the frames with the largest ego motion per scene (alignment matters most there)
    motion = []
    for a, b in zip(records[:-1], records[1:]):
        if a["scene_name"] == b["scene_name"]:
            motion.append((float(np.linalg.norm(b["lidar2global"][:3, 3] - a["lidar2global"][:3, 3])), b["scene_name"], b["frame_index"]))
    vis_frames = set()
    for scene in args.scenes:
        ranked = sorted([m for m in motion if m[1] == scene], reverse=True)
        vis_frames |= {(m[1], m[2]) for m in ranked[:2]}
    prev_samples = {m: None for m in modes}

    for record in records:
        bev = load_bev(args.cache, record["token"], dev)
        gt = record["gt"]
        valid = gt_valid(gt)
        # ---- 1. layout check on the frozen heatmap
        heat = runner.heatmap(bev)
        if valid.any():
            xy = torch.tensor(gt["boxes"][valid, :2], dtype=torch.float32, device=dev)
            labels = torch.tensor(gt["labels"][valid], device=dev)
            for name, values in (
                ("xy (ours)", sample_at(heat, BEVFUSION_GRID, xy, mode="nearest")[0]),
                ("yx", sample_at(heat, GRID_YX, xy, mode="nearest")[0]),
                ("checkpoint-1 (rows=+y..-y, cols=x)", ckpt1_sample(heat, gt["boxes"][valid, :2])),
            ):
                layout_hits[name].extend(values[labels, torch.arange(len(labels))].cpu().tolist())
            rand_xy = torch.empty(200, 2, device=dev).uniform_(-50, 50)
            layout_rand.extend(sample_at(heat, BEVFUSION_GRID, rand_xy, mode="nearest")[0].max(0).values.cpu().tolist())
        # ---- 2. warp validation against the previous keyframe of the same scene
        if prev is not None and prev["scene_name"] == record["scene_name"]:
            prev_bev = load_bev(args.cache, prev["token"], dev)
            aligned, valid_mask = warp_to_current(prev_bev, prev["lidar2global"], record["lidar2global"])
            old_aligned, _ = ckpt1_warp(prev_bev, prev["lidar2global"], record["lidar2global"], (-54.0, -54.0, 54.0, 54.0))
            norm = bev.norm(dim=1)[0]
            occupied = valid_mask & (norm > norm[valid_mask].median())
            for name, other in (("no alignment", prev_bev), ("aligned (ours)", aligned), ("aligned (checkpoint-1 function)", old_aligned)):
                cos = cosine_map(bev, other)[0]
                warp_cos[name].append(float(cos[valid_mask].mean()))
                warp_cos_occ[name].append(float(cos[occupied].mean()))
            prev_tokens = {t: k for k, t in enumerate(prev["gt"]["instance_tokens"])}
            for k, token in enumerate(gt["instance_tokens"]):
                j = prev_tokens.get(token)
                speed = np.hypot(*gt["boxes"][k, 7:9])
                if j is None or not valid[k] or speed > 0.2 or not gt["velocity_valid"][k]:
                    continue
                world = prev["lidar2global"] @ np.r_[prev["gt"]["boxes"][j, :3], 1.0]
                in_cur = np.linalg.inv(record["lidar2global"]) @ world
                static_pose_err.append(float(np.linalg.norm(in_cur[:2] - gt["boxes"][k, :2])))
                c = torch.tensor(gt["boxes"][k : k + 1, :2], dtype=torch.float32, device=dev)
                f_cur = sample_at(bev, BEVFUSION_GRID, c)[0, :, 0]
                for name, other in (("raw", prev_bev), ("aligned", aligned)):
                    f_prev = sample_at(other, BEVFUSION_GRID, c)[0, :, 0]
                    static_feat[name].append(float(torch.nn.functional.cosine_similarity(f_cur, f_prev, dim=0)))
        # ---- 3. aggregation modes -> frozen head, identity separability, ghosting
        for mode in modes:
            buf = buffers[mode]
            if buf.scene != record["scene_name"]:
                buf.reset(record["scene_name"])
                prev_samples[mode] = None
            hist = buf.aligned_history(record["lidar2global"], record["timestamp"])
            fused, diag = aggregate(bev, hist, mode=mode)
            if mode == "gated" and "gate_newest" in diag:
                gate_means.append(float(diag["gate_newest"].mean()))
            det = runner.decode_bev(fused)
            names = [DETECTION_CLASSES[int(label)] for label in det.labels]
            submissions[mode][record["token"]] = detection_entries(record["token"], det.boxes, det.scores, names, record["lidar2global"])
            if valid.any():
                xy = torch.tensor(gt["boxes"][valid, :2], dtype=torch.float32, device=dev)
                emb = torch.nn.functional.normalize(sample_at(fused, BEVFUSION_GRID, xy)[0].T, dim=1).cpu().numpy()
                current = {"tokens": gt["instance_tokens"][valid], "names": gt["names"][valid], "xy": gt["boxes"][valid, :2], "emb": emb}
                last = prev_samples[mode]
                if last is not None:
                    # express previous positions in the current frame for the 10 m negative radius
                    t = np.linalg.inv(record["lidar2global"]) @ prev["lidar2global"]
                    last_xy = last["xy"] @ t[:2, :2].T + t[:2, 3]
                    sims = current["emb"] @ last["emb"].T
                    for a, token in enumerate(current["tokens"]):
                        for b, token_b in enumerate(last["tokens"]):
                            if token_b == token:
                                id_pairs[mode]["pos"].append(sims[a, b])
                            elif last["names"][b] == current["names"][a] and np.linalg.norm(last_xy[b] - current["xy"][a]) < 10.0:
                                id_pairs[mode]["neg"].append(sims[a, b])
                    # ghost energy: fused feature norm at the vacated position of fast movers
                    for a, token in enumerate(current["tokens"]):
                        idx = np.where(last["tokens"] == token)[0]
                        if len(idx) and np.linalg.norm(last_xy[idx[0]] - current["xy"][a]) > 1.5:
                            # object-likeness of the vacated cell: cos(cell, object feature now);
                            # fused minus single-frame > 0 means the mover left a trail (smearing)
                            pts = torch.tensor(np.stack([last_xy[idx[0]], current["xy"][a]]), dtype=torch.float32, device=dev)
                            f_f = sample_at(fused, BEVFUSION_GRID, pts)[0]
                            f_c = sample_at(bev, BEVFUSION_GRID, pts)[0]
                            obj = f_c[:, 1]
                            cos_fused = torch.nn.functional.cosine_similarity(f_f[:, 0], obj, dim=0)
                            cos_single = torch.nn.functional.cosine_similarity(f_c[:, 0], obj, dim=0)
                            ghost[mode].append(float(cos_fused - cos_single))
                prev_samples[mode] = current
            if mode == "gated" and (record["scene_name"], record["frame_index"]) in vis_frames and hist:
                render_panels(args.vis_dir, record, bev, hist, diag, fused, prev, runner)
            buf.push(bev, record["lidar2global"], record["timestamp"], record["scene_name"])
        prev = record
        if record["frame_index"] % 10 == 0:
            print(f"{record['scene_name']} f{record['frame_index']}", flush=True)

    summary = {
        "frames": len(records),
        "history": args.history,
        "layout_check_mean_gt_heatmap": {k: float(np.mean(v)) for k, v in layout_hits.items()},
        "layout_check_random_location_max_heatmap": float(np.mean(layout_rand)),
        "layout_check_n_gt": len(layout_hits["xy (ours)"]),
        "warp_cosine_valid_region": {k: float(np.mean(v)) for k, v in warp_cos.items()},
        "warp_cosine_occupied_cells": {k: float(np.mean(v)) for k, v in warp_cos_occ.items()},
        "warp_pairs": len(warp_cos["aligned (ours)"]),
        "static_gt_pose_chain_error_m": {"mean": float(np.mean(static_pose_err)), "p95": float(np.percentile(static_pose_err, 95)), "n": len(static_pose_err)},
        "static_gt_feature_cosine": {k: float(np.mean(v)) for k, v in static_feat.items()},
        "gated_mean_gate_newest": float(np.mean(gate_means)) if gate_means else None,
        "identity_roc_auc": {m: roc_auc(np.array(v["pos"]), np.array(v["neg"])) for m, v in id_pairs.items()},
        "identity_pairs": {m: {"pos": len(v["pos"]), "neg": len(v["neg"])} for m, v in id_pairs.items()},
        "ghost_energy_vacated_cell": {m: (float(np.mean(v)) if v else None) for m, v in ghost.items()},
        "ghost_n": len(ghost["mean"]),
    }
    for mode in modes:
        write_submission(args.artifacts_dir / f"submission_det_{mode}.json", submissions[mode])
    (args.results_dir / "temporal_bev_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def _pca_rgb(features: list[torch.Tensor]) -> list[np.ndarray]:
    ref = features[0][0].reshape(features[0].shape[1], -1).T
    mean = ref.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(ref - mean, q=3)
    out = []
    lo = hi = None
    for f in features:
        proj = ((f[0].reshape(f.shape[1], -1).T - mean) @ v).reshape(f.shape[2], f.shape[3], 3)
        if lo is None:
            lo, hi = proj.reshape(-1, 3).quantile(0.01, 0), proj.reshape(-1, 3).quantile(0.99, 0)
        img = ((proj - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()
        out.append(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]))  # [x, y] -> image rows=+y top
    return out


def render_panels(vis_dir: Path, record, bev, hist, diag, fused, prev, runner) -> None:
    import cv2

    aligned = hist[0][0]
    prev_raw = load_bev(Path(REPO / "artifacts/checkpoint_2/bev_cache/v1.0-mini"), prev["token"], bev.device)
    old_aligned, _ = ckpt1_warp(prev_raw, prev["lidar2global"], record["lidar2global"], (-54.0, -54.0, 54.0, 54.0))
    imgs = _pca_rgb([bev, prev_raw, aligned, old_aligned, fused])
    gate = diag["gate_newest"][0, 0].cpu().numpy().transpose(1, 0)[::-1]
    gate_img = cv2.applyColorMap((gate * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1] / 255.0
    resid = (bev - aligned).norm(dim=1)[0].cpu().numpy().transpose(1, 0)[::-1]
    resid_img = cv2.applyColorMap((np.clip(resid / np.percentile(resid, 99), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1] / 255.0
    heat_cur = runner.heatmap(bev).max(1, keepdim=True).values
    heat_prev = runner.heatmap(prev_raw).max(1, keepdim=True).values
    heat_al, _ = warp_to_current(heat_prev, prev["lidar2global"], record["lidar2global"])
    heat_old, _ = ckpt1_warp(heat_prev, prev["lidar2global"], record["lidar2global"], (-54.0, -54.0, 54.0, 54.0))

    def overlay(other):
        r = heat_cur[0, 0].cpu().numpy().transpose(1, 0)[::-1]
        g = other[0, 0].cpu().numpy().transpose(1, 0)[::-1]
        return np.clip(np.stack([r, g, np.zeros_like(r)], -1) * 1.5, 0, 1)

    shift = float(np.linalg.norm(record["lidar2global"][:3, 3] - prev["lidar2global"][:3, 3]))
    titles = ["current t", "previous t-1 (raw)", "t-1 aligned (ours)", "t-1 aligned (ckpt-1 fn)", "heat t(R) vs t-1 raw(G)",
              "gated temporal BEV", "gate w(t-1)", "|current - aligned|", "heat t(R) vs t-1 ALIGNED(G)", "heat t(R) vs ckpt-1 warp(G)"]
    panels = imgs[:4] + [overlay(heat_prev)] + [imgs[4], gate_img, resid_img, overlay(heat_al), overlay(heat_old)]
    scale = 3
    tiles = []
    for img, title in zip(panels, titles):
        tile = cv2.resize((np.asarray(img)[:, :, ::-1] * 255).astype(np.uint8), (180 * scale, 180 * scale), interpolation=cv2.INTER_NEAREST)
        for box, name, npts in zip(record["gt"]["boxes"], record["gt"]["names"], record["gt"]["num_pts"]):
            if npts == 0:
                continue
            from bevdrive.visualization.render3d import box_corners

            c = box_corners(box)[:4, :2]
            px = np.stack([(c[:, 0] + 54) / 0.6 * scale, (54 - c[:, 1]) / 0.6 * scale], 1).astype(np.int32)
            color = (0, 0, 255) if name in ("pedestrian", "bicycle", "motorcycle") else (255, 255, 255)
            cv2.polylines(tile, [px], True, color, 1, cv2.LINE_AA)
        cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:10])])
    header = np.full((36, grid.shape[1], 3), 30, np.uint8)
    cv2.putText(header, f"{record['scene_name']} frame {record['frame_index']} | ego moved {shift:.1f} m since t-1 | LiDAR frame, +y (forward) up | GT: red=VRU white=other | PCA of 512-ch fused BEV; heat = frozen TransFusion class-max heatmap (yellow = overlap)", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(vis_dir / f"aligned_bev_{record['scene_name']}_f{record['frame_index']:02d}.jpg"), np.vstack([header, grid]), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    raise SystemExit(main())
