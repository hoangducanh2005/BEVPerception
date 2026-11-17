#!/usr/bin/env python3
"""Checkpoint-2 Phase 4: train BEVFusion-TrackFormer (BEVFusion stays frozen).

Training data = cached fused BEV maps + BEVFusion proposals + nuScenes GT of
the *training* scenes only (default: nuScenes-mini ``mini_train``). The
evaluation scenes (mini_val) are never seen: no early stopping or model
selection on them; the final epoch is used.

Recipe (UniAD stage-1 / MOTR): clips of ``--clip-len`` consecutive keyframes,
track queries propagated through the clip with gradients (QIM + memory bank),
ClipMatcher losses on every decoder layer, AdamW, cosine LR, grad-clip 35.
Initialisation: UniAD stage-1 decoder/heads/QIM/memory bank; new modules
(BEV adapter, proposal encoders) from scratch with a 10x larger LR.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.temporal.bev_alignment import aggregate, warp_to_current  # noqa: E402
from bevdrive.trackformer import BEVFusionTrackFormer, ClipMatcher, TrackFormerConfig, boxes_to_code, load_uniad_track_weights  # noqa: E402

UNIAD_RANGE = 51.2


class ClipDataset(torch.utils.data.Dataset):
    """Each item: ``clip_len`` consecutive keyframes (+ ``history`` earlier ones for the temporal BEV)."""

    def __init__(self, records: list[dict], detections: dict, cache: Path, clip_len: int, history: int) -> None:
        self.records, self.detections, self.cache = records, detections, cache
        self.clip_len, self.history = clip_len, history
        by_scene: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            by_scene.setdefault(r["scene_name"], []).append(i)
        self.starts = [idx[k] for idx in by_scene.values() for k in range(0, len(idx) - clip_len + 1)]
        self.instance_ids: dict[str, int] = {}
        for r in records:
            for token in r["gt"]["instance_tokens"]:
                self.instance_ids.setdefault(token, len(self.instance_ids))

    def __len__(self) -> int:
        return len(self.starts)

    def _frame(self, i: int, with_gt: bool) -> dict:
        r = self.records[i]
        item = {"bev": np.load(self.cache / f"{r['token']}.npy"), "l2g": r["lidar2global"], "timestamp": r["timestamp"], "scene": r["scene_name"], "token": r["token"]}
        if with_gt:
            gt = r["gt"]
            keep = (gt["num_pts"] > 0) & (np.abs(gt["boxes"][:, 0]) < UNIAD_RANGE) & (np.abs(gt["boxes"][:, 1]) < UNIAD_RANGE)
            item["gt"] = {
                "boxes": gt["boxes"][keep].astype(np.float32),
                "labels": gt["labels"][keep],
                "obj_ids": np.asarray([self.instance_ids[t] for t in gt["instance_tokens"][keep]], dtype=np.int64),
                "vel_valid": gt["velocity_valid"][keep],
            }
            det = self.detections[r["token"]]
            item["proposals"] = {"boxes": det["boxes"], "scores": det["scores"], "labels": det["labels"]}
        return item

    def __getitem__(self, k: int) -> dict:
        start = self.starts[k]
        clip = [self._frame(start + j, True) for j in range(self.clip_len)]
        hist = []
        for j in range(1, self.history + 1):
            i = start - j
            if i >= 0 and self.records[i]["scene_name"] == self.records[start]["scene_name"]:
                hist.append(self._frame(i, False))
        return {"clip": clip, "history": hist[::-1]}


def identity_collate(item):
    return item


def build_frames(item: dict, device: str, use_temporal: bool, history: int) -> list[dict]:
    """Move a clip to the GPU and compute the motion-gated temporal context for every frame."""

    seq = item["history"] + item["clip"]
    bevs = [torch.from_numpy(f["bev"]).to(device).float()[None] for f in seq]
    frames = []
    offset = len(item["history"])
    for j, f in enumerate(item["clip"]):
        idx = offset + j
        hist_ctx = None
        if use_temporal:
            hist = []
            for k in range(1, history + 1):
                if idx - k < 0:
                    break
                feat, valid = warp_to_current(bevs[idx - k], seq[idx - k]["l2g"], f["l2g"])
                hist.append((feat, valid, f["timestamp"] - seq[idx - k]["timestamp"]))
            hist_ctx, _ = aggregate(bevs[idx], hist, mode="gated", include_current=False) if hist else (torch.zeros_like(bevs[idx]), None)
        gt = f["gt"]
        boxes = torch.from_numpy(gt["boxes"]).to(device)
        frames.append(
            {
                "bev": bevs[idx],
                "history": hist_ctx,
                "proposals": f["proposals"],
                "l2g": f["l2g"],
                "timestamp": f["timestamp"],
                "gt": {
                    "code": boxes_to_code(boxes) if len(boxes) else torch.zeros(0, 10, device=device),
                    "labels": torch.from_numpy(gt["labels"]).to(device),
                    "obj_ids": torch.from_numpy(gt["obj_ids"]).to(device),
                    "vel_valid": torch.from_numpy(gt["vel_valid"]).to(device),
                },
            }
        )
    return frames


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, nargs="+", default=[REPO / "artifacts/checkpoint_2/phase_01/mini_all/records.pkl"])
    p.add_argument("--detections", type=Path, nargs="+", default=[REPO / "artifacts/checkpoint_2/phase_01/mini_all/detections.pkl"])
    p.add_argument("--cache", type=Path, nargs="+", default=[REPO / "artifacts/checkpoint_2/bev_cache/v1.0-mini"])
    p.add_argument("--train-scenes", nargs="+", default=["mini_train"])
    p.add_argument("--exclude-scenes", nargs="*", default=["scene-0103", "scene-0916"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--clip-len", type=int, default=3)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-pretrained-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--query-mode", default="proposal")
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--no-uniad-init", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    wanted = {s for item in args.train_scenes for s in (splits[item] if item in splits else [item])} - set(args.exclude_scenes)
    records, datasets = [], []
    for rec_path, det_path, cache in zip(args.records, args.detections, args.cache):
        recs = [r for r in pickle.load(rec_path.open("rb"))["records"] if r["scene_name"] in wanted]
        dets = pickle.load(det_path.open("rb"))
        datasets.append(ClipDataset(recs, dets, cache, args.clip_len, args.history))
        records += recs
    dataset = torch.utils.data.ConcatDataset(datasets)
    scenes = sorted({r["scene_name"] for r in records})
    print(f"[data] {len(scenes)} train scenes, {len(records)} keyframes, {len(dataset)} clips", flush=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=identity_collate)

    cfg = TrackFormerConfig(query_mode=args.query_mode, use_temporal_bev=not args.no_temporal)
    model = BEVFusionTrackFormer(cfg)
    init = {"loaded": [], "not_initialised": []} if args.no_uniad_init else load_uniad_track_weights(model)
    model.to(args.device).train()
    pretrained = set(init["loaded"])
    groups = [
        {"params": [p_ for n, p_ in model.named_parameters() if n in pretrained], "lr": args.lr * args.lr_pretrained_mult},
        {"params": [p_ for n, p_ in model.named_parameters() if n not in pretrained], "lr": args.lr},
    ]
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(dataset)
    base_lrs = [g["lr"] for g in optim.param_groups]
    criterion = ClipMatcher()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "model": asdict(cfg), "train_scenes": scenes, "init_loaded": len(init["loaded"]), "init_new": init["not_initialised"]}, indent=2))
    log = (args.output_dir / "train_log.jsonl").open("a")
    step = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        t0, sums, n = time.time(), {}, 0
        for item in loader:
            warm = min(1.0, (step + 1) / 200)
            cos = 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
            for g, base in zip(optim.param_groups, base_lrs):
                g["lr"] = base * warm * max(cos, 0.02)
            frames = build_frames(item, args.device, cfg.use_temporal_bev, args.history)
            losses = model.forward_train_clip(frames, criterion)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0))
            optim.step()
            step += 1
            n += 1
            for key in ("loss", "cls_d5", "box_d5"):
                sums[key] = sums.get(key, 0.0) + float(losses[key])
            for key, value in criterion.stats.items():
                sums[key] = sums.get(key, 0.0) + value
            if step % 50 == 0:
                print(f"ep {epoch} step {step} loss {float(losses['loss']):.3f} cls5 {float(losses['cls_d5']):.3f} box5 {float(losses['box_d5']):.3f} gn {grad_norm:.1f} "
                      f"match new/track/gt {criterion.stats['matched_new']}/{criterion.stats['matched_track']}/{criterion.stats['gt']} mem {torch.cuda.max_memory_allocated() / 2**20:.0f}MB", flush=True)
        entry = {"epoch": epoch, "steps": n, "sec": time.time() - t0, **{k: v / max(n, 1) for k, v in sums.items()}, "track_query_match_ratio": sums.get("matched_track", 0) / max(sums.get("gt", 1), 1), "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print("[epoch]", json.dumps(entry), flush=True)
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, args.output_dir / "latest.pth")
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": args.epochs - 1, "train_scenes": scenes}, args.output_dir / "final.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
