#!/usr/bin/env python3
"""Dump UniAD stage-1 TrackFormer inputs/outputs on nuScenes-mini val (env ``uniad2.0``).

Run from the UniAD repo root (it imports UniAD's plugin):

  cd ~/GIT/UniAD && PYTHONPATH=. conda run -n uniad2.0 python \
     ~/GIT/BEVPerception/scripts/uniad_dump_track_io.py \
     research_outputs/configs/mini_track_map.py ckpts/uniad_base_track_map.py OUT_DIR

Per frame it stores the BEVFormer ``bev_embed`` (the TrackFormer value map),
the ego poses / timestamp handed to the tracker, and UniAD's own track
outputs. ``cp2_validate_port.py`` feeds the same BEV into the BEVPerception
port with the same weights and compares ids/boxes. The UniAD model is run
unchanged (``simple_test_track`` is wrapped, not modified).
"""

import argparse
import functools
import importlib
import os
import pickle
import warnings

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--fp32", action="store_true", help="store bev_embed in float32 (exact port check)")
    args = ap.parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
    m = model.module
    rec = {}
    orig = m.simple_test_track

    @functools.wraps(orig)
    def wrapped(img=None, l2g_t=None, l2g_r_mat=None, img_metas=None, timestamp=None):
        out = orig(img=img, l2g_t=l2g_t, l2g_r_mat=l2g_r_mat, img_metas=img_metas, timestamp=timestamp)
        r = out[0]
        rec["bev_embed"] = (r["bev_embed"][:, 0].detach().float() if args.fp32 else r["bev_embed"][:, 0].detach().half()).cpu().numpy()  # [H*W, 256], (h w) row-major
        rec["l2g_t"] = l2g_t.detach().double().cpu().numpy().reshape(3)
        rec["l2g_r_mat"] = l2g_r_mat.detach().double().cpu().numpy().reshape(3, 3)
        rec["timestamp"] = float(timestamp.detach().double().cpu().numpy().reshape(-1)[0])
        rec["scene_token"] = img_metas[0]["scene_token"]
        rec["sample_token"] = img_metas[0]["sample_idx"]
        rec["boxes"] = r["boxes_3d"].tensor.detach().cpu().numpy()
        rec["scores"] = r["scores_3d"].detach().cpu().numpy()
        rec["labels"] = r["labels_3d"].detach().cpu().numpy()
        rec["track_ids"] = r["track_ids"].detach().cpu().numpy()
        return out

    m.simple_test_track = wrapped
    os.makedirs(args.out_dir, exist_ok=True)
    index = []
    for i, data in enumerate(loader):
        rec.clear()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        np.savez(os.path.join(args.out_dir, "frame_%03d.npz" % i), **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)})
        index.append({k: v for k, v in rec.items() if not isinstance(v, np.ndarray)})
        if i % 10 == 0:
            print(i, rec["scene_token"][:8], len(rec["track_ids"]), flush=True)
    with open(os.path.join(args.out_dir, "index.pkl"), "wb") as f:
        pickle.dump(index, f)


if __name__ == "__main__":
    main()
