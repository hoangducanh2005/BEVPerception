#!/usr/bin/env python3
"""Official nuScenes devkit evaluation (detection or tracking) for checkpoint 2.

Runs ``DetectionEval`` (detection_cvpr_2019) or ``TrackingEval``
(tracking_nips_2019) unchanged and writes the devkit's own summary plus a
compact VRU table. Python-3.9 compatible so it can run in the ``uniad2.0``
env, which has the ``motmetrics`` version the tracking eval needs.

``--scenes`` restricts the GT split to a subset of scenes (e.g. the frames
actually processed); it patches the split table the devkit loaders read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VRU = ("pedestrian", "bicycle", "motorcycle")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("det", "track"), required=True)
    p.add_argument("--result-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--data-root", default="/home/anhhd/dataset/nuscenes-mini")
    p.add_argument("--eval-set", default="mini_val")
    p.add_argument("--scenes", nargs="*", default=None, help="restrict eval_set to these scene names")
    p.add_argument("--render-curves", action="store_true")
    return p.parse_args()


def patch_split(eval_set, scenes):
    import nuscenes.eval.common.loaders as loaders
    from nuscenes.utils import splits

    original = splits.create_splits_scenes

    def patched(verbose=False):
        table = original(verbose)
        table[eval_set] = [s for s in table[eval_set] if s in set(scenes)] if eval_set in table else list(scenes)
        return table

    loaders.create_splits_scenes = patched
    try:
        import nuscenes.eval.tracking.loaders as tloaders  # noqa: F401
        if hasattr(tloaders, "create_splits_scenes"):
            tloaders.create_splits_scenes = patched
    except ImportError:
        pass


def restrict_to_eval_set(nusc, args):
    """Keep only sample tokens of the evaluated scenes (the devkit requires an exact match)."""

    from nuscenes.eval.common import loaders

    scene_names = set(loaders.create_splits_scenes()[args.eval_set])
    tokens = {s["token"] for s in nusc.sample if nusc.get("scene", s["scene_token"])["name"] in scene_names}
    data = json.loads(Path(args.result_json).read_text())
    kept = {k: v for k, v in data["results"].items() if k in tokens}
    missing = tokens - set(kept)
    if missing:
        raise SystemExit("%d evaluated samples have no prediction entry" % len(missing))
    out = args.output_dir / ("filtered_" + Path(args.result_json).name)
    out.write_text(json.dumps({"meta": data["meta"], "results": kept}))
    return out


def main():
    args = parse_args()
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory

    if args.scenes:
        patch_split(args.eval_set, args.scenes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    args.result_json = restrict_to_eval_set(nusc, args)
    if args.task == "det":
        from nuscenes.eval.detection.evaluate import DetectionEval

        cfg = config_factory("detection_cvpr_2019")
        evaluator = DetectionEval(nusc, cfg, str(args.result_json), args.eval_set, str(args.output_dir), verbose=False)
        metrics, metric_data = evaluator.evaluate()
        summary = metrics.serialize()
        (args.output_dir / "metrics_details.json").write_text(json.dumps(metric_data.serialize()))
        (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        rows = {}
        for name in VRU:
            rows[name] = {
                "AP": summary["mean_dist_aps"][name],
                "AP@0.5m": summary["label_aps"][name][0.5],
                "AP@1m": summary["label_aps"][name][1.0],
                "AP@2m": summary["label_aps"][name][2.0],
                "AP@4m": summary["label_aps"][name][4.0],
                "ATE": summary["label_tp_errors"][name]["trans_err"],
                "ASE": summary["label_tp_errors"][name]["scale_err"],
                "AOE": summary["label_tp_errors"][name]["orient_err"],
                "AVE": summary["label_tp_errors"][name]["vel_err"],
            }
        compact = {
            "mAP": summary["mean_ap"],
            "NDS": summary["nd_score"],
            "mATE": summary["tp_errors"]["trans_err"],
            "mAVE": summary["tp_errors"]["vel_err"],
            "VRU_mAP": sum(r["AP"] for r in rows.values()) / len(rows),
            "per_class_vru": rows,
            "per_class_ap_all": summary["mean_dist_aps"],
        }
    else:
        from nuscenes.eval.tracking.evaluate import TrackingEval

        cfg = config_factory("tracking_nips_2019")
        evaluator = TrackingEval(cfg, str(args.result_json), args.eval_set, str(args.output_dir), args.version, args.data_root, verbose=False, render_classes=None)
        summary = evaluator.main(render_curves=args.render_curves)
        keys = ("amota", "amotp", "recall", "motar", "mota", "motp", "ids", "frag", "tp", "fp", "fn", "gt", "tid", "lgd", "faf")
        rows = {name: {k: summary["label_metrics"][k][name] for k in keys if k in summary["label_metrics"]} for name in VRU}
        compact = {k: summary.get(k) for k in keys}
        compact["per_class_vru"] = rows
        vru_amota = [rows[n]["amota"] for n in VRU if rows[n]["amota"] == rows[n]["amota"]]
        compact["VRU_mean_amota"] = sum(vru_amota) / len(vru_amota) if vru_amota else None
    compact["eval_set"] = args.eval_set
    compact["scenes"] = args.scenes
    compact["result_json"] = str(args.result_json)
    (args.output_dir / "compact_metrics.json").write_text(json.dumps(compact, indent=2))
    print(json.dumps({k: v for k, v in compact.items() if k not in ("per_class_ap_all",)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
