#!/usr/bin/env python3
"""Phase 5: collect A vs B (and references) into results/phase_05/comparison.{json,md}.

Every number is read from devkit outputs / runtime files written by the
earlier scripts; nothing is recomputed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VRU = ("pedestrian", "bicycle", "motorcycle")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def track_row(eval_dir: Path) -> dict | None:
    s = load(eval_dir / "metrics_summary.json")
    if s is None:
        return None
    lm = s["label_metrics"]
    row = {"AMOTA": s["amota"], "AMOTP": s["amotp"], "recall": s["recall"], "MOTA": s["mota"], "IDS": s["ids"], "FRAG": s["frag"]}
    for c in VRU:
        row[c] = {k: lm[k][c] for k in ("amota", "amotp", "recall", "ids", "frag")}
    row["VRU_AMOTA"] = sum(lm["amota"][c] for c in VRU) / 3
    row["VRU_IDS"] = sum(lm["ids"][c] for c in VRU)
    row["VRU_FRAG"] = sum(lm["frag"][c] for c in VRU)
    return row


def det_row(eval_dir: Path) -> dict | None:
    s = load(eval_dir / "metrics_summary.json")
    if s is None:
        return None
    row = {"mAP": s["mean_ap"], "NDS": s["nd_score"], "mATE": s["tp_errors"]["trans_err"], "mAVE": s["tp_errors"]["vel_err"]}
    for c in VRU:
        row[c] = {"AP": s["mean_dist_aps"][c], "ATE": s["label_tp_errors"][c]["trans_err"], "AVE": s["label_tp_errors"][c]["vel_err"]}
    row["VRU_mAP"] = sum(s["mean_dist_aps"][c] for c in VRU) / 3
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", type=Path, default=REPO / "results/phase_05/methods.json", help="name -> {track_eval, det_eval, runtime, failures}")
    p.add_argument("--out", type=Path, default=REPO / "results/phase_05/comparison")
    args = p.parse_args()
    methods = json.loads(args.methods.read_text())
    out = {}
    for name, m in methods.items():
        entry = {"description": m.get("description", "")}
        if m.get("track_eval"):
            entry["tracking"] = track_row(REPO / m["track_eval"])
        if m.get("det_eval"):
            entry["detection"] = det_row(REPO / m["det_eval"])
        for key in ("runtime", "detector_runtime"):
            if m.get(key):
                entry[key] = load(REPO / m[key])
        if m.get("failures"):
            f = load(REPO / m["failures"])
            entry["failures"] = f["counts"] if f else None
        out[name] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=2))
    lines = ["| Method | AMOTA | AMOTP | Recall | IDS | FRAG | Ped AMOTA (IDS) | Bicycle AMOTA (IDS) | Moto AMOTA (IDS) | VRU AMOTA | VRU IDS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in out.items():
        t = e.get("tracking")
        if not t:
            continue
        cells = [f"{t[c]['amota']:.3f} ({int(t[c]['ids'])})" for c in VRU]
        lines.append(f"| {name} | {t['AMOTA']:.3f} | {t['AMOTP']:.3f} | {t['recall']:.3f} | {int(t['IDS'])} | {int(t['FRAG'])} | " + " | ".join(cells) + f" | {t['VRU_AMOTA']:.3f} | {int(t['VRU_IDS'])} |")
    lines += ["", "| Method | mAP | NDS | mATE | mAVE | Ped AP | Bicycle AP | Moto AP | VRU mAP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in out.items():
        d = e.get("detection")
        if not d:
            continue
        lines.append(f"| {name} | {d['mAP']:.3f} | {d['NDS']:.3f} | {d['mATE']:.3f} | {d['mAVE']:.3f} | " + " | ".join(f"{d[c]['AP']:.3f}" for c in VRU) + f" | {d['VRU_mAP']:.3f} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
