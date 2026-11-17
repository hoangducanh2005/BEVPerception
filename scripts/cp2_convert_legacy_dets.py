#!/usr/bin/env python3
"""Apply the legacy->mmdet3d-v1 box re-parametrisation to an existing detection run.

Runs produced before the runner's ``legacy_box_convention`` fix stored raw
checkpoint boxes. This rewrites ``detections.pkl`` / ``submission_det.json``
in place (keeping ``detections_raw_legacy.pkl``) and is idempotent.
Optionally re-renders the six-camera/BEV visualisations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bevdrive.datasets.nuscenes_infos import DETECTION_CLASSES  # noqa: E402
from bevdrive.evaluation.nuscenes_submission import detection_entries, write_submission  # noqa: E402
from bevdrive.perception.bevfusion_runner import BOX_CONVENTION_TAG, legacy_to_v1_boxes  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--records", type=Path, default=None)
    p.add_argument("--vis-dir", type=Path, default=None)
    p.add_argument("--vis-every", type=int, default=10)
    args = p.parse_args()
    runtime_path = args.run_dir / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    raw_path = args.run_dir / "detections_raw_legacy.pkl"
    tag = runtime.get("box_convention", "raw checkpoint output")
    if tag == BOX_CONVENTION_TAG:
        print("already converted")
        detections = pickle.load((args.run_dir / "detections.pkl").open("rb"))
    else:
        if raw_path.exists():  # convert from the untouched checkpoint output
            detections = pickle.load(raw_path.open("rb"))
            fix_dims_yaw = True
        else:
            detections = pickle.load((args.run_dir / "detections.pkl").open("rb"))
            fix_dims_yaw = not tag.startswith("mmdet3d_v1")  # runner v1 fixed dims/yaw but not z
            if not fix_dims_yaw:
                pass
            else:
                shutil.copy(args.run_dir / "detections.pkl", raw_path)
        for det in detections.values():
            det["boxes"] = legacy_to_v1_boxes(det["boxes"], fix_dims_yaw=fix_dims_yaw, fix_gravity_z=True)
        with (args.run_dir / "detections.pkl").open("wb") as handle:
            pickle.dump(detections, handle)
        runtime["box_convention"] = BOX_CONVENTION_TAG
        runtime_path.write_text(json.dumps(runtime, indent=2))
    records = pickle.load((args.records or args.run_dir / "records.pkl").open("rb"))["records"]
    submission = {}
    for record in records:
        det = detections[record["token"]]
        names = [DETECTION_CLASSES[int(label)] for label in det["labels"]]
        submission[record["token"]] = detection_entries(record["token"], det["boxes"], det["scores"], names, record["lidar2global"])
    write_submission(args.run_dir / "submission_det.json", submission)
    if args.vis_dir:
        from types import SimpleNamespace

        from cp2_detect import render_frame

        for i, record in enumerate(records):
            if i % args.vis_every == 0:
                det = detections[record["token"]]
                names = [DETECTION_CLASSES[int(label)] for label in det["labels"]]
                render_frame(args.vis_dir, i, record, SimpleNamespace(**det), names, 0.3)
    print(f"converted {len(detections)} frames in {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "scripts"))
    raise SystemExit(main())
