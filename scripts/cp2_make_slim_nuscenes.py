#!/usr/bin/env python3
"""Write a slim copy of nuScenes metadata restricted to a few scenes (for RAM-bounded devkit evaluation).

Loading the full v1.0-trainval tables takes ~10-12 GB of RAM inside the devkit
(OOM on a 15 GB laptop). The evaluation only needs the tables of the
evaluated scenes, so this copies scene/sample/sample_data/ego_pose/
sample_annotation/instance rows of those scenes (plus the small global tables)
into ``<out-root>/<out-version>``. The version name must end with ``trainval``
so the devkit accepts ``--eval-set val``. Row contents are copied verbatim.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path


def load(path: Path) -> list:
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("/media/anhhd/New Volume/nuscenes/v1.0-trainval"))
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--out-version", default="v1.0-val23-trainval")
    args = p.parse_args()
    assert args.out_version.endswith("trainval")
    out = args.out_root / args.out_version
    out.mkdir(parents=True, exist_ok=True)
    for name in ("category", "attribute", "visibility", "sensor", "calibrated_sensor", "log", "map"):
        shutil.copy(args.src / f"{name}.json", out / f"{name}.json")
    scenes = [s for s in load(args.src / "scene.json") if s["name"] in set(args.scenes)]
    assert len(scenes) == len(set(args.scenes)), "some scenes not found"
    scene_tokens = {s["token"] for s in scenes}
    samples = [s for s in load(args.src / "sample.json") if s["scene_token"] in scene_tokens]
    sample_tokens = {s["token"] for s in samples}
    sample_data = [d for d in load(args.src / "sample_data.json") if d["sample_token"] in sample_tokens]
    gc.collect()
    pose_tokens = {d["ego_pose_token"] for d in sample_data}
    ego_pose = [e for e in load(args.src / "ego_pose.json") if e["token"] in pose_tokens]
    gc.collect()
    anns = [a for a in load(args.src / "sample_annotation.json") if a["sample_token"] in sample_tokens]
    gc.collect()
    instance_tokens = {a["instance_token"] for a in anns}
    instances = [i for i in load(args.src / "instance.json") if i["token"] in instance_tokens]
    for name, rows in (("scene", scenes), ("sample", samples), ("sample_data", sample_data), ("ego_pose", ego_pose), ("sample_annotation", anns), ("instance", instances)):
        (out / f"{name}.json").write_text(json.dumps(rows))
    print(f"wrote {out}: scenes {len(scenes)}, samples {len(samples)}, sample_data {len(sample_data)}, annotations {len(anns)}, instances {len(instances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
