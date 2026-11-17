# Phase 2 — Tracking Evaluation Audit and GT-Box Oracle

## A. Objective

Separate detector dropout from tracker/association errors and validate ID
switch, fragmentation and scene-reset accounting.

## B. Work completed

- Added `scripts/vru_tracking_oracle.py` using the cached raw detections from
  Phase 1, without rerunning or modifying BEVFusion.
- Re-ran the same `TrackManager` on predicted boxes and on GT boxes.
- Added global one-to-one Hungarian evaluation with exact native class names.
- Separated consecutive-frame ID switches from reappearance after a gap.
- Logged ID-switch/fragmentation events and per-scene/per-class metrics.
- Used the same 80 frames, score threshold `0.20`, tracker gate `3.0m`,
  max-missed `3`, and evaluation distance threshold `2.0m`.

## C. Experimental results

### Predicted boxes versus GT-box oracle

| Mode | Matched | Predicted | GT | Precision | Recall | F1 | Consecutive ID switches | Gap fragmentations | Gap reappearances | Lost frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predicted boxes + Kalman/Hungarian | 1237 | 1337 | 5338 | 0.925 | 0.232 | 0.371 | 133 | 83 | 163 | 351 |
| GT boxes + same tracker | 5338 | 5338 | 5338 | 1.000 | 1.000 | 1.000 | 36 | 0 | 0 | 0 |

### Per-scene corrected identity metrics

| Mode | Scene | Consecutive ID switches | Gap fragmentations | Gap reappearances | Lost frames |
|---|---|---:|---:|---:|---:|
| Predicted boxes | scene-0003 | 6 | 24 | 61 | 172 |
| Predicted boxes | scene-0004 | 127 | 59 | 102 | 179 |
| GT boxes | scene-0003 | 15 | 0 | 0 | 0 |
| GT boxes | scene-0004 | 21 | 0 | 0 | 0 |

### VRU per-class oracle sanity check

| Mode | Pedestrian P/R/F1 | Bicycle P/R/F1 | Motorcycle P/R/F1 |
|---|---|---|---|
| Predicted boxes | 0.913 / 0.198 / 0.326 | 0.000 / 0.000 / 0.000 | 0.889 / 0.229 / 0.364 |
| GT boxes | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

The old evaluator reported 232 identity changes. Under the corrected global
matching/consecutive protocol, the cached detection stream produces 133
consecutive ID switches and 83 gap fragmentations. The numbers are not expected
to be identical because the corrected protocol uses native classes, Hungarian
GT matching and an explicit gap distinction.

| Dataset | Tracking F1 | Reported ID switches | Gap-related identity changes found in audit |
|---|---:|---:|---:|
| nuScenes 80 frames | 0.380 | 232 | 88 |
| NavSim 10 frames | 0.229 | 0 | Not measured from native GT protocol |

The 88 gap-related changes show why the existing `232` number cannot be
interpreted as 232 true consecutive-frame identity switches.

## D. Technical findings

- The GT-box oracle reaches perfect detection precision/recall by construction,
  but still has 36 consecutive ID switches. This proves the tracker itself has
  an association/kinematic ambiguity component independent of detector dropout.
- Predicted boxes have 351 lost frames and 83 gap fragmentations, indicating
  detection continuity is a much larger practical issue than the oracle case.
- Scene-0004 is the dominant source of identity failures: 127 consecutive
  switches and 59 fragmentations versus 6 and 24 in scene-0003.
- The old 232 count is not a reliable pure ID-switch metric because it counts
  changes after arbitrary gaps.
- The corrected matching is still a local diagnostic, not official nuScenes
  AMOTA/AMOTP/HOTA/AssA.

## E. Changes relative to previous phase

- Added corrected identity accounting and GT-box oracle evaluation.
- Added per-scene/per-class CSV output and event logs.
- No detector weights, tracker parameters or association algorithm were changed.

## F. Limitations

- Oracle metrics use GT boxes but do not pass GT instance IDs to the tracker.
- The oracle uses exact native class names and the existing 3m tracker gate;
  it does not represent a learned tracker.
- No official AMOTA/AMOTP/HOTA/AssA result is available.
- The subset has only 40 frames per scene and is not a full-split benchmark.

## G. Recommendation

Keep the corrected evaluation protocol as the versioned reference. The large
drop from oracle F1 `1.000` to predicted-box F1 `0.371`, plus the 36 oracle ID
switches, says the next work should address both detector continuity and
class-aware motion association. Do not evaluate query-assisted association
until Phase 3 measures positive/negative identity separation.

## H. Completion status

**PASS**
