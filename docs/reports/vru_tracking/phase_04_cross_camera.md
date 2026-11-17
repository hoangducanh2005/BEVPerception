# Phase 4 — Cross-Camera Handoff Evaluation

## A. Objective

Measure identity preservation when a VRU moves between camera views, separate
from ordinary same-camera tracking and detection dropout.

## B. Work completed

- Added `scripts/vru_cross_camera.py` using the cached detections, nuScenes
  calibration, GT instance tokens and image files.
- Projected GT 3D VRU boxes into all six camera views.
- Mined adjacent-frame candidates where the geometric visible-camera set
  changed with no camera intersection.
- Attached predicted track IDs before and after each candidate using the same
  classical tracker and 2m GT matching.
- Generated synchronized before/after six-camera montages for the first 10
  candidates.
- Manually inspected representative montages, including a preserved-ID event
  and a detection-dropout event. The runner intentionally leaves all events
  marked `geometric_candidate_not_manually_validated`.

## C. Experimental results

| Quantity | Result |
|---|---:|
| Frames | 80 |
| Geometric handoff candidates | 18 |
| scene-0003 candidates | 4 |
| scene-0004 candidates | 14 |
| Candidates with same predicted track ID before/after | 12 |
| Candidates with missing track on one/both sides | 6 |
| Manually validated handoffs | 0 |

The `12/18 = 66.7%` value is only geometric-candidate track continuity; it is
not a validated cross-camera handoff success rate. The event list and images
are in:

```text
artifacts/experiments/vru_tracking/phase_04/cross_camera/cross_camera_events.json
artifacts/experiments/vru_tracking/phase_04/cross_camera/visualizations/
```

## D. Technical findings

- The 80-frame subset contains geometric transitions, mostly in scene-0004.
- A global 3D tracker can preserve an ID through a camera-set transition when
  detections remain available; 12 candidates did so.
- Six candidates lose one or both track IDs, consistent with the detection
  recall/dropout bottleneck identified in Phase 1.
- Projection and visibility are geometry-based. The code does not perform
  occlusion reasoning or establish that a projected person is visually visible.
- Manual inspection confirms the montage is useful for review, but the current
  sample is insufficient to claim a statistically reliable handoff rate.

## E. Changes relative to previous phase

- Added camera projection, transition-event mining and synchronized diagnostic
  montages.
- No detector weights, tracker parameters or ID-association algorithm changed.

## F. Limitations

- `manually_validated_handoffs=0`; therefore validated success rate and
  transition-specific ID-switch rate are intentionally not reported.
- Geometric projection does not prove visibility under occlusion.
- Only two bounded scenes and 18 candidates are available.
- The event detector requires adjacent frames with disjoint camera sets and
  may miss transitions where camera overlap exists.

## G. Recommendation

Keep this phase **PARTIAL**. For a valid handoff metric, manually validate a
larger labeled subset of candidates or select additional bounded scenes only
with explicit authorization. Until then, use the 12/18 value only as a review
queue, not as a product metric.

## H. Completion status

**PARTIAL**
