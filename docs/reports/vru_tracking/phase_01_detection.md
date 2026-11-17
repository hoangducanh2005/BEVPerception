# Phase 1 — VRU Detection Recall Analysis

## A. Objective

Measure whether low pedestrian recall is caused by score thresholding, data or
evaluation issues, or detector limitations.

## B. Work completed

- Added `scripts/vru_detection_experiment.py` without changing the frozen
  detector or the existing Phase 1/2 runners.
- Ran BEVFusion once on 80 frames from `scene-0003` and `scene-0004` with seed
  `0` and cached the model output before experiment score filtering. The cache
  is still after the detector's own NMS/top-K behavior.
- Evaluated thresholds `[0.05, 0.10, 0.20, 0.30, 0.50]` with the same 2m XY
  matching threshold.
- Evaluated VRU recall in 0–15m, 15–30m and 30–50m ego-centric planar range
  bins.
- Classified unmatched VRU GT boxes using only evidence in the raw cache.
- Generated representative BEV/camera diagnostics for each false-negative
  category.

## C. Experimental results

### Confidence threshold sweep — native class evaluation

| Threshold | Class | Detections | TP | FP | FN | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | pedestrian | 544 | 359 | 185 | 645 | 0.660 | 0.358 | 0.464 |
| 0.10 | pedestrian | 368 | 293 | 75 | 711 | 0.796 | 0.292 | 0.427 |
| 0.20 | pedestrian | 218 | 199 | 19 | 805 | 0.913 | 0.198 | 0.326 |
| 0.30 | pedestrian | 143 | 137 | 6 | 867 | 0.958 | 0.136 | 0.239 |
| 0.50 | pedestrian | 53 | 52 | 1 | 952 | 0.981 | 0.052 | 0.098 |
| 0.05 | bicycle | 20 | 4 | 16 | 91 | 0.200 | 0.042 | 0.070 |
| 0.20 | bicycle | 1 | 0 | 1 | 95 | 0.000 | 0.000 | 0.000 |
| 0.05 | motorcycle | 29 | 12 | 17 | 23 | 0.414 | 0.343 | 0.375 |
| 0.20 | motorcycle | 9 | 8 | 1 | 27 | 0.889 | 0.229 | 0.364 |
| 0.05 | vru | 593 | 377 | 216 | 757 | 0.636 | 0.332 | 0.437 |
| 0.20 | vru | 228 | 210 | 18 | 924 | 0.921 | 0.185 | 0.308 |
| 0.05 | vehicle | 1253 | 774 | 479 | 530 | 0.618 | 0.594 | 0.605 |
| 0.20 | vehicle | 595 | 552 | 43 | 752 | 0.928 | 0.423 | 0.581 |

Full sweep: `artifacts/experiments/vru_tracking/phase_01/nuscenes_detection/threshold_sweep.csv`.

### Distance-based recall at threshold 0.20

| Class | 0–15m | 15–30m | 30–50m |
|---|---:|---:|---:|
| pedestrian | 0.736 (92/125) | 0.294 (96/326) | 0.037 (11/297) |
| bicycle | 0.000 (0/2) | 0.118 (2/17) | 0.000 (0/75) |
| motorcycle | 0.667 (2/3) | 0.700 (7/10) | 0.000 (0/13) |

The distance denominator is the number of GT boxes in the ego-centric planar
range bin. Counts are shown as `TP/GT`; bins omit GT beyond 50m.

### False-negative analysis at threshold 0.20

| Category | Count | Share of 924 unmatched VRU GT boxes |
|---|---:|---:|
| Candidate below threshold | 446 | 48.3% |
| No same-class raw candidate within 2m | 424 | 45.9% |
| Wrong-class raw candidate within 2m | 37 | 4.0% |
| Candidate competed/matching conflict | 17 | 1.8% |

The `no same-class raw candidate` category cannot be further labeled as domain
shift, occlusion or pre-NMS/top-K loss from the current output alone. The raw
cache proves only that no retained model output of that class was within 2m.

Existing reference only:

| Dataset | Pedestrian precision | Pedestrian recall | Pedestrian F1 |
|---|---:|---:|---:|
| nuScenes bounded artifact | 0.913 | 0.198 | 0.326 |
| NavSim bounded artifact | 0.400 | 0.068 | 0.117 |

## D. Technical findings

- Thresholding is a significant bottleneck for pedestrians: threshold `0.05`
  recovers 160 additional TP over `0.20`, at the cost of 166 additional FP.
- Threshold `0.05` gives the best pedestrian and aggregate VRU F1 among the
  tested values, but pedestrian precision drops from 0.913 to 0.660.
- Distance is a stronger bottleneck than threshold at long range: pedestrian
  recall falls from 0.736 at 0–15m to 0.037 at 30–50m at threshold 0.20.
- Bicycle predictions are nearly absent: only 20 retained outputs at 0.05 and
  1 at 0.20 for 95 GT boxes.
- 48.3% of unmatched VRU GT boxes have a same-class raw candidate below 0.20,
  so a lower score threshold can recover some recall.
- 45.9% have no same-class raw candidate within 2m even before score filtering;
  thresholding cannot recover these cases. They remain potential detector
  limitations, localization errors, occlusion or top-K/NMS losses.
- Native pedestrian, bicycle and motorcycle metrics are now reported separately
  in this experiment; this does not alter the old baseline protocol.

## E. Changes relative to previous phase

- Added an experiment-only raw-detection cache and evaluation script.
- Added explicit native VRU class metrics and distance bins.
- Added false-negative categorization and diagnostic visualizations.
- The pretrained checkpoint, detector config, model weights and selected scene
  split were unchanged.

## F. Limitations

- Raw detections are after the detector's own NMS/top-K, so candidate-missing
  cannot be attributed specifically to pre-NMS suppression.
- The 80-frame, two-scene subset is not a full nuScenes evaluation.
- Visual false-negative cases are representative, not exhaustive.
- No threshold sweep was run on NavSim because the Phase 1 native-class and
  range analysis targets the nuScenes GT protocol.

## G. Recommendation

For recall-first VRU inference, keep `0.05` as an experimental candidate, not a
global production default: it improves pedestrian/VRU recall and F1 on this
bounded subset but materially lowers precision. The next decision should use
the tracking oracle and false-positive cost, with separate thresholds or
post-processing for VRUs if justified. Do not claim detector recovery for the
45.9% no-candidate category without additional detector evidence.

## H. Completion status

**PASS**
