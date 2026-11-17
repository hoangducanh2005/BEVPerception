# Phase 3 — Temporal BEV and Query Embedding Analysis

## A. Objective

Test whether frozen BEVFusion features discriminate VRU identities, rather
than treating a high same-track cosine as proof of identity quality.

## B. Work completed

- Added `scripts/vru_query_analysis.py` and reran the frozen BEVFusion model on
  the same 80 cached-detection frames.
- Extracted GT-identity embeddings using single-cell sampling and 3×3 ROI
  pooling around each 3D box.
- Built same-instance positive pairs across consecutive observed frames and
  different-instance same-frame negative pairs.
- Measured global and VRU object-region temporal warp similarity.
- Logged tracker-active versus query-bank state counts and scene resets.

## C. Experimental results

### Positive/negative identity similarity

| Feature | Positive pairs | Negative pairs | Positive mean | Negative mean | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Single-cell | 890 | 5249 | 0.556 | 0.328 | 0.800 |
| 3×3 ROI pooled | 890 | 5249 | 0.683 | 0.409 | 0.824 |

ROI pooling improves both the positive similarity and the bounded pairwise
ROC-AUC, but the distributions still overlap; it is not a learned identity
embedding.

### Temporal alignment

| Diagnostic | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Global BEV cosine | 0.4994 | 0.4603 | 0.2559 | 0.9775 |
| Global BEV mean absolute delta | 0.02136 | 0.02216 | 0.00483 | 0.02775 |
| Warp valid fraction | 0.9780 | 0.9816 | 0.9477 | 1.0000 |
| VRU object-region cosine | 0.6191 | 0.5888 | 0.3864 | 0.9740 |

There are 78 aligned transitions in the 80-frame, two-scene run.

### Query lifecycle

| Quantity | Value |
|---|---:|
| Max active tracker tracks | 48 |
| Max query states | 190 |
| Max query states minus active tracks | 179 |
| Query expiration implemented | No |

Existing temporal diagnostics only:

| Dataset | Mean BEV cosine | Mean query cosine | Max query states |
|---|---:|---:|---:|
| NavSim, 10 frames | 0.9243 | 0.8941 | 22 |
| nuScenes, 80 frames | 0.4994 | 0.6181 | 190 |

The old global/query cosine values remain diagnostics; the positive/negative
pair analysis above is the identity evidence used for Phase 5 gating.

## D. Technical findings

- Frozen BEV features contain useful but imperfect identity signal on this
  bounded subset: ROI AUC 0.824 is better than single-cell AUC 0.800.
- ROI pooling is preferable to a single cell because it is less sensitive to a
  small center localization error.
- The global BEV cosine is not an identity metric; object-region cosine is lower
  and more relevant.
- The query bank has a lifecycle bug: it retains historical query states after
  `TrackManager` has deleted tracks. This can cause stale identity candidates
  and makes `max_active_queries` misleading.
- The evidence justifies a conservative, gated query-assisted association
  ablation, not replacing classical motion association or claiming TrackFormer.

## E. Changes relative to previous phase

- Added experiment-only single-cell/ROI embedding extraction and pair metrics.
- Added temporal object-region diagnostics and query lifecycle logging.
- No detector or pretrained weight was changed.

## F. Limitations

- Pair construction uses same-frame negatives and consecutive same-instance
  positives; it does not establish performance across unseen scenes.
- AUC is computed on this bounded subset and is not a MOT metric.
- The BEV feature is frozen and sampled without a learned projection head.
- Query expiration is not implemented in the existing Phase 2 prototype.

## G. Recommendation

Proceed to Phase 5 with ROI-pooled query features as an optional cost term,
strict class/motion gating, and the original Kalman/Hungarian path as the
default control. Add lifecycle expiration in the experimental association
runner rather than changing the existing Phase 2 baseline.

## H. Completion status

**PASS**
