# Phase 2 — Temporal BEV alignment

Status: **done**. Ego-motion-compensated BEV history with a configurable
length and motion-gated aggregation, validated on real fused features.
Per-frame detection stays available and unchanged (mode `none`).

## 1. What was implemented (`src/bevdrive/temporal/bev_alignment.py`)

* `BEVGrid`: explicit metric grid with a **tensor-layout flag**.
  BEVFusion = `layout="xy"` (`feature[..., ix, iy]`, 180², 0.6 m, ±54 m).
  UniAD = `layout="yx"` (200², 0.512 m, ±51.2 m). Cell centres are
  `min + (i + 0.5)·res`, which is exactly what
  `grid_sample(align_corners=False)` addresses.
* `resample(feature, src_grid, dst_grid, dst_to_src)`: one metric resampler
  used for both layout/grid conversion (BEVFusion → UniAD in Phase 4) and
  rigid warping.
* `warp_to_current(prev, T_prev, T_cur)`: for each current cell centre
  (z = 0), `p_prev = T_prev⁻¹ · T_cur · p_cur`, with `T = lidar2global` built
  from nuScenes `calibrated_sensor` + `ego_pose` (Phase 1 records). Bilinear
  sampling; also returns a validity mask.
* `TemporalBEVBuffer(history=H)`: per-scene FIFO of (BEV, pose, timestamp).
  Resets on scene change, and drops entries after a temporal gap.
* `aggregate(current, history, mode)`:
  * `none`: the current map only, i.e. per-frame detection;
  * `mean`: uniform average (the naive baseline; smears movers);
  * `decay`: exponential age weights `0.5^k`;
  * **`gated`** (default): `w_k = 0.5^k · valid · σ(10·(cos(F_t, F̂_{t−k}) − 0.6))`
    per cell. Static structure, which agrees after warping, is integrated.
    Cells whose content changed (moving VRUs, disocclusions) get weight ≈ 0,
    so they are not smeared.
  * `include_current=False` returns a pure history-context map. Phase 4 uses
    it as the second input of the query tracker, so the current evidence is
    never diluted.

## 2. Data flow

```text
fused BEV_t (Phase-1 cache) ─┬───────────────────────────────────────────► per-frame detection (unchanged)
                             │
buffer {BEV_{t-1}, BEV_{t-2}} ─► warp to LiDAR_t (poses) ─► gate vs BEV_t ─► temporal context C_t
                                                                           └► [BEV_t ; C_t] → Phase-4 tracker
```

## 3. Files

Created `src/bevdrive/temporal/{__init__,bev_alignment}.py`,
`scripts/cp2_temporal_bev.py` and `tests/test_bev_alignment.py`. Modified
`src/bevdrive/perception/bevfusion_runner.py` (`decode_bev` works on cached
maps; `heatmap()`).

## 4. Commands

```bash
CUDA_VISIBLE_DEVICES="" conda run -n ai_core python -m pytest -q tests/test_bev_alignment.py   # 5 passed
conda run -n ai_core python scripts/cp2_temporal_bev.py            # mini_val, history 2
for m in none mean decay gated; do
  conda run -n ai_core python scripts/cp2_eval_nuscenes.py --task det \
    --result-json artifacts/checkpoint_2/phase_02/submission_det_$m.json --output-dir results/phase_02/det_$m --eval-set mini_val
done
```

## 5. Validation of the coordinate transforms

### 5.1 Unit tests (synthetic, exact)

Identity resampling is exact. `xy` → `yx` resampling is exactly a transpose.
A blob at a known world point lands in the expected cell after a +3 m / 90°
ego motion. The gate suppresses a changed cell (gate < 0.01) and leaves
static cells (gate > 0.97).

### 5.2 Which tensor axis is x? (frozen TransFusion heatmap at GT centres, 3,537 GT, mini_val)

| Layout hypothesis | Mean heatmap of the GT class at the GT centre |
|---|---:|
| **`xy` (dim −2 = x, dim −1 = y)** | **0.631** |
| `yx` | 0.016 |
| Checkpoint-1 convention (rows +y → −y, columns x) | 0.007 |
| Random locations (max over classes) | 0.012 |

The BEVFusion BEV is `[C, x, y]`. Checkpoint 1's `temporal_query.py` sampled
and warped under a wrong convention: its numbers are at chance level.

### 5.3 Pose chain and feature alignment on real data (79 consecutive pairs)

| Check | Result |
|---|---|
| Static GT (|v| < 0.2 m/s), t−1 centre mapped through `T_t⁻¹·T_{t−1}` vs t centre | mean **2.9 cm**, p95 8.7 cm (n = 2,338) |
| Feature cosine at static objects, BEV_t vs raw BEV_{t−1} | 0.407 |
| Feature cosine at static objects, BEV_t vs **aligned** BEV_{t−1} | **0.872** |
| Whole-map cosine (valid region): raw / aligned (ours) / aligned (checkpoint-1 fn) | 0.648 / 0.603 / 0.450 |
| Same, on the 50 % highest-norm cells | 0.713 / 0.606 / 0.473 |

Whole-map cosine is **higher without alignment**. A large part of the fused
BEV is *sensor-centric*: LiDAR ring density, camera-frustum and
out-of-coverage patterns (the large hatched regions in the PCA panels) move
with the ego vehicle. Warping misaligns those patterns while it aligns the
world. Whole-map temporal cosine is therefore **not** a valid alignment
metric; this also applies to checkpoint 1's "0.4994". The object-level test
(0.41 → 0.87) and the heatmap overlays (§6) are the meaningful checks.

## 6. Single-frame vs temporal BEV

### 6.1 Frozen detector on aggregated BEV (no retraining; history 2; devkit, mini_val)

| Aggregation | mAP | NDS | mAVE | VRU mAP | Ped AP | Bicycle AP | Moto AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` (per-frame; identical to the live Phase-1 run) | 0.587 | 0.589 | 0.429 | 0.743 | 0.926 | 0.575 | 0.728 |
| `mean` | 0.469 | 0.513 | 0.487 | 0.540 | 0.728 | 0.228 | 0.663 |
| `decay` | 0.567 | 0.571 | 0.477 | 0.713 | 0.907 | 0.527 | 0.705 |
| **`gated`** | **0.588** | **0.591** | **0.426** | **0.746** | **0.928** | **0.597** | 0.714 |

### 6.2 Motion smearing and identity (VRU + others, mini_val)

| Aggregation | Ghost at vacated cell† (↓) | Same-instance vs different-instance ROC-AUC (t vs t−1, ≤10 m negatives) |
|---|---:|---:|
| `none` | 0.000 | 0.826 |
| `mean` | +0.399 | 0.940 |
| `decay` | +0.336 | 0.912 |
| **`gated`** | **+0.051** | 0.875 |

† For 100 objects that moved > 1.5 m between frames: cos(fused feature at
the cell the object left, object's feature now) minus the same quantity for
the single-frame map. Positive means a trail of the object remains (smearing).

Interpretation:

* Naive averaging **smears movers**. Ghost +0.40, and the frozen head loses
  0.20 VRU mAP; bicycle AP falls 0.58 → 0.23 and its velocity error doubles.
* Motion gating removes about 87 % of the ghosting (+0.05) and matches the
  single-frame detector (VRU mAP 0.746 vs 0.743, within noise).
* Without retraining the head, temporal aggregation is **not** a detection
  gain. Its value is as context for a learned temporal model (Phase 4),
  where the history enters as a separate input rather than being mixed into
  the current evidence.
* The identity ROC-AUC rises with aggregation because the fused map at t
  already contains t−1, which makes consecutive embeddings trivially closer.
  It is reported for completeness and is **not** evidence of better
  identity features.

## 7. Visualisations

`outputs/phase_02/aligned_bev_<scene>_fNN.jpg`, chosen as the frames with
the largest ego motion per scene. Ten panels:

* PCA-RGB of the current, raw previous, aligned previous (ours) and aligned
  previous (checkpoint-1 function) maps;
* the gated temporal BEV;
* the gate map w(t−1);
* |current − aligned|;
* three heatmap overlays: current class-max heatmap in red, previous in green
  (raw / aligned ours / checkpoint-1 warp). Yellow means they coincide.

GT boxes are overlaid (red = VRU). With our warp, static objects turn
yellow; raw and checkpoint-1 warps leave split red/green pairs.

## 8. Limitations / next steps

* The warp is 2D (z = 0 plane, yaw/translation from the full SE(3) pose).
  Pitch and roll are negligible on nuScenes roads, but not on ramps.
* The gate threshold (τ = 0.6, sharpness 10) and history length 2 were set
  by hand. History > 2 was not explored.
* Moving objects are excluded from aggregation, not motion-compensated.
  Warping each object by its predicted velocity (flow-guided aggregation) is
  the natural extension; the query tracker does this per object instead
  (Phase 4, `velo_update`).
