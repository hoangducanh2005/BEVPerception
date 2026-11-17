# Phase 5 — Evaluation and visualisation: Baseline A vs Proposed B

Status: **done** for nuScenes-mini. The scale-up to the locally available
trainval scenes is in §6.

## 1. What is compared

| ID | Pipeline | Learned components trained here |
|---|---|---|
| **A** | 6 cams + LiDAR → BEVFusion (frozen) → Kalman + class-aware two-pass Hungarian (Phase 3) | none |
| **B** | 6 cams + LiDAR → BEVFusion (frozen) → temporal BEV (Phase 2, gated, history 2) → BEVFusion-TrackFormer (Phase 4: detection queries from BEVFusion proposals, persistent track queries, 6-layer deformable decoder, QIM, MemoryBank, query lifecycle) | BEV adapter + proposal encoders (new); decoder, QIM, MemoryBank fine-tuned from UniAD stage 1 |
| B-noTemp | B with the temporal-context input zeroed (separately trained) | same |
| Ref. | UniAD-B stage-1 (camera-only BEVFormer + its own TrackFormer), official checkpoint | — (run in `~/GIT/UniAD/research_outputs`) |

### Fairness protocol

* Same split: nuScenes-mini `mini_val` (scene-0103, scene-0916, 81
  keyframes). Neither scene is in BEVFusion's training split.
* A and B consume the **same** BEVFusion run. A uses its boxes; B uses the
  same boxes as proposals plus the fused BEV.
* Same evaluator: nuScenes devkit TrackingEval `tracking_nips_2019` (AMOTA,
  AMOTP, IDS, FRAG, recall) and DetectionEval `detection_cvpr_2019`, run
  through `scripts/cp2_eval_nuscenes.py`.
* No tuning on mini_val. A's defaults were fixed a priori; B uses UniAD's
  lifecycle thresholds. In the "tuned" rows, thresholds were chosen on
  **mini_train** for *both* methods. B was trained only on mini_train
  scenes, and the final epoch was used.

## 2. Files and commands

New: `scripts/cp2_track_query.py` (B inference), `scripts/cp2_online_pipeline.py`
(live end-to-end A/B, no caches), `scripts/cp2_render_tracks.py`,
`scripts/cp2_failure_analysis.py`, `scripts/cp2_compare.py`,
`results/phase_05/methods.json`.

```bash
# B on mini_val (cached BEV) and evaluation
conda run -n ai_core python scripts/cp2_track_query.py --checkpoint artifacts/checkpoint_2/phase_04/train_main_temporal/final.pth \
    --output-dir artifacts/checkpoint_2/phase_04/infer_main_temporal
conda run -n uniad2.0 python scripts/cp2_eval_nuscenes.py --task track --result-json artifacts/checkpoint_2/phase_04/infer_main_temporal/submission_track.json \
    --output-dir results/phase_04/main_temporal_track --eval-set mini_val
# live end-to-end runs (latency / memory), both trackers
conda run -n ai_core python scripts/cp2_online_pipeline.py --tracker query --output-dir artifacts/checkpoint_2/phase_05/online_B --render outputs/phase_05/online_B
conda run -n ai_core python scripts/cp2_online_pipeline.py --tracker kf    --output-dir artifacts/checkpoint_2/phase_05/online_A
# failure analysis, comparison tables
conda run -n ai_core python scripts/cp2_failure_analysis.py --tracks <tracks.pkl> --out-json results/phase_05/failures_B.json --vis-dir outputs/phase_05/failures_B
python3 scripts/cp2_compare.py --out results/phase_05/comparison_mini_val
# threshold grids on mini_train (never on val)
artifacts/checkpoint_2/tune_A_on_train.sh ; artifacts/checkpoint_2/tune_B_on_train.sh
```

## 3. Results on mini_val (official devkit)

### 3.1 Tracking

| Method | AMOTA ↑ | AMOTP ↓ | Recall | IDS ↓ | FRAG ↓ | Ped AMOTA (IDS) | Bicycle AMOTA (IDS) | Moto AMOTA (IDS) | **VRU AMOTA** | VRU IDS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** BEVFusion + KF/Hungarian | **0.741** | **0.626** | **0.767** | **26** | **26** | **0.856 (4)** | **0.475 (0)** | **0.697 (1)** | **0.676** | **5** |
| A, thresholds tuned on mini_train | 0.782 | 0.525 | 0.805 | 28 | 30 | 0.862 (6) | 0.550 (0) | 0.767 (2) | 0.726 | 8 |
| **B** BEVFusion + temporal BEV + TrackFormer | 0.530 | 0.987 | 0.619 | 32 | 35 | 0.673 (11) | 0.000 (0) | 0.289 (0) | 0.321 | 11 |
| B, thresholds tuned on mini_train (= default) | 0.530 | 0.987 | 0.619 | 32 | 35 | 0.673 (11) | 0.000 (0) | 0.289 (0) | 0.321 | 11 |
| B-noTemp (ablation) | 0.557 | 0.959 | 0.630 | 31 | 27 | 0.709 (17) | 0.000 (0) | 0.294 (0) | 0.334 | 17 |
| Ref. UniAD-B stage-1 (camera-only) | 0.486 | 1.088 | 0.653 | 38 | 22 | 0.532 (28) | 0.148 (0) | 0.454 (5) | 0.378 | 33 |

The live online runs, which compute the BEV in fp32 without caches,
reproduce the cache-based numbers: A 0.741 (identical), B 0.535 vs 0.530.

### 3.2 Detection (devkit DetectionEval; A = BEVFusion's boxes, B = decoder's top-300 query boxes)

| Method | mAP | NDS | mATE | mAVE | Ped AP | Bicycle AP | Moto AP | VRU mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (BEVFusion) | **0.587** | **0.589** | **0.401** | 0.429 | **0.926** | **0.575** | **0.728** | **0.743** |
| B (TrackFormer decoder) | 0.463 | 0.500 | 0.474 | **0.409** | 0.817 | 0.058 | 0.559 | 0.478 |
| B-noTemp | 0.460 | 0.501 | 0.464 | 0.444 | 0.842 | 0.009 | 0.482 | 0.444 |
| Ref. UniAD-B stage-1 | 0.372 | 0.396 | 0.683 | 0.492 | 0.565 | 0.259 | 0.447 | 0.424 |

### 3.3 Performance (RTX 4050 Laptop 6 GB, fp32, batch 1, live run over 81 frames)

| Stage | A | B |
|---|---:|---:|
| Disk I/O + pre-processing (6 images + 10 sweeps) | 185 ms | 206 ms |
| BEVFusion forward | 425 ms | 424 ms |
| Temporal BEV (warp 2 maps + gate) | — (not used) | 12 ms |
| Tracker | 4.6 ms (KF, CPU) | 31 ms (decoder + QIM + MemoryBank, GPU) |
| **Total / FPS** | **≈615 ms / 1.63 FPS** | **673 ms / 1.49 FPS** |
| Model-only FPS (no I/O) | 2.33 | 2.14 |
| Peak GPU memory (allocated / reserved) | 2.55 / 3.15 GB | 2.58 / 3.18 GB |
| Training cost | none | 32 min (12 epochs on mini_train), 2.8 GB |

For reference, UniAD-B stage 1 runs at about 1.5 s/frame (tools/test.py)
with 4.0 GB peak on the same GPU.

### 3.4 Generalisation: training scenes vs held-out scenes

| Method | mini_train AMOTA (B was trained on these) | mini_val AMOTA | Gap |
|---|---:|---:|---:|
| A | 0.766 | 0.741 | −0.03 |
| B | **0.851** | 0.530 | **−0.32** |

B is the best tracker on the scenes it was trained on. On unseen scenes it
falls far behind A, mostly through VRU detection quality: bicycle AP 0.058,
motorcycle AMOTA 0.289. mini_train has only 149 bicycle and 135 motorcycle
GT boxes. This is overfitting from data scarcity: 323 training keyframes vs
about 28k for UniAD.

### 3.5 Failure analysis (diagnostic counts, score ≥ 0.3, 2 m, VRU GT with ≥1 point within 40 m)

| Pedestrians (1,060 GT) | A | B |
|---|---:|---:|
| TP | **884** | 813 |
| ID switch (consecutive frames) | **27** | 37 |
| Re-ID failure after a gap | **9** | 19 |
| Fragmentation | 41 | **32** |
| Miss — occluded (vis ≤ 40 % or < 5 pts) | 130 | **114** |
| Miss — visible | **46** | 133 |
| False positives | **168** | 294 |

| Motorcycles (212 GT) / bicycles (36 GT) | A | B |
|---|---:|---:|
| Motorcycle fragmentation | 21 | **5** |
| Motorcycle miss (occluded / visible) | **70 / 19** | 86 / 47 |
| Bicycle TP / FP | **18 / 8** | 8 / 35 |

Observed in the renders:

* B's persistent queries reduce fragmentation and occlusion misses, because
  sleeping queries survive 5 frames.
* B's learned re-classification creates duplicates with the wrong class,
  e.g. a pedestrian also emitted as `mot #162` (scene-0103 f30, CAM_FRONT_RIGHT).
* B inflates motorcycle boxes in the scene-0916 motorcycle park.
* A's ID switches concentrate in the dense pedestrian group at the
  scene-0916 bus stop. There, purely geometric association swaps neighbours
  when detections jitter.

## 4. Visual evidence

| What | Where |
|---|---|
| Six-camera + BEV tracking video, A | `outputs/phase_03/kf_default/tracking.mp4` (81 frames, 2 fps) |
| Six-camera + BEV tracking video, B (cached BEV) | `outputs/phase_04/B_main_temporal/tracking.mp4` |
| Six-camera + BEV tracking video, B (live end-to-end run) | `outputs/phase_05/online_B/tracking.mp4` |
| Per-frame images (ids consistent across cameras/frames, 12-frame trails, 1 s velocity arrows) | `outputs/phase_03/kf_default/`, `outputs/phase_04/B_main_temporal/` |
| Failure cases (red = miss, yellow = ID switch, orange = FP) | `outputs/phase_05/failures_A/`, `outputs/phase_05/failures_B/` |
| Detection (six cameras + BEV with GT) | `outputs/phase_01/mini_all/` |
| Temporal alignment panels | `outputs/phase_02/` |

## 5. Conclusions on nuScenes-mini

1. **Baseline A is the better VRU tracker on held-out data**: AMOTA 0.741 vs
   0.530, VRU AMOTA 0.676 vs 0.321. Its strength is BEVFusion's detection
   quality (VRU mAP 0.743), which a Kalman filter preserves.
2. **B works and learned to track**: 0.851 AMOTA on its training scenes,
   fewer fragmentations, and persistence through occlusion. But trained on
   323 keyframes it does not generalise. Its decoder degrades the proposals
   (VRU mAP 0.743 → 0.478).
3. The temporal-BEV input gives no measurable benefit in this regime: B vs
   B-noTemp differ within noise. Phase 2 showed it does not hurt
   per-frame detection when gated.
4. Both BEVFusion-based pipelines beat the camera-only UniAD reference on
   mini_val. That gap comes from the detector, not the tracker.
5. The required next step is more training data for B (§6).

## 6. Scale-up to the locally available nuScenes trainval scenes

### 6.1 What ran and what did not

* **Ran.** BEVFusion inference with fused-BEV caching on every trainval scene
  whose keyframes and 9 sweeps are on the data disk, excluding the 10 mini
  scenes: **84 scenes, 3,337 keyframes**. Of these, **61 are train-split and
  23 val-split** (`artifacts/checkpoint_2/phase_01/trainval_partial/scenes.json`).
  Runtime: 459 ms/frame model, 1.51 FPS end-to-end, 2.3 GB peak; cache
  about 110 GB.
* **Ran.** Inference and official evaluation of A, train-tuned A, and the
  **existing mini-trained B** on the 23 held-out val scenes (914 keyframes,
  "val-23"). None of these scenes is in BEVFusion's training split or in B's
  training data. This test set is 11× larger than mini_val.
* **Not run (paused at the user's request).** Retraining B on the 61
  train-split scenes plus mini_train, with the same recipe and 4 epochs fixed
  a priori. Everything is prepared in `artifacts/checkpoint_2/run_scaleup.sh`
  (training → inference on mini_val and val-23 → evaluation). No B-large
  numbers exist.
* **Engineering note.** The devkit's TrackingEval on full `v1.0-trainval`
  metadata was OOM-killed (11.7 GB RSS on a 15 GB machine). Evaluation now
  uses a verbatim metadata subset for the 23 scenes
  (`scripts/cp2_make_slim_nuscenes.py` → `v1.0-val23-trainval`; maps
  symlinked). The evaluation code is unchanged.

```bash
python scripts/cp2_detect.py --data-root "/media/anhhd/New Volume/nuscenes" --version v1.0-trainval --scenes train val \
  --complete-only --exclude-scenes <10 mini scenes> --output-dir artifacts/checkpoint_2/phase_01/trainval_partial \
  --cache-bev artifacts/checkpoint_2/bev_cache/v1.0-trainval
python scripts/cp2_make_slim_nuscenes.py --scenes <23 val scenes> --out-root artifacts/checkpoint_2/nuscenes_slim
python scripts/cp2_track_baseline.py --records .../trainval_partial/records.pkl --detections .../trainval_partial/detections.pkl --scenes <val-23> --output-dir artifacts/checkpoint_2/phase_05/A_val23
python scripts/cp2_track_query.py --checkpoint .../train_main_temporal/final.pth --records ... --detections ... --cache artifacts/checkpoint_2/bev_cache/v1.0-trainval --scenes <val-23> --output-dir artifacts/checkpoint_2/phase_05/B_mini_val23
conda run -n uniad2.0 python scripts/cp2_eval_nuscenes.py --task track --result-json <submission> --output-dir <dir> \
  --version v1.0-val23-trainval --data-root artifacts/checkpoint_2/nuscenes_slim --eval-set val
```

### 6.2 Results on val-23 (23 val-split scenes, 914 keyframes; 5,774 pedestrian, 558 bicycle, 110 motorcycle GT)

| Method | AMOTA | AMOTP | Recall | IDS | FRAG | Ped AMOTA (IDS) | Bicycle AMOTA (IDS) | Moto AMOTA (IDS) | **VRU AMOTA** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** BEVFusion + KF/Hungarian | **0.589** | 0.684 | 0.774 | **123** | **103** | 0.752 (67) | 0.519 (0) | 0.606 (1) | 0.626 |
| A, thresholds tuned on mini_train | **0.619** | **0.584** | **0.817** | 143 | 88 | **0.793** (80) | **0.567** (0) | **0.643** (0) | **0.668** |
| **B** TrackFormer (trained on mini_train only) | 0.536 | 0.849 | 0.619 | 131 | 129 | 0.604 (86) | 0.332 (0) | 0.519 (0) | 0.485 |

The devkit's all-class AMOTA includes trailer (26 GT), where A scores 0.0
and B 0.347. On cars the two are close: A 0.787 vs B 0.775, with fewer car
ID switches for B (39 vs 54).

| Detection (val-23) | mAP | NDS | mATE | mAVE | Ped AP | Bicycle AP | Moto AP | VRU mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BEVFusion (A's boxes) | **0.653** | **0.693** | **0.241** | **0.348** | **0.905** | **0.682** | **0.696** | **0.761** |
| B decoder (top-300 queries) | 0.536 | 0.612 | 0.347 | 0.415 | 0.786 | 0.300 | 0.580 | 0.556 |

BEVFusion's val-23 mAP/NDS (0.653 / 0.693) is in line with the published
BEVFusion full-val results. This is an independent sanity check that the
fixed inputs and the box conversion are right.

Diagnostic VRU counts (score ≥ 0.3, val-23), A vs B:

| | A | B |
|---|---:|---:|
| Ped TP | 4,013 | 3,885 |
| Ped ID switches (consecutive) | **135** | 331 |
| Ped fragmentations | 268 | **211** |
| Ped false positives | **631** | 1,540 |
| Bicycle misses (occluded / visible) | 175 / **92** | **150** / 179 |
| Bicycle fragmentations | 28 | **15** |

Files: `results/phase_05/comparison_val23.{md,json}`,
`results/phase_05/failures_{A_val23,B_mini_val23}.json`,
`outputs/phase_05/failures_{A_val23,B_mini_val23}/`.

### 6.3 What changes with the larger test set

* The ranking is unchanged: **A > B** for VRU tracking on held-out data.
  VRU AMOTA is 0.626 vs 0.485, or 0.668 with train-tuned thresholds.
* The gap is much smaller than on mini_val: overall AMOTA 0.589 vs 0.536
  (mini_val: 0.741 vs 0.530). Bicycle AMOTA is 0.332, not 0.000. mini_val's
  2 scenes (41 bicycle GT) exaggerated B's failure.
* B's qualitative profile is stable across both test sets. It has fewer
  fragmentations and occlusion misses thanks to persistent queries, and
  roughly 2.4× more false positives plus more ID switches in dense
  pedestrian groups, from decoder re-scoring and re-classification learned
  on 8 scenes.
* Whether B closes the gap with 61 more training scenes is **open**. That
  training was prepared but paused (§6.1).
