# Phase 4 — UniAD-inspired query-based 3D tracking on BEVFusion features

Status: **done**. A trainable TrackFormer was implemented, proven numerically
equivalent to UniAD's tracker, adapted to BEVFusion's fused BEV, **trained**
(BEVFusion frozen) on nuScenes-mini `mini_train`, and evaluated on `mini_val`.
Untrained smoke tests and trained results are kept separate below.

## 1. What was implemented (`src/bevdrive/trackformer/`)

| Module | Origin | Role |
|---|---|---|
| `layers.py`: `TrackDecoder`, `TrackDecoderLayer`, `CustomMSDeformableAttention` | Port of UniAD `DetectionTransformerDecoder` + mmcv `DetrTransformerDecoderLayer` + `CustomMSDeformableAttention` | 6 layers of [query self-attention → deformable cross-attention into the BEV (8 heads × 4 points around each query's reference point) → FFN], with iterative reference refinement (`with_box_refine`) |
| `query_modules.py`: `QueryInteractionModule` | Port of UniAD/MOTR QIM | Updates each surviving track's query (position + content) from its decoder embedding. Training: random track drop (0.1) and false-positive track injection (0.3). |
| `query_modules.py`: `MemoryBank` | Port of UniAD | Per-track FIFO of 4 past embeddings + temporal attention: the learned track embedding history |
| `query_modules.py`: `RuntimeTrackerBase` | Port of UniAD | Query lifecycle: a detection query with score ≥ 0.4 is born as a new id; a track with score < 0.35 sleeps and dies after 5 misses (occlusion tolerance) |
| `criterion.py`: `ClipMatcher` | Port of UniAD `losses/track_loss.py` | MOTR-style clip loss: track queries supervised by GT identity, new GT Hungarian-matched to detection queries, focal + L1 on all 6 layers |
| `model.py`: `BEVFusionTrackFormer` | New, following `UniADTrack` | Query generation, `velo_update` (ego motion + v·dt), online inference, clip training |
| `bev_adapter.py`: `BEVAdapter` | **New** | Interface between BEVFusion's fused BEV and the UniAD decoder (§2) |
| `weights.py` | New | Loads UniAD stage-1 weights into every interface-compatible module |

Kalman filtering and Hungarian association are **not** used at inference. A
track's identity *is* its query: the query persists across frames, is
updated by the QIM, attends to its own history in the memory bank, and
competes with new detection queries through decoder self-attention. That
competition is what suppresses duplicate births. Hungarian matching appears
only in the training loss, as in MOTR and UniAD.

### Query types

* **Detection queries (new objects).** One per BEVFusion proposal: the
  TransFusion top-200 with score ≥ 0.02, inside ±51.2 m.
  * Reference point = proposal centre.
  * `query_pos = MLP(sinusoidal(x, y, z))`.
  * `query_content = MLP([adapted-BEV feature sampled at the centre, class
    one-hot, score, log-size, sin/cos yaw, velocity])`.
  * UniAD's 900 learned queries are also supported (`query_mode="learned"`
    or `"hybrid"`); the port check uses them.
* **Track queries (existing objects).** Carried from t−1 after the QIM.
  Reference points are propagated with ego motion and the predicted velocity:
  `p_t = T_t⁻¹ · T_{t−1} · (p_{t−1} + v·Δt)`. As in UniAD, the z reference
  is re-predicted from `query_pos`.

## 2. BEVFusion ↔ UniAD interface gap and how it is bridged

| Aspect | BEVFusion fused BEV | UniAD TrackFormer expects | Bridge (`BEVAdapter`) |
|---|---|---|---|
| Channels | 512 (`pts_neck`) | 256 | GroupNorm → 1×1 conv → GN/ReLU → 3×3 conv → GN/ReLU → 1×1 conv (trained) |
| Grid / range | 180² @ 0.6 m, ±54 m | 200² @ 0.512 m, ±51.2 m | Exact metric resampling (`resample`, bilinear), no learned warp |
| Tensor layout | `[C, x, y]` (verified, Phase 2) | `[C, y, x]`, flattened `(h w)` | Handled by `BEVGrid.layout` |
| Positional encoding | None | BEV queries carry learned row/col embeddings | UniAD's `LearnedPositionalEncoding` (200×128 ×2) is loaded and added to the value map |
| Reference points | TransFusion heatmap peaks (grid indices) | Sigmoid-normalised xyz in pc_range [-51.2, 51.2] × [-5, 3] | Proposal centres normalised to UniAD's pc_range; same `inverse_sigmoid` convention |
| Box code | Legacy dims/yaw, gravity z (fixed in Phase 1) | `(cx, cy, log l, log w, z_bottom, log h, sin, cos, vx, vy)` | `boxes_to_code` / `code_to_boxes` |
| Temporal memory | None | BEVFormer temporal self-attention (prev_bev) + track queries + MemoryBank | Phase-2 motion-gated, ego-aligned history map concatenated with the current map at the adapter input; track queries + MemoryBank unchanged |
| Features | LiDAR + camera, trained for TransFusion | BEVFormer camera features | Not reusable as-is; the adapter is learned |

**Weight reuse.** 271 tensors load from `uniad_base_track_map.pth`: the
decoder, cls/reg branches, QIM, memory bank, reference-point head and
positional embedding. The adapter and the proposal encoders are new. UniAD's
BEVFormer encoder and image backbone are not used.

## 3. Port verification (before any training)

Test: feed **UniAD's own** BEVFormer `bev_embed` and ego poses, dumped from
the unmodified UniAD with `scripts/uniad_dump_track_io.py` in env
`uniad2.0`, into our port with UniAD's weights (`query_mode="learned"`, 900
queries + ego query). Compare with UniAD's own outputs on mini_val.

| Check (81 frames) | fp16 BEV dump | fp32 BEV dump |
|---|---:|---:|
| Output track ids identical to UniAD | 48.5 %* | **99.4 %** |
| Median per-frame worst box-centre error | 0.000 m (first scene) | **6.5 mm** |
| AMOTA (devkit) — port / UniAD official | 0.487 / 0.486 | **0.488 / 0.486** |
| Pedestrian / bicycle / motorcycle AMOTA — port / UniAD | 0.533 / 0.148 / 0.454 vs 0.532 / 0.148 / 0.454 | 0.535 / 0.148 / 0.454 |

\*The fp16 run also reset ids per scene; UniAD keeps counting, so every
scene-2 id differed by a constant offset. Fixed with
`reset_ids_per_scene=False` for the fp32 check.

Frames 0–2 agree to 1e-4 m. The remaining differences are floating-point
drift (UniAD's CUDA deformable kernel vs our `grid_sample` path) amplified
through the recurrent query state; occasionally a query crosses the 0.4
birth threshold in one implementation and not the other. **Conclusion: the
port is a faithful UniAD TrackFormer.** Results:
`results/phase_04/port_check{,_fp32}/port_check.json`.

## 4. Training (BEVFusion frozen)

| Item | Value |
|---|---|
| Training data | nuScenes-mini `mini_train`: 8 scenes, 323 keyframes, 307 clips of 3 frames. mini_val is never used for training, early stopping or model selection; the **final** epoch is used. |
| Inputs | Cached fused BEV (Phase 1, fp16) + Phase-2 gated history (2 frames) + BEVFusion proposals + GT (≥1 point, inside ±51.2 m, 10 classes, instance ids) |
| Recipe | UniAD stage-1 / MOTR: clip length 3, track queries propagated through the clip with gradients, QIM drop 0.1 / FP 0.3, ClipMatcher on all 6 decoder layers (focal ×2, L1 ×0.25, code weights `[1]*8 + [0.2, 0.2]`) |
| Optimiser | AdamW, wd 0.01. LR 2e-4 for new modules, 2e-5 for UniAD-initialised modules. 200-step warm-up, cosine decay, grad-clip 35. 12 epochs (3,684 steps). |
| Deviation from UniAD | QIM keeps a track during training if its matched prediction is within 1 m of GT (UniAD: 3D IoU > 0.5). This avoids dropping small VRU boxes early in training. |
| Cost | About 3 min/epoch, peak 2.8 GB GPU (RTX 4050), 37 min total |

Commands:

```bash
conda run -n ai_core python scripts/cp2_validate_port.py --dump artifacts/checkpoint_2/phase_04/uniad_port_check_fp32 --output-dir results/phase_04/port_check_fp32
conda run -n ai_core python scripts/cp2_train_trackformer.py --output-dir artifacts/checkpoint_2/phase_04/train_main_temporal --epochs 12
conda run -n ai_core python scripts/cp2_track_query.py --checkpoint artifacts/checkpoint_2/phase_04/train_main_temporal/final.pth --output-dir artifacts/checkpoint_2/phase_04/infer_main_temporal
conda run -n uniad2.0 python scripts/cp2_eval_nuscenes.py --task track --result-json artifacts/checkpoint_2/phase_04/infer_main_temporal/submission_track.json --output-dir results/phase_04/main_temporal_track --eval-set mini_val
```

## 5. Tests

`tests/test_trackformer.py` (6 tests): box-code round trip; deformable
attention equal to mmcv's reference implementation; `velo_update`
geometry; runtime lifecycle (birth and death); identity persistence and
end-to-end gradient flow (adapter, decoder, QIM) on a random BEV; UniAD
weight loading. These are **smoke tests with untrained or random inputs**:
they validate mechanics, not tracking quality.

## 6. Training curve (mini_train, 12 epochs)

| Epoch | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clip loss | 7.68 | 3.72 | 3.03 | 2.68 | 2.43 | 2.28 | 2.24 |
| Focal (last layer) | 0.433 | 0.156 | 0.115 | 0.092 | 0.075 | 0.063 | 0.060 |
| L1 box (last layer) | 0.863 | 0.447 | 0.369 | 0.330 | 0.307 | 0.292 | 0.288 |
| GT kept by a track query (identity carried) | 0.44 | 0.50 | 0.50 | 0.51 | 0.51 | 0.51 | 0.50 |

Log: `artifacts/checkpoint_2/phase_04/train_main_temporal/train_log.jsonl`.

## 7. Trained results — Proposed B (12-epoch final checkpoint, UniAD thresholds birth 0.4 / filter 0.35)

### 7.1 mini_val (held out) — official devkit

| Class | AMOTA | AMOTP | Recall | IDS | FRAG | Baseline A AMOTA (IDS) |
|---|---:|---:|---:|---:|---:|---:|
| Pedestrian | 0.673 | 0.731 | 0.742 | 11 | 7 | 0.856 (4) |
| Bicycle | 0.000 | 1.766 | 0.220 | 0 | 1 | 0.475 (0) |
| Motorcycle | 0.289 | 1.548 | 0.375 | 0 | 1 | 0.697 (1) |
| Car | 0.638 | 0.762 | 0.736 | 21 | 25 | 0.795 |
| **All (devkit)** | **0.530** | 0.987 | 0.619 | 32 | 35 | 0.741 (26) |

The decoder's detection output (top-300 queries per frame, devkit
DetectionEval) scores mAP 0.463, NDS 0.500, VRU mAP 0.478 (pedestrian 0.817,
bicycle 0.058, motorcycle 0.559). The BEVFusion proposals it refines score
0.736. **The learned decoder degrades the detections it was given.**

### 7.2 The same model on its training scenes (mini_train) — overfitting diagnosis

| Split | AMOTA | Recall | Pedestrian | Bicycle | Motorcycle |
|---|---:|---:|---:|---:|---:|
| B on mini_train (training scenes) | **0.851** | 0.911 | 0.796 | 0.669 | 0.909 |
| B on mini_val | 0.530 | 0.619 | 0.673 | 0.000 | 0.289 |
| A on mini_train | 0.766 | 0.791 | 0.841 | 0.415 | 0.831 |
| A on mini_val | 0.741 | 0.767 | 0.856 | 0.475 | 0.697 |

On its training scenes B beats A (0.851 vs 0.766). On held-out scenes it
loses 0.32 AMOTA, while A loses 0.03. mini_train contains only 149 bicycle
and 135 motorcycle GT boxes, from a handful of instances. mini_val's
motorcycle-parking scene (scene-0916) is out of distribution for them.
**The query tracker overfits the 8 training scenes.** This is a data-scale
limitation, not an inference bug: the mechanics are verified against UniAD
(§3), and train-scene performance is high. UniAD trained the same decoder on
about 28k keyframes; here it has 323.

Lifecycle thresholds swept on **mini_train** (never on val): birth/filter
0.4/0.35 → 0.851, 0.3/0.2 → 0.839, 0.2/0.1 → 0.813, 0.3/0.1 → 0.830. The
UniAD defaults are best, so "B tuned on train" is the same model as B
default.

### 7.3 Where B is better and worse (diagnostic counts, score ≥ 0.3, mini_val VRU)

| | Baseline A | Proposed B |
|---|---:|---:|
| Pedestrian fragmentations | 41 | **32** |
| Motorcycle fragmentations | 21 | **5** |
| Pedestrian misses — occluded | 130 | **114** |
| Pedestrian misses — visible | **46** | 133 |
| Pedestrian false positives | **168** | 294 |
| Bicycle false positives | **8** | 35 |
| Pedestrian ID switches (consecutive) | **27** | 37 |

Persistent track queries do what they are designed for: fewer fragmentations
and fewer occlusion misses, because a sleeping query survives up to 5 frames
and re-attaches. The weak point is the learned re-scoring and classification
of detection queries: more visible misses, false positives and class
confusion. That is the part that needs more training data.

Latency (mini_val, RTX 4050): temporal BEV 17 ms + tracker 50 ms per frame
(median 41 ms), peak 0.65 GB on top of BEVFusion. KF baseline: 11.5 ms (CPU).

## 8. Visualisations

* `outputs/phase_04/B_main_temporal/tracking.mp4` and per-frame
  `scene-*_fNN.jpg`: six cameras + BEV, coloured by query track id, with
  trails and velocity arrows.
* `outputs/phase_05/failures_B/*.jpg`: worst frames (red = miss, yellow =
  ID switch, orange = FP).

## 9. Limitations and next steps

1. **Training data scale is the dominant limitation.** On the 23-scene val
   set the mini-trained B reaches VRU AMOTA 0.485 vs 0.626 for A
   (Phase 5 §6). Retraining on the 61 locally available train-split scenes
   with the same recipe is prepared (`artifacts/checkpoint_2/run_scaleup.sh`)
   but **paused at the user's request**.
2. A score prior from the detector is a principled architectural fix, but it
   was *not* tried, to avoid design iteration on the evaluation split. The
   idea: initialise each detection query's class logit with
   `logit(BEVFusion score)` and learn a residual. Validate it on a held-out
   *train* scene first.
3. UniAD's IoU-based QIM selection was replaced by a 1 m distance criterion
   (documented deviation).
4. The temporal-context input is evaluated by the no-temporal ablation
   (Phase 5).
