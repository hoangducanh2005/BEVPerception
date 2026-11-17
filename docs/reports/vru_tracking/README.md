# VRU Detection & Multi-View Tracking Evaluation

This directory records the controlled, inference-only evaluation requested for
BEVPerception. No neural network training or checkpoint modification is
allowed in these experiments. Each phase gets its own report and artifact
directory; prior results must not be overwritten.

## Phase status

| Phase | Status | Main result | Report |
|---|---|---|---|
| 0. Audit | **PASS** | Code/artifact audit found ID-switch and query-lifecycle protocol issues; data mount was restored read-only and fresh 10/80-frame baselines were reproduced. | [phase_00_audit.md](phase_00_audit.md) |
| 1. Detection | **PASS** | Cached raw BEVFusion outputs; threshold 0.05 raises pedestrian recall 0.198→0.358 and F1 0.326→0.464, but 424/924 VRU misses have no same-class raw candidate within 2m. | [phase_01_detection.md](phase_01_detection.md) |
| 2. Tracking oracle | **PASS** | Corrected protocol reports 133 consecutive ID switches and 83 gap fragmentations; GT-box oracle has 36 ID switches with perfect detection continuity. | [phase_02_tracking_oracle.md](phase_02_tracking_oracle.md) |
| 3. Query analysis | **PASS** | ROI-pooled BEV embeddings reach AUC 0.824 versus single-cell 0.800, but query lifecycle leaks historical states (190 queries vs 48 active tracks). | [phase_03_query_analysis.md](phase_03_query_analysis.md) |
| 4. Cross-camera | **PARTIAL** | 18 geometry-based transition candidates found; 12/18 keep the same predicted ID, but no event has been fully manually validated, so no official handoff rate is claimed. | [phase_04_cross_camera.md](phase_04_cross_camera.md) |
| 5. Association | **BLOCKED** | Must wait for Phase 3 evidence and a verified baseline protocol. | [phase_05_association.md](phase_05_association.md) |

The dependent phase reports are placeholders documenting the exact blocker;
they do not claim that the experiments were executed.

## Existing baseline evidence

The previous local bounded run is retained under the ignored directory:

```text
artifacts/phase_runs/navsim_shard5/
artifacts/phase_runs/nuscenes_shard1/
```

It contains 10 NavSim frames and 80 nuScenes frames from `scene-0003` and
`scene-0004`. Its metrics are custom bounded diagnostics, not official
nuScenes benchmark metrics. The current audit found that its `232` ID-switch
count includes identity changes after gaps and that Phase 2 query counts are
historical accumulation, not active-track counts.

## Artifact layout

```text
artifacts/experiments/vru_tracking/
├── phase_01/
├── phase_02/
├── phase_03/
├── phase_04/
└── phase_05/
```

Once the data mount is restored, each phase must be rerun with explicit
commands, seeds/configuration, raw detection caches where applicable, and
phase-local metrics.
