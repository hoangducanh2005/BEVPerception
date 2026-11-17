| Method | AMOTA | AMOTP | Recall | IDS | FRAG | Ped AMOTA (IDS) | Bicycle AMOTA (IDS) | Moto AMOTA (IDS) | VRU AMOTA | VRU IDS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: BEVFusion + KF/Hungarian (default) | 0.741 | 0.626 | 0.767 | 26 | 26 | 0.856 (4) | 0.475 (0) | 0.697 (1) | 0.676 | 5 |
| A (thresholds tuned on mini_train) | 0.782 | 0.525 | 0.805 | 28 | 30 | 0.862 (6) | 0.550 (0) | 0.767 (2) | 0.726 | 8 |
| B: BEVFusion + temporal BEV + TrackFormer (UniAD thresholds) | 0.530 | 0.987 | 0.619 | 32 | 35 | 0.673 (11) | 0.000 (0) | 0.289 (0) | 0.321 | 11 |
| B (thresholds tuned on mini_train) | 0.530 | 0.987 | 0.619 | 32 | 35 | 0.673 (11) | 0.000 (0) | 0.289 (0) | 0.321 | 11 |
| B without temporal BEV (ablation) | 0.557 | 0.959 | 0.630 | 31 | 27 | 0.709 (17) | 0.000 (0) | 0.294 (0) | 0.334 | 17 |
| Reference: UniAD-B stage-1 (camera-only, own detector) | 0.486 | 1.088 | 0.653 | 38 | 22 | 0.532 (28) | 0.148 (0) | 0.454 (5) | 0.378 | 33 |
| A (live online run) | 0.741 | 0.627 | 0.767 | 26 | 26 | 0.856 (4) | 0.475 (0) | 0.697 (1) | 0.676 | 5 |
| B (live online run) | 0.535 | 0.979 | 0.622 | 33 | 34 | 0.675 (11) | 0.000 (0) | 0.289 (0) | 0.321 | 11 |

| Method | mAP | NDS | mATE | mAVE | Ped AP | Bicycle AP | Moto AP | VRU mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: BEVFusion + KF/Hungarian (default) | 0.587 | 0.589 | 0.401 | 0.429 | 0.926 | 0.575 | 0.728 | 0.743 |
| B: BEVFusion + temporal BEV + TrackFormer (UniAD thresholds) | 0.463 | 0.500 | 0.474 | 0.409 | 0.817 | 0.058 | 0.559 | 0.478 |
| B without temporal BEV (ablation) | 0.460 | 0.501 | 0.464 | 0.444 | 0.842 | 0.009 | 0.482 | 0.444 |
| Reference: UniAD-B stage-1 (camera-only, own detector) | 0.372 | 0.396 | 0.683 | 0.492 | 0.565 | 0.259 | 0.447 | 0.424 |
