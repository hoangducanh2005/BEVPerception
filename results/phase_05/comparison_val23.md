| Method | AMOTA | AMOTP | Recall | IDS | FRAG | Ped AMOTA (IDS) | Bicycle AMOTA (IDS) | Moto AMOTA (IDS) | VRU AMOTA | VRU IDS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: BEVFusion + KF/Hungarian (default) | 0.589 | 0.684 | 0.774 | 123 | 103 | 0.752 (67) | 0.519 (0) | 0.606 (1) | 0.626 | 68 |
| A (thresholds tuned on mini_train) | 0.619 | 0.584 | 0.817 | 143 | 88 | 0.793 (80) | 0.567 (0) | 0.643 (0) | 0.668 | 80 |
| B: BEVFusion + temporal BEV + TrackFormer (trained on mini_train only) | 0.536 | 0.849 | 0.619 | 131 | 129 | 0.604 (86) | 0.332 (0) | 0.519 (0) | 0.485 | 86 |

| Method | mAP | NDS | mATE | mAVE | Ped AP | Bicycle AP | Moto AP | VRU mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: BEVFusion + KF/Hungarian (default) | 0.653 | 0.693 | 0.241 | 0.348 | 0.905 | 0.682 | 0.696 | 0.761 |
| B: BEVFusion + temporal BEV + TrackFormer (trained on mini_train only) | 0.536 | 0.612 | 0.347 | 0.415 | 0.786 | 0.300 | 0.580 | 0.556 |
