# BEVDrive architecture

The system is planned as a research stack with calibrated camera and LiDAR
inputs. Phase 1 shares camera and LiDAR features in BEVFusion for unified
vehicle/VRU 3D detection. The existing YOLO26m model is an independent 2D
camera baseline and is not on the main inference path. A LiDAR-only 3D detector
is the comparison baseline.

Later phases consume a versioned 3D detection contract for online tracking,
causal motion/behavior prediction, and simulation-only occupancy/risk/planning.
Those phases are not being implemented in this environment preparation step.

Data conventions from `objectives.md`: timestamps in seconds (preserve source
microseconds in dataset records), translations in meters, velocities in m/s,
angles in radians, and explicit coordinate frames and sensor transforms.
