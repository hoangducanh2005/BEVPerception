"""Model adapters for Phase 1."""

from .bev_detector import BEVFusionDetector
from .lidar_baseline import LidarOnlyDetector

__all__ = ["BEVFusionDetector", "LidarOnlyDetector"]
