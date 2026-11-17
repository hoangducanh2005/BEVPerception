"""Small constant-velocity Kalman filter for ground-plane tracking."""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """Track ``[x, y, vx, vy]`` in a metric world/odometry frame."""

    def __init__(self, position_xy: np.ndarray, *, position_var: float = 1.0) -> None:
        position = np.asarray(position_xy, dtype=np.float64).reshape(2)
        self.state = np.array([position[0], position[1], 0.0, 0.0], dtype=np.float64)
        self.covariance = np.diag([position_var, position_var, 25.0, 25.0]).astype(np.float64)

    def predict(self, dt: float, *, process_accel_std: float = 4.0) -> None:
        dt = max(float(dt), 1e-3)
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        q = float(process_accel_std) ** 2
        process = q * np.array(
            [[dt**4 / 4, 0.0, dt**3 / 2, 0.0], [0.0, dt**4 / 4, 0.0, dt**3 / 2], [dt**3 / 2, 0.0, dt**2, 0.0], [0.0, dt**3 / 2, 0.0, dt**2]],
            dtype=np.float64,
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

    def update_position_velocity(
        self, position_xy: np.ndarray, velocity_xy: np.ndarray, *, position_std: float = 0.5, velocity_std: float = 1.0
    ) -> None:
        """Joint update with a measured position and a measured (detector) velocity."""

        measurement = np.r_[np.asarray(position_xy, dtype=np.float64).reshape(2), np.asarray(velocity_xy, dtype=np.float64).reshape(2)]
        noise = np.diag([position_std**2, position_std**2, velocity_std**2, velocity_std**2])
        innovation = measurement - self.state
        gain = self.covariance @ np.linalg.inv(self.covariance + noise)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(4) - gain) @ self.covariance

    def update(self, position_xy: np.ndarray, *, measurement_std: float = 0.7) -> None:
        measurement = np.asarray(position_xy, dtype=np.float64).reshape(2)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        noise = np.eye(2, dtype=np.float64) * float(measurement_std) ** 2
        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - gain @ observation) @ self.covariance
