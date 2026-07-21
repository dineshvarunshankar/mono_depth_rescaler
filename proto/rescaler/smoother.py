"""Temporal smoothers for fitted curve parameters (splines pass through)."""
from __future__ import annotations
import numpy as np


class EMASmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x: np.ndarray | None = None

    def update(self, params: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        if self._x is None or self._x.shape != p.shape:
            self._x = p.copy()
        else:
            self._x = (1.0 - self.alpha) * self._x + self.alpha * p
        return self._x

    def reset(self) -> None:
        self._x = None


class KalmanSmoother:
    """Full-covariance random-walk Kalman filter on the parameter vector.

    R is each fit's own parameter covariance from WLS; measurement_noise is the
    fallback R for fits without one.
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.q = process_noise
        self.r = measurement_noise
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None

    def update(self, params: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(params, dtype=float)
        n = len(z)
        R = np.asarray(cov, dtype=float) if cov is not None else self.r * np.eye(n)
        if self._x is None or self._x.shape != z.shape:
            self._x = z.copy()
            self._P = R.copy()
            return self._x
        P_pred = self._P + self.q * np.eye(n)
        K = P_pred @ np.linalg.inv(P_pred + R)   # gain: full matrix, correlations included
        self._x = self._x + K @ (z - self._x)
        P_new = (np.eye(n) - K) @ P_pred
        self._P = 0.5 * (P_new + P_new.T)        # keep symmetric against round-off
        return self._x

    def reset(self) -> None:
        self._x = self._P = None
