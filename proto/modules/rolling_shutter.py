"""Rolling-shutter correction for anchor projection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..rescaler.types import VioPose
from ..rescaler.preprocess import Preprocessor


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


@dataclass
class RollingShutter:
    readout_s: float
    top_to_bottom: bool = True

    def correct(
        self,
        P_vio: np.ndarray,        # (N, 3) landmark positions, world frame
        v_row: np.ndarray,        # (N,) rows from the uncorrected projection
        pose: VioPose,
        R_cam_to_imu: np.ndarray,
        T_cam_wrt_imu: np.ndarray,
        pre: Preprocessor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reproject each landmark with the pose at its row's capture time.

        Returns (u, v, front, depth) like the uncorrected projection.
        """
        frac = np.clip(v_row, 0.0, pre.dst_h) / pre.dst_h
        if not self.top_to_bottom:
            frac = 1.0 - frac
        dt = frac * self.readout_s                                       # (N,)

        T = pose.T_imu_wrt_vio + pose.vel[None, :] * dt[:, None]         # (N, 3)
        dR = np.eye(3)[None] + _skew(pose.omega)[None] * dt[:, None, None]
        R = pose.R_imu_to_vio[None] @ dR                                 # (N, 3, 3)

        P_imu = np.einsum("nj,njk->nk", P_vio - T, R)
        P_cam = (P_imu - T_cam_wrt_imu) @ R_cam_to_imu
        u, v, front = pre.project(P_cam)
        return u, v, front, P_cam[:, 2]
