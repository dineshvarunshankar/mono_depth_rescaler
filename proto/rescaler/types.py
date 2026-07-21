from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Feature:
    xyz_vio: np.ndarray   # (3,) 3D position in VIO world frame, metres
    cam_id: int           # 0=tracking_front, 1=tracking_down
    quality: int          # 1=MEDIUM(MSCKF), 2=HIGH(SLAM)
    depth: float          # distance from observing camera, metres
    depth_stddev: float   # depth uncertainty, metres
    id: int = 0                      # stable across frames; used by the anchor window
    p_tsf: np.ndarray | None = None  # (3, 3) position covariance, VIO world frame
    pix_loc: np.ndarray | None = None  # (2,) raw distorted pixel in the observing tracking camera


@dataclass
class VioPose:
    # P_vio = R_imu_to_vio @ P_imu + T_imu_wrt_vio
    R_imu_to_vio: np.ndarray   # (3, 3)
    T_imu_wrt_vio: np.ndarray  # (3,)
    vel: np.ndarray | None = None    # (3,) IMU velocity in VIO frame, m/s
    omega: np.ndarray | None = None  # (3,) body angular rate, rad/s

    @property
    def valid(self) -> bool:
        """A real orientation (det ~ 1). qVIO zeroes R when tracking has failed;
        such a pose cannot place anchors, so callers skip the frame."""
        return bool(np.isfinite(self.R_imu_to_vio).all()
                    and abs(np.linalg.det(self.R_imu_to_vio) - 1.0) < 0.02)


@dataclass
class TofFrame:
    t_ns: int
    xyz: np.ndarray     # (N, 3) points in the ToF camera frame, metres
    noise: np.ndarray   # (N,)  per-point depth noise, metres (-> var = noise^2)
    conf: np.ndarray    # (N,)  per-point confidence, 0..255 (quality gate)


@dataclass
class Frame:
    t_ns: int
    image: np.ndarray       # (H, W, 3) uint8 RGB, raw fisheye at capture resolution
    pose: VioPose
    features: list[Feature]
    tof: "TofFrame | None" = None   # populated only when anchors.use_tof


@dataclass
class RescaleResult:
    depth: np.ndarray            # (H, W) metric depth, metres, at model output resolution
    method: str                  # rescale curve: disp_metric = f(disp_rel)
    params: np.ndarray | None    # fitted curve parameters; None for splines
    n_anchors: int
    inlier_ratio: float
    held: bool = False           # reused an earlier fit (scale up to calib_age_ns old)
    calib_age_ns: int = 0        # ns since the fit in use was computed; 0 if fresh
    anchors: tuple | None = None # (disp_rel, depth, weights) the fit consumed
