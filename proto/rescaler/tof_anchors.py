"""ToF anchor provider: quality ToF points -> model-pixel anchors.

Mirrors the VIO anchor contract (geometry.project_features): produces
(uv, depth, var) in the depth model's pixel space, so the pipeline can
concatenate ToF and VIO anchors and run the existing compute_weights / fit
path unchanged. ToF weight comes from per-point noise (var = noise^2, the same
variance-of-metric-depth scale VIO uses); confidence is the quality gate.
"""
from __future__ import annotations
import numpy as np

from .types import TofFrame, VioPose
from .preprocess import Preprocessor


def _to_hires_cam(P_tof: np.ndarray, pose_tof: VioPose, pose_hires: VioPose,
                  R_tof: np.ndarray, T_tof: np.ndarray,
                  R_hires: np.ndarray, T_hires: np.ndarray) -> np.ndarray:
    """ToF-cam -> imu(tof time) -> world -> imu(hires time) -> hires-cam.

    Extrinsics are child->imu (P_imu = P_cam @ R.T + T). With pose_tof ==
    pose_hires (synchronised / rigid instant) the world hop cancels and this
    reduces to the fixed ToF->hires extrinsic.
    """
    P_imu_t = P_tof @ R_tof.T + T_tof
    P_world = P_imu_t @ pose_tof.R_imu_to_vio.T + pose_tof.T_imu_wrt_vio
    P_imu_h = (P_world - pose_hires.T_imu_wrt_vio) @ pose_hires.R_imu_to_vio
    return (P_imu_h - T_hires) @ R_hires


def tof_anchors(tof: TofFrame, pose_tof: VioPose, pose_hires: VioPose,
                R_tof: np.ndarray, T_tof: np.ndarray,
                R_hires: np.ndarray, T_hires: np.ndarray,
                pre: Preprocessor, conf_min: int, max_points: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (uv (M,2), depth (M,), var (M,)) for in-view quality ToF points."""
    keep = ((tof.conf >= conf_min)
            & np.isfinite(tof.xyz).all(axis=1)
            & (tof.noise > 0))
    xyz, noise, conf = tof.xyz[keep], tof.noise[keep], tof.conf[keep]
    if len(xyz) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)

    if len(xyz) > max_points:
        # even stride over the sensor grid (confidence is binary 0/255)
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(int)
        xyz, noise = xyz[idx], noise[idx]

    P_cam = _to_hires_cam(xyz, pose_tof, pose_hires, R_tof, T_tof, R_hires, T_hires)
    u, v, front = pre.project(P_cam)
    depth = P_cam[:, 2]
    m = (front & (depth > 0)
         & (u >= 0) & (u < pre.dst_w) & (v >= 0) & (v < pre.dst_h))
    return np.stack([u[m], v[m]], axis=1), depth[m], noise[m] ** 2
