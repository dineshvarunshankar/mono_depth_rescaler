"""Project VIO landmarks into the depth model's image.

Row-vector convention (matches vio_data_t):
    P_imu = (P_vio - T_imu_wrt_vio) @ R_imu_to_vio
    P_cam = (P_imu - T_cam_wrt_imu) @ R_cam_to_imu
    (u, v) = preprocessor.project(P_cam)
"""
from __future__ import annotations
import cv2
import numpy as np
from .types import Feature, VioPose
from .preprocess import Preprocessor

WEIGHTING_MODES = ("none", "stddev", "covariance")


def project_features(
    features: list[Feature],
    pose: VioPose,
    R_cam_to_imu: np.ndarray,   # (3, 3)  from extrinsics R_child_to_parent
    T_cam_wrt_imu: np.ndarray,  # (3,)    from extrinsics T_child_wrt_parent
    pre: Preprocessor,
    min_quality: int = 1,
    weighting: str = "none",
    rolling_shutter=None,       # modules.rolling_shutter.RollingShutter | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project in-view features and estimate per-anchor depth variance.

    Returns:
        uv:    (N, 2) pixel coordinates in the model's image
        depth: (N,)  metric Z in camera frame, metres
        var:   (N,)  variance of that depth (ones when weighting="none");
                     "stddev" uses qVIO's scalar depth_error_stddev,
                     "covariance" projects the full 3x3 p_tsf onto the viewing
                     axis: sigma^2 = r^T P r with r the camera z-axis in world
        idx:   (N,)  positions into `features` for the returned anchors
    """
    if weighting not in WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {WEIGHTING_MODES}")

    # stddev/p_tsf are qVIO-only (OV leaves them unset); gate only when used
    sel = np.array([
        i for i, f in enumerate(features)
        if f.quality >= min_quality
        and (weighting != "stddev" or f.depth_stddev > 0)
        and (weighting != "covariance" or f.p_tsf is not None)
    ], dtype=int)
    if len(sel) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0), sel

    feats = [features[i] for i in sel]
    # world->imu->cam
    P_vio = np.stack([f.xyz_vio for f in feats])
    P_imu = (P_vio - pose.T_imu_wrt_vio) @ pose.R_imu_to_vio
    P_cam = (P_imu - T_cam_wrt_imu) @ R_cam_to_imu

    u, v, front = pre.project(P_cam)
    depth = P_cam[:, 2]

    if rolling_shutter is not None:
        u, v, front, depth = rolling_shutter.correct(
            P_vio, v, pose, R_cam_to_imu, T_cam_wrt_imu, pre
        )

    if weighting == "stddev":
        var = np.array([f.depth_stddev for f in feats]) ** 2
    elif weighting == "covariance":
        # camera z-axis in world coordinates: z_cam = P_vio . r + const
        r = (pose.R_imu_to_vio @ R_cam_to_imu)[:, 2]
        P = np.stack([f.p_tsf for f in feats])
        var = np.einsum("i,nij,j->n", r, P, r)
    else:
        var = np.ones(len(feats))

    keep = (
        front & (var > 0)
        & (u >= 0) & (u < pre.dst_w) & (v >= 0) & (v < pre.dst_h) # & - element-wise comparison
    )
    return np.stack([u[keep], v[keep]], axis=1), depth[keep], var[keep], sel[keep]


def compute_weights(depth: np.ndarray, var: np.ndarray, weighting: str) -> np.ndarray:
    """Inverse-variance weights for fitting metric disparity y = 1/d.

    Var(1/d) = Var(d)/d^4, so w = d^4/Var(d). Counterintuitively this favours
    FAR anchors when sigma is equal: a +-sigma depth error at 1 m moves 1/d by
    ~sigma, at 10 m by ~sigma/100, so far points pin the disparity curve more
    tightly. In practice sigma grows roughly with d^2 for triangulated points,
    which is why the uniform weighting ("none") is a reasonable default.

    Mean-normalised so the spline penalty terms keep the same scale in every
    mode.
    """
    if weighting == "none" or len(depth) == 0:
        return np.ones(len(depth))
    w = depth ** 4 / var
    return w * (len(w) / w.sum())


def project_features_tracking_extrinsic(
    features: list[Feature],
    cams: dict,          # {cam_id: CameraModel} tracking-camera lens models
    extrs: dict,         # {cam_id: (R_track_to_imu (3,3), T_track_wrt_imu (3,))}
    R_hires: np.ndarray, # hires R_cam_to_imu (3,3)
    T_hires: np.ndarray, # hires T_cam_wrt_imu (3,)
    pre: Preprocessor,
    min_quality: int = 1,
    weighting: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place VIO anchors into the model image WITHOUT the VIO world pose.

    Back-projects each feature's raw pixel (pix_loc) in its observing tracking
    camera to a normalised ray, scales by the feature depth (assumed Z along the
    tracking-camera optical axis), then applies the FIXED tracking->hires
    extrinsic composed through the IMU. No R_imu_to_vio -- so it is robust to a
    failed VIO pose, but assumes the tracking observation and hires frame are
    effectively simultaneous (no motion compensation).

    Returns (uv (N,2), depth (N,), var (N,), idx (N,)) like project_features.
    weighting none|stddev; covariance needs the world pose, so it falls back to
    uniform here (gate it to world_pose upstream).
    """
    use_stddev = (weighting == "stddev")
    by_cam: dict = {}
    for i, f in enumerate(features):
        if f.quality < min_quality or f.pix_loc is None:
            continue
        if f.cam_id not in cams or f.cam_id not in extrs or not (f.depth > 0):
            continue
        if use_stddev and not (f.depth_stddev > 0):
            continue
        by_cam.setdefault(f.cam_id, []).append(i)

    uv_out, depth_out, var_out, idx_out = [], [], [], []
    for cam_id, idxs in by_cam.items():
        cam = cams[cam_id]
        R_t, T_t = extrs[cam_id]
        pts = np.array([features[i].pix_loc for i in idxs], np.float64).reshape(-1, 1, 2)
        if cam.model == "fisheye":
            norm = cv2.fisheye.undistortPoints(pts, cam.K, cam.D)
        else:
            norm = cv2.undistortPoints(pts, cam.K, cam.D)
        norm = norm.reshape(-1, 2)
        d = np.array([features[i].depth for i in idxs])
        P_track = np.column_stack([norm[:, 0] * d, norm[:, 1] * d, d])
        P_imu = P_track @ R_t.T + T_t
        P_hires = (P_imu - T_hires) @ R_hires
        u, v, front = pre.project(P_hires)
        z = P_hires[:, 2]
        for k, i in enumerate(idxs):
            if not front[k] or z[k] <= 0:
                continue
            if not (0 <= u[k] < pre.dst_w and 0 <= v[k] < pre.dst_h):
                continue
            var = features[i].depth_stddev ** 2 if use_stddev else 1.0
            if not (var > 0):
                continue
            uv_out.append((u[k], v[k])); depth_out.append(z[k])
            var_out.append(var); idx_out.append(i)

    if not uv_out:
        return np.empty((0, 2)), np.empty(0), np.empty(0), np.array([], dtype=int)
    return (np.array(uv_out), np.array(depth_out),
            np.array(var_out), np.array(idx_out, dtype=int))
