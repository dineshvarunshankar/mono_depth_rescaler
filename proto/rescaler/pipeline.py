"""Core rescaling pipeline: VIO anchors + relative disparity -> metric depth."""
from __future__ import annotations
from typing import Optional
import numpy as np

from ..config import Config
from .types import Frame, RescaleResult
from .camera_model import CameraModel
from .preprocess import Preprocessor
from .geometry import project_features, compute_weights, project_features_tracking_extrinsic
from .tof_anchors import tof_anchors
from . import fits
from .smoother import EMASmoother, KalmanSmoother
from ..modules.rolling_shutter import RollingShutter
from ..modules.multi_frame_window import FrameWindow


BALANCE_MODES = ("off", "adaptive_source", "adaptive_count")


def _spatial_subsample(
    uv: np.ndarray, target: int, width: int, height: int
) -> np.ndarray:
    """Deterministic farthest-point sample in normalized model-image space."""
    n = len(uv)
    if target >= n:
        return np.arange(n)
    if target <= 0:
        return np.empty(0, dtype=int)
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=float)
    points = uv / scale
    center = np.array([0.5, 0.5])
    first = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    selected = np.empty(target, dtype=int)
    selected[0] = first
    min_d2 = np.sum((points - points[first]) ** 2, axis=1)
    min_d2[first] = -1.0
    for i in range(1, target):
        chosen = int(np.argmax(min_d2))
        selected[i] = chosen
        min_d2 = np.minimum(
            min_d2, np.sum((points - points[chosen]) ** 2, axis=1)
        )
        min_d2[selected[: i + 1]] = -1.0
    return selected


def _adaptive_source_weights(
    vio_weights: np.ndarray,
    tof_weights: np.ndarray,
    vio_fraction: float,
    vio_sufficient: int,
) -> np.ndarray:
    """Raise VIO to an adaptive source-mass floor without downweighting it."""
    nv, nt = len(vio_weights), len(tof_weights)
    if nv == 0:
        return np.ones(nt)
    if nt == 0:
        return np.ones(nv)
    if not 0.0 <= vio_fraction <= 1.0:
        raise ValueError("balance_vio_fraction must be in [0, 1]")
    if vio_sufficient <= 0:
        raise ValueError("balance_vio_sufficient must be positive")
    natural_vio_mass = nv / (nv + nt)
    adaptive_floor = vio_fraction * min(1.0, nv / vio_sufficient)
    vio_mass = max(natural_vio_mass, adaptive_floor)
    tof_mass = 1.0 - vio_mass
    wv = vio_weights / np.sum(vio_weights) * vio_mass
    wt = tof_weights / np.sum(tof_weights) * tof_mass
    return np.concatenate([wv, wt]) * (nv + nt)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.anchors.balance_mode not in BALANCE_MODES:
            raise ValueError(
                f"anchors.balance_mode must be one of {BALANCE_MODES}"
            )
        self.camera = CameraModel(cfg.hires.K, cfg.hires.D, cfg.hires.distortion_model)
        self.pre = Preprocessor(
            self.camera,
            src_size=(cfg.hires.width, cfg.hires.height),
            dst_size=cfg.inference.input_resolution,
            mode=cfg.inference.preprocess,
            fov=cfg.inference.fov,
            antialias=cfg.inference.antialias,
        )
        self.R_cam = cfg.extr_hires.R
        self.T_cam = cfg.extr_hires.T
        self.R_tof = cfg.extr_tof.R
        self.T_tof = cfg.extr_tof.T

        # tracking-camera lens models + extrinsics (cam_id: 0=front, 1=down)
        self._track_cams: dict = {}
        self._track_extrs: dict = {}
        if cfg.tracking_front and cfg.extr_tracking_front:
            self._track_cams[0] = CameraModel(cfg.tracking_front.K, cfg.tracking_front.D,
                                              cfg.tracking_front.distortion_model)
            self._track_extrs[0] = (cfg.extr_tracking_front.R, cfg.extr_tracking_front.T)
        if cfg.tracking_down and cfg.extr_tracking_down:
            self._track_cams[1] = CameraModel(cfg.tracking_down.K, cfg.tracking_down.D,
                                              cfg.tracking_down.distortion_model)
            self._track_extrs[1] = (cfg.extr_tracking_down.R, cfg.extr_tracking_down.T)

        rs = cfg.modules.rolling_shutter
        self._rs = RollingShutter(
            readout_s=rs.readout_time_us * 1e-6,
            top_to_bottom=rs.readout_direction == "top_to_bottom",
        ) if rs.enable else None

        mf = cfg.modules.multi_frame_window
        self._window = FrameWindow(mf.n_frames) if mf.enable else None

        if cfg.smoother.type == "kalman":
            self.smoother = KalmanSmoother(
                cfg.smoother.kalman_process_noise,
                cfg.smoother.kalman_measurement_noise,
            )
        else:
            self.smoother = EMASmoother(cfg.smoother.ema_alpha)

        self._held: Optional[fits.Fit] = None
        self._held_t_ns: int = 0

    def _sample(self, disparity: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """uv is already in model pixel coordinates -- the preprocessor put it there."""
        H, W = disparity.shape
        u, v = uv[:, 0], uv[:, 1]
        if not self.cfg.rescale.subpixel_2x2:
            return disparity[
                np.clip(v.astype(int), 0, H - 1),
                np.clip(u.astype(int), 0, W - 1),
            ]
        # closest surface wins: max disparity over the floor/ceil neighbourhood
        uf = np.clip(np.floor(u).astype(int), 0, W - 1)
        uc = np.clip(np.ceil(u).astype(int), 0, W - 1)
        vf = np.clip(np.floor(v).astype(int), 0, H - 1)
        vc = np.clip(np.ceil(v).astype(int), 0, H - 1)
        return np.maximum.reduce([
            disparity[vf, uf], disparity[vf, uc],
            disparity[vc, uf], disparity[vc, uc],
        ])

    def anchors(self, frame: Frame, disparity: np.ndarray) -> Optional[tuple]:
        """(disp_rel, depth, weights) for in-view, in-range anchors.

        Mutates the multi-frame window when it is enabled, so call it at most
        once per frame. Callers that also run process() must read the anchor
        set from RescaleResult.anchors instead of calling this again.
        """
        r = self.cfg.rescale
        if self.cfg.anchors.projection == "tracking_extrinsic":
            uv, depth, var, idx = project_features_tracking_extrinsic(
                frame.features, self._track_cams, self._track_extrs,
                self.R_cam, self.T_cam, self.pre,
                self.cfg.vio.min_quality, r.weighting)
        else:
            uv, depth, var, idx = project_features(
                frame.features,
                frame.pose,
                self.R_cam,
                self.T_cam,
                self.pre,
                self.cfg.vio.min_quality,
                weighting=r.weighting,
                rolling_shutter=self._rs,
            )
        keep = (depth >= r.anchor_depth_min) & (depth <= r.anchor_depth_max)
        uv, depth, var, idx = uv[keep], depth[keep], var[keep], idx[keep]

        if self.cfg.anchors.use_tof and frame.tof is not None:
            # union VIO + quality ToF anchors; optional balancing (BALANCE_MODES)
            a = self.cfg.anchors
            uv_t, depth_t, var_t = tof_anchors(
                frame.tof, frame.pose, frame.pose,
                self.R_tof, self.T_tof, self.R_cam, self.T_cam,
                self.pre, a.tof_confidence_min, a.tof_max_points)
            keep_t = (depth_t >= r.anchor_depth_min) & (depth_t <= r.anchor_depth_max)
            uv_t, depth_t, var_t = uv_t[keep_t], depth_t[keep_t], var_t[keep_t]

            if a.balance_mode == "adaptive_count":
                target = (
                    len(uv)
                    if len(uv) >= a.balance_vio_sufficient
                    else max(a.balance_tof_fallback, r.min_anchors - len(uv))
                )
                selected = _spatial_subsample(
                    uv_t, target, self.pre.dst_w, self.pre.dst_h
                )
                uv_t, depth_t, var_t = (
                    uv_t[selected], depth_t[selected], var_t[selected]
                )

            vio_n = len(depth)
            uv = np.concatenate([uv, uv_t])
            depth = np.concatenate([depth, depth_t])
            var = np.concatenate([var, var_t])
            disp_rel = self._sample(disparity, uv)
            if a.balance_mode == "adaptive_source":
                weights = _adaptive_source_weights(
                    compute_weights(depth[:vio_n], var[:vio_n], r.weighting),
                    compute_weights(depth[vio_n:], var[vio_n:], r.weighting),
                    a.balance_vio_fraction,
                    a.balance_vio_sufficient,
                )
            else:
                weights = compute_weights(depth, var, r.weighting)
            if len(depth) < r.min_anchors:
                return None
            return disp_rel, depth, weights

        disp_rel = self._sample(disparity, uv)
        weights = compute_weights(depth, var, r.weighting)

        if self._window is not None:
            ids = np.array([frame.features[i].id for i in idx])
            disp_rel, depth, weights = self._window.update(ids, disp_rel, depth, weights)

        if len(depth) < r.min_anchors:
            return None
        return disp_rel, depth, weights

    def process(self, frame: Frame, disparity: np.ndarray) -> Optional[RescaleResult]:
        r = self.cfg.rescale

        # fresh fit needs enough anchors and a valid curve; else hold last valid
        fresh = None
        anchors = self.anchors(frame, disparity)
        if anchors is not None:
            disp_rel, depth, weights = anchors
            candidate = fits.create_robust(
                r.method, disp_rel, 1.0 / depth, weights,
                outlier_rejection=r.outlier_rejection,
                outlier_k=r.outlier_k,
                **r.args,
            )
            if candidate.valid:
                fresh = candidate

        if fresh is not None:
            # drop stale smoother state before blending a fit after a long gap
            if self._held is not None and frame.t_ns - self._held_t_ns > r.max_hold_age_ns:
                self.smoother.reset()
            if self.cfg.smoother.enable and fresh.params is not None:
                fresh.params = self.smoother.update(fresh.params, fresh.params_cov)
                fresh.predict = fresh.rebuild(fresh.params)
            self._held = fresh
            self._held_t_ns = frame.t_ns
            fit, held, calib_age_ns, res_anchors = fresh, False, 0, anchors
        elif self._held is not None and frame.t_ns - self._held_t_ns <= r.max_hold_age_ns:
            # no fresh fit: reuse held calibration (structure current, scale old)
            held_fit: fits.Fit = self._held
            fit, held, calib_age_ns, res_anchors = held_fit, True, frame.t_ns - self._held_t_ns, None
        else:
            # calibration older than max_hold_age: drop it
            self._held = None
            self.smoother.reset()
            return None

        disp_metric = fit.predict(np.clip(disparity, fit.x_min, fit.x_max))
        depth_map = 1.0 / np.clip(disp_metric, 1.0 / r.depth_max, 1.0 / r.depth_min)

        return RescaleResult(
            depth=depth_map,
            method=r.method,
            params=fit.params,
            n_anchors=res_anchors[1].size if res_anchors is not None else 0,
            inlier_ratio=float(fit.inliers.mean()) if fit.inliers is not None else 1.0,
            held=held,
            calib_age_ns=calib_age_ns,
            anchors=res_anchors,
        )
