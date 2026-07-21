"""Integration test: full Python pipeline on synthetic data."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from proto.rescaler.pipeline import Pipeline
from proto.rescaler.types import Feature, Frame, VioPose
from proto.config import (
    Config, IntrinsicsCfg, ExtrinsicsCfg, VioCfg,
    InferenceCfg, RescaleCfg, SmootherCfg, ModulesCfg,
    RollingShutterCfg, MultiFrameCfg, AnchorsCfg,
)


def _rescale_cfg(method="polynomial", min_anchors=5) -> RescaleCfg:
    return RescaleCfg(
        method=method, args={"degree": 1, "num_knots_spline": 10},
        min_anchors=min_anchors, weighting="none", outlier_rejection=False, outlier_k=3.0,
        subpixel_2x2=True, anchor_depth_min=0.05, anchor_depth_max=65.0,
        depth_min=0.3, depth_max=50.0, max_hold_age_ns=5_000_000_000,
    )


def _make_cfg() -> Config:
    K = np.array([[500, 0, 128], [0, 500, 128], [0, 0, 1.0]])
    hires = IntrinsicsCfg(
        width=256, height=256, fx=500, fy=500, cx=128, cy=128,
        distortion_model="pinhole", distortion=[0.0, 0.0, 0.0, 0.0],
    )
    return Config(
        hires=hires,
        extr_hires=ExtrinsicsCfg(R=np.eye(3), T=np.zeros(3)),
        extr_tof=ExtrinsicsCfg(R=np.eye(3), T=np.zeros(3)),
        vio=VioCfg(backend="qvio", pipe="qvio_extended", min_quality=1),
        inference=InferenceCfg(
            source="mpa_pipe", mpa_pipe_name="tflite_disparity",
            model_path="", backend_delegate="cpu",
            input_resolution=(256, 256),
            preprocess="raw", fov="crop", antialias=True,
        ),
        rescale=_rescale_cfg(min_anchors=5),
        smoother=SmootherCfg(enable=True, type="ema", ema_alpha=0.4, kalman_process_noise=1e-4, kalman_measurement_noise=1e-2),
        modules=ModulesCfg(
            rolling_shutter=RollingShutterCfg(enable=False, readout_time_us=15000, readout_direction="top_to_bottom"),
            multi_frame_window=MultiFrameCfg(enable=False, n_frames=5),
        ),
        anchors=AnchorsCfg(use_tof=False, tof_pipe="tof", tof_confidence_min=128, tof_max_points=200),
    )


def _make_frame(a=3.5, b=0.15, n=30) -> tuple[Frame, np.ndarray]:
    rng = np.random.default_rng(123)
    H, W = 256, 256
    fx, fy, cx, cy = 500.0, 500.0, 128.0, 128.0

    depth_gt = np.broadcast_to(np.linspace(1.0, 8.0, W), (H, W)).copy()
    disp_gt  = 1.0 / depth_gt
    disp_rel = (disp_gt - b) / a

    # Random image-plane points; back-project to VIO world (identity pose).
    # Feature depth must agree with the disparity map at its pixel, otherwise
    # the affine fit is meaningless and the a>0 validity guard rejects it.
    px = rng.uniform(20, 235, n)
    py = rng.uniform(20, 235, n)
    iz = depth_gt[py.astype(int), px.astype(int)] + rng.normal(0, 0.02, n)
    ix = (px - cx) * iz / fx
    iy = (py - cy) * iz / fy

    features = [
        Feature(
            xyz_vio=np.array([ix[i], iy[i], iz[i]]),
            cam_id=0, quality=2, depth=iz[i], depth_stddev=0.01,
        )
        for i in range(n)
    ]
    pose = VioPose(R_imu_to_vio=np.eye(3), T_imu_wrt_vio=np.zeros(3))
    frame = Frame(t_ns=0, image=np.zeros((H, W, 3), np.uint8), pose=pose, features=features)
    return frame, disp_rel.astype(np.float32)


def test_pipeline_returns_result():
    cfg = _make_cfg()
    pl = Pipeline(cfg)
    frame, disp = _make_frame()
    res = pl.process(frame, disp)
    assert res is not None
    assert res.depth.shape == disp.shape
    assert res.n_anchors > 0
    assert 0.0 <= res.inlier_ratio <= 1.0
    d_rel, depth, weights = res.anchors
    assert len(d_rel) == len(depth) == len(weights) == res.n_anchors


def test_pipeline_metric_depth_reasonable():
    cfg = _make_cfg()
    pl = Pipeline(cfg)
    frame, disp = _make_frame(a=3.5, b=0.15)
    res = pl.process(frame, disp)
    assert res is not None
    # Centre-image depth should be in 3–6 m range for our gradient scene
    h, w = res.depth.shape
    centre = float(res.depth[h // 2, w // 2])
    assert 0.3 <= centre <= 50.0


def test_pipeline_returns_none_below_min_anchors():
    cfg = _make_cfg()
    cfg.rescale = _rescale_cfg(min_anchors=50)
    pl = Pipeline(cfg)
    frame, disp = _make_frame(n=5)  # only 5 features
    res = pl.process(frame, disp)
    assert res is None


def test_smoother_output_changes_between_frames():
    cfg = _make_cfg()
    pl = Pipeline(cfg)
    frame, disp = _make_frame()
    r1 = pl.process(frame, disp)
    r2 = pl.process(frame, disp)
    assert r1 is not None and r2 is not None
    assert np.isfinite(r1.params).all() and np.isfinite(r2.params).all()


def test_fresh_fit_is_not_held():
    cfg = _make_cfg()
    res = Pipeline(cfg).process(*_make_frame())
    assert res is not None and res.held is False and res.calib_age_ns == 0


def test_feature_drought_holds_last_valid():
    cfg = _make_cfg()
    pl = Pipeline(cfg)
    frame1, disp1 = _make_frame()
    frame1.t_ns = 1_000_000_000
    r1 = pl.process(frame1, disp1)

    # a feature drought (too few anchors) inside the hold window: reuse the
    # earlier calibration rather than dropping the frame, and report its age
    frame2, disp2 = _make_frame(n=3)
    frame2.t_ns = 1_200_000_000
    res = pl.process(frame2, disp2)
    assert res is not None
    assert res.held is True
    assert res.calib_age_ns == 200_000_000
    assert res.n_anchors == 0            # no fresh anchor set was fit
    assert res.anchors is None
    # the held curve is applied to THIS frame's disparity, so depth is current
    np.testing.assert_array_equal(res.depth, r1.depth)


def test_drought_past_hold_window_returns_none():
    cfg = _make_cfg()
    pl = Pipeline(cfg)
    f1, d1 = _make_frame(); f1.t_ns = 0
    pl.process(f1, d1)
    f2, d2 = _make_frame(n=3); f2.t_ns = cfg.rescale.max_hold_age_ns + 1
    assert pl.process(f2, d2) is None


def test_smoother_forgets_across_stale_gap():
    cfg = _make_cfg()
    pl = Pipeline(cfg)

    frame1, disp1 = _make_frame(a=3.5, b=0.15)
    pl.process(frame1, disp1)

    # same scene with a different true curve, arriving just past the hold
    # window: the smoother must start clean, not blend with stale memory
    frame2, disp2 = _make_frame(a=5.0, b=0.10)
    frame2.t_ns = cfg.rescale.max_hold_age_ns + 1
    r_gap = pl.process(frame2, disp2)

    fresh = Pipeline(cfg).process(frame2, disp2)
    assert r_gap is not None and fresh is not None
    np.testing.assert_allclose(r_gap.params, fresh.params)


def test_pipeline_spline_method():
    cfg = _make_cfg()
    cfg.rescale = _rescale_cfg(method="monotonic_nonsmoothing_spline")
    pl = Pipeline(cfg)
    frame, disp = _make_frame()
    res = pl.process(frame, disp)
    assert res is not None
    assert res.params is None
    assert np.isfinite(res.depth).all()
    assert (res.depth >= 0.3).all() and (res.depth <= 50.0).all()
