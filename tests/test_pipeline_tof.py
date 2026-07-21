import numpy as np
from proto.config import load
from proto.rescaler.pipeline import (
    Pipeline,
    _adaptive_source_weights,
    _spatial_subsample,
)
from proto.rescaler.types import Frame, VioPose, TofFrame


def _disp(cfg):
    w, h = cfg.inference.input_resolution
    return np.full((h, w), 0.5, np.float32)


def _frame(with_tof, n=30):
    pose = VioPose(R_imu_to_vio=np.eye(3), T_imu_wrt_vio=np.zeros(3))
    tof = None
    if with_tof:
        xyz = np.tile([0.0, 0.0, 2.0], (n, 1)) + np.random.default_rng(0).normal(0, 0.01, (n, 3))
        tof = TofFrame(0, xyz, np.full(n, 0.02), np.full(n, 200, np.uint8))
    return Frame(t_ns=0, image=np.zeros((768, 1024, 3), np.uint8),
                 pose=pose, features=[], tof=tof)


def test_use_tof_off_is_vio_only():
    cfg = load()
    cfg.anchors.use_tof = False
    # no VIO features and ToF ignored -> no anchors
    assert Pipeline(cfg).anchors(_frame(True), _disp(cfg)) is None


def test_use_tof_on_adds_anchors():
    cfg = load()
    cfg.anchors.use_tof = True
    cfg.rescale.min_anchors = 5
    out = Pipeline(cfg).anchors(_frame(True), _disp(cfg))
    assert out is not None
    disp_rel, depth, weights = out
    assert len(depth) >= 5
    assert len(disp_rel) == len(depth) == len(weights)


def test_adaptive_source_weights_balance_and_fade_on_scarcity():
    balanced = _adaptive_source_weights(np.ones(10), np.ones(100), 0.5, 10)
    assert np.isclose(balanced[:10].sum(), balanced[10:].sum())

    scarce = _adaptive_source_weights(np.ones(2), np.ones(100), 0.5, 10)
    assert np.isclose(scarce[:2].sum() / scarce.sum(), 0.1)
    assert np.isclose(scarce.sum(), len(scarce))

    # A low configured floor must not reduce VIO below its ordinary uniform
    # point-count contribution.
    natural = _adaptive_source_weights(np.ones(10), np.ones(10), 0.05, 20)
    assert np.isclose(natural[:10].sum() / natural.sum(), 0.5)


def test_spatial_subsample_is_deterministic_and_spread():
    yy, xx = np.mgrid[0:10, 0:10]
    uv = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)
    a = _spatial_subsample(uv, 8, 10, 10)
    b = _spatial_subsample(uv, 8, 10, 10)
    assert np.array_equal(a, b)
    selected = uv[a]
    assert np.ptp(selected[:, 0]) >= 8
    assert np.ptp(selected[:, 1]) >= 8
