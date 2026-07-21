"""Unit tests for proto/rescaler/geometry.py."""
import numpy as np
import pytest
from proto.rescaler.geometry import project_features, compute_weights
from proto.rescaler.types import Feature, VioPose
from proto.rescaler.camera_model import CameraModel
from proto.rescaler.preprocess import Preprocessor


def _pinhole_camera(fx=500.0, fy=500.0, cx=128.0, cy=128.0, W=256, H=256):
    """Undistorted pinhole at model resolution: project() is the identity mapping,
    so these tests exercise the transform chain, not the lens model."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.zeros(4)
    cam = CameraModel(K, D, model="pinhole")
    pre = Preprocessor(cam, src_size=(W, H), dst_size=(W, H),
                       mode="undistort", fov="stretch", antialias=False)
    return pre, W, H


def _identity_pose() -> VioPose:
    return VioPose(R_imu_to_vio=np.eye(3), T_imu_wrt_vio=np.zeros(3))


def _feature_at(xyz, depth=1.0, stddev=0.01, quality=2) -> Feature:
    return Feature(xyz_vio=np.array(xyz, dtype=float),
                   cam_id=0, quality=quality, depth=depth, depth_stddev=stddev)


def test_points_in_front_project():
    pre, W, H = _pinhole_camera()
    pose = _identity_pose()
    R_ci = np.eye(3)
    T_ci = np.zeros(3)
    # points 5 m in front along camera z
    feats = [_feature_at([0.0, 0.0, 5.0], depth=5.0) for _ in range(5)]
    uv, depth, var, idx = project_features(feats, pose, R_ci, T_ci, pre)
    assert len(depth) == 5
    assert np.allclose(uv[:, 0], 128.0)
    assert np.allclose(uv[:, 1], 128.0)
    assert np.allclose(depth, 5.0)


def test_points_behind_camera_rejected():
    pre, W, H = _pinhole_camera()
    pose = _identity_pose()
    feats = [_feature_at([0.0, 0.0, -1.0], depth=1.0)]
    uv, depth, var, idx = project_features(feats, pose, np.eye(3), np.zeros(3), pre)
    assert len(depth) == 0


def test_out_of_frame_rejected():
    pre, W, H = _pinhole_camera(fx=500, cx=128, cy=128)
    pose = _identity_pose()
    # offset so projected u >> W
    feats = [_feature_at([1000.0, 0.0, 1.0], depth=1.0)]
    uv, depth, var, idx = project_features(feats, pose, np.eye(3), np.zeros(3), pre)
    assert len(depth) == 0


def test_quality_filter():
    pre, W, H = _pinhole_camera()
    pose = _identity_pose()
    feats = [
        _feature_at([0, 0, 3.0], depth=3.0, quality=1),
        _feature_at([0, 0, 4.0], depth=4.0, quality=2),
    ]
    uv, depth, var, idx = project_features(feats, pose, np.eye(3), np.zeros(3), pre, min_quality=2)
    assert len(depth) == 1
    assert np.isclose(depth[0], 4.0)


def test_weights_mean_one():
    depth = np.array([1.0, 3.0, 8.0])
    var = np.full(3, 0.01 ** 2)
    w = compute_weights(depth, var, "stddev")
    assert np.isclose(w.mean(), 1.0)


def test_equal_sigma_far_anchors_weigh_more():
    # Fitting y = 1/d: Var(1/d) = Var(d)/d^4, so with equal sigma a far anchor
    # constrains the disparity curve more tightly than a near one.
    depth = np.array([1.0, 8.0])
    var = np.full(2, 0.01 ** 2)
    w = compute_weights(depth, var, "stddev")
    assert w[1] > w[0]


def test_uniform_mode_is_paper_faithful():
    w = compute_weights(np.array([1.0, 5.0]), np.array([1.0, 9.0]), "none")
    assert np.array_equal(w, np.ones(2))


def test_idx_references_input_positions():
    pre, W, H = _pinhole_camera()
    pose = _identity_pose()
    feats = [
        _feature_at([0, 0, 3.0], depth=3.0, quality=1),   # filtered by min_quality
        _feature_at([0, 0, 4.0], depth=4.0, quality=2),
        _feature_at([0, 0, -1.0], depth=1.0, quality=2),  # behind camera
    ]
    _, _, _, idx = project_features(feats, pose, np.eye(3), np.zeros(3), pre, min_quality=2)
    assert idx.tolist() == [1]
