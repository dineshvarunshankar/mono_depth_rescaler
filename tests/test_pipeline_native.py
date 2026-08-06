"""Anchors taken natively from the camera that observed them."""
import numpy as np
import pytest

from proto.rescaler.geometry import project_features_native
from proto.rescaler.camera_model import CameraModel
from proto.rescaler.preprocess import Preprocessor
from proto.rescaler.types import Feature, VioPose


def _pre() -> Preprocessor:
    K = np.array([[500.0, 0, 128], [0, 500.0, 128], [0, 0, 1.0]])
    cam = CameraModel(K, np.zeros(4), "pinhole")
    return Preprocessor(cam, src_size=(256, 256), dst_size=(256, 256),
                        mode="raw", fov="crop", antialias=False)


def _feature(pix, depth, cam_id=0, xyz=(0.0, 0.0, 0.0)):
    return Feature(xyz_vio=np.array(xyz), cam_id=cam_id, quality=2,
                   depth=depth, depth_stddev=0.01,
                   pix_loc=np.array(pix, dtype=float))


def test_pixel_comes_from_pix_loc():
    pre = _pre()
    feats = [_feature((128.0, 128.0), 3.0), _feature((180.0, 64.0), 5.0)]
    uv, depth, _, idx = project_features_native(
        feats, 0, pre.camera, pre, VioPose(np.eye(3), np.zeros(3)),
        np.eye(3), np.zeros(3), "feature")

    assert len(idx) == 2
    assert np.allclose(uv[0], [128.0, 128.0])
    assert np.allclose(uv[1], [180.0, 64.0])
    assert np.allclose(depth, [3.0, 5.0])


def test_other_cameras_are_ignored():
    pre = _pre()
    feats = [_feature((128.0, 128.0), 3.0, cam_id=0),
             _feature((100.0, 100.0), 4.0, cam_id=1)]
    _, _, _, idx = project_features_native(
        feats, 0, pre.camera, pre, VioPose(np.eye(3), np.zeros(3)),
        np.eye(3), np.zeros(3), "feature")

    assert list(idx) == [0]


def test_pose_depth_matches_feature_depth():
    """With an identity pose and extrinsic, both sources agree."""
    pre = _pre()
    feats = [_feature((128.0, 128.0), 3.0, xyz=(0.0, 0.0, 3.0))]
    pose = VioPose(np.eye(3), np.zeros(3))

    _, d_feat, _, _ = project_features_native(
        feats, 0, pre.camera, pre, pose, np.eye(3), np.zeros(3), "feature")
    _, d_pose, _, _ = project_features_native(
        feats, 0, pre.camera, pre, pose, np.eye(3), np.zeros(3), "pose")

    assert np.allclose(d_feat, d_pose)


def test_features_without_depth_are_dropped_when_reading_the_field():
    pre = _pre()
    feats = [_feature((128.0, 128.0), 0.0)]
    _, _, _, idx = project_features_native(
        feats, 0, pre.camera, pre, VioPose(np.eye(3), np.zeros(3)),
        np.eye(3), np.zeros(3), "feature")

    assert len(idx) == 0


def test_covariance_weighting_is_rejected():
    pre = _pre()
    with pytest.raises(ValueError):
        project_features_native(
            [_feature((128.0, 128.0), 3.0)], 0, pre.camera, pre,
            VioPose(np.eye(3), np.zeros(3)), np.eye(3), np.zeros(3),
            "feature", weighting="covariance")
