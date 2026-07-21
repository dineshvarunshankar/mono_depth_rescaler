import numpy as np
from proto.rescaler.types import TofFrame, VioPose
from proto.rescaler.camera_model import CameraModel
from proto.rescaler.preprocess import Preprocessor
from proto.rescaler.tof_anchors import tof_anchors

I3 = np.eye(3)
Z3 = np.zeros(3)


def _pre():
    cam = CameraModel(np.array([[300., 0, 192], [0, 300, 192], [0, 0, 1]]),
                      np.zeros(5), "pinhole")
    return Preprocessor(cam, (384, 384), (384, 384), mode="undistort", fov="crop")


def _id_pose():
    return VioPose(R_imu_to_vio=I3, T_imu_wrt_vio=Z3)


def test_confidence_gate_and_var():
    tof = TofFrame(0, np.array([[0., 0., 2.], [0., 0., 3.]]),
                   np.array([0.02, 0.05]), np.array([200, 10], np.uint8))
    uv, depth, var = tof_anchors(tof, _id_pose(), _id_pose(),
                                 I3, Z3, I3, Z3, _pre(),
                                 conf_min=128, max_points=200)
    assert len(depth) == 1                       # low-confidence point gated out
    assert np.isclose(depth[0], 2.0)
    assert np.isclose(var[0], 0.02 ** 2)         # var = noise^2
    assert np.allclose(uv[0], [192.0, 192.0], atol=1.0)   # on-axis -> principal point


def test_max_points_caps_with_even_coverage():
    # confidence is binary in this sensor's data, so the cap takes an even
    # stride over the gated points (spatial coverage), not a confidence rank
    xyz = np.tile([0., 0., 2.], (5, 1))
    tof = TofFrame(0, xyz, np.full(5, 0.02),
                   np.array([200, 199, 198, 197, 196], np.uint8))
    _, depth, _ = tof_anchors(tof, _id_pose(), _id_pose(), I3, Z3, I3, Z3,
                              _pre(), conf_min=128, max_points=3)
    assert len(depth) == 3


def test_empty_when_all_gated():
    tof = TofFrame(0, np.array([[0., 0., 2.]]), np.array([0.02]),
                   np.array([10], np.uint8))
    uv, depth, var = tof_anchors(tof, _id_pose(), _id_pose(), I3, Z3, I3, Z3,
                                 _pre(), conf_min=128, max_points=200)
    assert len(depth) == 0 and uv.shape == (0, 2)
