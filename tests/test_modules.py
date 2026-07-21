"""Tests for the rolling-shutter and multi-frame-window modules, and the
covariance weighting mode."""
import numpy as np
import pytest

from proto.rescaler.types import Feature, VioPose
from proto.rescaler.camera_model import CameraModel
from proto.rescaler.preprocess import Preprocessor
from proto.rescaler.geometry import project_features
from proto.modules.rolling_shutter import RollingShutter
from proto.modules.multi_frame_window import FrameWindow


def _pre(W=256, H=256, f=500.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    cam = CameraModel(K, np.zeros(4), model="pinhole")
    return Preprocessor(cam, (W, H), (W, H), mode="undistort", fov="stretch", antialias=False)


def _pose(vel=None, omega=None):
    return VioPose(
        R_imu_to_vio=np.eye(3), T_imu_wrt_vio=np.zeros(3),
        vel=np.zeros(3) if vel is None else np.asarray(vel, float),
        omega=np.zeros(3) if omega is None else np.asarray(omega, float),
    )


def _grid_points(n=5, z=4.0):
    xs = np.linspace(-0.3, 0.3, n) * z
    ys = np.linspace(-0.3, 0.3, n) * z
    return np.array([[x, y, z] for y in ys for x in xs])


class TestRollingShutter:
    def test_static_camera_changes_nothing(self):
        pre = _pre()
        P = _grid_points()
        u0, v0, _ = pre.project(P)
        rs = RollingShutter(readout_s=0.015)
        u, v, front, depth = rs.correct(P, v0, _pose(), np.eye(3), np.zeros(3), pre)
        np.testing.assert_allclose(u, u0, atol=1e-12)
        np.testing.assert_allclose(v, v0, atol=1e-12)

    def test_correction_grows_with_row(self):
        # camera translating +x: later rows were captured further along, so the
        # correction shifts them further, and the top row barely moves
        pre = _pre()
        P = _grid_points()
        u0, v0, _ = pre.project(P)
        rs = RollingShutter(readout_s=0.015)
        u, v, _, _ = rs.correct(P, v0, _pose(vel=[3.0, 0, 0]), np.eye(3), np.zeros(3), pre)
        shift = np.abs(u - u0)
        top, bottom = v0 < 100, v0 > 156
        assert shift[bottom].mean() > 2.0 * shift[top].mean()

    def test_magnitude_matches_first_order_prediction(self):
        # pixel shift ~= f * v * dt / z for translation parallel to the image
        pre = _pre(f=500.0)
        z, vel, readout = 4.0, 3.0, 0.015
        P = np.array([[0.0, 0.736, z]])          # v = 128 + 500*0.184 = 220, in frame
        u0, v0, _ = pre.project(P)
        rs = RollingShutter(readout_s=readout)
        u, v, _, _ = rs.correct(P, v0, _pose(vel=[vel, 0, 0]), np.eye(3), np.zeros(3), pre)
        dt = (v0[0] / pre.dst_h) * readout
        expected = 500.0 * vel * dt / z
        assert abs((u0[0] - u[0]) - expected) < 0.15 * expected

    def test_bottom_to_top_reverses_the_gradient(self):
        pre = _pre()
        P = _grid_points()
        u0, v0, _ = pre.project(P)
        down = RollingShutter(readout_s=0.015, top_to_bottom=True)
        up = RollingShutter(readout_s=0.015, top_to_bottom=False)
        ud, *_ = down.correct(P, v0, _pose(vel=[3.0, 0, 0]), np.eye(3), np.zeros(3), pre)
        uu, *_ = up.correct(P, v0, _pose(vel=[3.0, 0, 0]), np.eye(3), np.zeros(3), pre)
        top = v0 < 100
        assert np.abs(uu - u0)[top].mean() > np.abs(ud - u0)[top].mean()


class TestFrameWindow:
    def test_pools_across_frames(self):
        w = FrameWindow(n_frames=3)
        w.update(np.array([1, 2]), np.array([0.1, 0.2]), np.array([1.0, 2.0]), np.ones(2))
        d_rel, depth, wt = w.update(
            np.array([3, 4]), np.array([0.3, 0.4]), np.array([3.0, 4.0]), np.ones(2)
        )
        assert len(depth) == 4

    def test_newest_observation_wins(self):
        w = FrameWindow(n_frames=3)
        w.update(np.array([7]), np.array([0.1]), np.array([1.0]), np.ones(1))
        d_rel, depth, _ = w.update(np.array([7]), np.array([0.9]), np.array([9.0]), np.ones(1))
        assert len(depth) == 1
        assert depth[0] == 9.0 and d_rel[0] == 0.9

    def test_old_frames_expire(self):
        w = FrameWindow(n_frames=2)
        w.update(np.array([1]), np.array([0.1]), np.array([1.0]), np.ones(1))
        w.update(np.array([2]), np.array([0.2]), np.array([2.0]), np.ones(1))
        _, depth, _ = w.update(np.array([3]), np.array([0.3]), np.array([3.0]), np.ones(1))
        assert sorted(depth.tolist()) == [2.0, 3.0]


class TestCovarianceWeighting:
    def _feature(self, z, P):
        return Feature(xyz_vio=np.array([0.0, 0.0, z]), cam_id=0, quality=2,
                       depth=z, depth_stddev=0.05, p_tsf=P)

    def test_variance_is_the_viewing_axis_component(self):
        # identity rotations: the camera looks along world z, so only P[2,2]
        # should matter -- huge lateral uncertainty must not change var
        pre = _pre()
        pose = _pose()
        f_axial = self._feature(4.0, np.diag([1e-6, 1e-6, 0.04]))
        f_lateral = self._feature(4.0, np.diag([9.0, 9.0, 0.04]))
        _, _, var, _ = project_features(
            [f_axial, f_lateral], pose, np.eye(3), np.zeros(3), pre,
            weighting="covariance",
        )
        np.testing.assert_allclose(var, [0.04, 0.04], rtol=1e-9)

    def test_features_without_covariance_are_excluded(self):
        pre = _pre()
        feats = [
            self._feature(4.0, np.diag([0.1, 0.1, 0.1])),
            Feature(xyz_vio=np.array([0.0, 0.0, 4.0]), cam_id=0, quality=2,
                    depth=4.0, depth_stddev=0.05),   # p_tsf defaults to None
        ]
        _, _, var, idx = project_features(
            feats, _pose(), np.eye(3), np.zeros(3), pre, weighting="covariance",
        )
        assert idx.tolist() == [0]

    def test_zero_covariance_cannot_hijack_the_fit(self):
        pre = _pre()
        feats = [
            self._feature(4.0, np.zeros((3, 3))),    # "perfect" = no information
            self._feature(4.0, np.diag([0.1, 0.1, 0.1])),
        ]
        _, _, var, idx = project_features(
            feats, _pose(), np.eye(3), np.zeros(3), pre, weighting="covariance",
        )
        assert idx.tolist() == [1]


class TestPosePropagation:
    def test_zero_rates_zero_dt_is_identity(self):
        from proto.vio.ext_vio import propagate
        pose = _pose()
        out = propagate(pose, 0.05)
        np.testing.assert_array_equal(out.T_imu_wrt_vio, pose.T_imu_wrt_vio)
        np.testing.assert_array_equal(out.R_imu_to_vio, pose.R_imu_to_vio)

    def test_translation_follows_velocity(self):
        from proto.vio.ext_vio import propagate
        out = propagate(_pose(vel=[2.0, 0.0, -1.0]), 0.05)
        np.testing.assert_allclose(out.T_imu_wrt_vio, [0.1, 0.0, -0.05])

    def test_rotation_matches_exact_for_small_angle(self):
        from proto.vio.ext_vio import propagate
        from scipy.spatial.transform import Rotation
        omega, dt = np.array([0.0, 0.0, 1.0]), 0.02        # 1 rad/s yaw, 20 ms
        out = propagate(_pose(omega=omega), dt)
        exact = Rotation.from_rotvec(omega * dt).as_matrix()
        np.testing.assert_allclose(out.R_imu_to_vio, exact, atol=5e-4)

    def test_negative_dt_propagates_backward(self):
        from proto.vio.ext_vio import propagate
        pose = _pose(vel=[1.0, 0, 0], omega=[0, 0, 0.5])
        back = propagate(propagate(pose, 0.03), -0.03)
        np.testing.assert_allclose(back.T_imu_wrt_vio, pose.T_imu_wrt_vio, atol=1e-12)
        np.testing.assert_allclose(back.R_imu_to_vio, pose.R_imu_to_vio, atol=1e-3)
