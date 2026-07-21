"""Decode ext_vio_data_t packets (qvio_extended / ov_extended MPA pipe).

Packed little-endian, matches vio_data_t.h. Sizes: 324 / 76 / 5268 B.
"""
from __future__ import annotations
import numpy as np
from ..rescaler.types import Feature, VioPose

VIO_MAX_FEATURES = 64

_vio_data_t = np.dtype([
    ("magic_number",       "<u4"),
    ("quality",            "<i4"),
    ("timestamp_ns",       "<i8"),
    ("T_imu_wrt_vio",      "<f4", (3,)),
    ("R_imu_to_vio",       "<f4", (3, 3)),
    ("pose_covariance",    "<f4", (21,)),
    ("vel_imu_wrt_vio",    "<f4", (3,)),
    ("velocity_covariance","<f4", (21,)),
    ("imu_angular_vel",    "<f4", (3,)),
    ("gravity_vector",     "<f4", (3,)),
    ("T_cam_wrt_imu",      "<f4", (3,)),
    ("R_cam_to_imu",       "<f4", (3, 3)),
    ("error_code",         "<u4"),
    ("n_feature_points",   "<u2"),
    ("state",              "u1"),
    ("frame",              "u1"),
], align=False)

_vio_feature_t = np.dtype([
    ("id",                 "<u4"),
    ("cam_id",             "<i4"),
    ("pix_loc",            "<f4", (2,)),
    ("tsf",                "<f4", (3,)),   # 3D position in VIO world frame
    ("p_tsf",              "<f4", (3, 3)), # position covariance
    ("depth",              "<f4"),
    ("depth_error_stddev", "<f4"),
    ("point_quality",      "<i4"),
], align=False)

_ext_vio_data_t = np.dtype([
    ("v",                    _vio_data_t),
    ("last_cam_frame_id",    "<i4"),
    ("last_cam_timestamp_ns","<i8"),
    ("imu_cam_time_shift_s", "<f4"),
    ("gravity_covariance",   "<f4", (3, 3)),
    ("gyro_bias",            "<f4", (3,)),
    ("accl_bias",            "<f4", (3,)),
    ("n_total_features",     "<u4"),
    ("features",             _vio_feature_t, (VIO_MAX_FEATURES,)),
], align=False)

assert _vio_data_t.itemsize    == 324,  _vio_data_t.itemsize
assert _vio_feature_t.itemsize == 76,   _vio_feature_t.itemsize
assert _ext_vio_data_t.itemsize == 5268, _ext_vio_data_t.itemsize


def parse_buffer(buf: bytes) -> np.ndarray:
    """View a raw byte buffer as an array of ext_vio_data_t records."""
    n = len(buf) // _ext_vio_data_t.itemsize
    return np.frombuffer(buf[:n * _ext_vio_data_t.itemsize], dtype=_ext_vio_data_t)


def unpack(rec: np.ndarray) -> tuple[int, VioPose, list[Feature]]:
    """Unpack one ext_vio_data_t record into typed objects."""
    v = rec["v"]
    pose = VioPose(
        R_imu_to_vio=v["R_imu_to_vio"].astype(np.float64),
        T_imu_wrt_vio=v["T_imu_wrt_vio"].astype(np.float64),
        vel=v["vel_imu_wrt_vio"].astype(np.float64),
        omega=v["imu_angular_vel"].astype(np.float64),
    )
    n = min(int(rec["n_total_features"]), VIO_MAX_FEATURES)
    features = [
        Feature(
            xyz_vio=rec["features"][i]["tsf"].astype(np.float64),
            cam_id=int(rec["features"][i]["cam_id"]),
            quality=int(rec["features"][i]["point_quality"]),
            depth=float(rec["features"][i]["depth"]),
            depth_stddev=float(rec["features"][i]["depth_error_stddev"]),
            id=int(rec["features"][i]["id"]),
            p_tsf=rec["features"][i]["p_tsf"].astype(np.float64),
            pix_loc=rec["features"][i]["pix_loc"].astype(np.float64),
        )
        for i in range(n)
    ]
    return int(v["timestamp_ns"]), pose, features


def propagate(pose: VioPose, dt_s: float) -> VioPose:
    """First-order pose propagation to a nearby instant: T moves with velocity,
    R integrates the body angular rate."""
    w = pose.omega
    skew = np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])
    return VioPose(
        R_imu_to_vio=pose.R_imu_to_vio @ (np.eye(3) + skew * dt_s),
        T_imu_wrt_vio=pose.T_imu_wrt_vio + pose.vel * dt_s,
        vel=pose.vel,
        omega=pose.omega,
    )
