import numpy as np
from proto.config import load


def test_tof_extrinsic_is_valid_rotation():
    cfg = load()
    R = cfg.extr_tof.R
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)      # orthonormal
    assert abs(np.linalg.det(R) - 1.0) < 1e-6              # proper rotation
    assert cfg.extr_tof.T.shape == (3,)
    assert np.allclose(cfg.extr_tof.T, [0.066, 0.009, -0.012])
