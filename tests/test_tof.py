import os
import numpy as np
from proto.sources import tof

LOG = "/home/ubuntu/voxl_logs/log0000/run/mpa/tof"


def test_packet_size_is_known():
    assert tof.PKT_BYTES == 777620
    assert tof.N_POINTS == 43200


def test_packet_size_divides_file():
    assert os.path.getsize(f"{LOG}/data.raw") % tof.PKT_BYTES == 0


def test_decode_is_physically_sane():
    t, xyz, noise, conf = next(tof.frames(LOG))
    assert xyz.shape == (tof.N_POINTS, 3)
    assert noise.shape == (tof.N_POINTS,)
    assert conf.shape == (tof.N_POINTS,)
    # frame 0 timestamp matches data.csv (header layout verification)
    assert t == 633623571787
    z = xyz[:, 2]
    pos = z > 0
    assert pos.sum() > 0                       # some valid returns
    assert (z[pos] < 8.0).mean() > 0.8         # ToF is near-field
    assert conf.min() >= 0
