import itertools
from proto.config import load
from proto.sources.offline import frames

LOG = "/home/ubuntu/voxl_logs/log0000"


def test_tof_attached_when_enabled():
    # The ToF stream starts ~0.7 s after the hires stream, so the very first
    # frames have no ToF within tolerance; scan until the streams overlap.
    cfg = load()
    cfg.anchors.use_tof = True
    f = next((f for f in itertools.islice(frames(cfg, LOG, vio_pipe="ov_extended"), 300)
              if f.tof is not None), None)
    assert f is not None, "no hires frame paired with a ToF frame in first 300"
    assert f.tof.xyz.shape[1] == 3
    assert len(f.tof.noise) == len(f.tof.conf) == len(f.tof.xyz)


def test_tof_absent_when_disabled():
    cfg = load()
    cfg.anchors.use_tof = False
    f = next(frames(cfg, LOG, vio_pipe="ov_extended"))
    assert f.tof is None
