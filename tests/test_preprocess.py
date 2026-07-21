"""Preprocessor: the image transform and the anchor projection must agree.

If they ever disagree, anchors sample the disparity of a different object and
the fit converges confidently to the wrong curve, with nothing to signal it.
So the test is end-to-end: put a mark where a 3D point really appears in the
raw fisheye frame, run the frame through prepare(), and require the mark to
land exactly where project() says it will.
"""
import cv2
import numpy as np
import pytest

from proto.rescaler.camera_model import CameraModel
from proto.rescaler.preprocess import Preprocessor

SRC = (1024, 768)
DST = (256, 256)

# Real AR0144 calibration (config/intrinsics/tracking_front.yaml), rescaled to
# 1024x768 -- the hires values are placeholders and unfit to test against.
_K = np.array([[458.43 * 0.8, 0, 636.90 * 0.8],
               [0, 457.87 * 0.96, 385.31 * 0.96],
               [0, 0, 1.0]])
_D = np.array([0.051385, 0.049631, -0.063327, 0.029654])


def _camera():
    return CameraModel(_K, _D, model="fisheye")


def _points():
    rng = np.random.default_rng(0)
    z = rng.uniform(2.0, 8.0, 12)
    return np.column_stack([
        rng.uniform(-0.35, 0.35, 12) * z,
        rng.uniform(-0.25, 0.25, 12) * z,
        z,
    ])


def _gaussian_spot(u: float, v: float, sigma: float = 5.0) -> np.ndarray:
    """A single-peaked spot. A filled disc would be a plateau, and the centroid
    of a plateau is decided by tie-breaking rather than by geometry."""
    ys, xs = np.mgrid[0:SRC[1], 0:SRC[0]]
    g = np.exp(-(((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma ** 2)))
    return (255 * g).astype(np.uint8)


def _centroid(img: np.ndarray) -> tuple[float, float]:
    w = img.astype(np.float64)
    w[w < 0.2 * w.max()] = 0.0            # drop the tails, keep the peak
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    total = w.sum()
    return float((w * xs).sum() / total), float((w * ys).sum() / total)


@pytest.mark.parametrize("mode,fov", [
    ("raw", "crop"),
    ("undistort", "crop"),
    ("undistort", "fit"),
    ("undistort", "stretch"),
])
def test_projection_matches_image_transform(mode, fov):
    cam = _camera()
    pre = Preprocessor(cam, SRC, DST, mode=mode, fov=fov, antialias=False)

    P = _points()
    u_src, v_src, ok = cam.project_distorted(P)     # where each point really appears
    u_dst, v_dst, _ = pre.project(P)                # where we claim it lands

    checked = 0
    for i in range(len(P)):
        if not ok[i] or not (8 <= u_dst[i] < DST[0] - 8 and 8 <= v_dst[i] < DST[1] - 8):
            continue
        out = pre.prepare(_gaussian_spot(u_src[i], v_src[i]))
        cx, cy = _centroid(out)
        assert abs(cx - u_dst[i]) <= 1.0 and abs(cy - v_dst[i]) <= 1.0, (
            f"{mode}/{fov}: image puts the mark at ({cx:.1f}, {cy:.1f}) "
            f"but project() says ({u_dst[i]:.1f}, {v_dst[i]:.1f})"
        )
        checked += 1
    assert checked >= 5, f"{mode}/{fov}: only {checked} points in view"


def test_undistort_straightens_lines():
    """A straight world line bows in the fisheye frame and must come back straight."""
    cam = _camera()
    raw = Preprocessor(cam, SRC, DST, mode="raw", antialias=False)
    und = Preprocessor(cam, SRC, DST, mode="undistort", fov="stretch", antialias=False)

    # collinear 3D points spanning the view, on a line parallel to the image plane
    P = np.column_stack([np.linspace(-2.5, 2.5, 25), np.full(25, -1.2), np.full(25, 3.0)])

    def bow(pre):
        u, v, _ = pre.project(P)
        keep = (u >= 0) & (u < DST[0]) & (v >= 0) & (v < DST[1])
        u, v = u[keep], v[keep]
        # max deviation from the straight line through the endpoints
        t = np.polyfit([u[0], u[-1]], [v[0], v[-1]], 1)
        return np.abs(v - np.polyval(t, u)).max()

    assert bow(und) < 0.5          # rectilinear: the line is straight again
    assert bow(raw) > 2.0 * bow(und)   # fisheye: it visibly bows


def test_fov_modes_order_by_view_width():
    cam = _camera()
    hfov = {}
    for fov in ("crop", "fit", "stretch"):
        pre = Preprocessor(cam, SRC, DST, mode="undistort", fov=fov)
        hfov[fov] = 2 * np.degrees(np.arctan(DST[0] / 2 / pre.K_model[0, 0]))

    # 'fit' keeps the whole view; 'crop' discards the wider axis to avoid blanks
    assert hfov["fit"] > hfov["crop"]
    assert np.isclose(hfov["stretch"], hfov["fit"])


def test_raw_mode_has_no_k_model():
    pre = Preprocessor(_camera(), SRC, DST, mode="raw")
    assert pre.K_model is None      # raw has no rectilinear camera to speak of


def test_rejects_unknown_modes():
    with pytest.raises(ValueError):
        Preprocessor(_camera(), SRC, DST, mode="rectify")
    with pytest.raises(ValueError):
        Preprocessor(_camera(), SRC, DST, mode="undistort", fov="balance")
