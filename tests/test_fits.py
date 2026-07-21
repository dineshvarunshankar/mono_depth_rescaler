"""Unit tests for proto/rescaler/fits: all six rescaling methods."""
import numpy as np
import pytest
from proto.rescaler import fits

_AFFINE_METHODS = [
    "polynomial",
    "polynomial_monotonic",
    "smoothing_spline",
    "monotonic_smoothing_spline",
    "monotonic_nonsmoothing_spline",
]


def _affine_data(n=60, a=3.5, b=0.15, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.05, 2.0, n))
    return x, a * x + b, np.ones(n) / n


@pytest.mark.parametrize("method", _AFFINE_METHODS)
def test_recovers_affine_data(method):
    x, y, w = _affine_data()
    fit = fits.create(method, x, y, w)
    assert fit.valid
    xx = np.linspace(x.min(), x.max(), 100)
    np.testing.assert_allclose(fit.predict(xx), 3.5 * xx + 0.15, rtol=0.05, atol=0.02)


def test_exponential_recovers_exponential_data():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.05, 2.0, 60))
    y = 0.2 * np.exp(1.5 * x)
    w = np.ones(60) / 60
    fit = fits.create("exponential", x, y, w)
    assert fit.valid
    np.testing.assert_allclose(fit.params, [0.2, 1.5], rtol=1e-6)


def test_unknown_method_raises():
    x, y, w = _affine_data()
    with pytest.raises(ValueError):
        fits.create("cubic_hermite", x, y, w)


def test_decreasing_curve_is_invalid():
    x, y, w = _affine_data()
    fit = fits.create("polynomial", x, y[::-1].copy(), w)
    assert not fit.valid


def test_spline_extrapolation_is_nan_without_clip():
    x, y, w = _affine_data()
    fit = fits.create("monotonic_nonsmoothing_spline", x, y, w)
    assert np.isnan(fit.predict(np.array([fit.x_max + 1.0])))
    assert np.isfinite(fit.predict(np.clip(np.array([fit.x_max + 1.0]), fit.x_min, fit.x_max)))


def test_outlier_rejection_recovers_truth():
    x, y, w = _affine_data()
    y_corrupt = y.copy()
    y_corrupt[::10] *= 3.0
    fit = fits.create_robust("polynomial", x, y_corrupt, w,
                             outlier_rejection=True, outlier_k=3.0)
    assert fit.valid
    assert not fit.inliers.all()
    np.testing.assert_allclose(fit.params, [3.5, 0.15], rtol=0.05, atol=0.02)


def test_rebuild_matches_predict():
    x, y, w = _affine_data()
    for method in ("polynomial", "exponential"):
        fit = fits.create(method, x, y, w)
        np.testing.assert_array_equal(fit.rebuild(fit.params)(x), fit.predict(x))
