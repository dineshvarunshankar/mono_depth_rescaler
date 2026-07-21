"""Tests for the parameter smoothers and the WLS covariance that feeds them."""
import numpy as np
from proto.rescaler import fits
from proto.rescaler.fits.polynomial import _wls
from proto.rescaler.smoother import EMASmoother, KalmanSmoother


def test_wls_covariance_matches_analytic():
    # unit weights: Cov must equal sigma^2 (A^T A)^-1 computed independently
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.1, 2.0, 200))
    y = 3.5 * x + 0.15 + rng.normal(0, 0.02, 200)
    w = np.ones(200)

    coeffs, cov = _wls(x, y, w, degree=1)

    A = np.vander(x, 2)
    resid = y - A @ coeffs
    sigma2 = np.sum(resid**2) / (200 - 2)
    np.testing.assert_allclose(cov, sigma2 * np.linalg.inv(A.T @ A), rtol=1e-10)


def test_slope_intercept_anticorrelated_for_positive_x():
    # disparity data is all positive, so the fit pivots about the data centroid:
    # a slope error must be paid back by an intercept error of opposite sign
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.1, 2.0, 100))
    y = 3.5 * x + 0.15 + rng.normal(0, 0.02, 100)
    _, cov = _wls(x, y, np.ones(100), degree=1)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert corr < -0.7


def test_exponential_cov_is_symmetric_positive():
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.05, 2.0, 80))
    y = 0.2 * np.exp(1.5 * x) * np.exp(rng.normal(0, 0.01, 80))
    fit = fits.create("exponential", x, y, np.ones(80))
    c = fit.params_cov
    np.testing.assert_allclose(c, c.T)
    assert c[0, 0] > 0 and c[1, 1] > 0


def test_kalman_matches_scalar_recursion_when_diagonal():
    # with diagonal R and Q the matrix filter must reproduce the elementwise
    # scalar formulae exactly
    q, r = 1e-4, 1e-2
    kf = KalmanSmoother(process_noise=q, measurement_noise=r)
    zs = [np.array([3.5, 0.15]), np.array([3.7, 0.10]), np.array([3.4, 0.20])]

    x, p = zs[0].copy(), np.full(2, r)   # scalar reference
    kf.update(zs[0])
    for z in zs[1:]:
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x + k * (z - x)
        p = (1 - k) * p_pred
        np.testing.assert_allclose(kf.update(z), x, rtol=1e-12)


def test_kalman_converges_on_constant_truth():
    rng = np.random.default_rng(3)
    truth = np.array([3.5, 0.15])
    R = np.array([[4e-4, -1.5e-4], [-1.5e-4, 1e-4]])   # anti-correlated, like a line fit
    kf = KalmanSmoother(process_noise=1e-6)
    L = np.linalg.cholesky(R)
    est = None
    for _ in range(300):
        z = truth + L @ rng.normal(size=2)
        est = kf.update(z, cov=R)
    assert np.abs(est - truth).max() < 0.01


def test_kalman_trusts_precise_frames_more():
    kf = KalmanSmoother(process_noise=1e-6)
    kf.update(np.array([3.5, 0.15]), cov=np.eye(2) * 1e-4)
    # a wildly uncertain measurement should barely move the estimate
    out = kf.update(np.array([10.0, 5.0]), cov=np.eye(2) * 1e2)
    assert np.abs(out - [3.5, 0.15]).max() < 0.05


def test_ema_accepts_and_ignores_cov():
    ema = EMASmoother(alpha=0.4)
    ema.update(np.array([1.0, 1.0]), cov=np.eye(2))
    out = ema.update(np.array([2.0, 2.0]), cov=np.eye(2) * 99)
    np.testing.assert_allclose(out, [1.4, 1.4])
