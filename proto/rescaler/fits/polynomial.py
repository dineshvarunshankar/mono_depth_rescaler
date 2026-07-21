"""Polynomial fits. degree=1 is the affine model of the MonoDepth Rescaler paper."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from . import Fit, is_valid

_MONOTONIC_PENALTY = 1e8


def _predict_from(coeffs: np.ndarray):
    return lambda xx: np.polyval(coeffs, xx)


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray, degree: int):
    """Weighted least squares; returns (coeffs, covariance).

    Cov = sigma^2 (A^T W A)^-1 with sigma^2 from the weighted residuals -- the
    fit's own uncertainty, used as the Kalman smoother's per-frame R. For
    uncentred disparity data the slope/intercept errors are strongly
    anti-correlated, which the off-diagonal term carries.
    """
    sw = np.sqrt(w)
    A = np.vander(x, N=degree + 1, increasing=False)
    Aw = A * sw[:, None]
    coeffs = np.linalg.lstsq(Aw, sw * y, rcond=None)[0]
    resid = y - A @ coeffs
    dof = max(len(x) - (degree + 1), 1)
    sigma2 = float(np.sum(w * resid**2) / dof)
    cov = sigma2 * np.linalg.inv(Aw.T @ Aw)
    return coeffs, cov


def fit(x, y, w, degree: int = 1, **_):
    coeffs, cov = _wls(x, y, w, degree)
    predict = _predict_from(coeffs)
    x_min, x_max = float(x.min()), float(x.max())
    return Fit(
        predict=predict,
        x_min=x_min,
        x_max=x_max,
        valid=is_valid(predict, x_min, x_max),
        params=coeffs,
        params_cov=cov,
        rebuild=_predict_from,
    )


def fit_monotonic(x, y, w, degree: int = 1,
                  poly_monotonic_penalty: float = _MONOTONIC_PENALTY, **_):
    """Weighted least squares with a soft monotonicity penalty (Nelder-Mead).

    The derivative is checked at the interval endpoints and at the critical
    points of p'(x); a negative derivative anywhere there is penalised.
    """
    x_min, x_max = float(x.min()), float(x.max())
    initial, cov = _wls(x, y, w, degree)

    def loss(params):
        data = np.sum(w * (y - np.polyval(params, x)) ** 2) #weighted sum of squared residuals, ls error
        der = np.polyder(params)
        crit = [r.real for r in np.roots(np.polyder(der))
                if np.isreal(r) and x_min <= r.real <= x_max]
        der_vals = np.polyval(der, [x_min, x_max] + crit)
        return data + poly_monotonic_penalty * np.sum(np.minimum(0.0, der_vals) ** 2)

    res = minimize(loss, initial, method="Nelder-Mead")
    coeffs = res.x if res.success else initial

    predict = _predict_from(coeffs)
    # cov is the unconstrained WLS covariance (approximate; ignores the penalty)
    return Fit(
        predict=predict,
        x_min=x_min,
        x_max=x_max,
        valid=is_valid(predict, x_min, x_max),
        params=coeffs,
        params_cov=cov,
        rebuild=_predict_from,
    )
