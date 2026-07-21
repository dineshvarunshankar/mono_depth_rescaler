"""Exponential fit: metric_disparity = a * exp(b * relative_disparity).

Linearised as log(y) = log(a) + b*x and solved by weighted least squares.
"""
from __future__ import annotations
import numpy as np
from . import Fit, is_valid


def _predict_from(params: np.ndarray):
    return lambda xx: params[0] * np.exp(params[1] * np.asarray(xx))


def fit(x, y, w, **_):

    pos = y > 0                       # positive points - log() is undefined for non-positive disparity
    if pos.sum() < 2:
        return Fit(predict=lambda xx: np.zeros_like(xx), x_min=0.0, x_max=0.0, valid=False)

    xp, yp, wp = x[pos], y[pos], w[pos]
    sw = np.sqrt(wp)
    A = np.column_stack([xp, np.ones_like(xp)])
    Aw = A * sw[:, None]
    ylog = np.log(yp)
    sol = np.linalg.lstsq(Aw, sw * ylog, rcond=None)[0]   # (b, log_a)
    b, log_a = float(sol[0]), float(sol[1])
    a = float(np.exp(log_a))
    params = np.array([a, b])

    # covariance in (b, log_a) space, then delta method to (a, b) ordering:
    # a = exp(log_a) so var(a) = a^2 var(log_a), cov(a, b) = a cov(log_a, b)
    resid = ylog - A @ np.array([b, log_a])
    dof = max(len(xp) - 2, 1)
    sigma2 = float(np.sum(wp * resid**2) / dof)
    c = sigma2 * np.linalg.inv(Aw.T @ Aw)          # order: (b, log_a)
    #use delta method to get the covariance of the parameters
    #a = exp(log_a) so var(a) = a^2 var(log_a), cov(a, b) = a cov(log_a, b)
    cov = np.array([
        [a * a * c[1, 1], a * c[1, 0]], 
        [a * c[0, 1],     c[0, 0]],
    ])

    predict = _predict_from(params)
    x_min, x_max = float(xp.min()), float(xp.max())
    return Fit(
        predict=predict,
        x_min=x_min,
        x_max=x_max,
        valid=is_valid(predict, x_min, x_max),
        params=params,
        params_cov=cov,
        rebuild=_predict_from,
    )
