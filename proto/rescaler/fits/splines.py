"""Penalised B-spline (P-spline) fits.

Vendored from https://github.com/rislab/mono_depth_rescaler (MIT, (c) 2025 Carnegie
Mellon University), itself derived from https://github.com/fohrloop/penalized-splines.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline, UnivariateSpline

from . import Fit, is_valid

_DEGREE = 3 #cubic spline - smooth curve that is twice continuously differentiable
_KAPPA = 1e6 #penalty parameter for the monotonicity constraint
_MAXITER = 30 #maximum number of iterations for the active set method
_RIDGE = 1e-8 #ridge parameter for the regularization/stabilization


def _pspline(x, y, w, knot_segments, lambda_smoothing, kappa=_KAPPA):
    interval = (x[-1] - x[0]) / knot_segments

    knots = np.linspace(
        x[0] - (_DEGREE + 1) * interval,
        x[-1] + (_DEGREE + 1) * interval,
        _DEGREE * 2 + knot_segments + 1,
    )
    B = BSpline.design_matrix(x=x, t=knots, k=_DEGREE).toarray()
    n = B.shape[1] #number of basis functions/columns/control points
    I = np.eye(n)
    D1 = np.diff(I, n=1, axis=0)
    D3 = np.diff(I, n=3, axis=0)

    BW = B.T * w
    A = BW @ B + lambda_smoothing * (D3.T @ D3)
    if lambda_smoothing == 0.0:
        A += _RIDGE * I    # reference adds this in the non-smoothing variant only
    BTy = BW @ y

    def solve(V):
        return np.linalg.solve(A + D1.T @ np.diag(V * kappa) @ D1, BTy)

    V = np.zeros(n - 1) #adjacent intervals in n spline coefficients
    alphas = solve(V)
    for _ in range(_MAXITER):
        V_new = (D1 @ alphas < 0) * 1
        if np.array_equal(V, V_new):
            break
        V = V_new
        alphas = solve(V)
    else:
        # pin flags cumulatively so the active set only grows
        while True:
            V_new = np.maximum(V, (D1 @ alphas < 0) * 1)
            if np.array_equal(V, V_new):
                break
            V = V_new
            alphas = solve(V)

    return BSpline(knots, alphas, _DEGREE, extrapolate=False)


def _ascending(x, y, w):
    o = np.argsort(x)
    return x[o], y[o], w[o]


def _invalid() -> "Fit":
    return Fit(predict=lambda xx: np.zeros_like(xx), x_min=0.0, x_max=0.0, valid=False)


def _wrap(spline, x):
    lo, hi = float(x[0]), float(x[-1])
    return Fit(predict=spline, x_min=lo, x_max=hi, valid=is_valid(spline, lo, hi))


def fit_monotonic_nonsmoothing(x, y, w, num_knots_spline: int = 10,
                               spline_kappa: float = _KAPPA, **_):
    x, y, w = _ascending(x, y, w)
    # a cubic B-spline over num_knots_spline segments has knot_segments+degree
    # basis functions; fewer points than that is underdetermined -> invalid.
    if len(x) < num_knots_spline + _DEGREE:
        return _invalid()
    try:
        return _wrap(_pspline(x, y, w, num_knots_spline, 0.0, spline_kappa), x)
    except (np.linalg.LinAlgError, ValueError):
        return _invalid()


def fit_monotonic_smoothing(x, y, w, num_knots_spline: int = 10,
                            lambda_smoothing: float = 1e5,
                            spline_kappa: float = _KAPPA, **_):
    x, y, w = _ascending(x, y, w)
    if len(x) < num_knots_spline + _DEGREE:
        return _invalid()
    try:
        return _wrap(_pspline(x, y, w, num_knots_spline, lambda_smoothing, spline_kappa), x)
    except (np.linalg.LinAlgError, ValueError):
        return _invalid()


def fit_smoothing(x, y, w, spline_s: float = 1.0, **_):
    x, y, w = _ascending(x, y, w)
    if len(x) <= _DEGREE:                     # UnivariateSpline needs m > k
        return _invalid()
    try:
        return _wrap(UnivariateSpline(x, y, w=w, s=spline_s), x)
    except (np.linalg.LinAlgError, ValueError):
        return _invalid()
