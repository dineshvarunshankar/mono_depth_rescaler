"""Rescaling curve fits: relative disparity x -> metric disparity y = 1/depth."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

_MIN_POINTS = 2

@dataclass
class Fit:
    predict: Callable[[np.ndarray], np.ndarray] #function that takes a numpy array and returns a numpy array
    x_min: float
    x_max: float
    valid: bool
    params: Optional[np.ndarray] = None    # smoothable across frames; None for splines
    params_cov: Optional[np.ndarray] = None  # (p, p) WLS parameter covariance
    rebuild: Optional[Callable[[np.ndarray], Callable]] = None  # params -> predict
    inliers: Optional[np.ndarray] = None


def is_valid(predict: Callable, x_min: float, x_max: float, n: int = 32) -> bool:
    """A curve is usable only if it yields finite, positive, non-decreasing
    metric disparity over the fitted range. Subsumes the a > 0 guard for the
    degree-1 case: a negative slope would inverts near and far."""
    if not np.isfinite([x_min, x_max]).all() or x_max <= x_min:
        return False
    y = np.asarray(predict(np.linspace(x_min, x_max, n)), dtype=float)
    if not (np.isfinite(y).all() and (y > 0).all()):
        return False
    # soft monotonicity penalties leave O(scale/kappa) leakage; tolerance is relative
    #np.diff() -difference between consecutive elements of the array - curve must be non-decreasing
    #y.max() - y.min() + 1e-12 - range of the curve - to avoid division by zero
    return bool((np.diff(y) >= -1e-6 * (y.max() - y.min() + 1e-12)).all())


from . import polynomial, exponential, splines   # after Fit/is_valid: submodules import them

_REGISTRY: dict[str, Callable] = {
    "polynomial":                    polynomial.fit,
    "polynomial_monotonic":          polynomial.fit_monotonic,
    "exponential":                   exponential.fit,
    "smoothing_spline":              splines.fit_smoothing,
    "monotonic_smoothing_spline":    splines.fit_monotonic_smoothing,
    "monotonic_nonsmoothing_spline": splines.fit_monotonic_nonsmoothing,
}

METHODS = tuple(_REGISTRY)


def _invalid() -> Fit:
    return Fit(predict=lambda xx: np.zeros_like(xx), x_min=0.0, x_max=0.0, valid=False)


def create(method: str, x: np.ndarray, y: np.ndarray, w: np.ndarray, **args) -> Fit:
    if method not in _REGISTRY:
        raise ValueError(f"unknown rescale method {method!r}; expected one of {METHODS}")
    # drop non-finite rows (inf/near-zero VIO uncertainty -> non-finite weights)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if not finite.all():
        x, y, w = x[finite], y[finite], w[finite]
    if len(x) < _MIN_POINTS:
        return _invalid()
    try:
        return _REGISTRY[method](x, y, w, **args)
    except (np.linalg.LinAlgError, ValueError):
        return _invalid()


def create_robust(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    outlier_rejection: bool = False,
    outlier_k: float = 3.0,
    **args,
) -> Fit:
    """Fit, optionally re-fitting on the MAD inlier set.

    Rejection is applied against the chosen method's own residuals, so it works
    for every curve rather than assuming a straight line.
    """
    fit = create(method, x, y, w, **args)
    if not outlier_rejection or not fit.valid:
        return fit
    #residuals - difference between the actual and predicted values
    r = y - fit.predict(x)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-9
    inliers = np.abs(r - med) < outlier_k * 1.4826 * mad
    if inliers.sum() < _MIN_POINTS or inliers.all(): #min points and all inliers(first pass already used the perfect fit data)
        fit.inliers = inliers
        return fit

    refit = create(method, x[inliers], y[inliers], w[inliers], **args)
    refit.inliers = inliers
    return refit if refit.valid else fit
