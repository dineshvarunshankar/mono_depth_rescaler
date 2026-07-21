from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthMetrics:
    mean_absrel: float
    median_absrel: float
    rmse: float
    d1: float
    n: int


def _metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> DepthMetrics:
    valid = (
        np.isfinite(prediction)
        & np.isfinite(ground_truth)
        & (prediction > 0)
        & (ground_truth > 0)
    )
    pred = prediction[valid]
    gt = ground_truth[valid]
    if len(gt) == 0:
        nan = float("nan")
        return DepthMetrics(nan, nan, nan, nan, 0)
    relative = np.abs(pred - gt) / gt
    ratio = np.maximum(pred / gt, gt / pred)
    return DepthMetrics(
        mean_absrel=float(relative.mean()),
        median_absrel=float(np.median(relative)),
        rmse=float(np.sqrt(np.mean((pred - gt) ** 2))),
        d1=float(np.mean(ratio < 1.25)),
        n=len(gt),
    )


def dense_depth_metrics(
    depth_map: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray | None = None,
) -> DepthMetrics:
    if depth_map.shape != ground_truth.shape:
        raise ValueError("prediction and ground truth shapes must match")
    if mask is None:
        return _metrics(depth_map, ground_truth)
    if mask.shape != depth_map.shape:
        raise ValueError("mask shape must match depth maps")
    return _metrics(depth_map[mask], ground_truth[mask])


def depth_metrics(
    depth_map: np.ndarray,
    ground_truth_uv: np.ndarray,
    ground_truth_depth: np.ndarray,
) -> DepthMetrics:
    height, width = depth_map.shape
    u = np.clip(np.rint(ground_truth_uv[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.rint(ground_truth_uv[:, 1]).astype(int), 0, height - 1)
    return _metrics(depth_map[v, u], ground_truth_depth)
