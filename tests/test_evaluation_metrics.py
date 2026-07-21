import numpy as np
import pytest

from proto.evaluation.metrics import dense_depth_metrics, depth_metrics


def test_point_metrics_exact_match():
    depth = np.full((4, 4), 2.0, np.float32)
    uv = np.array([[1.0, 1.0], [2.0, 2.0]])
    metrics = depth_metrics(depth, uv, np.array([2.0, 2.0]))
    assert metrics.mean_absrel == 0.0
    assert metrics.d1 == 1.0
    assert metrics.n == 2


def test_dense_metrics_support_mask():
    prediction = np.array([[2.0, 4.0], [3.0, 8.0]])
    ground_truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, True], [False, False]])
    metrics = dense_depth_metrics(prediction, ground_truth, mask)
    assert metrics.n == 2
    assert metrics.mean_absrel == 1.0
    assert metrics.median_absrel == 1.0
    assert metrics.rmse == pytest.approx(np.sqrt(2.5))
    assert metrics.d1 == 0.0


def test_dense_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        dense_depth_metrics(np.ones((2, 2)), np.ones((3, 3)))
