import numpy as np
from proto.rescaler.anchor_grid import grid_thin

W = H = 256


def _d(n):
    return np.ones(n)  # uniform depth (pick rule irrelevant)


def test_one_per_cell_collapses():
    tof = np.random.default_rng(0).uniform(0, 15, size=(100, 2))
    keep = grid_thin(tof, _d(100), grid=16, k=1, width=W, height=H)
    assert len(keep) == 1


def test_distinct_cells_all_kept():
    tof = np.array([[8.0, 8.0], [40.0, 40.0], [200.0, 120.0]])
    keep = grid_thin(tof, _d(3), grid=16, k=1, width=W, height=H)
    assert len(keep) == 3


def test_nearest_pick_takes_min_depth():
    tof = np.zeros((3, 2))          # all in cell 0
    depth = np.array([5.0, 2.0, 9.0])
    keep = grid_thin(tof, depth, grid=16, k=1, width=W, height=H, pick="nearest")
    assert list(keep) == [1]        # depth 2.0


def test_median_pick_takes_middle_depth():
    tof = np.zeros((3, 2))
    depth = np.array([5.0, 2.0, 9.0])
    keep = grid_thin(tof, depth, grid=16, k=1, width=W, height=H, pick="median")
    assert list(keep) == [0]        # depth 5.0


def test_k_cap_even_stride():
    tof = np.stack([np.linspace(0, 15, 10), np.zeros(10)], axis=1)
    keep = grid_thin(tof, _d(10), grid=16, k=3, width=W, height=H)
    assert list(keep) == [0, 4, 9]


def test_empty():
    keep = grid_thin(np.empty((0, 2)), _d(0), grid=16, k=1, width=W, height=H)
    assert len(keep) == 0


def test_deterministic():
    tof = np.random.default_rng(1).uniform(0, 256, size=(300, 2))
    d = np.random.default_rng(2).uniform(1, 8, size=300)
    assert np.array_equal(grid_thin(tof, d, 16, 1, W, H),
                          grid_thin(tof, d, 16, 1, W, H))
