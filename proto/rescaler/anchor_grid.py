"""Grid-thin ToF anchors: keep up to K per GxG image cell."""
import numpy as np


def _cell_ids(uv: np.ndarray, grid: int, width: int, height: int) -> np.ndarray:
    if len(uv) == 0:
        return np.empty(0, dtype=np.int64)
    cu = np.clip((uv[:, 0] * grid / width).astype(np.int64), 0, grid - 1)
    cv = np.clip((uv[:, 1] * grid / height).astype(np.int64), 0, grid - 1)
    return cv * grid + cu


def _pick(members: list, depth: np.ndarray, k: int, rule: str) -> list:
    n = len(members)
    if n <= k:
        return members
    if k == 1:
        if rule == "nearest":
            return [min(members, key=lambda i: depth[i])]
        if rule == "median":
            s = sorted(members, key=lambda i: depth[i])
            return [s[n // 2]]
        return [members[0]]  # first (input order)
    return [members[int(i * (n - 1) / (k - 1))] for i in range(k)]


def grid_thin(uv: np.ndarray, depth: np.ndarray, grid: int, k: int,
              width: int, height: int, pick: str = "nearest"):
    """Indices into uv: up to k per cell. pick (k=1): first | nearest | median."""
    by: dict = {}
    for i, c in enumerate(_cell_ids(uv, grid, width, height)):
        by.setdefault(int(c), []).append(i)
    keep: list = []
    for c in sorted(by):
        keep += _pick(by[c], depth, k, pick)
    return np.array(keep, dtype=int)
