"""Pool anchors across the last n_frames, keeping newest observation per id."""
from __future__ import annotations
from collections import deque
import numpy as np


class FrameWindow:
    def __init__(self, n_frames: int = 5):
        self._frames: deque = deque(maxlen=n_frames)

    def update(
        self,
        ids: np.ndarray,
        disp_rel: np.ndarray,
        depth: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add this frame's anchors; return the pooled window, newest-per-id."""
        self._frames.append((ids, disp_rel, depth, weights))

        latest: dict[int, tuple[float, float, float]] = {}
        for f_ids, f_rel, f_depth, f_w in self._frames:
            for i in range(len(f_ids)):
                latest[int(f_ids[i])] = (f_rel[i], f_depth[i], f_w[i])
        if not latest:
            return np.empty(0), np.empty(0), np.empty(0)

        merged = np.array(list(latest.values()))
        return merged[:, 0], merged[:, 1], merged[:, 2]

    def reset(self) -> None:
        self._frames.clear()
