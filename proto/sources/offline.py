"""Frame source: pulled voxl-logger log directory.

Directory layout produced by voxl-logger:
    <log>/run/mpa/<cam_pipe>/   NNNNN.jpg  +  data.csv (i, timestamp(ns), ...)
    <log>/run/mpa/qvio_extended/ data.raw  (packed ext_vio_data_t, 5268 B each)

Both streams share the VOXL monotonic clock.  Each hires frame is paired with
the nearest VIO packet; pairs outside sync_tol_ms are dropped.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import csv
import cv2
import numpy as np

from ..vio import ext_vio
from . import tof as tof_reader
from ..config import Config
from ..rescaler.types import Frame, TofFrame

# ToF<->hires pairing tolerates a wider offset than the hires<->VIO sync
_TOF_TOL_NS = 200_000_000  # 200 ms


def frames(
    cfg: Config,
    log_dir: str | Path,
    cam_pipe: str = "hires_small_color",
    vio_pipe: str = "qvio_extended",
    sync_tol_ms: int = 50,
) -> Iterator[Frame]:
    """Yield time-synchronised Frames from a pulled voxl-logger directory."""
    log_dir = Path(log_dir)
    cam_dir  = log_dir / "run" / "mpa" / cam_pipe
    raw_path = log_dir / "run" / "mpa" / vio_pipe / "data.raw"

    recs  = ext_vio.parse_buffer(raw_path.read_bytes())
    vt    = recs["v"]["timestamp_ns"].astype(np.int64)
    order = np.argsort(vt)
    vt_s  = vt[order]

    cam_rows = list(csv.DictReader(open(cam_dir / "data.csv", newline="")))
    ht = np.array([int(r["timestamp(ns)"]) for r in cam_rows], dtype=np.int64)
    tol_ns = sync_tol_ms * 1_000_000

    tof_recs = tof_ts = None
    if cfg.anchors.use_tof:
        tof_recs = tof_reader.memmap(log_dir / "run" / "mpa" / cfg.anchors.tof_pipe)
        tof_ts = np.asarray(tof_recs["timestamp_ns"], dtype=np.int64)

    def _tof_at(t_ns: int):
        if tof_recs is None:
            return None
        k = int(np.clip(np.searchsorted(tof_ts, t_ns), 1, len(tof_ts) - 1))
        k = k - 1 if (t_ns - tof_ts[k - 1]) <= (tof_ts[k] - t_ns) else k
        if abs(int(tof_ts[k]) - t_ns) > _TOF_TOL_NS:
            return None
        r = tof_recs[k]
        return TofFrame(int(r["timestamp_ns"]),
                        np.array(r["points"], np.float32),
                        np.array(r["noises"], np.float32),
                        np.array(r["conf"], np.uint8))

    # Nearest VIO index for each hires timestamp (vectorised)
    j    = np.clip(np.searchsorted(vt_s, ht), 1, len(vt_s) - 1)
    pick = np.where(ht - vt_s[j - 1] <= vt_s[j] - ht, j - 1, j)
    gaps = np.abs(vt_s[pick] - ht)

    for row, p, gap in zip(cam_rows, pick, gaps):
        if gap > tol_ns:
            continue
        # unpack the (cheap) VIO packet first and skip invalid-pose frames before
        # decoding the JPG -- a frame with no valid pose produces no anchors, and
        # decoding it anyway (e.g. qVIO, valid only briefly) wastes most of the run
        t_pkt, pose, feats = ext_vio.unpack(recs[order[p]])
        if not pose.valid:
            continue
        pose = ext_vio.propagate(pose, (int(row["timestamp(ns)"]) - t_pkt) * 1e-9)
        bgr = cv2.imread(str(cam_dir / f"{int(row['i']):05d}.jpg"))
        if bgr is None:
            continue
        yield Frame(
            t_ns=int(row["timestamp(ns)"]),
            image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            pose=pose,
            features=feats,
            tof=_tof_at(int(row["timestamp(ns)"])),
        )


def frames_with_depth(
    cfg: Config,
    log_dir: str | Path,
    depth_pipe: str = "tflite_disparity",
    cam_pipe: str = "hires_small_color",
    vio_pipe: str = "qvio_extended",
    sync_tol_ms: int = 50,
) -> Iterator[tuple[Frame, np.ndarray]]:
    """Yield (Frame, disparity) using on-drone model output.

    Use this when the drone ran voxl-tflite-server during the log, with the
    disparity pipe captured via `voxl-logger --raw <depth_pipe>`.
    """
    from ..depth.mpa_disparity import frames as disparity_frames

    log_dir  = Path(log_dir)
    raw_path = log_dir / "run" / "mpa" / vio_pipe / "data.raw"
    cam_dir  = log_dir / "run" / "mpa" / cam_pipe

    recs  = ext_vio.parse_buffer(raw_path.read_bytes())
    vt    = recs["v"]["timestamp_ns"].astype(np.int64)
    order = np.argsort(vt)
    vt_s  = vt[order]
    tol_ns = sync_tol_ms * 1_000_000

    # Optional colour frames for visualisation
    cam_rows, cts = [], np.array([], dtype=np.int64)
    csv_path = cam_dir / "data.csv"
    if csv_path.exists():
        cam_rows = list(csv.DictReader(open(csv_path, newline="")))
        cts = np.array([int(r["timestamp(ns)"]) for r in cam_rows], dtype=np.int64)

    def _nearest(t: int):
        j = int(np.clip(np.searchsorted(vt_s, t), 1, len(vt_s) - 1))
        p = j - 1 if (t - vt_s[j - 1]) <= (vt_s[j] - t) else j
        return p, int(abs(int(vt_s[p]) - t))

    for t_ns, disparity in disparity_frames(log_dir / "run" / "mpa" / depth_pipe):
        p, gap = _nearest(t_ns)
        if gap > tol_ns:
            continue
        t_pkt, pose, feats = ext_vio.unpack(recs[order[p]])
        pose = ext_vio.propagate(pose, (t_ns - t_pkt) * 1e-9)

        image = np.zeros((cfg.hires.height, cfg.hires.width, 3), np.uint8)
        if len(cts):
            k = int(np.clip(np.searchsorted(cts, t_ns), 1, len(cts) - 1))
            k = k - 1 if (t_ns - cts[k - 1]) <= (cts[k] - t_ns) else k
            if abs(int(cts[k]) - t_ns) <= tol_ns:
                bgr = cv2.imread(str(cam_dir / f"{int(cam_rows[k]['i']):05d}.jpg"))
                if bgr is not None:
                    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        yield Frame(t_ns=t_ns, image=image, pose=pose, features=feats), disparity
