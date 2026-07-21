"""Decoder for the VOXL ToF stream (tof_data_t) logged by voxl-logger.

    magic          u32   @0
    timestamp_ns   i64   @4   (packed -- no alignment padding)
    (8 header bytes)     @12
    points[N][3]   f32        @20
    noises[N]      f32
    grayValues[N]  u8
    confidences[N] u8
    -> 20 + N*(12+4+1+1) = 20 + 43200*18 = 777620 bytes/packet

N = 43200 = 240 x 180. Points in ToF camera frame, metres; invalid pixels carry
sentinel non-positive z (filtered downstream).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np

N_POINTS = 43200

_DTYPE = np.dtype([
    ("magic",        "<u4"),
    ("timestamp_ns", "<i8"),
    ("_header",      "<u4", (2,)),
    ("points",       "<f4", (N_POINTS, 3)),
    ("noises",       "<f4", (N_POINTS,)),
    ("gray",         "u1",  (N_POINTS,)),
    ("conf",         "u1",  (N_POINTS,)),
], align=False)

PKT_BYTES = _DTYPE.itemsize
assert PKT_BYTES == 777620, f"tof_data_t layout mismatch: {PKT_BYTES}"


def parse_buffer(buf: bytes) -> np.ndarray:
    """Reinterpret a raw ToF byte buffer as an array of tof_data_t records."""
    n = len(buf) // PKT_BYTES
    return np.frombuffer(buf[: n * PKT_BYTES], dtype=_DTYPE, count=n)


def memmap(log_dir) -> np.ndarray:
    """Memory-map the ToF log as a record array (fields: timestamp_ns, points,
    noises, gray, conf) for random access by frame index."""
    path = Path(log_dir)
    raw = path / "data.raw" if path.is_dir() else path
    return np.memmap(raw, dtype=_DTYPE, mode="r")


def frames(log_dir) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (timestamp_ns, points(N,3), noise(N,), confidence(N,)) per ToF frame.

    log_dir may be the ToF pipe directory (containing data.raw) or the data.raw
    file itself. Uses a memmap so the 279 MB log is not read into RAM at once.
    """
    path = Path(log_dir)
    raw = path / "data.raw" if path.is_dir() else path
    recs = np.memmap(raw, dtype=_DTYPE, mode="r")
    for r in recs:
        yield (int(r["timestamp_ns"]),
               np.array(r["points"], dtype=np.float32),
               np.array(r["noises"], dtype=np.float32),
               np.array(r["conf"], dtype=np.uint8))
