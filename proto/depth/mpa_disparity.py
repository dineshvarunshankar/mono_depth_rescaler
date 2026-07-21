"""Read relative disparity from a logged MPA disparity pipe.

The on-drone model helper publishes dequantised float32 disparity on DISPARITY_CH
(<prefix>_tflite_disparity), carrying the source camera frame's timestamp.
- Quantisation is handled producer-side, so no scale or zero-point is needed here
and the reader is model-variant agnostic.
- Log with voxl-logger's raw mode, which writes the pipe bytes verbatim:

    voxl-logger --raw tflite_disparity

Note: The camera-typed logging path min-max normalises float32 down to an 8-bit PNG
and cannot be replayed. The <prefix>_tflite pipe likewise carries only a JET
colour map for voxl-portal: lossy, non-invertible, not a disparity source.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np

IMAGE_FORMAT_FLOAT32 = 11 #VOXL format registry value for float32

_meta_t = np.dtype([
    ("magic_number", "<u4"), #little-endian byte order for ARM
    ("timestamp_ns", "<i8"),
    ("frame_id",     "<i4"),
    ("width",        "<i2"),
    ("height",       "<i2"),
    ("size_bytes",   "<i4"),
    ("stride",       "<i4"),
    ("exposure_ns",  "<i4"),
    ("gain",         "<i2"),
    ("format",       "<i2"),
    ("framerate",    "<i2"),
    ("reserved",     "<i2"),
], align=False) #align=False means no padding between fields

assert _meta_t.itemsize == 40, _meta_t.itemsize


def frames(disparity_dir: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (mid_exposure_timestamp_ns, disparity) from a raw-logged pipe dir.

    timestamp_ns marks the start of exposure; the frame's effective time is its
    midpoint, which is what pose interpolation should target.
    """
    buf = memoryview((Path(disparity_dir) / "data.raw").read_bytes())
    off = 0
    while off + _meta_t.itemsize <= len(buf):
        meta = np.frombuffer(buf, dtype=_meta_t, count=1, offset=off)[0]
        off += _meta_t.itemsize
        n = int(meta["size_bytes"]) #number of bytes in the image
        if meta["format"] != IMAGE_FORMAT_FLOAT32:
            raise ValueError(
                f"expected FLOAT32 disparity pipe, got format {int(meta['format'])}"
            )
        disparity = np.frombuffer(buf, dtype="<f4", count=n // 4, offset=off).reshape(
            int(meta["height"]), int(meta["width"])
        )
        off += n
        yield int(meta["timestamp_ns"]) + int(meta["exposure_ns"]) // 2, disparity 
        #beginning of exposure + exposure duration / 2 since camera integrates light over the whole exposure - the image is averaged over that time.
