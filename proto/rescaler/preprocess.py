"""Prepare a camera frame for the depth model and project 3D points into it."""
from __future__ import annotations
import cv2
import numpy as np

from .camera_model import CameraModel

MODES = ("raw", "undistort")
FOV_MODES = ("crop", "fit", "stretch")


def _source_fov_tan(cam: CameraModel, w: int, h: int) -> tuple[float, float]:
    """tan of the widest view angle the source image sees, horizontally and vertically."""
    pts = np.array(
        [[0.0, h / 2], [w - 1.0, h / 2], [w / 2, 0.0], [w / 2, h - 1.0]] #center of pixel
    ).reshape(-1, 1, 2)
    #returns normalized image coordinates - tangent of the angle for those edge pixels
    if cam.model == "fisheye":
        n = cv2.fisheye.undistortPoints(pts, cam.K, cam.D)
    else:
        n = cv2.undistortPoints(pts, cam.K, cam.D.ravel())
    n = n.reshape(-1, 2)
    return max(abs(n[0, 0]), abs(n[1, 0])), max(abs(n[2, 1]), abs(n[3, 1]))


class Preprocessor:
    def __init__(
        self,
        camera: CameraModel,
        src_size: tuple[int, int],   # (width, height) at capture
        dst_size: tuple[int, int],   # (width, height) at model input
        mode: str = "raw",
        fov: str = "crop",
        antialias: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if fov not in FOV_MODES:
            raise ValueError(f"fov must be one of {FOV_MODES}")

        self.camera = camera
        self.mode = mode
        self.src_w, self.src_h = src_size
        self.dst_w, self.dst_h = dst_size
        self.antialias = antialias
        # 4x downscale through a 2x2 bilinear gather aliases badly; pre-blur first
        self._scale = max(self.src_w / self.dst_w, self.src_h / self.dst_h)
        self._xr = self.src_w / self.dst_w
        self._yr = self.src_h / self.dst_h

        self.K_model: np.ndarray | None = None
        if mode == "undistort":
            self.K_model = self._build_k_model(fov)
            self._maps = self._build_undistort_maps()
        else:
            self._maps = self._build_resize_maps()

    def _build_k_model(self, fov: str) -> np.ndarray:
        x_max, y_max = _source_fov_tan(self.camera, self.src_w, self.src_h)
        hw, hh = self.dst_w / 2.0, self.dst_h / 2.0 #cx, cy
        if fov == "stretch":
            fx, fy = hw / x_max, hh / y_max
        else:
            # 'fit' keeps the whole source (blank borders); 'crop' fills the frame
            f = min(hw / x_max, hh / y_max) if fov == "fit" else max(hw / x_max, hh / y_max)
            fx = fy = f
        return np.array([[fx, 0.0, hw], [0.0, fy, hh], [0.0, 0.0, 1.0]])

    def _build_undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        size = (self.dst_w, self.dst_h)
        if self.camera.model == "fisheye":
            return cv2.fisheye.initUndistortRectifyMap(
                self.camera.K, self.camera.D, np.eye(3), self.K_model, size, cv2.CV_32FC1
            )
        return cv2.initUndistortRectifyMap(
            self.camera.K, self.camera.D.ravel(), None, self.K_model, size, cv2.CV_32FC1
        )

    def _build_resize_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Half-pixel-centred resize: src = (dst + 0.5) * ratio - 0.5.
        (only for offline testing)
        """
        xs = (np.arange(self.dst_w, dtype=np.float32) + 0.5) * self._xr - 0.5
        ys = (np.arange(self.dst_h, dtype=np.float32) + 0.5) * self._yr - 0.5
        return np.tile(xs, (self.dst_h, 1)), np.repeat(ys[:, None], self.dst_w, axis=1)

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Raw capture frame -> model input, (dst_h, dst_w, C)."""
        if self.antialias and self._scale > 1.5:
            image = cv2.GaussianBlur(image, (0, 0), 0.5 * self._scale)
        return cv2.remap(image, *self._maps, cv2.INTER_LINEAR)

    def project(self, P_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project (N, 3) camera-frame points into MODEL pixel coordinates."""
        if self.mode == "undistort":
            z = P_cam[:, 2]
            valid = z > 1e-3
            u = np.zeros(len(P_cam))
            v = np.zeros(len(P_cam))
            u[valid] = self.K_model[0, 0] * P_cam[valid, 0] / z[valid] + self.K_model[0, 2]
            v[valid] = self.K_model[1, 1] * P_cam[valid, 1] / z[valid] + self.K_model[1, 2]
            return u, v, valid

        # exact inverse of _build_resize_maps
        u, v, valid = self.camera.project_distorted(P_cam)
        return (u + 0.5) / self._xr - 0.5, (v + 0.5) / self._yr - 0.5, valid
