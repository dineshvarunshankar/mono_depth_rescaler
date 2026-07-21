"""Fisheye / pinhole lens model.

Holds a camera's intrinsics (K), distortion (D), and lens type, and projects
3D camera-frame points into the raw distorted image. Image preparation and the
matching pinhole projection into the model grid live in preprocess.Preprocessor.
"""
from __future__ import annotations
import cv2
import numpy as np


class CameraModel:
    def __init__(self, K: np.ndarray, D: np.ndarray, model: str = "fisheye"):
        """
        K:     (3, 3) camera matrix at capture resolution.
        D:     distortion coefficients — (k1,k2,k3,k4) for fisheye,
               (k1,k2,p1,p2) for pinhole.
        model: "fisheye" | "pinhole"
        """
        self.K = K.astype(np.float64)
        self.D = D.astype(np.float64).reshape(-1, 1)
        self.model = model

    def project_distorted(self, P_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project (N, 3) camera-frame points into the raw distorted image.

        Returns (u, v, valid); valid masks points in front of the camera.
        (Points behind the camera can yield finite pixel coordinates)
        """
        z = P_cam[:, 2]
        valid = z > 1e-3
        u = np.zeros(len(P_cam), dtype=np.float64)
        v = np.zeros(len(P_cam), dtype=np.float64)
        if valid.any():
            pts = P_cam[valid].reshape(-1, 1, 3)
            zero = np.zeros(3)                      # points are already in camera frame
            if self.model == "fisheye":
                px, _ = cv2.fisheye.projectPoints(pts, zero, zero, self.K, self.D)      # Kannala-Brandt
            else:
                px, _ = cv2.projectPoints(pts, zero, zero, self.K, self.D.ravel())      # pinhole
            u[valid] = px[:, 0, 0]
            v[valid] = px[:, 0, 1]
        return u, v, valid
