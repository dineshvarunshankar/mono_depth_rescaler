"""Run MiDaS on captured frames. MidasTflite (drone .tflite), MidasTorch (PC hub);
both return (H_model, W_model) float32 relative disparity."""
from __future__ import annotations
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ..rescaler.preprocess import Preprocessor


def _check_output_grid(shape: tuple[int, int], pre: Preprocessor) -> None:
    """The map must land on the grid anchors are projected into.

    Both sides are fixed before the first frame (the model file and
    inference.input_resolution)
    """
    if shape != (pre.dst_h, pre.dst_w):
        raise ValueError(
            f"model outputs {shape}, config expects {(pre.dst_h, pre.dst_w)}; "
            f"set inference.input_resolution to match the model"
        )


class MidasTflite:
    """Quantised MiDaS via TFLite."""

    def __init__(self, model_path: str, pre: Preprocessor):
        self._pre = pre
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()  # calculate and freeze the memory layout for all graph layers
        self._interp = interp
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        self._in_scale, self._in_zp = inp["quantization"]
        self._out_scale, self._out_zp = out["quantization"]
        self._in_idx = inp["index"]
        self._out_idx = out["index"]

        # set_tensor() enforces the input shape; nothing enforces the output one.
        _, h, w, _ = out["shape"]
        _check_output_grid((int(h), int(w)), pre)

    def disparity(self, rgb: np.ndarray) -> np.ndarray:
        x = self._pre.prepare(rgb)
        # quantise exactly as the drone does
        q = np.clip(
            np.round(x.astype(np.float32) / 255.0 / self._in_scale + self._in_zp), 0, 255
        ).astype(np.uint8)
        # q[None]: (H, W, 3) -> (1, H, W, 3); the model expects a batch dimension
        self._interp.set_tensor(self._in_idx, q[None])
        self._interp.invoke()
        # [0, ..., 0]: (1, H, W, 1) -> (H, W); single batch, single channel
        qo = self._interp.get_tensor(self._out_idx)[0, ..., 0].astype(np.float32)
        return (qo - self._out_zp) * self._out_scale


class MidasTorch:
    """Float MiDaS via torch.hub."""

    def __init__(self, pre: Preprocessor):
        import torch

        self._torch = torch
        self._pre = pre
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self._device).eval()
        self._transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # probe the output grid by running once (torch hub sets its own resolution)
        probe = self.disparity(np.zeros((pre.src_h, pre.src_w, 3), np.uint8))
        _check_output_grid(probe.shape, pre)

    def disparity(self, rgb: np.ndarray) -> np.ndarray:
        x = self._pre.prepare(rgb)
        t = self._transform(x).to(self._device)  # to tensor, normalise, resize, add batch dim
        with self._torch.no_grad():
            pred = self._model(t).squeeze()
        return pred.cpu().numpy().astype(np.float32)
