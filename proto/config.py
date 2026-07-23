"""Load and assemble typed configuration from YAML files."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Typed config objects
# ---------------------------------------------------------------------------

@dataclass
class IntrinsicsCfg:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion: list[float]

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0,    1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.array(self.distortion, dtype=np.float64)


@dataclass
class ExtrinsicsCfg:
    # P_imu = R @ P_cam + T   (R_child_to_parent, T_child_wrt_parent)
    R: np.ndarray   # (3, 3)
    T: np.ndarray   # (3,)


@dataclass
class VioCfg:
    backend: str
    pipe: str
    min_quality: int


@dataclass
class InferenceCfg:
    source: str
    mpa_pipe_name: str
    model_path: str
    backend_delegate: str
    input_resolution: tuple[int, int]
    preprocess: str      # raw | undistort
    fov: str             # crop | fit | stretch  (undistort only)
    antialias: bool


@dataclass
class RescaleCfg:
    method: str
    args: dict
    min_anchors: int
    weighting: str        # none | stddev | covariance
    outlier_rejection: bool
    outlier_k: float
    subpixel_2x2: bool
    anchor_depth_min: float
    anchor_depth_max: float
    depth_min: float
    depth_max: float
    max_hold_age_ns: int


@dataclass
class AnchorsCfg:
    use_tof: bool             # ON = VIO + quality ToF anchors; OFF = VIO only
    tof_pipe: str
    tof_confidence_min: int   # gate: drop ToF points below this confidence
    tof_max_points: int       # compute cap; spatially distributed points retained
    projection: str = "world_pose"   # world_pose | tracking_extrinsic
    balance_mode: str = "off"        # off | adaptive_source | adaptive_count
    balance_vio_fraction: float = 0.5
    balance_vio_sufficient: int = 10
    balance_tof_fallback: int = 64
    tof_tolerance_ms: int = 200


@dataclass
class SmootherCfg:
    enable: bool
    type: str
    ema_alpha: float
    kalman_process_noise: float
    kalman_measurement_noise: float


@dataclass
class RollingShutterCfg:
    enable: bool
    readout_time_us: float
    readout_direction: str


@dataclass
class MultiFrameCfg:
    enable: bool
    n_frames: int


@dataclass
class ModulesCfg:
    rolling_shutter: RollingShutterCfg
    multi_frame_window: MultiFrameCfg


@dataclass
class Config:
    hires: IntrinsicsCfg
    extr_hires: ExtrinsicsCfg
    extr_tof: ExtrinsicsCfg
    vio: VioCfg
    inference: InferenceCfg
    rescale: RescaleCfg
    smoother: SmootherCfg
    modules: ModulesCfg
    anchors: AnchorsCfg
    profile: str = "qvio"
    # tracking-camera intrinsics (fisheye) for the tracking_extrinsic anchor
    # projection; optional so manually-built Configs need not supply them.
    tracking_front: IntrinsicsCfg | None = None
    tracking_down: IntrinsicsCfg | None = None
    extr_tracking_front: ExtrinsicsCfg | None = None
    extr_tracking_down: ExtrinsicsCfg | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _parse_intrinsics(d: dict) -> IntrinsicsCfg:
    return IntrinsicsCfg(
        width=d["width"], height=d["height"],
        fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
        distortion_model=d["distortion_model"],
        distortion=d["distortion"],
    )


def _parse_extrinsics(d: dict) -> ExtrinsicsCfg:
    return ExtrinsicsCfg(
        R=np.array(d["R_child_to_parent"], dtype=np.float64),
        T=np.array(d["T_child_wrt_parent"], dtype=np.float64),
    )


def load(
    pipeline_yaml: str | Path = "config/pipeline.yaml",
    intrinsics_dir: str | Path = "config/intrinsics",
    extrinsics_yaml: str | Path = "config/extrinsics/starling2.yaml",
    profile: str | None = None,
    fov: str | None = None,
) -> Config:
    pipeline_yaml = Path(pipeline_yaml)
    intrinsics_dir = Path(intrinsics_dir)
    extrinsics_yaml = Path(extrinsics_yaml)

    p = _load_yaml(pipeline_yaml)
    intr = _parse_intrinsics(_load_yaml(intrinsics_dir / "hires.yaml"))
    def _opt_intr(name):
        p = intrinsics_dir / name
        return _parse_intrinsics(_load_yaml(p)) if p.exists() else None
    tf_intr = _opt_intr("tracking_front.yaml")
    td_intr = _opt_intr("tracking_down.yaml")
    extr_all = _load_yaml(extrinsics_yaml)
    extr = _parse_extrinsics(extr_all["hires"])
    extr_tof = _parse_extrinsics(extr_all["tof"])

    deployment = p["deployment"]
    selected_profile = profile or deployment.get("profile", "qvio")
    profiles = deployment["profiles"]
    if selected_profile not in profiles:
        raise ValueError(f"unknown deployment profile: {selected_profile}")
    v = profiles[selected_profile]
    inf = p["inference"]
    r = p["rescale"]
    s = p.get("smoother", {})
    m = p.get("modules", {})
    an = p.get("anchors", {})
    selected_fov = fov or inf.get("fov", "crop")
    if selected_fov not in ("crop", "stretch"):
        raise ValueError(f"unsupported deployment fov: {selected_fov}")
    fit_args = {
        "degree": 1,
        "num_knots_spline": r.get("num_knots_spline", 10),
        "lambda_smoothing": 1.0e5,
        "spline_kappa": r.get("spline_kappa", 1.0e6),
        "poly_monotonic_penalty": 1.0e8,
    }
    rolling = m.get("rolling_shutter", {})
    multi_frame = m.get("multi_frame_window", {})

    return Config(
        hires=intr,
        extr_hires=extr,
        extr_tof=extr_tof,
        vio=VioCfg(
            backend=selected_profile,
            pipe=v["vio_pipe"],
            min_quality=v.get("min_quality", 1),
        ),
        inference=InferenceCfg(
            source="mpa_pipe",
            mpa_pipe_name=inf["mpa_pipe_name"],
            model_path="models/midas-tflite-w8a8/midas.tflite",
            backend_delegate="cpu",
            input_resolution=tuple(inf["input_resolution"]),
            preprocess=inf["preprocess"],
            fov=selected_fov,
            antialias=inf["antialias"],
        ),
        rescale=RescaleCfg(
            method=r["method"],
            args=fit_args,
            min_anchors=r["min_anchors"],
            weighting="none",
            outlier_rejection=r["outlier_rejection"],
            outlier_k=r["outlier_k"],
            subpixel_2x2=r["subpixel_2x2"],
            anchor_depth_min=r["anchor_depth_min"],
            anchor_depth_max=r["anchor_depth_max"],
            depth_min=r["depth_min"],
            depth_max=r["depth_max"],
            max_hold_age_ns=r["max_hold_age_ns"],
        ),
        smoother=SmootherCfg(
            enable=s.get("enable", False),
            type=s.get("type", "ema"),
            ema_alpha=s.get("ema_alpha", 0.4),
            kalman_process_noise=s.get("kalman_process_noise", 1.0e-4),
            kalman_measurement_noise=s.get("kalman_measurement_noise", 1.0e-2),
        ),
        modules=ModulesCfg(
            rolling_shutter=RollingShutterCfg(
                enable=rolling.get("enable", False),
                readout_time_us=rolling.get("readout_time_us", 0),
                readout_direction=rolling.get("readout_direction", "top_to_bottom"),
            ),
            multi_frame_window=MultiFrameCfg(
                enable=multi_frame.get("enable", False),
                n_frames=multi_frame.get("n_frames", 1),
            ),
        ),
        anchors=AnchorsCfg(
            use_tof=an.get("use_tof", False),
            tof_pipe=an.get("tof_pipe", "tof"),
            tof_confidence_min=an.get("tof_confidence_min", 128),
            tof_max_points=v["tof_max_points"],
            projection=v.get("projection", "world_pose"),
            balance_mode=an.get("balance_mode", "off"),
            balance_vio_fraction=an.get("balance_vio_fraction", 0.5),
            balance_vio_sufficient=an.get("balance_vio_sufficient", 10),
            balance_tof_fallback=an.get("balance_tof_fallback", 64),
            tof_tolerance_ms=an.get("tof_tolerance_ms", 200),
        ),
        profile=selected_profile,
        tracking_front=tf_intr,
        tracking_down=td_intr,
        extr_tracking_front=_parse_extrinsics(extr_all["tracking_front"]) if "tracking_front" in extr_all else None,
        extr_tracking_down=_parse_extrinsics(extr_all["tracking_down"]) if "tracking_down" in extr_all else None,
    )
