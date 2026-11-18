# mono_depth_rescaler

Convert monocular disparity to metric depth using VIO landmarks and ToF measurements on VOXL 2.
The C++ service is in `src/`. The Python prototype is in `proto/`.

The service reads disparity from `voxl-tflite-server`, VIO from `ov_extended` or `qvio_extended`, and ToF from `tof`.
It publishes depth on `metric_depth`.

## 1. Build

Use the ModalAI SDK and [voxl-cross build environment](https://gitlab.com/voxl-public/voxl-docker) for QRB5165.
From the repository directory on the host:

```bash
voxl-docker -i voxl-cross
./install_build_deps.sh qrb5165 dev
./build.sh qrb5165
./make_package.sh
exit
```

For a 2.x system image, use `qrb5165-2` as the dependency platform.

## 2. Install

On the host:

```bash
adb push mono-depth-rescaler_0.1.0_arm64.deb /tmp/
adb shell dpkg -i /tmp/mono-depth-rescaler_0.1.0_arm64.deb
```

The package installs the binary, configuration, and systemd service.

## 3. Configure

Edit `/etc/mono_depth_rescaler/pipeline.yaml` on the drone:

```yaml
deployment:
  profile: openvins
```

Use `qvio` for `voxl-qvio-server`. Enable the VIO service for the selected profile.
Enable ToF in `voxl-camera-server`.

Configure `voxl-tflite-server` to publish FLOAT32 relative disparity on `tflite_disparity`.
Use `allow_multiple: false` and `publish_disparity: 1` in the corresponding producer configuration files.
Match the camera calibration and `fov` between the producer and rescaler.
See [AI_VOXL2 configuration](https://github.com/dineshvarunshankar/AI_VOXL2#3-configure) for the deployment settings.

The service reads frame dimensions from each disparity packet.

| Output setting | Values |
|---|---|
| `size` | `0` preserves model resolution; a smaller `N` pools disparity to `N×N` |
| `format` | `float32` in metres or `uint16_mm` in millimetres, capped at 65.535 m |
| `viz` | `1` enables the `metric_depth_viz` colour image |

Defaults use the hires camera, crop undistortion, and a monotonic spline with 10 knots.
MAD rejection uses `k=3`. Calibration remains available for up to five seconds without a new fit.
ToF anchors use a four-pixel grid; VIO anchors have no additional grid cap.

Pose-based projection requires an OK VIO state and a pose within the timestamp tolerance.
With qVIO tracking cameras, native image coordinates and feature depths can supply anchors without a valid pose.

## 4. Run

On the drone, start the camera, disparity producer, and selected VIO service. Then start the rescaler:

```bash
systemctl start mono_depth_rescaler
systemctl status mono_depth_rescaler
```

Inspect each pipe separately:

```bash
voxl-inspect-cam tflite_disparity
voxl-inspect-cam metric_depth
```

For a manual profile override:

```bash
systemctl stop mono_depth_rescaler
mono_depth_rescaler --profile openvins --fov crop
```

## Local tests

Python prototype:

```bash
uv sync
uv run pytest -q
```

C++ core, with OpenCV and CMake installed:

```bash
cmake -S . -B build -DBUILD_VOXL_APP=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
