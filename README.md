# mono_depth_rescaler

Converts relative monocular disparity to metric depth using VIO landmarks and
ToF points, on VOXL 2 (Starling 2). The Python implementation in `proto/` is the
reference; `src/` is the C++ deploy build. Both use `config/pipeline.yaml`.

The rescaler consumes three MPA pipes — relative disparity (from
`voxl-tflite-server`), VIO (`qvio_extended` or `ov_extended`), and ToF (`tof`) —
and publishes metric depth on `metric_depth`. It runs no model itself.

## Requirements

- VOXL 2 (QRB5165), e.g. Starling 2, with the ModalAI SDK image
- The `voxl-cross` build container (see [step 1](#1-cross-compile))
- Running on the drone: `voxl-tflite-server` (disparity), a VIO server
  (`voxl-qvio-server` or `voxl-open-vins-server`), and `voxl-camera-server` with ToF enabled

## Deploy on VOXL 2

### 1. Cross-compile

`voxl-docker` is the host wrapper that launches the `voxl-cross` image (cross
compiler + VOXL SDK). Install both per ModalAI's
[voxl-docker guide](https://gitlab.com/voxl-public/voxl-docker), then:

```bash
voxl-docker -i voxl-cross                    # enter the build container
./install_build_deps.sh qrb5165 dev          # or qrb5165-2 for 2.x images
./build.sh qrb5165                           # ./build.sh native for host build + tests
./make_package.sh                            # build the .deb (qrb5165 build only)
```

### 2. Install on the drone

The `.deb` carries the binary, service, camera drop-in, and config; its postinst runs
`daemon-reload` and enables the service.

```bash
adb push mono-depth-rescaler_0.1.0_arm64.deb /tmp/
adb shell dpkg -i /tmp/mono-depth-rescaler_0.1.0_arm64.deb
adb shell systemctl start mono_depth_rescaler
adb shell systemctl status mono_depth_rescaler
```

The service runs `/usr/bin/mono_depth_rescaler` with **no profile/fov flags**.
Choose the VIO backend in `/etc/mono_depth_rescaler/pipeline.yaml`:

```yaml
deployment:
  profile: qvio      # or openvins
```

Also enable the matching VIO service (`voxl-qvio-server` or `voxl-open-vins-server`).
`inference.fov` in the same YAML sets crop/stretch.

For a temporary manual override (debug only), stop the service first:

```bash
systemctl stop mono_depth_rescaler
mono_depth_rescaler --profile openvins --fov crop
```

### 3. Configure the disparity producer

Run `voxl-tflite-server` with a depth model publishing FLOAT32 disparity — the
rescaler works with any (MiDaS, ZipDepth, …), it only reads the disparity pipe.
See [AI_VOXL2](https://github.com/dineshvarunshankar/AI_VOXL2) §4.4 for the
tflite-server config and patch guide.

Match on both sides: `allow_multiple: false` -> pipe `tflite_disparity` (rescaler
default), `publish_disparity: 1`, and `fov` / `inference.input_resolution`
(256 for MiDaS).

### 4. Verify

```bash
voxl-inspect-cam tflite_disparity
voxl-inspect-cam metric_depth
systemctl is-active mono_depth_rescaler
```

## Selected configuration

- Default YAML profile: `qvio` (`qvio_extended`); switch to `openvins` in YAML for `ov_extended`
- ToF anchors: image-space grid (`tof_cell_px=4`, nearest per cell); VIO uncapped
- Undistort `crop` by default (`stretch` keeps full FOV)
- Monotonic non-smoothing spline, 10 knots, uniform anchor weights
- MAD outlier rejection (k=3.0); five-second calibration hold
- VIO features used only when `v.state == VIO_STATE_OK` (qVIO and OpenVINS)

`config/pipeline.yaml` is shared by Python and C++.

## Tests

```bash
uv sync
uv run pytest -q
cmake -S . -B build -DBUILD_VOXL_APP=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The local build tests the platform-independent C++ core. The VOXL executable is
built automatically when the ModalAI SDK provides `modal_pipe`.
