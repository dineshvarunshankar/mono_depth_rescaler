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
```

### 2. Install on the drone

```bash
adb push build/mono_depth_rescaler /usr/bin/
adb push config/ /etc/mono_depth_rescaler/
adb push services/mono_depth_rescaler.service /etc/systemd/system/
adb shell mkdir -p /etc/systemd/system/voxl-tflite-server.service.d \
  /etc/systemd/system/voxl-camera-server.service.d
adb push services/voxl-tflite-server.service.d/partof-camera.conf \
  /etc/systemd/system/voxl-tflite-server.service.d/
adb push services/voxl-camera-server.service.d/wants-depth.conf \
  /etc/systemd/system/voxl-camera-server.service.d/

adb shell
systemctl daemon-reload
systemctl enable --now mono_depth_rescaler
systemctl status mono_depth_rescaler
```

The service runs `/usr/bin/mono_depth_rescaler` with **no profile/fov flags**.
Choose the VIO backend in `/etc/mono_depth_rescaler/pipeline.yaml`:

```yaml
deployment:
  profile: qvio      # or openvins
```

Also enable the matching VIO service (`voxl-qvio-server` or `voxl-open-vins-server`).
`inference.fov` in the same YAML sets crop/stretch.

Camera lifecycle recovery:
- `PartOf=voxl-camera-server` on the rescaler and on tflite (drop-in) so both
  restart when the camera server restarts.
- Camera drop-in `Wants=` tflite + rescaler so starting camera brings them back.
- The binary exits non-zero on disparity/VIO disconnect or failed pipe open so
  `Restart=always` can recover sticky disconnects.

For a temporary manual override (debug only), stop the service first:

```bash
systemctl stop mono_depth_rescaler
mono_depth_rescaler --profile openvins --fov crop
```

### 3. Configure the disparity producer

Run `voxl-tflite-server` with a depth model publishing FLOAT32 disparity — the
rescaler works with any (MiDaS, ZipDepth, …), it only reads the disparity pipe.
[AI_VOXL2](https://github.com/dineshvarunshankar/AI_VOXL2) §4.4 has the full patch
guide. Starling 2 MiDaS example:

`/etc/modalai/voxl-tflite-server.conf`:

```
model_architecture: MIDAS_V2
delegate:           gpu
allow_multiple:     false
skip_frames:        0
input_pipe:         /run/mpa/hires_small_color/
```

`/etc/voxl-tflite-server/undistort.yml`: `publish_image: 0`, `publish_disparity: 1`,
`fov: crop`. Then `allow_multiple: false` → pipe `tflite_disparity` (rescaler
default). Match `fov` and `inference.input_resolution` (256 for MiDaS) on both sides.

### 4. Verify

```bash
voxl-inspect-cam tflite_disparity
voxl-inspect-cam metric_depth
systemctl is-active mono_depth_rescaler
```

## Selected configuration

- Default YAML profile: `qvio` (`qvio_extended`); switch to `openvins` in YAML for `ov_extended`
- ToF cap 500; undistort `crop` by default (`stretch` keeps full FOV)
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
