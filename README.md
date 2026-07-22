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
adb push config/ /etc/mono_depth_rescaler/      # pipeline.yaml, intrinsics/, extrinsics/
```

### 3. Configure the disparity producer

Run `voxl-tflite-server` with a depth model publishing FLOAT32 disparity — the
rescaler works with any (MiDaS, ZipDepth, …), it only reads the disparity pipe.
[AI_VOXL2](https://github.com/dineshvarunshankar/AI_VOXL2) provides the
depth-model helpers and disparity output that `voxl-tflite-server` uses. In
`/etc/modalai/voxl-tflite-server.conf` (MiDaS shown):

```
model_architecture: MIDAS_V2
delegate:           gpu
allow_multiple:     false
```

and `publish_disparity: 1` in `/etc/voxl-tflite-server/undistort.yml`. With
`allow_multiple: false` the pipe is `tflite_disparity` (the rescaler default). Its
`fov` must match `inference.fov`, and `inference.input_resolution` must match the
model's output (e.g. 256 for MiDaS, 384 for ZipDepth).

### 4. Run

```bash
mono_depth_rescaler --profile qvio --fov crop
mono_depth_rescaler --profile openvins --fov crop
```

Service (unit defaults to qvio + crop):

```bash
adb push services/mono_depth_rescaler.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now mono_depth_rescaler
```

### 5. Verify

```bash
voxl-inspect-cam tflite_disparity
voxl-inspect-cam metric_depth
systemctl is-active mono_depth_rescaler
```

## Selected configuration

- qVIO profile: `qvio_extended`, ToF cap 500
- OpenVINS profile: `ov_extended`, ToF cap 500
- Undistort with `crop` by default; `stretch` retains full FOV
- Monotonic non-smoothing spline, 10 knots, uniform anchor weights
- MAD outlier rejection (k=3.0); five-second calibration hold

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
