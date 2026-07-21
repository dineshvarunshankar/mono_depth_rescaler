#include "core/camera_model.h"
#include "core/config.h"
#include "core/fits.h"
#include "core/pipeline.h"
#include "core/preprocess.h"
#include "core/tof_anchors.h"
#include "io/image_packet.h"
#include "tof/tof_types.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void identity(ExtrinsicsConfig& extrinsics) {
    for (int i = 0; i < 3; ++i) {
        extrinsics.T[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            extrinsics.R[i][j] = i == j ? 1.0f : 0.0f;
        }
    }
}

void identity_pose(float T[3], float R[3][3]) {
    for (int i = 0; i < 3; ++i) {
        T[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            R[i][j] = i == j ? 1.0f : 0.0f;
        }
    }
}

Config synthetic_config() {
    Config cfg;
    cfg.hires.width = 100;
    cfg.hires.height = 100;
    cfg.hires.fx = 50.0f;
    cfg.hires.fy = 50.0f;
    cfg.hires.cx = 50.0f;
    cfg.hires.cy = 50.0f;
    cfg.hires.distortion.fill(0.0f);
    cfg.inference.input_w = 100;
    cfg.inference.input_h = 100;
    cfg.inference.preprocess = "raw";
    cfg.inference.fov = "crop";
    cfg.inference.antialias = false;
    cfg.rescale.method = "polynomial";
    cfg.rescale.min_anchors = 2;
    cfg.rescale.subpixel_2x2 = false;
    cfg.anchors.tof_max_points = 3;
    identity(cfg.extr_hires);
    identity(cfg.extr_tof);
    return cfg;
}

void test_profiles() {
    const std::string root = PROJECT_SOURCE_DIR;
    Config openvins = Config::from_yaml(
        root + "/config/pipeline.yaml", root + "/config/intrinsics",
        root + "/config/extrinsics/starling2.yaml");
    assert(openvins.profile == "openvins");
    assert(openvins.vio.pipe == "ov_extended");
    assert(openvins.anchors.tof_max_points == 500);

    Config qvio = Config::from_yaml(
        root + "/config/pipeline.yaml", root + "/config/intrinsics",
        root + "/config/extrinsics/starling2.yaml", "qvio", "stretch");
    assert(qvio.vio.pipe == "qvio_extended");
    assert(qvio.anchors.tof_max_points == 500);
    assert(qvio.inference.fov == "stretch");
    assert(qvio.rescale.outlier_rejection);
    assert(qvio.rescale.outlier_k == 3.0f);

    for (const std::string& profile : {"openvins", "qvio"}) {
        for (const std::string& fov : {"crop", "stretch"}) {
            Config resolved = Config::from_yaml(
                root + "/config/pipeline.yaml", root + "/config/intrinsics",
                root + "/config/extrinsics/starling2.yaml", profile, fov);
            assert(resolved.profile == profile);
            assert(resolved.inference.fov == fov);
        }
    }
    bool rejected = false;
    try {
        Config::from_yaml(
            root + "/config/pipeline.yaml", root + "/config/intrinsics",
            root + "/config/extrinsics/starling2.yaml", "invalid", "crop");
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

void test_tof_decode() {
    std::vector<uint8_t> packet(TOF_PACKET_BYTES, 0);
    const int64_t timestamp = 123456789;
    const float point[3] = {1.0f, 2.0f, 3.0f};
    const float noise = 0.25f;
    std::memcpy(packet.data() + 4, &timestamp, sizeof(timestamp));
    std::memcpy(packet.data() + 20, point, sizeof(point));
    std::memcpy(
        packet.data() + 20 + TOF_POINT_COUNT * 3 * sizeof(float),
        &noise, sizeof(noise));
    packet.back() = 200;
    TofFrame frame = decode_tof_packet(packet.data(), packet.size());
    assert(frame.timestamp_ns == timestamp);
    assert(frame.points[0].z == 3.0f);
    assert(frame.noise[0] == noise);
    assert(frame.confidence.back() == 200);
}

void test_tof_projection_and_cap() {
    Config cfg = synthetic_config();
    CameraModel camera(
        cfg.hires.fx, cfg.hires.fy, cfg.hires.cx, cfg.hires.cy,
        cfg.hires.distortion, cfg.hires.distortion_model);
    Preprocessor pre(
        camera, 100, 100, 100, 100, "raw", "crop", false);
    TofFrame frame;
    frame.points.assign(10, TofPoint{0.0f, 0.0f, 2.0f});
    frame.noise.assign(10, 0.02f);
    frame.confidence.assign(10, 255);
    float T[3], R[3][3];
    identity_pose(T, R);
    ProjectedAnchors anchors = project_tof_anchors(
        frame, T, R, T, R, cfg.extr_tof, cfg.extr_hires, pre, 128, 3);
    assert(anchors.depth.size() == 3);
    for (size_t i = 0; i < anchors.depth.size(); ++i) {
        assert(std::abs(anchors.u[i] - 50.0f) < 1e-5f);
        assert(std::abs(anchors.depth[i] - 2.0f) < 1e-5f);
    }
}

void test_tof_projection_python_golden() {
    const std::string root = PROJECT_SOURCE_DIR;
    Config cfg = Config::from_yaml(
        root + "/config/pipeline.yaml", root + "/config/intrinsics",
        root + "/config/extrinsics/starling2.yaml");
    CameraModel camera(
        cfg.hires.fx, cfg.hires.fy, cfg.hires.cx, cfg.hires.cy,
        cfg.hires.distortion, cfg.hires.distortion_model);
    Preprocessor pre(
        camera, cfg.hires.width, cfg.hires.height,
        cfg.inference.input_w, cfg.inference.input_h,
        cfg.inference.preprocess, cfg.inference.fov, cfg.inference.antialias);

    const float pi = std::acos(-1.0f);
    const float tof_angle = 7.0f * pi / 180.0f;
    const float image_angle = -4.0f * pi / 180.0f;
    float tof_T[3] = {0.2f, -0.1f, 0.05f};
    float tof_R[3][3] = {
        {std::cos(tof_angle), -std::sin(tof_angle), 0.0f},
        {std::sin(tof_angle), std::cos(tof_angle), 0.0f},
        {0.0f, 0.0f, 1.0f}};
    float image_T[3] = {0.15f, -0.08f, 0.03f};
    float image_R[3][3] = {
        {std::cos(image_angle), 0.0f, std::sin(image_angle)},
        {0.0f, 1.0f, 0.0f},
        {-std::sin(image_angle), 0.0f, std::cos(image_angle)}};
    TofFrame frame;
    frame.points = {{0.1f, -0.05f, 2.3f}};
    frame.noise = {0.02f};
    frame.confidence = {255};
    ProjectedAnchors anchors = project_tof_anchors(
        frame, tof_T, tof_R, image_T, image_R, cfg.extr_tof,
        cfg.extr_hires, pre, 128, 500);
    assert(anchors.depth.size() == 1);
    assert(std::abs(anchors.u[0] - 155.81617453f) < 1e-3f);
    assert(std::abs(anchors.v[0] - 121.35762648f) < 1e-3f);
    assert(std::abs(anchors.depth[0] - 2.35388282f) < 1e-4f);
}

void test_spline() {
    std::vector<double> x, y, w;
    for (int i = 0; i < 30; ++i) {
        const double value = 0.05 + i * 0.01;
        x.push_back(value);
        y.push_back(0.1 + 2.0 * value);
        w.push_back(1.0);
    }
    fits::Fit fit = fits::create(
        "monotonic_nonsmoothing_spline", x, y, w, 1, 10, 1.0e6);
    assert(fit.valid);
    const double query[] = {0.08, 0.14, 0.20, 0.28, 0.33};
    const double python_golden[] = {
        0.2599999268, 0.3799999816, 0.4999999703, 0.6600003197,
        0.7600013423};
    for (size_t i = 0; i < std::size(query); ++i) {
        assert(std::abs(fit.predict(query[i]) - python_golden[i]) < 2e-5);
    }
}

void test_pipeline_union_and_hold() {
    Config cfg = synthetic_config();
    Pipeline pipeline(cfg);
    float T[3], R[3][3];
    identity_pose(T, R);

    ext_vio_data_t packet{};
    packet.n_total_features = 5;
    packet.v.timestamp_ns = 1'000'000'000;
    std::vector<float> disparity(100 * 100, 0.2f);
    for (int i = 0; i < 5; ++i) {
        const float depth = 1.0f + i;
        const int u = 20 + i * 15;
        packet.features[i].point_quality = VIO_POINT_HIGH;
        packet.features[i].tsf[0] = (u - 50.0f) * depth / 50.0f;
        packet.features[i].tsf[1] = 0.0f;
        packet.features[i].tsf[2] = depth;
        for (int dv = -1; dv <= 1; ++dv) {
            for (int du = -1; du <= 1; ++du) {
                disparity[(50 + dv) * 100 + u + du] = 1.0f / depth;
            }
        }
    }

    TofFrame tof;
    tof.points.assign(10, TofPoint{0.0f, 0.0f, 2.0f});
    tof.noise.assign(10, 0.02f);
    tof.confidence.assign(10, 255);
    ProjectedAnchors combined = pipeline.build_anchors(
        packet, T, R, &tof, T, R);
    assert(combined.depth.size() == 8);

    auto fresh = pipeline.process(
        1'000'000'000, packet, T, R, nullptr, T, R, disparity);
    assert(fresh && !fresh->held && fresh->n_anchors == 5);

    ext_vio_data_t empty{};
    auto held = pipeline.process(
        2'000'000'000, empty, T, R, nullptr, T, R, disparity);
    assert(held && held->held);
    auto stale = pipeline.process(
        8'000'000'000, empty, T, R, nullptr, T, R, disparity);
    assert(!stale);
}

void test_image_packet() {
    ImageMetadata metadata{};
    metadata.timestamp_ns = 42;
    metadata.exposure_ns = 100;
    std::vector<float> pixels(12, 2.0f);
    std::vector<uint8_t> packet =
        make_float_image_packet(metadata, pixels, 4, 3);
    assert(packet.size() == sizeof(ImageMetadata) + pixels.size() * sizeof(float));
    ImageMetadata decoded{};
    std::memcpy(&decoded, packet.data(), sizeof(decoded));
    assert(decoded.timestamp_ns == 42);
    assert(decoded.width == 4 && decoded.height == 3);
    assert(decoded.format == IMAGE_FORMAT_FLOAT32_VALUE);
    assert(decoded.size_bytes == 48);
}
}

int main() {
    test_profiles();
    test_tof_decode();
    test_tof_projection_and_cap();
    test_tof_projection_python_golden();
    test_spline();
    test_pipeline_union_and_hold();
    test_image_packet();
    std::cout << "rescaler_core_tests passed\n";
    return 0;
}
