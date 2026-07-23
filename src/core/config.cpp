#include "config.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace {
void load_extrinsics(
    const YAML::Node& root, const std::string& name, ExtrinsicsConfig& out) {
    const YAML::Node node = root[name];
    if (!node) {
        throw std::runtime_error("missing extrinsics block: " + name);
    }
    const YAML::Node rotation = node["R_child_to_parent"];
    const YAML::Node translation = node["T_child_wrt_parent"];
    if (!rotation || rotation.size() != 3 || !translation ||
        translation.size() != 3) {
        throw std::runtime_error("invalid extrinsics block: " + name);
    }
    for (int i = 0; i < 3; ++i) {
        if (rotation[i].size() != 3) {
            throw std::runtime_error("invalid rotation in: " + name);
        }
        for (int j = 0; j < 3; ++j) {
            out.R[i][j] = rotation[i][j].as<float>();
        }
        out.T[i] = translation[i].as<float>();
    }
}
}

Config Config::from_yaml(const std::string& pipeline_yaml,
                         const std::string& intrinsics_dir,
                         const std::string& extrinsics_yaml,
                         const std::string& profile_override,
                         const std::string& fov_override) {
    Config c;

    const YAML::Node pipeline = YAML::LoadFile(pipeline_yaml);
    const YAML::Node deployment = pipeline["deployment"];
    c.profile = profile_override.empty()
        ? deployment["profile"].as<std::string>("qvio")
        : profile_override;
    const YAML::Node profile = deployment["profiles"][c.profile];
    if (!profile || (c.profile != "openvins" && c.profile != "qvio")) {
        throw std::runtime_error("unknown deployment profile: " + c.profile);
    }
    c.vio.backend = c.profile;
    c.vio.pipe = profile["vio_pipe"].as<std::string>();
    c.vio.min_quality = profile["min_quality"].as<int>(1);

    const YAML::Node inference = pipeline["inference"];
    c.inference.mpa_pipe_name =
        inference["mpa_pipe_name"].as<std::string>("tflite_disparity");
    const YAML::Node res = inference["input_resolution"];
    c.inference.input_w = res[0].as<int>(256);
    c.inference.input_h = res[1].as<int>(256);
    c.inference.preprocess =
        inference["preprocess"].as<std::string>("undistort");
    c.inference.fov = fov_override.empty()
        ? inference["fov"].as<std::string>("crop")
        : fov_override;
    c.inference.antialias = inference["antialias"].as<bool>(false);
    if (c.inference.input_w <= 0 || c.inference.input_h <= 0 ||
        c.inference.preprocess != "undistort" ||
        (c.inference.fov != "crop" && c.inference.fov != "stretch")) {
        throw std::runtime_error(
            "deployment supports undistort with crop or stretch");
    }

    const YAML::Node anchors = pipeline["anchors"];
    c.anchors.use_tof = anchors["use_tof"].as<bool>(true);
    c.anchors.tof_pipe = anchors["tof_pipe"].as<std::string>("tof");
    c.anchors.tof_confidence_min =
        anchors["tof_confidence_min"].as<int>(128);
    c.anchors.tof_tolerance_ns =
        anchors["tof_tolerance_ms"].as<int64_t>(200) * 1'000'000LL;
    c.anchors.tof_max_points =
        profile["tof_max_points"].as<int>();
    c.anchors.projection =
        profile["projection"].as<std::string>("world_pose");
    if (!c.anchors.use_tof || c.anchors.tof_max_points <= 0 ||
        c.anchors.tof_confidence_min < 0 ||
        c.anchors.tof_confidence_min > 255 ||
        c.anchors.tof_tolerance_ns < 0 ||
        c.anchors.projection != "world_pose") {
        throw std::runtime_error("profile requires ToF and world_pose projection");
    }

    const YAML::Node scale = pipeline["rescale"];
    c.rescale.method =
        scale["method"].as<std::string>("monotonic_nonsmoothing_spline");
    c.rescale.num_knots_spline =
        scale["num_knots_spline"].as<int>(10);
    c.rescale.spline_kappa = scale["spline_kappa"].as<double>(1.0e6);
    c.rescale.min_anchors = scale["min_anchors"].as<int>(13);
    c.rescale.outlier_rejection =
        scale["outlier_rejection"].as<bool>(false);
    c.rescale.outlier_k = scale["outlier_k"].as<float>(3.0f);
    c.rescale.subpixel_2x2 = scale["subpixel_2x2"].as<bool>(true);
    c.rescale.anchor_depth_min =
        scale["anchor_depth_min"].as<float>(0.05f);
    c.rescale.anchor_depth_max =
        scale["anchor_depth_max"].as<float>(65.0f);
    c.rescale.depth_min = scale["depth_min"].as<float>(0.3f);
    c.rescale.depth_max = scale["depth_max"].as<float>(50.0f);
    c.rescale.max_hold_age_ns =
        scale["max_hold_age_ns"].as<int64_t>(5'000'000'000LL);
    if (c.rescale.method != "monotonic_nonsmoothing_spline" ||
        c.rescale.num_knots_spline < 2 || c.rescale.spline_kappa <= 0.0) {
        throw std::runtime_error("invalid deployment rescale configuration");
    }
    if (c.rescale.min_anchors < c.rescale.num_knots_spline + 3 ||
        c.rescale.anchor_depth_min <= 0.0f ||
        c.rescale.anchor_depth_min >= c.rescale.anchor_depth_max ||
        c.rescale.depth_min <= 0.0f ||
        c.rescale.depth_min >= c.rescale.depth_max ||
        c.rescale.max_hold_age_ns < 0) {
        throw std::runtime_error("invalid deployment depth limits");
    }

    c.output.pipe =
        pipeline["output"]["pipe"].as<std::string>("metric_depth");

    const YAML::Node intr = YAML::LoadFile(intrinsics_dir + "/hires.yaml");
    c.hires.width  = intr["width"].as<int>();
    c.hires.height = intr["height"].as<int>();
    c.hires.fx     = intr["fx"].as<float>();
    c.hires.fy     = intr["fy"].as<float>();
    c.hires.cx     = intr["cx"].as<float>();
    c.hires.cy     = intr["cy"].as<float>();
    c.hires.distortion_model = intr["distortion_model"].as<std::string>("pinhole");
    auto dist = intr["distortion"];
    c.hires.distortion.fill(0.0f);
    for (int i = 0;
         i < static_cast<int>(c.hires.distortion.size()) &&
         i < static_cast<int>(dist.size());
         ++i) {
        c.hires.distortion[i] = dist[i].as<float>();
    }

    const YAML::Node extrinsics = YAML::LoadFile(extrinsics_yaml);
    load_extrinsics(extrinsics, "hires", c.extr_hires);
    load_extrinsics(extrinsics, "tof", c.extr_tof);

    return c;
}
