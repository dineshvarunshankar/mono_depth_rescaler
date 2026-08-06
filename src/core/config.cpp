#include "config.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>

namespace {
void load_intrinsics(const std::string& path, IntrinsicsConfig& out) {
    const YAML::Node n = YAML::LoadFile(path);
    out.width  = n["width"].as<int>();
    out.height = n["height"].as<int>();
    out.fx     = n["fx"].as<float>();
    out.fy     = n["fy"].as<float>();
    out.cx     = n["cx"].as<float>();
    out.cy     = n["cy"].as<float>();
    out.distortion_model = n["distortion_model"].as<std::string>("pinhole");
    const YAML::Node dist = n["distortion"];
    out.distortion.fill(0.0f);
    for (int i = 0;
         i < static_cast<int>(out.distortion.size()) &&
         i < static_cast<int>(dist.size());
         ++i) {
        out.distortion[i] = dist[i].as<float>();
    }
}

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

bool load_optional_camera(const std::string& intrinsics_path,
                          const YAML::Node& extrinsics, const std::string& name,
                          IntrinsicsConfig& intr, ExtrinsicsConfig& extr) {
    if (!std::filesystem::exists(intrinsics_path) || !extrinsics[name]) {
        return false;
    }
    load_intrinsics(intrinsics_path, intr);
    load_extrinsics(extrinsics, name, extr);
    return true;
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
    c.inference.camera = inference["camera"].as<std::string>("hires");
    if (c.inference.camera != "hires" &&
        c.inference.camera != "tracking_front" &&
        c.inference.camera != "tracking_down") {
        throw std::runtime_error("unknown inference camera: " + c.inference.camera);
    }
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
        anchors["tof_tolerance_ms"].as<int64_t>(250) * 1'000'000LL;
    c.anchors.feature_tol_ns =
        anchors["feature_tol_ms"].as<int64_t>(100) * 1'000'000LL;
    c.anchors.tof_cell_px = anchors["tof_cell_px"].as<int>(4);
    c.anchors.max_per_cell = anchors["max_per_cell"].as<int>(1);
    c.anchors.tof_cell_pick = anchors["tof_cell_pick"].as<std::string>("nearest");
    c.anchors.tof_trust_range_m = anchors["tof_trust_range_m"].as<float>(10.0f);
    c.anchors.tof_max_points =
        profile["tof_max_points"].as<int>();
    c.anchors.projection =
        profile["projection"].as<std::string>("world_pose");
    if (!c.anchors.use_tof || c.anchors.tof_max_points <= 0 ||
        c.anchors.tof_confidence_min < 0 ||
        c.anchors.tof_confidence_min > 255 ||
        c.anchors.tof_tolerance_ns < 0 ||
        c.anchors.feature_tol_ns <= 0 ||
        c.anchors.tof_cell_px < 0 || c.anchors.max_per_cell < 1 ||
        c.anchors.tof_trust_range_m <= 0.0f ||
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

    load_intrinsics(intrinsics_dir + "/hires.yaml", c.hires);

    const YAML::Node extrinsics = YAML::LoadFile(extrinsics_yaml);
    load_extrinsics(extrinsics, "hires", c.extr_hires);
    load_extrinsics(extrinsics, "tof", c.extr_tof);

    c.has_tracking_front = load_optional_camera(
        intrinsics_dir + "/tracking_front.yaml", extrinsics, "tracking_front",
        c.tracking_front, c.extr_tracking_front);
    c.has_tracking_down = load_optional_camera(
        intrinsics_dir + "/tracking_down.yaml", extrinsics, "tracking_down",
        c.tracking_down, c.extr_tracking_down);

    if ((c.inference.camera == "tracking_front" && !c.has_tracking_front) ||
        (c.inference.camera == "tracking_down" && !c.has_tracking_down)) {
        throw std::runtime_error("no calibration for camera " + c.inference.camera);
    }

    return c;
}
