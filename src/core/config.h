#pragma once
#include <array>
#include <cstdint>
#include <string>

struct VioConfig {
    std::string backend;
    std::string pipe;
    int         min_quality{1};
};

struct InferenceConfig {
    std::string mpa_pipe_name;
    int         input_w{256}, input_h{256};
    std::string preprocess{"undistort"};
    std::string fov{"crop"};
    bool        antialias{false};
};

struct RescaleConfig {
    std::string method{"monotonic_nonsmoothing_spline"};
    int     num_knots_spline{10};
    double  spline_kappa{1.0e6};
    int     min_anchors{10};
    bool    outlier_rejection{false};
    float   outlier_k{3.0f};
    bool    subpixel_2x2{true};
    float   anchor_depth_min{0.05f};
    float   anchor_depth_max{65.0f};
    float   depth_min{0.3f};
    float   depth_max{50.0f};
    int64_t max_hold_age_ns{5'000'000'000LL};
};

struct IntrinsicsConfig {
    int   width{1024}, height{768};
    float fx{593.0478f}, fy{591.7126f}, cx{500.2332f}, cy{388.6283f};
    std::string distortion_model{"pinhole"};
    std::array<float, 5> distortion{0.1388804f, -0.2150293f, 0.0007813f, 0.0022019f, 0.1215502f};
};

struct ExtrinsicsConfig {
    float R[3][3]{};
    float T[3]{};
};

struct AnchorsConfig {
    bool use_tof{true};
    std::string tof_pipe{"tof"};
    int tof_confidence_min{128};
    int tof_max_points{500};
    int64_t tof_tolerance_ns{200'000'000LL};
    std::string projection{"world_pose"};
};

struct OutputConfig {
    std::string pipe{"metric_depth"};
};

struct Config {
    std::string      profile{"openvins"};
    VioConfig        vio;
    InferenceConfig  inference;
    RescaleConfig    rescale;
    AnchorsConfig    anchors;
    IntrinsicsConfig hires;
    ExtrinsicsConfig extr_hires;
    ExtrinsicsConfig extr_tof;
    OutputConfig     output;

    static Config from_yaml(const std::string& pipeline_yaml,
                            const std::string& intrinsics_dir,
                            const std::string& extrinsics_yaml,
                            const std::string& profile_override = "",
                            const std::string& fov_override = "");
};
