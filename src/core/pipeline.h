#pragma once
#include "config.h"
#include "camera_model.h"
#include "fits.h"
#include "geometry.h"
#include "preprocess.h"
#include "../tof/tof_types.h"
#include "../vio/vio_types.h"
#include <memory>
#include <vector>

struct RescaleResult {
    std::vector<float>  depth;         // (H_model x W_model) metric depth, metres
    std::vector<double> params;        // fitted curve params (empty for splines)
    int     n_anchors{0};              // 0 on a held frame (no fresh fit)
    float   inlier_ratio{1.0f};
    bool    held{false};               // reusing an earlier fit, not fitting fresh
    int64_t calib_age_ns{0};           // ns since the fit in use was last computed
};

class Pipeline {
public:
    explicit Pipeline(const Config& cfg);

    ProjectedAnchors build_anchors(
        const ext_vio_data_t& vio_pkt,
        const float T_imu_image_wrt_vio[3],
        const float R_imu_image_to_vio[3][3],
        const TofFrame* tof,
        const float T_imu_tof_wrt_vio[3],
        const float R_imu_tof_to_vio[3][3]) const;

    std::unique_ptr<RescaleResult> process(
        int64_t frame_timestamp_ns,
        const ext_vio_data_t& vio_pkt,
        const float T_imu_image_wrt_vio[3],
        const float R_imu_image_to_vio[3][3],
        const TofFrame* tof,
        const float T_imu_tof_wrt_vio[3],
        const float R_imu_tof_to_vio[3][3],
        const std::vector<float>& disparity);

private:
    float sample_disparity(const std::vector<float>& disp, float u, float v) const;

    const Config&             _cfg;
    CameraModel               _camera;
    Preprocessor              _pre;
    fits::Fit                 _held;
    bool                      _has_held{false};
    int64_t                   _held_t_ns{0};
};
