#pragma once
#include "../vio/vio_types.h"
#include "preprocess.h"
#include <vector>

struct ProjectedAnchors {
    std::vector<float> u, v;
    std::vector<float> depth;
    std::vector<float> var;
};

ProjectedAnchors project_features(
    const ext_vio_data_t& pkt,
    const float           T_imu_wrt_vio[3],
    const float           R_imu_to_vio[3][3],
    const float           R_cam_to_imu[3][3],
    const float           T_cam_wrt_imu[3],
    const Preprocessor&   pre,
    int                   min_quality = 1);
