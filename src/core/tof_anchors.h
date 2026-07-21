#pragma once

#include "config.h"
#include "geometry.h"
#include "preprocess.h"
#include "../tof/tof_types.h"

ProjectedAnchors project_tof_anchors(
    const TofFrame& frame,
    const float T_imu_tof_wrt_vio[3],
    const float R_imu_tof_to_vio[3][3],
    const float T_imu_image_wrt_vio[3],
    const float R_imu_image_to_vio[3][3],
    const ExtrinsicsConfig& extr_tof,
    const ExtrinsicsConfig& extr_hires,
    const Preprocessor& pre,
    int confidence_min,
    int max_points);
