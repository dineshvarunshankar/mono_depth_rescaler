#include "tof_anchors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
void rotate(const float R[3][3], const float p[3], float out[3]) {
    for (int i = 0; i < 3; ++i) {
        out[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            out[i] += R[i][j] * p[j];
        }
    }
}

void inverse_pose(
    const float p_world[3], const float T[3], const float R[3][3], float out[3]) {
    const float delta[3] = {
        p_world[0] - T[0], p_world[1] - T[1], p_world[2] - T[2]};
    for (int i = 0; i < 3; ++i) {
        out[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            out[i] += R[j][i] * delta[j];
        }
    }
}

std::vector<size_t> evenly_capped(
    const std::vector<size_t>& indices, int max_points) {
    if (max_points <= 0) {
        return {};
    }
    if (indices.size() <= static_cast<size_t>(max_points)) {
        return indices;
    }
    std::vector<size_t> selected;
    selected.reserve(max_points);
    if (max_points == 1) {
        selected.push_back(indices[indices.size() / 2]);
        return selected;
    }
    for (int i = 0; i < max_points; ++i) {
        const size_t j = static_cast<size_t>(
            static_cast<uint64_t>(i) * (indices.size() - 1) /
            static_cast<uint64_t>(max_points - 1));
        selected.push_back(indices[j]);
    }
    return selected;
}
}

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
    int max_points) {
    if (frame.points.size() != frame.noise.size() ||
        frame.points.size() != frame.confidence.size()) {
        throw std::invalid_argument("inconsistent ToF frame arrays");
    }

    std::vector<size_t> valid;
    valid.reserve(frame.points.size());
    for (size_t i = 0; i < frame.points.size(); ++i) {
        const auto& p = frame.points[i];
        if (frame.confidence[i] >= confidence_min && frame.noise[i] > 0.0f &&
            std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            valid.push_back(i);
        }
    }

    ProjectedAnchors out;
    for (const size_t i : evenly_capped(valid, max_points)) {
        const auto& point = frame.points[i];
        const float p_tof[3] = {point.x, point.y, point.z};
        float p_imu_tof[3], p_world[3], p_imu_image[3], p_hires[3];

        rotate(extr_tof.R, p_tof, p_imu_tof);
        for (int k = 0; k < 3; ++k) {
            p_imu_tof[k] += extr_tof.T[k];
        }
        rotate(R_imu_tof_to_vio, p_imu_tof, p_world);
        for (int k = 0; k < 3; ++k) {
            p_world[k] += T_imu_tof_wrt_vio[k];
        }
        inverse_pose(
            p_world, T_imu_image_wrt_vio, R_imu_image_to_vio, p_imu_image);
        inverse_pose(p_imu_image, extr_hires.T, extr_hires.R, p_hires);

        float u, v;
        if (!pre.project(p_hires[0], p_hires[1], p_hires[2], u, v) ||
            u < 0.0f || u >= pre.dst_w() || v < 0.0f || v >= pre.dst_h()) {
            continue;
        }
        out.u.push_back(u);
        out.v.push_back(v);
        out.depth.push_back(p_hires[2]);
        out.var.push_back(frame.noise[i] * frame.noise[i]);
    }
    return out;
}
