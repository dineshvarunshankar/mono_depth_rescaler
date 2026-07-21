#include "geometry.h"
#include <cstring>

static void transform(const float P_in[3], const float T[3],
                      const float R[3][3], float P_out[3]) {
    float d[3] = { P_in[0] - T[0], P_in[1] - T[1], P_in[2] - T[2] };
    for (int i = 0; i < 3; ++i) {
        P_out[i] = 0.0f;
        for (int j = 0; j < 3; ++j) P_out[i] += d[j] * R[j][i];
    }
}

ProjectedAnchors project_features(
    const ext_vio_data_t& pkt,
    const float           T_imu_wrt_vio[3],
    const float           R_imu_to_vio[3][3],
    const float           R_cam_to_imu[3][3],
    const float           T_cam_wrt_imu[3],
    const Preprocessor&   pre,
    int                   min_quality) {

    ProjectedAnchors out;
    for (uint32_t f = 0; f < pkt.n_total_features && f < VIO_MAX_FEATURES; ++f) {
        const auto& ft = pkt.features[f];
        if (ft.point_quality < min_quality) continue;

        float P_imu[3], P_cam[3];
        float point[3];
        std::memcpy(point, ft.tsf, sizeof(point));
        transform(point, T_imu_wrt_vio, R_imu_to_vio, P_imu);
        transform(P_imu, T_cam_wrt_imu, R_cam_to_imu, P_cam);

        float u, v;
        bool front = pre.project(P_cam[0], P_cam[1], P_cam[2], u, v);
        if (!front) continue;
        if (u < 0 || u >= pre.dst_w() || v < 0 || v >= pre.dst_h()) continue;

        out.u.push_back(u);
        out.v.push_back(v);
        out.depth.push_back(P_cam[2]);
        out.var.push_back(1.0f);
    }
    return out;
}
