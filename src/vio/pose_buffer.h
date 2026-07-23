#pragma once
#include "vio_types.h"
#include <deque>
#include <mutex>
#include <cstdint>

// Buffers ext_vio_data_t packets: nearest-in-time feature lookup + first-order
// pose interpolation (vel_imu_wrt_vio, imu_angular_vel).
class PoseBuffer {
public:
    explicit PoseBuffer(int capacity = 32);

    void push(const ext_vio_data_t& pkt);

    // Query at t_ns. false if empty or nearest packet farther than feature_tol_ns.
    // pkt_out: nearest packet (copied under lock). T_out/R_out: pose at t_ns.
    bool get(int64_t t_ns, int64_t feature_tol_ns,
             ext_vio_data_t& pkt_out,
             float T_out[3], float R_out[3][3]) const;

    // Pose only (no feature packet copy). Same sync rules as get().
    bool get_pose(int64_t t_ns, int64_t feature_tol_ns,
                  float T_out[3], float R_out[3][3]) const;

private:
    int                        _cap;
    mutable std::mutex         _mtx;
    std::deque<ext_vio_data_t> _buf;  // newest last, sorted by v.timestamp_ns
};
