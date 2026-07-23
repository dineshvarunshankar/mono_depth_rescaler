#include "pose_buffer.h"
#include <climits>
#include <cmath>
#include <cstring>

PoseBuffer::PoseBuffer(int capacity) : _cap(capacity) {}

void PoseBuffer::push(const ext_vio_data_t& pkt) {
    std::lock_guard<std::mutex> lk(_mtx);
    _buf.push_back(pkt);
    while (static_cast<int>(_buf.size()) > _cap) _buf.pop_front();
}

// First-order rotation propagation.
// R_imu_to_vio is the body-to-world rotation stored in column convention.
// With body-frame angular velocity ω: Ṙ = R·[ω]×, so R_new = R·(I + [ω]×·dt).
static void propagate_R(const float R[3][3], const float omega[3], float dt,
                        float R_out[3][3]) {
    const float dR[3][3] = {
        { 1.0f,           -omega[2] * dt,   omega[1] * dt},
        { omega[2] * dt,   1.0f,           -omega[0] * dt},
        {-omega[1] * dt,   omega[0] * dt,   1.0f         }
    };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R_out[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) R_out[i][j] += R[i][k] * dR[k][j];
        }
}

static void pose_at(
    const float T0[3], const float V0[3], const float R0[3][3],
    const float omega[3], float dt, float T_out[3], float R_out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        T_out[i] = T0[i] + V0[i] * dt;
    }
    propagate_R(R0, omega, dt, R_out);
}

bool PoseBuffer::get_pose(
    int64_t t_ns, int64_t feature_tol_ns,
    float T_out[3], float R_out[3][3]) const {
    float T0[3], V0[3], R0[3][3], omega[3];
    int64_t t0 = 0;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (_buf.empty()) return false;

        const ext_vio_data_t* nearest = nullptr;
        int64_t nearest_gap = INT64_MAX;
        for (const auto& p : _buf) {
            int64_t gap = std::abs(p.v.timestamp_ns - t_ns);
            if (gap < nearest_gap) {
                nearest_gap = gap;
                nearest = &p;
            }
        }
        if (!nearest || nearest_gap > feature_tol_ns) return false;

        const ext_vio_data_t* base = nullptr;
        for (const auto& p : _buf) {
            if (p.v.timestamp_ns <= t_ns &&
                (!base || p.v.timestamp_ns > base->v.timestamp_ns)) {
                base = &p;
            }
        }
        if (!base) base = nearest;

        t0 = base->v.timestamp_ns;
        std::memcpy(T0, base->v.T_imu_wrt_vio, sizeof(T0));
        std::memcpy(V0, base->v.vel_imu_wrt_vio, sizeof(V0));
        std::memcpy(R0, base->v.R_imu_to_vio, sizeof(R0));
        std::memcpy(omega, base->v.imu_angular_vel, sizeof(omega));
    }
    const float dt = static_cast<float>(t_ns - t0) * 1e-9f;
    pose_at(T0, V0, R0, omega, dt, T_out, R_out);
    return true;
}

bool PoseBuffer::get(int64_t t_ns, int64_t feature_tol_ns,
                     ext_vio_data_t& pkt_out,
                     float T_out[3], float R_out[3][3]) const {
    float T0[3], V0[3], R0[3][3], omega[3];
    int64_t t0 = 0;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (_buf.empty()) return false;

        const ext_vio_data_t* nearest = nullptr;
        int64_t nearest_gap = INT64_MAX;
        for (const auto& p : _buf) {
            int64_t gap = std::abs(p.v.timestamp_ns - t_ns);
            if (gap < nearest_gap) {
                nearest_gap = gap;
                nearest = &p;
            }
        }
        if (!nearest || nearest_gap > feature_tol_ns) return false;
        pkt_out = *nearest;

        const ext_vio_data_t* base = nullptr;
        for (const auto& p : _buf) {
            if (p.v.timestamp_ns <= t_ns &&
                (!base || p.v.timestamp_ns > base->v.timestamp_ns)) {
                base = &p;
            }
        }
        if (!base) base = nearest;

        t0 = base->v.timestamp_ns;
        std::memcpy(T0, base->v.T_imu_wrt_vio, sizeof(T0));
        std::memcpy(V0, base->v.vel_imu_wrt_vio, sizeof(V0));
        std::memcpy(R0, base->v.R_imu_to_vio, sizeof(R0));
        std::memcpy(omega, base->v.imu_angular_vel, sizeof(omega));
    }
    const float dt = static_cast<float>(t_ns - t0) * 1e-9f;
    pose_at(T0, V0, R0, omega, dt, T_out, R_out);
    return true;
}
