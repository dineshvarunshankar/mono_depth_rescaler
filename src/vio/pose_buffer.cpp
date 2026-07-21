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
// dR = I + [ω]×·dt:
//   [ω]× = [[0, -ω₂, ω₁], [ω₂, 0, -ω₀], [-ω₁, ω₀, 0]]
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

bool PoseBuffer::get(int64_t t_ns, int64_t feature_tol_ns,
                     ext_vio_data_t& pkt_out,
                     float T_out[3], float R_out[3][3]) const {
    std::lock_guard<std::mutex> lk(_mtx);
    if (_buf.empty()) return false;

    // nearest packet, copied under the lock (VIO thread may pop_front concurrently)
    const ext_vio_data_t* nearest = nullptr;
    int64_t nearest_gap = INT64_MAX;
    for (const auto& p : _buf) {
        int64_t gap = std::abs(p.v.timestamp_ns - t_ns);
        if (gap < nearest_gap) { nearest_gap = gap; nearest = &p; }
    }
    if (nearest_gap > feature_tol_ns) return false;
    pkt_out = *nearest;  // copy under lock

    // Find the latest packet at or before t_ns for pose propagation.
    // If t_ns precedes all buffered packets, propagate backward from the oldest.
    const ext_vio_data_t* base = nullptr;
    for (const auto& p : _buf)
        if (p.v.timestamp_ns <= t_ns && (!base || p.v.timestamp_ns > base->v.timestamp_ns))
            base = &p;
    if (!base) base = nearest;

    float dt = static_cast<float>(t_ns - base->v.timestamp_ns) * 1e-9f;
    float position[3], velocity[3];
    std::memcpy(position, base->v.T_imu_wrt_vio, sizeof(position));
    std::memcpy(velocity, base->v.vel_imu_wrt_vio, sizeof(velocity));
    for (int i = 0; i < 3; ++i)
        T_out[i] = position[i] + velocity[i] * dt;
    float rotation[3][3], angular_velocity[3];
    std::memcpy(rotation, base->v.R_imu_to_vio, sizeof(rotation));
    std::memcpy(
        angular_velocity, base->v.imu_angular_vel, sizeof(angular_velocity));
    propagate_R(rotation, angular_velocity, dt, R_out);
    return true;
}
