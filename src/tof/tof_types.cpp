#include "tof_types.h"

#include <cstring>
#include <stdexcept>

namespace {
constexpr size_t TIMESTAMP_OFFSET = 4;
constexpr size_t POINTS_OFFSET = 20;
constexpr size_t POINT_BYTES = TOF_POINT_COUNT * 3 * sizeof(float);
constexpr size_t NOISE_OFFSET = POINTS_OFFSET + POINT_BYTES;
constexpr size_t NOISE_BYTES = TOF_POINT_COUNT * sizeof(float);
constexpr size_t GRAY_OFFSET = NOISE_OFFSET + NOISE_BYTES;
constexpr size_t CONFIDENCE_OFFSET = GRAY_OFFSET + TOF_POINT_COUNT;
}

TofFrame decode_tof_packet(const uint8_t* data, size_t size) {
    if (!data || size != TOF_PACKET_BYTES) {
        throw std::invalid_argument("invalid ToF packet size");
    }

    TofFrame frame;
    std::memcpy(&frame.timestamp_ns, data + TIMESTAMP_OFFSET, sizeof(frame.timestamp_ns));
    frame.points.resize(TOF_POINT_COUNT);
    frame.noise.resize(TOF_POINT_COUNT);
    frame.confidence.resize(TOF_POINT_COUNT);
    std::memcpy(frame.points.data(), data + POINTS_OFFSET, POINT_BYTES);
    std::memcpy(frame.noise.data(), data + NOISE_OFFSET, NOISE_BYTES);
    std::memcpy(
        frame.confidence.data(), data + CONFIDENCE_OFFSET, TOF_POINT_COUNT);
    return frame;
}
