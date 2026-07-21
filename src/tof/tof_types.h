#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr size_t TOF_POINT_COUNT = 43'200;
constexpr size_t TOF_PACKET_BYTES = 777'620;

struct TofPoint {
    float x;
    float y;
    float z;
};
static_assert(sizeof(TofPoint) == 12, "ToF point layout mismatch");

struct TofFrame {
    int64_t timestamp_ns{0};
    std::vector<TofPoint> points;
    std::vector<float> noise;
    std::vector<uint8_t> confidence;
};

TofFrame decode_tof_packet(const uint8_t* data, size_t size);
