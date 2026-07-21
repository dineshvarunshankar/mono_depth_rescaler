#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

constexpr int16_t IMAGE_FORMAT_FLOAT32_VALUE = 11;

#pragma pack(push, 1)
struct ImageMetadata {
    uint32_t magic_number;
    int64_t timestamp_ns;
    int32_t frame_id;
    int16_t width;
    int16_t height;
    int32_t size_bytes;
    int32_t stride;
    int32_t exposure_ns;
    int16_t gain;
    int16_t format;
    int16_t framerate;
    int16_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(ImageMetadata) == 40, "image metadata layout mismatch");

inline std::vector<uint8_t> make_float_image_packet(
    ImageMetadata metadata, const std::vector<float>& pixels, int width, int height) {
    const size_t expected = static_cast<size_t>(width) * height;
    if (pixels.size() != expected) {
        throw std::invalid_argument("pixel count does not match image dimensions");
    }
    metadata.width = static_cast<int16_t>(width);
    metadata.height = static_cast<int16_t>(height);
    metadata.size_bytes = static_cast<int32_t>(pixels.size() * sizeof(float));
    metadata.stride = width * static_cast<int>(sizeof(float));
    metadata.format = IMAGE_FORMAT_FLOAT32_VALUE;

    std::vector<uint8_t> packet(sizeof(metadata) + metadata.size_bytes);
    std::memcpy(packet.data(), &metadata, sizeof(metadata));
    std::memcpy(packet.data() + sizeof(metadata), pixels.data(), metadata.size_bytes);
    return packet;
}
