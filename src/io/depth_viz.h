#pragma once

#include "image_packet.h"

#include <cstdint>
#include <vector>

// JET colourmap over a fixed depth range, so a colour always means the same
// distance. Publishes IMAGE_FORMAT_RGB.
void fill_depth_viz_packet(
    ImageMetadata metadata,
    const std::vector<float>& depth_m,
    int width,
    int height,
    float depth_min,
    float depth_max,
    std::vector<uint8_t>& packet_out);
