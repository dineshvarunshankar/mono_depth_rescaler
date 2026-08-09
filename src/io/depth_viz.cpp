#include "depth_viz.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <stdexcept>

void fill_depth_viz_packet(
    ImageMetadata metadata,
    const std::vector<float>& depth_m,
    int width,
    int height,
    float depth_min,
    float depth_max,
    std::vector<uint8_t>& packet_out) {
    const size_t expected = static_cast<size_t>(width) * height;
    if (depth_m.size() != expected) {
        throw std::invalid_argument("pixel count does not match image dimensions");
    }

    cv::Mat depth(height, width, CV_32F, const_cast<float*>(depth_m.data()));
    cv::Mat gray;
    // inverted so near reads red, matching the disparity view
    const double scale = -255.0 / (depth_max - depth_min);
    depth.convertTo(gray, CV_8U, scale, -depth_max * scale);
    cv::Mat colour;
    cv::applyColorMap(gray, colour, cv::COLORMAP_JET);

    metadata.width = static_cast<int16_t>(width);
    metadata.height = static_cast<int16_t>(height);
    metadata.size_bytes = static_cast<int32_t>(expected * 3);
    metadata.stride = width * 3;
    metadata.format = IMAGE_FORMAT_RGB_VALUE;

    packet_out.resize(sizeof(metadata) + static_cast<size_t>(metadata.size_bytes));
    std::memcpy(packet_out.data(), &metadata, sizeof(metadata));
    std::memcpy(
        packet_out.data() + sizeof(metadata), colour.data,
        static_cast<size_t>(metadata.size_bytes));
}
