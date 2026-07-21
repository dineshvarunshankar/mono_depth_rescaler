#pragma once
#include "../io/image_packet.h"
#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <deque>
#include <functional>
#include <cstdint>

// Relative disparity (float32) from voxl-tflite-server's DISPARITY_CH pipe.
// NOT the <prefix>_tflite pipe -- that is a lossy JET colour map.
class MpaBackend {
public:
    struct Frame {
        ImageMetadata metadata{};
        int64_t mid_timestamp_ns() const {
            return metadata.timestamp_ns + metadata.exposure_ns / 2;
        }
        std::vector<float> disparity;
    };

    using FrameCallback = std::function<void(const Frame&)>;

    MpaBackend(std::string pipe, int model_w, int model_h,
               int ring_size = 8, int ch = 2);
    ~MpaBackend();

    void set_frame_callback(FrameCallback cb);
    void start();
    void stop();

private:
    static void helper_cb(int ch, char* data, int bytes, void* context);
    void on_data(char* data, int bytes);

    std::string        _pipe;
    int                _mw, _mh, _ring, _ch;
    std::atomic<bool>  _running{false};
    mutable std::mutex _mtx;
    std::deque<Frame>  _buf;
    FrameCallback      _callback;
};
