#pragma once
#include "../io/image_packet.h"
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
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

    MpaBackend(std::string pipe, int model_w, int model_h, int ring_size = 8);
    ~MpaBackend();

    void set_frame_callback(FrameCallback cb);
    void start();
    void stop();

private:
    void _run();
    std::string        _pipe;
    int                _mw, _mh, _ring;
    std::atomic<bool>  _running{false};
    std::thread        _thread;
    mutable std::mutex _mtx;
    std::deque<Frame>  _buf;
    FrameCallback      _callback;
    int                _fd{-1};
};
