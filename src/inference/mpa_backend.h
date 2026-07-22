#pragma once
#include "../io/image_packet.h"
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <deque>
#include <functional>
#include <cstdint>

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
    void worker_loop();

    std::string        _pipe;
    int                _mw, _mh, _ring, _ch;
    std::atomic<bool>  _running{false};
    mutable std::mutex _mtx;
    std::condition_variable _cv;
    std::deque<Frame>  _buf;
    FrameCallback      _callback;
    std::thread        _worker;
};
