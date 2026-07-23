#pragma once
#include "../io/image_packet.h"

#include <modal_pipe.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    using DisconnectCallback = std::function<void()>;

    MpaBackend(std::string pipe, int model_w, int model_h, int ch = 2);
    ~MpaBackend();

    void set_frame_callback(FrameCallback cb);
    void set_disconnect_callback(DisconnectCallback cb);

    // Returns false if the pipe could not be opened.
    bool start();
    void stop();

private:
    static void camera_cb(
        int ch, camera_image_metadata_t meta, char* frame, void* context);
    static void disc_cb(int ch, void* context);

    void on_frame(const ImageMetadata& meta, const float* pixels, int npix);
    void on_disconnect();
    void worker_loop();

    std::string        _pipe;
    int                _mw, _mh, _ch;
    std::atomic<bool>  _running{false};
    mutable std::mutex _mtx;
    std::condition_variable _cv;
    bool               _have_latest{false};
    Frame              _latest;
    FrameCallback      _callback;
    DisconnectCallback _on_disconnect;
    std::thread        _worker;
};
