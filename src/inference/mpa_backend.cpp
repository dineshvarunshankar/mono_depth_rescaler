#include "mpa_backend.h"

#include <cstdio>
#include <cstring>

static_assert(
    sizeof(ImageMetadata) == sizeof(camera_image_metadata_t),
    "ImageMetadata must match camera_image_metadata_t");

MpaBackend::MpaBackend(std::string pipe, int model_w, int model_h, int ch)
    : _pipe(std::move(pipe)), _mw(model_w), _mh(model_h), _ch(ch) {
    _latest.disparity.resize(static_cast<size_t>(_mw) * _mh);
}

MpaBackend::~MpaBackend() {
    stop();
}

void MpaBackend::stop() {
    if (!_running.exchange(false)) {
        return;
    }
    _cv.notify_all();
    pipe_client_close(_ch);
    if (_worker.joinable()) {
        _worker.join();
    }
}

void MpaBackend::set_frame_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(_mtx);
    _callback = std::move(cb);
}

void MpaBackend::set_disconnect_callback(DisconnectCallback cb) {
    std::lock_guard<std::mutex> lk(_mtx);
    _on_disconnect = std::move(cb);
}

bool MpaBackend::start() {
    if (_running.exchange(true)) {
        return true;
    }
    _worker = std::thread(&MpaBackend::worker_loop, this);
    pipe_client_set_camera_helper_cb(_ch, &MpaBackend::camera_cb, this);
    pipe_client_set_disconnect_cb(_ch, &MpaBackend::disc_cb, this);
    // Camera helper allocates its own read buffer; length argument is unused.
    const int rc = pipe_client_open(
        _ch,
        _pipe.c_str(),
        "mono_depth_rescaler",
        CLIENT_FLAG_EN_CAMERA_HELPER,
        0);
    if (rc < 0) {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: failed to open disparity pipe '%s'\n",
            _pipe.c_str());
        _running = false;
        _cv.notify_all();
        if (_worker.joinable()) {
            _worker.join();
        }
        return false;
    }
    return true;
}

void MpaBackend::camera_cb(
    int /*ch*/, camera_image_metadata_t meta, char* frame, void* context) {
    auto* self = static_cast<MpaBackend*>(context);
    ImageMetadata image_meta;
    std::memcpy(&image_meta, &meta, sizeof(image_meta));
    self->on_frame(
        image_meta, reinterpret_cast<const float*>(frame),
        self->_mw * self->_mh);
}

void MpaBackend::disc_cb(int /*ch*/, void* context) {
    static_cast<MpaBackend*>(context)->on_disconnect();
}

void MpaBackend::on_disconnect() {
    DisconnectCallback cb;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        cb = _on_disconnect;
    }
    std::fprintf(
        stderr,
        "mono_depth_rescaler: disconnected from disparity pipe '%s'\n",
        _pipe.c_str());
    if (cb) {
        cb();
    }
}

void MpaBackend::on_frame(
    const ImageMetadata& meta, const float* pixels, int npix) {
    if (!_running || !pixels) {
        return;
    }
    if (meta.format != IMAGE_FORMAT_FLOAT32_VALUE ||
        meta.width != _mw || meta.height != _mh ||
        meta.size_bytes != npix * static_cast<int>(sizeof(float))) {
        return;
    }

    {
        std::lock_guard<std::mutex> lk(_mtx);
        _latest.metadata = meta;
        std::memcpy(
            _latest.disparity.data(), pixels,
            static_cast<size_t>(npix) * sizeof(float));
        _have_latest = true;
    }
    _cv.notify_one();
}

void MpaBackend::worker_loop() {
    while (true) {
        Frame frame;
        FrameCallback cb;
        {
            std::unique_lock<std::mutex> lk(_mtx);
            _cv.wait(lk, [&] { return _have_latest || !_running; });
            if (!_have_latest) {
                if (!_running) {
                    return;
                }
                continue;
            }
            frame.metadata = _latest.metadata;
            frame.disparity = _latest.disparity;  // copy out for worker
            _have_latest = false;
            cb = _callback;
        }
        if (cb) {
            cb(frame);
        }
    }
}
