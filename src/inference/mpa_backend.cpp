#include "mpa_backend.h"
#include <modal_pipe.h>
#include <cstring>

MpaBackend::MpaBackend(std::string pipe, int model_w, int model_h, int ring_size)
    : _pipe(std::move(pipe)), _mw(model_w), _mh(model_h), _ring(ring_size) {}

MpaBackend::~MpaBackend() {
    stop();
}

void MpaBackend::stop() {
    _running = false;
    if (_fd >= 0) {
        pipe_client_close(_fd);
        _fd = -1;
    }
    if (_thread.joinable()) _thread.join();
}

void MpaBackend::set_frame_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(_mtx);
    _callback = std::move(cb);
}

void MpaBackend::start() {
    if (_running.exchange(true)) {
        return;
    }
    _thread = std::thread(&MpaBackend::_run, this);
}

void MpaBackend::_run() {
    _fd = pipe_client_open(_pipe.c_str(), PIPE_CLIENT_FLAGS_EN_SIMPLE_HELPER, 0);
    if (_fd < 0) return;

    // DISPARITY_CH publishes float32 disparity, dequantised producer-side, so no
    // scale or zero-point is applied here and the reader is model-agnostic.
    const int npix   = _mw * _mh;
    const int hdr_sz = sizeof(ImageMetadata);
    std::vector<uint8_t> raw(hdr_sz + npix * sizeof(float));

    while (_running) {
        int n = pipe_client_read(_fd, raw.data(), static_cast<int>(raw.size()));
        if (n != static_cast<int>(raw.size())) continue;

        ImageMetadata meta;
        std::memcpy(&meta, raw.data(), sizeof(meta));
        if (meta.format != IMAGE_FORMAT_FLOAT32_VALUE ||
            meta.width != _mw || meta.height != _mh ||
            meta.size_bytes != npix * static_cast<int>(sizeof(float))) {
            continue;
        }

        const float* px = reinterpret_cast<const float*>(raw.data() + hdr_sz);

        Frame f;
        f.metadata = meta;
        f.disparity.assign(px, px + npix);

        FrameCallback cb;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            _buf.push_back(f);
            while (static_cast<int>(_buf.size()) > _ring) _buf.pop_front();
            cb = _callback;
        }
        if (cb) cb(f);
    }
}
