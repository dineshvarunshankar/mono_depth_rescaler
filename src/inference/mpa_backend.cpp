#include "mpa_backend.h"
#include <modal_pipe.h>
#include <cstring>

MpaBackend::MpaBackend(
    std::string pipe, int model_w, int model_h, int ring_size, int ch)
    : _pipe(std::move(pipe)),
      _mw(model_w),
      _mh(model_h),
      _ring(ring_size),
      _ch(ch) {}

MpaBackend::~MpaBackend() {
    stop();
}

void MpaBackend::stop() {
    if (!_running.exchange(false)) {
        return;
    }
    pipe_client_close(_ch);
}

void MpaBackend::set_frame_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(_mtx);
    _callback = std::move(cb);
}

void MpaBackend::start() {
    if (_running.exchange(true)) {
        return;
    }
    const int npix = _mw * _mh;
    const int frame_bytes =
        static_cast<int>(sizeof(ImageMetadata)) +
        npix * static_cast<int>(sizeof(float));
    pipe_client_set_simple_helper_cb(_ch, &MpaBackend::helper_cb, this);
    const int rc = pipe_client_open(
        _ch,
        _pipe.c_str(),
        "mono_depth_rescaler",
        CLIENT_FLAG_EN_SIMPLE_HELPER,
        frame_bytes);
    if (rc < 0) {
        _running = false;
    }
}

void MpaBackend::helper_cb(int /*ch*/, char* data, int bytes, void* context) {
    static_cast<MpaBackend*>(context)->on_data(data, bytes);
}

void MpaBackend::on_data(char* data, int bytes) {
    if (!_running) {
        return;
    }

    const int npix = _mw * _mh;
    const int hdr_sz = static_cast<int>(sizeof(ImageMetadata));
    const int need = hdr_sz + npix * static_cast<int>(sizeof(float));
    if (bytes != need) {
        return;
    }

    ImageMetadata meta;
    std::memcpy(&meta, data, sizeof(meta));
    if (meta.format != IMAGE_FORMAT_FLOAT32_VALUE ||
        meta.width != _mw || meta.height != _mh ||
        meta.size_bytes != npix * static_cast<int>(sizeof(float))) {
        return;
    }

    const float* px = reinterpret_cast<const float*>(data + hdr_sz);

    Frame f;
    f.metadata = meta;
    f.disparity.assign(px, px + npix);

    FrameCallback cb;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _buf.push_back(f);
        while (static_cast<int>(_buf.size()) > _ring) {
            _buf.pop_front();
        }
        cb = _callback;
    }
    if (cb) {
        cb(f);
    }
}
