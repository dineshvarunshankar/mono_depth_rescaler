#include "tof_source.h"

#include <modal_pipe.h>

#include <cmath>
#include <cstdint>
#include <utility>

TofSource::TofSource(std::string pipe, int ch)
    : _pipe(std::move(pipe)), _ch(ch) {}

TofSource::~TofSource() {
    stop();
}

void TofSource::start() {
    if (_running.exchange(true)) {
        return;
    }
    pipe_client_set_simple_helper_cb(_ch, &TofSource::helper_cb, this);
    const int rc = pipe_client_open(
        _ch,
        _pipe.c_str(),
        "mono_depth_rescaler",
        CLIENT_FLAG_EN_SIMPLE_HELPER,
        static_cast<int>(TOF_PACKET_BYTES) * 4);
    if (rc < 0) {
        _running = false;
    }
}

void TofSource::stop() {
    if (!_running.exchange(false)) {
        return;
    }
    pipe_client_close(_ch);
}

std::shared_ptr<const TofFrame> TofSource::nearest(
    int64_t timestamp_ns, int64_t tolerance_ns) const {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_latest ||
        std::llabs(_latest->timestamp_ns - timestamp_ns) > tolerance_ns) {
        return {};
    }
    return _latest;
}

void TofSource::helper_cb(int /*ch*/, char* data, int bytes, void* context) {
    static_cast<TofSource*>(context)->on_data(data, bytes);
}

void TofSource::on_data(char* data, int bytes) {
    if (!_running || bytes != static_cast<int>(TOF_PACKET_BYTES)) {
        return;
    }
    auto frame = std::make_shared<TofFrame>(decode_tof_packet(
        reinterpret_cast<const uint8_t*>(data),
        static_cast<size_t>(bytes)));
    std::lock_guard<std::mutex> lock(_mutex);
    _latest = std::move(frame);
}
