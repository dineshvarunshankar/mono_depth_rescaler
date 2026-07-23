#include "tof_source.h"

#include <modal_pipe.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

TofSource::TofSource(std::string pipe, int ch)
    : _pipe(std::move(pipe)), _ch(ch) {
    _raw_latest.resize(TOF_PACKET_BYTES);
}

TofSource::~TofSource() {
    stop();
}

void TofSource::set_disconnect_callback(DisconnectCallback cb) {
    std::lock_guard<std::mutex> lock(_mutex);
    _on_disconnect = std::move(cb);
}

bool TofSource::start() {
    if (_running.exchange(true)) {
        return true;
    }
    pipe_client_set_simple_helper_cb(_ch, &TofSource::helper_cb, this);
    pipe_client_set_disconnect_cb(_ch, &TofSource::disc_cb, this);
    const int rc = pipe_client_open(
        _ch,
        _pipe.c_str(),
        "mono_depth_rescaler",
        CLIENT_FLAG_EN_SIMPLE_HELPER,
        static_cast<int>(TOF_RECOMMENDED_READ_BUF_SIZE));
    if (rc < 0) {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: failed to open ToF pipe '%s'\n",
            _pipe.c_str());
        _running = false;
        return false;
    }
    return true;
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
    if (_raw_dirty) {
        try {
            _latest = std::make_shared<TofFrame>(
                decode_tof_packet(_raw_latest.data(), _raw_latest.size()));
            _raw_dirty = false;
        } catch (const std::exception& e) {
            std::fprintf(
                stderr, "mono_depth_rescaler: ToF decode failed: %s\n",
                e.what());
            _raw_dirty = false;
            _latest.reset();
            return {};
        }
    }
    if (!_latest ||
        std::llabs(_latest->timestamp_ns - timestamp_ns) > tolerance_ns) {
        return {};
    }
    return _latest;
}

void TofSource::helper_cb(int /*ch*/, char* data, int bytes, void* context) {
    static_cast<TofSource*>(context)->on_data(data, bytes);
}

void TofSource::disc_cb(int /*ch*/, void* context) {
    static_cast<TofSource*>(context)->on_disconnect();
}

void TofSource::on_disconnect() {
    DisconnectCallback cb;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        cb = _on_disconnect;
    }
    std::fprintf(
        stderr, "mono_depth_rescaler: disconnected from ToF pipe '%s'\n",
        _pipe.c_str());
    if (cb) {
        cb();
    }
}

void TofSource::on_data(char* data, int bytes) {
    if (!_running || !data || bytes <= 0) {
        return;
    }

    int n_packets = 0;
    tof_data_t* packets = pipe_validate_tof_data_t(data, bytes, &n_packets);
    if (!packets || n_packets <= 0) {
        // Fallback: accept exact multiples of one gen-1 packet.
        if (bytes % static_cast<int>(TOF_PACKET_BYTES) != 0) {
            return;
        }
        n_packets = bytes / static_cast<int>(TOF_PACKET_BYTES);
        if (n_packets <= 0) {
            return;
        }
        const char* last =
            data + (n_packets - 1) * static_cast<int>(TOF_PACKET_BYTES);
        std::lock_guard<std::mutex> lock(_mutex);
        std::memcpy(_raw_latest.data(), last, TOF_PACKET_BYTES);
        _raw_dirty = true;
        return;
    }

    // Keep only the newest packet; decode lazily in nearest().
    const tof_data_t& latest = packets[n_packets - 1];
    std::lock_guard<std::mutex> lock(_mutex);
    std::memcpy(&_raw_latest[0], &latest, TOF_PACKET_BYTES);
    _raw_dirty = true;
}
