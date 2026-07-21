#include "tof_source.h"

#include <modal_pipe.h>

#include <cmath>
#include <utility>
#include <vector>

TofSource::TofSource(std::string pipe) : _pipe(std::move(pipe)) {}

TofSource::~TofSource() {
    stop();
}

void TofSource::start() {
    if (_running.exchange(true)) {
        return;
    }
    _thread = std::thread(&TofSource::run, this);
}

void TofSource::stop() {
    _running = false;
    if (_fd >= 0) {
        pipe_client_close(_fd);
        _fd = -1;
    }
    if (_thread.joinable()) {
        _thread.join();
    }
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

void TofSource::run() {
    _fd = pipe_client_open(_pipe.c_str(), PIPE_CLIENT_FLAGS_EN_SIMPLE_HELPER, 0);
    if (_fd < 0) {
        _running = false;
        return;
    }

    std::vector<uint8_t> packet(TOF_PACKET_BYTES);
    while (_running) {
        const int n = pipe_client_read(
            _fd, packet.data(), static_cast<int>(packet.size()));
        if (n != static_cast<int>(packet.size())) {
            continue;
        }
        auto frame = std::make_shared<TofFrame>(
            decode_tof_packet(packet.data(), packet.size()));
        std::lock_guard<std::mutex> lock(_mutex);
        _latest = std::move(frame);
    }
}
