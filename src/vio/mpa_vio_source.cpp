#include "mpa_vio_source.h"

#include <modal_pipe.h>

#include <utility>

MpaVioSource::MpaVioSource(std::string pipe) : _pipe(std::move(pipe)) {}

MpaVioSource::~MpaVioSource() {
    stop();
}

void MpaVioSource::start(Callback callback) {
    if (_running.exchange(true)) {
        return;
    }
    _callback = std::move(callback);
    _thread = std::thread(&MpaVioSource::run, this);
}

void MpaVioSource::stop() {
    _running = false;
    if (_fd >= 0) {
        pipe_client_close(_fd);
        _fd = -1;
    }
    if (_thread.joinable()) {
        _thread.join();
    }
}

void MpaVioSource::run() {
    _fd = pipe_client_open(_pipe.c_str(), PIPE_CLIENT_FLAGS_EN_SIMPLE_HELPER, 0);
    if (_fd < 0) {
        _running = false;
        return;
    }
    ext_vio_data_t packet;
    while (_running) {
        const int bytes = pipe_client_read(_fd, &packet, sizeof(packet));
        if (bytes == sizeof(packet)) {
            _callback(packet);
        }
    }
}
