#include "mpa_vio_source.h"

#include <modal_pipe.h>

#include <utility>

MpaVioSource::MpaVioSource(std::string pipe, int ch)
    : _pipe(std::move(pipe)), _ch(ch) {}

MpaVioSource::~MpaVioSource() {
    stop();
}

void MpaVioSource::helper_cb(int /*ch*/, char* data, int bytes, void* context) {
    static_cast<MpaVioSource*>(context)->on_data(data, bytes);
}

void MpaVioSource::on_data(char* data, int bytes) {
    if (!_running || !_callback || bytes <= 0) {
        return;
    }

    _remnant.insert(_remnant.end(), data, data + bytes);
    const size_t pkt = sizeof(ext_vio_data_t);
    const size_t n = _remnant.size() / pkt;
    if (n == 0) {
        return;
    }
    const size_t use = n * pkt;
    int n_packets = 0;
    ext_vio_data_t* packets = pipe_validate_ext_vio_data_t(
        _remnant.data(), static_cast<int>(use), &n_packets);
    if (!packets || n_packets <= 0) {
        _remnant.clear();
        return;
    }
    for (int i = 0; i < n_packets; ++i) {
        _callback(packets[i]);
    }
    _remnant.erase(_remnant.begin(), _remnant.begin() + static_cast<std::ptrdiff_t>(use));
}

void MpaVioSource::start(Callback callback) {
    if (_running.exchange(true)) {
        return;
    }
    _callback = std::move(callback);
    _remnant.clear();
    pipe_client_set_simple_helper_cb(_ch, &MpaVioSource::helper_cb, this);
    const int rc = pipe_client_open(
        _ch,
        _pipe.c_str(),
        "mono_depth_rescaler",
        CLIENT_FLAG_EN_SIMPLE_HELPER,
        EXT_VIO_RECOMMENDED_READ_BUF_SIZE);
    if (rc < 0) {
        _running = false;
        _callback = {};
    }
}

void MpaVioSource::stop() {
    if (!_running.exchange(false)) {
        return;
    }
    pipe_client_close(_ch);
    _callback = {};
    _remnant.clear();
}
