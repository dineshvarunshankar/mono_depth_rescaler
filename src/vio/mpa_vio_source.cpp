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
    if (!_running || !_callback) {
        return;
    }
    int n = 0;
    ext_vio_data_t* packets = pipe_validate_ext_vio_data_t(data, bytes, &n);
    if (!packets || n <= 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        _callback(packets[i]);
    }
}

void MpaVioSource::start(Callback callback) {
    if (_running.exchange(true)) {
        return;
    }
    _callback = std::move(callback);
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
}
