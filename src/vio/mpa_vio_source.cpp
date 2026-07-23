#include "mpa_vio_source.h"

#include <modal_pipe.h>

#include <cstdio>
#include <utility>

MpaVioSource::MpaVioSource(std::string pipe, int ch)
    : _pipe(std::move(pipe)), _ch(ch) {}

MpaVioSource::~MpaVioSource() {
    stop();
}

void MpaVioSource::set_disconnect_callback(DisconnectCallback cb) {
    std::lock_guard<std::mutex> lk(_mtx);
    _on_disconnect = std::move(cb);
}

void MpaVioSource::helper_cb(int /*ch*/, char* data, int bytes, void* context) {
    static_cast<MpaVioSource*>(context)->on_data(data, bytes);
}

void MpaVioSource::disc_cb(int /*ch*/, void* context) {
    static_cast<MpaVioSource*>(context)->on_disconnect();
}

void MpaVioSource::on_disconnect() {
    DisconnectCallback cb;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        cb = _on_disconnect;
    }
    std::fprintf(
        stderr, "mono_depth_rescaler: disconnected from VIO pipe '%s'\n",
        _pipe.c_str());
    if (cb) {
        cb();
    }
}

void MpaVioSource::on_data(char* data, int bytes) {
    Callback cb;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (!_running || !_callback || bytes <= 0) {
            return;
        }
        cb = _callback;
        _remnant.insert(_remnant.end(), data, data + bytes);
        constexpr size_t kMaxRemnant =
            static_cast<size_t>(EXT_VIO_RECOMMENDED_READ_BUF_SIZE) * 2;
        if (_remnant.size() > kMaxRemnant) {
            _remnant.clear();
            return;
        }
    }

    // Process under lock only for remnant mutations; invoke callback outside.
    while (true) {
        std::vector<ext_vio_data_t> batch;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            if (!_running) {
                return;
            }
            const size_t pkt = sizeof(ext_vio_data_t);
            if (_remnant.size() < pkt) {
                return;
            }

            const size_t n = _remnant.size() / pkt;
            const size_t use = n * pkt;
            int n_packets = 0;
            ext_vio_data_t* packets = pipe_validate_ext_vio_data_t(
                _remnant.data(), static_cast<int>(use), &n_packets);
            if (!packets || n_packets <= 0) {
                // Resync: drop one byte and try again later.
                _remnant.erase(_remnant.begin());
                if (_remnant.size() < pkt) {
                    return;
                }
                continue;
            }
            batch.assign(packets, packets + n_packets);
            _remnant.erase(
                _remnant.begin(),
                _remnant.begin() + static_cast<std::ptrdiff_t>(use));
            cb = _callback;
        }
        if (!cb) {
            return;
        }
        for (const auto& p : batch) {
            cb(p);
        }
    }
}

bool MpaVioSource::start(Callback callback) {
    if (_running.exchange(true)) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _callback = std::move(callback);
        _remnant.clear();
    }
    pipe_client_set_simple_helper_cb(_ch, &MpaVioSource::helper_cb, this);
    pipe_client_set_disconnect_cb(_ch, &MpaVioSource::disc_cb, this);
    const int rc = pipe_client_open(
        _ch,
        _pipe.c_str(),
        "mono_depth_rescaler",
        CLIENT_FLAG_EN_SIMPLE_HELPER,
        EXT_VIO_RECOMMENDED_READ_BUF_SIZE);
    if (rc < 0) {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: failed to open VIO pipe '%s'\n",
            _pipe.c_str());
        _running = false;
        std::lock_guard<std::mutex> lk(_mtx);
        _callback = {};
        return false;
    }
    return true;
}

void MpaVioSource::stop() {
    if (!_running.exchange(false)) {
        return;
    }
    pipe_client_close(_ch);
    std::lock_guard<std::mutex> lk(_mtx);
    _callback = {};
    _remnant.clear();
}
