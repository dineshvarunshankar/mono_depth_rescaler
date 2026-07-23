#pragma once

#include "tof_types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class TofSource {
public:
    using DisconnectCallback = std::function<void()>;

    explicit TofSource(std::string pipe, int ch = 1);
    ~TofSource();

    void set_disconnect_callback(DisconnectCallback cb);

    // Returns false if the pipe could not be opened.
    bool start();
    void stop();
    std::shared_ptr<const TofFrame> nearest(
        int64_t timestamp_ns, int64_t tolerance_ns) const;

private:
    static void helper_cb(int ch, char* data, int bytes, void* context);
    static void disc_cb(int ch, void* context);
    void on_data(char* data, int bytes);
    void on_disconnect();

    std::string _pipe;
    int _ch;
    std::atomic<bool> _running{false};
    mutable std::mutex _mutex;
    mutable std::vector<uint8_t> _raw_latest;
    mutable bool _raw_dirty{false};
    mutable std::shared_ptr<const TofFrame> _latest;
    DisconnectCallback _on_disconnect;
};
