#pragma once

#include "tof_types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

class TofSource {
public:
    explicit TofSource(std::string pipe, int ch = 1);
    ~TofSource();

    void start();
    void stop();
    std::shared_ptr<const TofFrame> nearest(
        int64_t timestamp_ns, int64_t tolerance_ns) const;

private:
    static void helper_cb(int ch, char* data, int bytes, void* context);
    void on_data(char* data, int bytes);

    std::string _pipe;
    int _ch;
    std::atomic<bool> _running{false};
    mutable std::mutex _mutex;
    std::shared_ptr<const TofFrame> _latest;
};
