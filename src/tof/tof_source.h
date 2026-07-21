#pragma once

#include "tof_types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class TofSource {
public:
    explicit TofSource(std::string pipe);
    ~TofSource();

    void start();
    void stop();
    std::shared_ptr<const TofFrame> nearest(
        int64_t timestamp_ns, int64_t tolerance_ns) const;

private:
    void run();

    std::string _pipe;
    std::atomic<bool> _running{false};
    std::thread _thread;
    mutable std::mutex _mutex;
    std::shared_ptr<const TofFrame> _latest;
    int _fd{-1};
};
