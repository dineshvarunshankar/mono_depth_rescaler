#pragma once

#include "vio_source.h"

#include <atomic>
#include <string>
#include <thread>

class MpaVioSource : public VioSource {
public:
    explicit MpaVioSource(std::string pipe);
    ~MpaVioSource() override;

    void start(Callback callback) override;
    void stop() override;

private:
    void run();

    std::string _pipe;
    Callback _callback;
    std::atomic<bool> _running{false};
    std::thread _thread;
    int _fd{-1};
};
