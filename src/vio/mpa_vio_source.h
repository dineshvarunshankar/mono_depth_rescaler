#pragma once

#include "vio_source.h"

#include <atomic>
#include <string>

class MpaVioSource : public VioSource {
public:
    MpaVioSource(std::string pipe, int ch = 0);
    ~MpaVioSource() override;

    void start(Callback callback) override;
    void stop() override;

private:
    static void helper_cb(int ch, char* data, int bytes, void* context);
    void on_data(char* data, int bytes);

    std::string _pipe;
    int _ch;
    Callback _callback;
    std::atomic<bool> _running{false};
};
