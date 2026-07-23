#pragma once

#include "vio_source.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

class MpaVioSource : public VioSource {
public:
    using DisconnectCallback = std::function<void()>;

    MpaVioSource(std::string pipe, int ch = 0);
    ~MpaVioSource() override;

    void set_disconnect_callback(DisconnectCallback cb);

    bool start(Callback callback) override;
    void stop() override;

private:
    static void helper_cb(int ch, char* data, int bytes, void* context);
    static void disc_cb(int ch, void* context);
    void on_data(char* data, int bytes);
    void on_disconnect();

    std::string _pipe;
    int _ch;
    mutable std::mutex _mtx;
    Callback _callback;
    DisconnectCallback _on_disconnect;
    std::atomic<bool> _running{false};
    std::vector<char> _remnant;
};
