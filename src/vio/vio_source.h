#pragma once
#include "vio_types.h"
#include <functional>

// Abstract VIO source: QVIO or OpenVINS, both publish ext_vio_data_t.
class VioSource {
public:
    using Callback = std::function<void(const ext_vio_data_t&)>;
    virtual ~VioSource() = default;
    virtual void start(Callback cb) = 0;
    virtual void stop() = 0;
};
