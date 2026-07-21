#include "core/config.h"
#include "core/pipeline.h"
#include "io/image_packet.h"
#include "vio/mpa_vio_source.h"
#include "vio/pose_buffer.h"
#include "inference/mpa_backend.h"
#include "tof/tof_source.h"

#include <modal_pipe.h>

#include <signal.h>

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

static constexpr int64_t FEATURE_TOL_NS = 200'000'000;

struct Arguments {
    std::string config{"/etc/mono_depth_rescaler/pipeline.yaml"};
    std::string intrinsics{"/etc/mono_depth_rescaler/intrinsics"};
    std::string extrinsics{"/etc/mono_depth_rescaler/extrinsics/starling2.yaml"};
    std::string profile;
    std::string fov;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::printf(
                "mono_depth_rescaler [--config PATH] [--intrinsics DIR] "
                "[--extrinsics PATH] [--profile openvins|qvio] "
                "[--fov crop|stretch]\n");
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--config") args.config = value;
        else if (key == "--intrinsics") args.intrinsics = value;
        else if (key == "--extrinsics") args.extrinsics = value;
        else if (key == "--profile") args.profile = value;
        else if (key == "--fov") args.fov = value;
        else throw std::invalid_argument("unknown argument: " + key);
    }
    return args;
}

int run(const Arguments& args) {
    Config cfg = Config::from_yaml(
        args.config, args.intrinsics, args.extrinsics, args.profile, args.fov);
    Pipeline pipeline(cfg);
    PoseBuffer pose_buf;

    int out_fd = pipe_server_create(cfg.output.pipe.c_str(), 0, 0);
    if (out_fd < 0) {
        throw std::runtime_error("failed to create output pipe");
    }

    auto vio_cb = [&](const ext_vio_data_t& pkt) {
        pose_buf.push(pkt);
    };

    TofSource tof_source(cfg.anchors.tof_pipe);
    MpaBackend depth_source(
        cfg.inference.mpa_pipe_name,
        cfg.inference.input_w, cfg.inference.input_h);

    depth_source.set_frame_callback([&](const MpaBackend::Frame& frame) {
        const int64_t image_time = frame.mid_timestamp_ns();
        ext_vio_data_t vio_pkt;
        float image_T[3], image_R[3][3];
        if (!pose_buf.get(
                image_time, FEATURE_TOL_NS, vio_pkt, image_T, image_R)) {
            return;
        }

        std::shared_ptr<const TofFrame> tof = tof_source.nearest(
            image_time, cfg.anchors.tof_tolerance_ns);
        float tof_T[3] = {0.0f, 0.0f, 0.0f};
        float tof_R[3][3] = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}};
        if (tof) {
            ext_vio_data_t unused;
            if (!pose_buf.get(
                    tof->timestamp_ns, FEATURE_TOL_NS, unused, tof_T, tof_R)) {
                tof.reset();
            }
        }

        auto result = pipeline.process(
            image_time, vio_pkt, image_T, image_R, tof.get(), tof_T, tof_R,
            frame.disparity);
        if (!result) {
            return;
        }
        const std::vector<uint8_t> packet = make_float_image_packet(
            frame.metadata, result->depth,
            cfg.inference.input_w, cfg.inference.input_h);
        pipe_server_write(
            out_fd, packet.data(), static_cast<int>(packet.size()));
    });

    MpaVioSource vio(cfg.vio.pipe);

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    vio.start(vio_cb);
    tof_source.start();
    depth_source.start();
    std::printf(
        "profile=%s fov=%s vio=%s tof_cap=%d output=%s\n",
        cfg.profile.c_str(), cfg.inference.fov.c_str(), cfg.vio.pipe.c_str(),
        cfg.anchors.tof_max_points, cfg.output.pipe.c_str());

    while (g_running) {
        struct timespec ts = {0, 50'000'000};
        nanosleep(&ts, nullptr);
    }

    depth_source.stop();
    tof_source.stop();
    vio.stop();
    pipe_server_close(out_fd);
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(parse_arguments(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mono_depth_rescaler: %s\n", error.what());
        return 1;
    }
}
