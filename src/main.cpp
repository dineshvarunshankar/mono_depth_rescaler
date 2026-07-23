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
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::atomic<bool> g_running{true};
static std::atomic<int> g_exit_code{0};
static void on_signal(int) { g_running = false; }

static void request_restart(const char* why) {
    std::fprintf(stderr, "mono_depth_rescaler: restart requested (%s)\n", why);
    g_exit_code = 1;
    g_running = false;
}

static constexpr int64_t FEATURE_TOL_NS = 200'000'000;

static constexpr int CH_VIO = 0;
static constexpr int CH_TOF = 1;
static constexpr int CH_DEPTH = 2;
static constexpr int CH_OUT = 0;

struct Arguments {
    std::string config{"/etc/mono_depth_rescaler/pipeline.yaml"};
    std::string intrinsics{"/etc/mono_depth_rescaler/intrinsics"};
    std::string extrinsics{"/etc/mono_depth_rescaler/extrinsics/starling2.yaml"};
    // Optional overrides for manual debug only. Service uses YAML alone.
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
                "[--extrinsics PATH]\n"
                "  Profile and fov come from pipeline.yaml "
                "(deployment.profile / inference.fov).\n"
                "  Optional overrides for manual runs only: "
                "[--profile openvins|qvio] [--fov crop|stretch]\n");
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

static void fill_output_pipe_info(pipe_info_t* info, const std::string& name) {
    std::memset(info, 0, sizeof(*info));
    std::snprintf(info->name, sizeof(info->name), "%s", name.c_str());
    std::snprintf(
        info->location, sizeof(info->location), "%s%s/",
        MODAL_PIPE_DEFAULT_BASE_DIR, name.c_str());
    std::snprintf(
        info->type, sizeof(info->type), "camera_image_metadata_t");
    std::snprintf(
        info->server_name, sizeof(info->server_name), "mono_depth_rescaler");
    info->size_bytes = 16 * 1024 * 1024;
    info->server_pid = 0;
}

struct Stats {
    std::atomic<uint64_t> disp_in{0};
    std::atomic<uint64_t> pose_miss{0};
    std::atomic<uint64_t> vio_bad_state{0};
    std::atomic<uint64_t> tof_miss{0};
    std::atomic<uint64_t> fit_fail{0};
    std::atomic<uint64_t> out_written{0};
};

int run(const Arguments& args) {
    Config cfg = Config::from_yaml(
        args.config, args.intrinsics, args.extrinsics, args.profile, args.fov);
    Pipeline pipeline(cfg);
    PoseBuffer pose_buf;
    Stats stats;

    pipe_info_t out_info;
    fill_output_pipe_info(&out_info, cfg.output.pipe);
    if (pipe_server_create(CH_OUT, out_info, 0)) {
        throw std::runtime_error("failed to create output pipe");
    }

    auto vio_cb = [&](const ext_vio_data_t& pkt) {
        pose_buf.push(pkt);
    };

    TofSource tof_source(cfg.anchors.tof_pipe, CH_TOF);
    MpaBackend depth_source(
        cfg.inference.mpa_pipe_name,
        cfg.inference.input_w, cfg.inference.input_h,
        CH_DEPTH);

    std::vector<uint8_t> out_packet;
    depth_source.set_frame_callback([&](const MpaBackend::Frame& frame) {
        stats.disp_in.fetch_add(1, std::memory_order_relaxed);
        const int64_t image_time = frame.mid_timestamp_ns();
        ext_vio_data_t vio_pkt;
        float image_T[3], image_R[3][3];
        if (!pose_buf.get(
                image_time, FEATURE_TOL_NS, vio_pkt, image_T, image_R)) {
            stats.pose_miss.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (vio_pkt.v.state != VIO_STATE_OK) {
            stats.vio_bad_state.fetch_add(1, std::memory_order_relaxed);
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
            if (!pose_buf.get_pose(
                    tof->timestamp_ns, FEATURE_TOL_NS, tof_T, tof_R)) {
                tof.reset();
                stats.tof_miss.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            stats.tof_miss.fetch_add(1, std::memory_order_relaxed);
        }

        auto result = pipeline.process(
            image_time, vio_pkt, image_T, image_R, tof.get(), tof_T, tof_R,
            frame.disparity);
        if (!result) {
            stats.fit_fail.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        fill_float_image_packet(
            frame.metadata, result->depth,
            cfg.inference.input_w, cfg.inference.input_h, out_packet);
        pipe_server_write(
            CH_OUT,
            reinterpret_cast<char*>(out_packet.data()),
            static_cast<int>(out_packet.size()));
        stats.out_written.fetch_add(1, std::memory_order_relaxed);
    });

    depth_source.set_disconnect_callback(
        [] { request_restart("disparity pipe disconnect"); });
    // ToF disconnect is soft: keep running on VIO anchors alone.
    tof_source.set_disconnect_callback([] {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: ToF disconnected; continuing without ToF\n");
    });

    MpaVioSource vio(cfg.vio.pipe, CH_VIO);
    vio.set_disconnect_callback(
        [] { request_restart("VIO pipe disconnect"); });

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    if (!vio.start(vio_cb)) {
        pipe_server_close(CH_OUT);
        return 1;
    }
    if (!tof_source.start()) {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: warning: ToF unavailable at start; "
            "continuing without ToF anchors\n");
    }
    if (!depth_source.start()) {
        vio.stop();
        tof_source.stop();
        pipe_server_close(CH_OUT);
        return 1;
    }

    std::printf(
        "profile=%s fov=%s vio=%s disparity=%s tof_cap=%d output=%s "
        "(from YAML%s)\n",
        cfg.profile.c_str(), cfg.inference.fov.c_str(), cfg.vio.pipe.c_str(),
        cfg.inference.mpa_pipe_name.c_str(),
        cfg.anchors.tof_max_points, cfg.output.pipe.c_str(),
        args.profile.empty() && args.fov.empty() ? "" : "; CLI override set");

    uint64_t ticks = 0;
    while (g_running) {
        struct timespec ts = {0, 50'000'000};
        nanosleep(&ts, nullptr);
        if ((++ticks % 40) == 0) {  // ~2 s
            std::fprintf(
                stderr,
                "mono_depth_rescaler stats: disp_in=%llu pose_miss=%llu "
                "vio_bad=%llu tof_miss=%llu fit_fail=%llu out=%llu\n",
                static_cast<unsigned long long>(
                    stats.disp_in.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    stats.pose_miss.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    stats.vio_bad_state.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    stats.tof_miss.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    stats.fit_fail.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    stats.out_written.load(std::memory_order_relaxed)));
        }
    }

    depth_source.stop();
    tof_source.stop();
    vio.stop();
    pipe_server_close(CH_OUT);
    return g_exit_code.load();
}

int main(int argc, char** argv) {
    try {
        return run(parse_arguments(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mono_depth_rescaler: %s\n", error.what());
        return 1;
    }
}
