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
static void on_signal(int) { g_running = false; }

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
    std::atomic<int>      anchors{0};   // last fitted frame
    std::atomic<int>      vio{0}, tof{0};
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
        const bool have_pose = pose_buf.get(
            image_time, cfg.anchors.feature_tol_ns, vio_pkt, image_T, image_R);

        // No valid pose: rescale from the held fit.
        std::unique_ptr<RescaleResult> result;
        if (!have_pose) {
            stats.pose_miss.fetch_add(1, std::memory_order_relaxed);
            result = pipeline.apply_held(image_time, frame.disparity);
        } else if (vio_pkt.v.state != VIO_STATE_OK) {
            stats.vio_bad_state.fetch_add(1, std::memory_order_relaxed);
            result = pipeline.apply_held(image_time, frame.disparity);
        } else {
            std::shared_ptr<const TofFrame> tof = tof_source.nearest(
                image_time, cfg.anchors.tof_tolerance_ns);
            float tof_T[3] = {0.0f, 0.0f, 0.0f};
            float tof_R[3][3] = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f}};
            if (tof) {
                if (!pose_buf.get_pose(
                        tof->timestamp_ns, cfg.anchors.feature_tol_ns,
                        tof_T, tof_R)) {
                    tof.reset();
                    stats.tof_miss.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                stats.tof_miss.fetch_add(1, std::memory_order_relaxed);
            }

            result = pipeline.process(
                image_time, vio_pkt, image_T, image_R, tof.get(), tof_T, tof_R,
                frame.disparity);
            stats.anchors.store(
                pipeline.last_anchor_count(), std::memory_order_relaxed);
            int n_vio, n_tof;
            pipeline.last_anchor_split(n_vio, n_tof);
            stats.vio.store(n_vio, std::memory_order_relaxed);
            stats.tof.store(n_tof, std::memory_order_relaxed);
            if (!result) {
                stats.fit_fail.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        if (!result) {
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

    // Disparity disconnect is soft: auto-reconnect, no restart.
    depth_source.set_disconnect_callback([] {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: disparity disconnected; awaiting reconnect\n");
    });
    // ToF disconnect is soft: keep running on VIO anchors alone.
    tof_source.set_disconnect_callback([] {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: ToF disconnected; continuing without ToF\n");
    });

    MpaVioSource vio(cfg.vio.pipe, CH_VIO);
    // VIO disconnect is soft: auto-reconnect; held fit covers the gap.
    vio.set_disconnect_callback([] {
        std::fprintf(
            stderr,
            "mono_depth_rescaler: VIO disconnected; awaiting reconnect\n");
    });

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    bool vio_up  = vio.start(vio_cb);
    bool tof_up  = tof_source.start();
    bool disp_up = depth_source.start();

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
        ++ticks;
        if ((ticks % 100) == 0) {  // ~5 s: retry pipes still down
            if (!vio_up)  vio_up  = vio.start(vio_cb);
            if (!tof_up)  tof_up  = tof_source.start();
            if (!disp_up) disp_up = depth_source.start();
        }
        if ((ticks % 40) == 0) {  // ~2 s
            std::fprintf(
                stderr,
                "mono_depth_rescaler stats: disp_in=%llu pose_miss=%llu "
                "vio_bad=%llu tof_miss=%llu fit_fail=%llu out=%llu "
                "anchors=%d vio=%d tof=%d\n",
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
                    stats.out_written.load(std::memory_order_relaxed)),
                stats.anchors.load(std::memory_order_relaxed),
                stats.vio.load(std::memory_order_relaxed),
                stats.tof.load(std::memory_order_relaxed));
        }
    }

    depth_source.stop();
    tof_source.stop();
    vio.stop();
    pipe_server_close(CH_OUT);
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
