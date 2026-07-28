#pragma once
#include "config.h"
#include "camera_model.h"
#include "fits.h"
#include "geometry.h"
#include "preprocess.h"
#include "../tof/tof_types.h"
#include "../vio/vio_types.h"
#include <memory>
#include <vector>

struct RescaleResult {
    std::vector<float>  depth;         // (H_model x W_model) metric depth, metres
    std::vector<double> params;        // fitted curve params (empty for splines)
    int     n_anchors{0};              // 0 on a held frame (no fresh fit)
    float   inlier_ratio{1.0f};
    bool    held{false};               // reusing an earlier fit, not fitting fresh
    int64_t calib_age_ns{0};           // ns since the fit in use was last computed
};

class Pipeline {
public:
    explicit Pipeline(const Config& cfg);

    ProjectedAnchors build_anchors(
        const ext_vio_data_t& vio_pkt,
        const float T_imu_image_wrt_vio[3],
        const float R_imu_image_to_vio[3][3],
        const TofFrame* tof,
        const float T_imu_tof_wrt_vio[3],
        const float R_imu_tof_to_vio[3][3]) const;

    std::unique_ptr<RescaleResult> process(
        int64_t frame_timestamp_ns,
        const ext_vio_data_t& vio_pkt,
        const float T_imu_image_wrt_vio[3],
        const float R_imu_image_to_vio[3][3],
        const TofFrame* tof,
        const float T_imu_tof_wrt_vio[3],
        const float R_imu_tof_to_vio[3][3],
        const std::vector<float>& disparity);

    // Rescale from the held fit; nullptr if none or aged past max_hold_age_ns.
    std::unique_ptr<RescaleResult> apply_held(
        int64_t frame_timestamp_ns, const std::vector<float>& disparity);

    // Anchors the last process() call fed to the fit.
    int last_anchor_count() const { return _last_anchors; }

    // Anchor counts from the last build_anchors: VIO kept, ToF in-FOV,
    // ToF surviving the depth gate, ToF after grid thinning.
    void last_anchor_split(int& vio, int& tof_fov, int& tof_gated,
                           int& tof_kept) const {
        vio = _n_vio; tof_fov = _n_tof_fov;
        tof_gated = _n_tof_gated; tof_kept = _n_tof_kept;
    }

    // Why the last fresh fit was rejected, and the ranges it was given.
    int last_fit_reason() const { return _last_fit_reason; }
    void last_fit_ranges(double& x_lo, double& x_hi,
                         double& y_lo, double& y_hi) const {
        x_lo = _last_x_lo; x_hi = _last_x_hi;
        y_lo = _last_y_lo; y_hi = _last_y_hi;
    }

private:
    float sample_disparity(const std::vector<float>& disp, float u, float v) const;
    std::unique_ptr<RescaleResult> render(
        const fits::Fit& fit, const std::vector<float>& disparity,
        bool held, int n_anchors, int64_t calib_age_ns) const;

    const Config&             _cfg;
    CameraModel               _camera;
    Preprocessor              _pre;
    fits::Fit                 _held;
    bool                      _has_held{false};
    int64_t                   _held_t_ns{0};
    int                       _last_anchors{0};
    int                       _last_fit_reason{0};
    mutable int               _n_vio{0}, _n_tof_fov{0};
    mutable int               _n_tof_gated{0}, _n_tof_kept{0};
    double                    _last_x_lo{0.0}, _last_x_hi{0.0};
    double                    _last_y_lo{0.0}, _last_y_hi{0.0};
};
