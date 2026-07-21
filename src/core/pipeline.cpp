#include "pipeline.h"
#include "tof_anchors.h"

#include <algorithm>
#include <cmath>

Pipeline::Pipeline(const Config& cfg)
    : _cfg(cfg),
      _camera(cfg.hires.fx, cfg.hires.fy, cfg.hires.cx, cfg.hires.cy,
              cfg.hires.distortion, cfg.hires.distortion_model),
      _pre(_camera, cfg.hires.width, cfg.hires.height,
           cfg.inference.input_w, cfg.inference.input_h,
           cfg.inference.preprocess, cfg.inference.fov, cfg.inference.antialias) {}

float Pipeline::sample_disparity(const std::vector<float>& disp, float u, float v) const {
    const int W = _cfg.inference.input_w, H = _cfg.inference.input_h;
    auto clampi = [](int x, int lo, int hi) { return std::max(lo, std::min(x, hi)); };
    if (!_cfg.rescale.subpixel_2x2) {
        int ui = clampi(int(u), 0, W - 1);
        int vi = clampi(int(v), 0, H - 1);
        return disp[vi * W + ui];
    }
    // closest surface wins: max disparity over the floor/ceil neighbourhood
    int uf = clampi(int(std::floor(u)), 0, W - 1);
    int uc = clampi(int(std::ceil(u)),  0, W - 1);
    int vf = clampi(int(std::floor(v)), 0, H - 1);
    int vc = clampi(int(std::ceil(v)),  0, H - 1);
    return std::max(std::max(disp[vf * W + uf], disp[vf * W + uc]),
                    std::max(disp[vc * W + uf], disp[vc * W + uc]));
}

ProjectedAnchors Pipeline::build_anchors(
    const ext_vio_data_t& pkt,
    const float T_imu_image_wrt_vio[3],
    const float R_imu_image_to_vio[3][3],
    const TofFrame* tof,
    const float T_imu_tof_wrt_vio[3],
    const float R_imu_tof_to_vio[3][3]) const {
    ProjectedAnchors anchors = project_features(
        pkt, T_imu_image_wrt_vio, R_imu_image_to_vio,
        _cfg.extr_hires.R, _cfg.extr_hires.T,
        _pre, _cfg.vio.min_quality);
    if (!tof) {
        return anchors;
    }

    ProjectedAnchors tof_anchors = project_tof_anchors(
        *tof, T_imu_tof_wrt_vio, R_imu_tof_to_vio,
        T_imu_image_wrt_vio, R_imu_image_to_vio,
        _cfg.extr_tof, _cfg.extr_hires, _pre,
        _cfg.anchors.tof_confidence_min, _cfg.anchors.tof_max_points);
    anchors.u.insert(anchors.u.end(), tof_anchors.u.begin(), tof_anchors.u.end());
    anchors.v.insert(anchors.v.end(), tof_anchors.v.begin(), tof_anchors.v.end());
    anchors.depth.insert(
        anchors.depth.end(), tof_anchors.depth.begin(), tof_anchors.depth.end());
    anchors.var.insert(
        anchors.var.end(), tof_anchors.var.begin(), tof_anchors.var.end());
    return anchors;
}

std::unique_ptr<RescaleResult> Pipeline::process(
    int64_t frame_timestamp_ns,
    const ext_vio_data_t& pkt,
    const float T_imu_image_wrt_vio[3],
    const float R_imu_image_to_vio[3][3],
    const TofFrame* tof,
    const float T_imu_tof_wrt_vio[3],
    const float R_imu_tof_to_vio[3][3],
    const std::vector<float>& disparity) {
    const auto& r = _cfg.rescale;
    if (disparity.size() !=
        static_cast<size_t>(_cfg.inference.input_w * _cfg.inference.input_h)) {
        return nullptr;
    }
    ProjectedAnchors a = build_anchors(
        pkt, T_imu_image_wrt_vio, R_imu_image_to_vio,
        tof, T_imu_tof_wrt_vio, R_imu_tof_to_vio);

    std::vector<double> disp_rel, y, weights_d;
    for (size_t i = 0; i < a.depth.size(); ++i) {
        if (a.depth[i] < r.anchor_depth_min || a.depth[i] > r.anchor_depth_max) continue;
        disp_rel.push_back(sample_disparity(disparity, a.u[i], a.v[i]));
        y.push_back(1.0 / a.depth[i]);
        weights_d.push_back(1.0);
    }

    bool have_fresh = false;
    fits::Fit fresh;
    if (static_cast<int>(y.size()) >= r.min_anchors) {
        fresh = fits::create_robust(
            r.method, disp_rel, y, weights_d,
            1, r.num_knots_spline, r.outlier_rejection, r.outlier_k,
            r.spline_kappa);
        have_fresh = fresh.valid;
    }

    const fits::Fit* fit = nullptr;
    bool    held;
    int64_t calib_age_ns;
    int     n_anchors;

    if (have_fresh) {
        _held = fresh;
        _has_held = true;
        _held_t_ns = frame_timestamp_ns;
        fit = &_held;
        held = false;
        calib_age_ns = 0;
        n_anchors = static_cast<int>(y.size());
    } else if (
        _has_held &&
        frame_timestamp_ns - _held_t_ns <= r.max_hold_age_ns) {
        fit = &_held;
        held = true;
        calib_age_ns = frame_timestamp_ns - _held_t_ns;
        n_anchors = 0;
    } else {
        _has_held = false;
        return nullptr;
    }

    const int npix = _cfg.inference.input_w * _cfg.inference.input_h;
    const double d_lo = 1.0 / r.depth_max, d_hi = 1.0 / r.depth_min;
    auto res = std::make_unique<RescaleResult>();
    res->depth.resize(npix);
    for (int i = 0; i < npix; ++i) {
        double x = std::min(std::max(double(disparity[i]), fit->x_min), fit->x_max);
        double md = fit->predict(x);
        md = std::min(std::max(md, d_lo), d_hi);
        res->depth[i] = float(1.0 / md);
    }
    res->params       = fit->params;
    res->n_anchors    = n_anchors;
    res->inlier_ratio = held ? 0.0f : fit->inlier_ratio;
    res->held         = held;
    res->calib_age_ns = calib_age_ns;
    return res;
}
