#include "pipeline.h"
#include "tof_anchors.h"
#include "anchor_grid.h"

#include <algorithm>
#include <cmath>

namespace {
void append(ProjectedAnchors& dst, const ProjectedAnchors& src) {
    dst.u.insert(dst.u.end(), src.u.begin(), src.u.end());
    dst.v.insert(dst.v.end(), src.v.begin(), src.v.end());
    dst.depth.insert(dst.depth.end(), src.depth.begin(), src.depth.end());
    dst.var.insert(dst.var.end(), src.var.begin(), src.var.end());
}

ProjectedAnchors gate(const ProjectedAnchors& a, float lo, float hi) {
    ProjectedAnchors out;
    for (size_t i = 0; i < a.depth.size(); ++i) {
        if (a.depth[i] >= lo && a.depth[i] <= hi) {
            out.u.push_back(a.u[i]);
            out.v.push_back(a.v[i]);
            out.depth.push_back(a.depth[i]);
            out.var.push_back(a.var[i]);
        }
    }
    return out;
}

ProjectedAnchors gather(const ProjectedAnchors& a, const std::vector<size_t>& idx) {
    ProjectedAnchors out;
    for (const size_t i : idx) {
        out.u.push_back(a.u[i]);
        out.v.push_back(a.v[i]);
        out.depth.push_back(a.depth[i]);
        out.var.push_back(a.var[i]);
    }
    return out;
}
}  // namespace

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
    ProjectedAnchors vio = project_features(
        pkt, T_imu_image_wrt_vio, R_imu_image_to_vio,
        _cfg.extr_hires.R, _cfg.extr_hires.T,
        _pre, _cfg.vio.min_quality);
    if (!tof) {
        return vio;  // no ToF: keep all VIO
    }

    const auto& a = _cfg.anchors;
    const int grid = a.tof_cell_px > 0 ? _pre.dst_w() / a.tof_cell_px : 0;
    const int cap = grid > 0 ? 0 : a.tof_max_points;
    ProjectedAnchors tof_a = project_tof_anchors(
        *tof, T_imu_tof_wrt_vio, R_imu_tof_to_vio,
        T_imu_image_wrt_vio, R_imu_image_to_vio,
        _cfg.extr_tof, _cfg.extr_hires, _pre,
        a.tof_confidence_min, cap);

    if (grid <= 0) {
        append(vio, tof_a);  // no grid: union VIO + capped ToF
        return vio;
    }

    const auto& r = _cfg.rescale;
    ProjectedAnchors v = gate(vio, r.anchor_depth_min, r.anchor_depth_max);
    ProjectedAnchors t = gate(
        tof_a, r.anchor_depth_min,
        std::min(r.anchor_depth_max, a.tof_trust_range_m));
    std::vector<size_t> tk;
    grid_thin(t.u, t.v, t.depth, grid, a.max_per_cell,
              static_cast<float>(_pre.dst_w()), static_cast<float>(_pre.dst_h()),
              a.tof_cell_pick, tk);
    ProjectedAnchors out = v;  // all VIO + thinned ToF
    append(out, gather(t, tk));
    if (out.depth.size() < static_cast<size_t>(r.min_anchors)) {
        out = v;  // scarce fallback: keep all
        append(out, t);
    }
    return out;
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

    _last_anchors = static_cast<int>(y.size());
    if (!disp_rel.empty()) {
        auto xr = std::minmax_element(disp_rel.begin(), disp_rel.end());
        auto yr = std::minmax_element(y.begin(), y.end());
        _last_x_lo = *xr.first; _last_x_hi = *xr.second;
        _last_y_lo = *yr.first; _last_y_hi = *yr.second;
    }

    bool have_fresh = false;
    fits::Fit fresh;
    if (static_cast<int>(y.size()) >= r.min_anchors) {
        fresh = fits::create_robust(
            r.method, disp_rel, y, weights_d,
            1, r.num_knots_spline, r.outlier_rejection, r.outlier_k,
            r.spline_kappa);
        have_fresh = fresh.valid;
        _last_fit_reason = fresh.invalid_reason;
    }

    if (have_fresh) {
        _held = fresh;
        _has_held = true;
        _held_t_ns = frame_timestamp_ns;
        return render(_held, disparity, false, static_cast<int>(y.size()), 0);
    }
    return apply_held(frame_timestamp_ns, disparity);
}

std::unique_ptr<RescaleResult> Pipeline::apply_held(
    int64_t frame_timestamp_ns, const std::vector<float>& disparity) {
    if (disparity.size() !=
        static_cast<size_t>(_cfg.inference.input_w * _cfg.inference.input_h)) {
        return nullptr;
    }
    if (!_has_held ||
        frame_timestamp_ns - _held_t_ns > _cfg.rescale.max_hold_age_ns) {
        _has_held = false;
        return nullptr;
    }
    return render(
        _held, disparity, true, 0, frame_timestamp_ns - _held_t_ns);
}

std::unique_ptr<RescaleResult> Pipeline::render(
    const fits::Fit& fit, const std::vector<float>& disparity,
    bool held, int n_anchors, int64_t calib_age_ns) const {
    const auto& r = _cfg.rescale;
    const int npix = _cfg.inference.input_w * _cfg.inference.input_h;
    const double d_lo = 1.0 / r.depth_max, d_hi = 1.0 / r.depth_min;
    auto res = std::make_unique<RescaleResult>();
    res->depth.resize(npix);
    for (int i = 0; i < npix; ++i) {
        double x = std::min(std::max(double(disparity[i]), fit.x_min), fit.x_max);
        double md = fit.predict(x);
        md = std::min(std::max(md, d_lo), d_hi);
        res->depth[i] = float(1.0 / md);
    }
    res->params       = fit.params;
    res->n_anchors    = n_anchors;
    res->inlier_ratio = held ? 0.0f : fit.inlier_ratio;
    res->held         = held;
    res->calib_age_ns = calib_age_ns;
    return res;
}
