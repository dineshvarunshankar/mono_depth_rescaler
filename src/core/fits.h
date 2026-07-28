#pragma once
#include <vector>
#include <string>
#include <functional>

// Rescaling curve fits: relative disparity x -> metric disparity y = 1/depth.
// Port of proto/rescaler/fits/.
namespace fits {

struct Fit {
    bool   valid{false};
    double x_min{0.0}, x_max{0.0};
    std::function<double(double)> predict;   // metric disparity from relative
    std::vector<double> params;              // empty for splines (not smoothable)
    std::vector<double> params_cov;          // n*n row-major; empty if unavailable
    bool  has_inliers{false};
    float inlier_ratio{1.0f};
};

// Usable iff finite, positive, non-decreasing over [x_min, x_max].
bool is_valid(const std::function<double(double)>& predict,
              double x_min, double x_max, int n = 32);

// Fit `method` to (x, y, w); invalid Fit on too few points or solver failure.
Fit create(const std::string& method,
           const std::vector<double>& x,
           const std::vector<double>& y,
           const std::vector<double>& w,
           int degree = 1, int num_knots_spline = 10,
           double spline_kappa = 1.0e6);

// create(), then optionally re-fit on the MAD inlier set (method's own residuals).
Fit create_robust(const std::string& method,
                  const std::vector<double>& x,
                  const std::vector<double>& y,
                  const std::vector<double>& w,
                  int degree = 1, int num_knots_spline = 10,
                  bool outlier_rejection = false, float outlier_k = 3.0f,
                  double spline_kappa = 1.0e6);

// Rebuild predictor from a params vector (polynomial/exponential only).
std::function<double(double)> rebuild(const std::string& method,
                                      const std::vector<double>& params);

bool is_known_method(const std::string& method);

}  // namespace fits
