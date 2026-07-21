#include "preprocess.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <stdexcept>

Preprocessor::Preprocessor(const CameraModel& camera,
                           int src_w, int src_h,
                           int dst_w, int dst_h,
                           const std::string& mode,
                           const std::string& fov,
                           bool antialias)
    : _camera(camera), _mode(mode),
      _src_w(src_w), _src_h(src_h), _dst_w(dst_w), _dst_h(dst_h),
      _antialias(antialias) {
    if (mode != "raw" && mode != "undistort")
        throw std::invalid_argument("preprocess mode must be raw|undistort");
    if (fov != "crop" && fov != "fit" && fov != "stretch")
        throw std::invalid_argument("fov must be crop|fit|stretch");

    // 4x downscale through a 2x2 bilinear gather aliases; pre-blur first.
    _scale = std::max(double(_src_w) / _dst_w, double(_src_h) / _dst_h);
    _xr    = double(_src_w) / _dst_w;
    _yr    = double(_src_h) / _dst_h;

    if (mode == "undistort") {
        _K_model = _build_k_model(fov);
        _fx_m = _K_model.at<double>(0, 0);
        _fy_m = _K_model.at<double>(1, 1);
        _cx_m = _K_model.at<double>(0, 2);
        _cy_m = _K_model.at<double>(1, 2);
        cv::Size size(_dst_w, _dst_h);
        if (_camera.model() == "fisheye")
            cv::fisheye::initUndistortRectifyMap(
                _camera.K(), _camera.D(), cv::Mat::eye(3, 3, CV_64F),
                _K_model, size, CV_32FC1, _map1, _map2);
        else
            cv::initUndistortRectifyMap(
                _camera.K(), _camera.D(), cv::Mat(),
                _K_model, size, CV_32FC1, _map1, _map2);
    } else {
        // Half-pixel-centred resize maps: src = (dst + 0.5) * ratio - 0.5.
        _map1.create(_dst_h, _dst_w, CV_32FC1);
        _map2.create(_dst_h, _dst_w, CV_32FC1);
        for (int y = 0; y < _dst_h; ++y) {
            float sy = float((y + 0.5) * _yr - 0.5);
            for (int x = 0; x < _dst_w; ++x) {
                _map1.at<float>(y, x) = float((x + 0.5) * _xr - 0.5);
                _map2.at<float>(y, x) = sy;
            }
        }
    }
}

void Preprocessor::_source_fov_tan(double& x_max, double& y_max) const {
    std::vector<cv::Point2f> pts = {
        {0.0f,             _src_h / 2.0f},
        {_src_w - 1.0f,    _src_h / 2.0f},
        {_src_w / 2.0f,    0.0f},
        {_src_w / 2.0f,    _src_h - 1.0f},
    };
    std::vector<cv::Point2f> n;
    if (_camera.model() == "fisheye")
        cv::fisheye::undistortPoints(pts, n, _camera.K(), _camera.D());
    else
        cv::undistortPoints(pts, n, _camera.K(), _camera.D());
    x_max = std::max(std::abs(n[0].x), std::abs(n[1].x));
    y_max = std::max(std::abs(n[2].y), std::abs(n[3].y));
}

cv::Mat Preprocessor::_build_k_model(const std::string& fov) const {
    double x_max, y_max;
    _source_fov_tan(x_max, y_max);
    double hw = _dst_w / 2.0, hh = _dst_h / 2.0;   // cx, cy
    double fx, fy;
    if (fov == "stretch") {
        fx = hw / x_max;
        fy = hh / y_max;
    } else {
        // 'fit' keeps the whole source (blank borders); 'crop' fills the frame
        // (discards the wider axis)
        double f = (fov == "fit") ? std::min(hw / x_max, hh / y_max)
                                  : std::max(hw / x_max, hh / y_max);
        fx = fy = f;
    }
    return (cv::Mat_<double>(3, 3) << fx, 0.0, hw, 0.0, fy, hh, 0.0, 0.0, 1.0);
}

cv::Mat Preprocessor::prepare(const cv::Mat& image) const {
    cv::Mat src;   // blur into a fresh buffer -- never mutate the caller's image
    if (_antialias && _scale > 1.5)
        cv::GaussianBlur(image, src, cv::Size(0, 0), 0.5 * _scale);
    else
        src = image;
    cv::Mat dst;
    cv::remap(src, dst, _map1, _map2, cv::INTER_LINEAR);
    return dst;
}

bool Preprocessor::project(float X, float Y, float Z, float& u, float& v) const {
    if (Z <= 1e-3f) return false;
    if (_mode == "undistort") {
        u = float(_fx_m * X / Z + _cx_m);
        v = float(_fy_m * Y / Z + _cy_m);
        return true;
    }
    // raw: distorted projection into source pixels, then the exact inverse of
    // the half-pixel-centred resize maps.
    float us, vs;
    if (!_camera.project_distorted(X, Y, Z, us, vs)) return false;
    u = float((us + 0.5) / _xr - 0.5);
    v = float((vs + 0.5) / _yr - 0.5);
    return true;
}
