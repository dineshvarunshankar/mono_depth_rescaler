#pragma once
#include "camera_model.h"
#include <opencv2/core.hpp>
#include <string>

// Image transform + 3D-point projection into model pixel space.
// Port of proto/rescaler/preprocess.py.
//   raw        resize only; project through the fisheye model
//   undistort  un-bend + resize; project via pinhole K_model
class Preprocessor {
public:
    Preprocessor(const CameraModel& camera,
                 int src_w, int src_h,
                 int dst_w, int dst_h,
                 const std::string& mode = "raw",   // raw | undistort
                 const std::string& fov  = "crop",  // crop | fit | stretch
                 bool antialias = true);

    // Raw capture frame -> model input (dst_h x dst_w). Used by the inline
    // TFLite path; the MPA path receives disparity already in model space.
    cv::Mat prepare(const cv::Mat& image) const;

    // Project a camera-frame point into MODEL pixel coordinates.
    // Returns true when the point is in front of the camera (usable);
    // (u, v) are only meaningful then.
    bool project(float X, float Y, float Z, float& u, float& v) const;

    int dst_w() const { return _dst_w; }
    int dst_h() const { return _dst_h; }

private:
    cv::Mat _build_k_model(const std::string& fov) const;
    void    _source_fov_tan(double& x_max, double& y_max) const;

    const CameraModel& _camera;
    std::string _mode;
    int   _src_w, _src_h, _dst_w, _dst_h;
    bool  _antialias;
    double _scale, _xr, _yr;

    cv::Mat _K_model;   // 3x3, only in undistort mode
    double  _fx_m{0}, _fy_m{0}, _cx_m{0}, _cy_m{0};
    cv::Mat _map1, _map2;
};
