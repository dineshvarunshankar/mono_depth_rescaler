#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <string>

class CameraModel {
public:
    CameraModel(float fx, float fy, float cx, float cy,
                const std::array<float, 5>& D, const std::string& model);

    // Undistort a raw fisheye frame to the virtual pinhole.
    cv::Mat undistort(const cv::Mat& src) const;

    // Project to undistorted (pinhole) pixel; for the undistorted-frame path.
    bool project(float x, float y, float z, float& u, float& v) const;

    // Project to distorted (fisheye) pixel; for the raw-frame (MPA) path.
    bool project_distorted(float x, float y, float z, float& u, float& v) const;

    float fx() const { return _K.at<double>(0, 0); }
    float fy() const { return _K.at<double>(1, 1); }
    float cx() const { return _K.at<double>(0, 2); }
    float cy() const { return _K.at<double>(1, 2); }

    const cv::Mat&     K() const { return _K; }
    const cv::Mat&     D() const { return _D; }
    const std::string& model() const { return _model; }

private:
    cv::Mat  _K, _D;
    std::string _model;
    mutable cv::Mat _map1, _map2;
    mutable bool    _maps_ready{false};
    mutable int     _width{0}, _height{0};   // undistort() lazily caches its map size
};
