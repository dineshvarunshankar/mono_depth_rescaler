#include "camera_model.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

CameraModel::CameraModel(float fx, float fy, float cx, float cy,
                         const std::array<float, 5>& D, const std::string& model)
    : _model(model) {
    _K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    // pinhole (Brown-Conrady) uses all 5: k1,k2,p1,p2,k3.
    // fisheye (Kannala-Brandt) uses only the first 4: k1..k4 (D[4] left 0).
    _D = _model == "fisheye"
        ? (cv::Mat_<double>(4, 1) << D[0], D[1], D[2], D[3])
        : (cv::Mat_<double>(5, 1) << D[0], D[1], D[2], D[3], D[4]);
}

cv::Mat CameraModel::undistort(const cv::Mat& src) const {
    if (!_maps_ready || _width != src.cols || _height != src.rows) {
        _width  = src.cols;
        _height = src.rows;
        if (_model == "fisheye") {
            cv::fisheye::initUndistortRectifyMap(
                _K, _D, cv::Mat::eye(3, 3, CV_64F), _K,
                src.size(), CV_32FC1, _map1, _map2);
        } else {
            cv::initUndistortRectifyMap(
                _K, _D, cv::Mat::eye(3, 3, CV_64F), _K,
                src.size(), CV_32FC1, _map1, _map2);
        }
        _maps_ready = true;
    }
    cv::Mat dst;
    cv::remap(src, dst, _map1, _map2, cv::INTER_LINEAR);
    return dst;
}

bool CameraModel::project(float x, float y, float z, float& u, float& v) const {
    if (z <= 0.0f) return false;
    u = static_cast<float>(_K.at<double>(0, 0)) * (x / z) + static_cast<float>(_K.at<double>(0, 2));
    v = static_cast<float>(_K.at<double>(1, 1)) * (y / z) + static_cast<float>(_K.at<double>(1, 2));
    return true;
}

bool CameraModel::project_distorted(float x, float y, float z, float& u, float& v) const {
    if (z <= 0.0f) return false;
    std::vector<cv::Point3f> pts3d = {{x, y, z}};
    std::vector<cv::Point2f> pts2d;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    if (_model == "fisheye")
        cv::fisheye::projectPoints(pts3d, pts2d, rvec, tvec, _K, _D);
    else
        cv::projectPoints(pts3d, rvec, tvec, _K, _D, pts2d);
    u = pts2d[0].x;
    v = pts2d[0].y;
    return true;
}
