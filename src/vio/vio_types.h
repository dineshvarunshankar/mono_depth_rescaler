#pragma once

#if __has_include(<pipe_interfaces/vio_data_t.h>)
#include <cstddef>
#include <pipe_interfaces/vio_data_t.h>

#ifndef VIO_MAX_FEATURES
#define VIO_MAX_FEATURES VIO_MAX_REPORTED_FEATURES
#endif

#ifndef VIO_POINT_LOW
#define VIO_POINT_LOW LOW
#define VIO_POINT_MEDIUM MEDIUM
#define VIO_POINT_HIGH HIGH
#endif

#else

#include <stdint.h>

#define VIO_MAX_FEATURES 64

typedef enum vio_point_quality_t {
    VIO_POINT_LOW = 0,
    VIO_POINT_MEDIUM,
    VIO_POINT_HIGH
} vio_point_quality_t;

#ifndef VIO_STATE_FAILED
#define VIO_STATE_FAILED 0
#define VIO_STATE_INITIALIZING 1
#define VIO_STATE_OK 2
#endif

typedef struct __attribute__((packed)) vio_data_t {
    uint32_t magic_number;
    int32_t  quality;
    int64_t  timestamp_ns;
    float    T_imu_wrt_vio[3];
    float    R_imu_to_vio[3][3];
    float    pose_covariance[21];
    float    vel_imu_wrt_vio[3];
    float    velocity_covariance[21];
    float    imu_angular_vel[3];
    float    gravity_vector[3];
    float    T_cam_wrt_imu[3];
    float    R_cam_to_imu[3][3];
    uint32_t error_code;
    uint16_t n_feature_points;
    uint8_t  state;
    uint8_t  frame;
} vio_data_t;

typedef struct __attribute__((packed)) vio_feature_t {
    uint32_t id;
    int32_t  cam_id;
    float    pix_loc[2];
    float    tsf[3];
    float    p_tsf[3][3];
    float    depth;
    float    depth_error_stddev;
    int32_t  point_quality;
} vio_feature_t;

typedef struct __attribute__((packed)) ext_vio_data_t {
    vio_data_t    v;
    int32_t       last_cam_frame_id;
    int64_t       last_cam_timestamp_ns;
    float         imu_cam_time_shift_s;
    float         gravity_covariance[3][3];
    float         gyro_bias[3];
    float         accl_bias[3];
    uint32_t      n_total_features;
    vio_feature_t features[VIO_MAX_FEATURES];
} ext_vio_data_t;

static_assert(sizeof(vio_data_t) == 324, "vio_data_t size mismatch");
static_assert(sizeof(vio_feature_t) == 76, "vio_feature_t size mismatch");
static_assert(sizeof(ext_vio_data_t) == 5268, "ext_vio_data_t size mismatch");

#endif
