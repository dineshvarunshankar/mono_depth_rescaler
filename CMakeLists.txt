cmake_minimum_required(VERSION 3.16)
project(mono_depth_rescaler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(yaml-cpp REQUIRED)

add_library(rescaler_core
    src/core/config.cpp
    src/core/camera_model.cpp
    src/core/preprocess.cpp
    src/core/geometry.cpp
    src/core/fits.cpp
    src/core/pipeline.cpp
    src/core/tof_anchors.cpp
    src/tof/tof_types.cpp
    src/vio/pose_buffer.cpp)
target_include_directories(rescaler_core PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rescaler_core PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp)
target_compile_options(rescaler_core PRIVATE -Wall -Wextra -Wpedantic)

option(BUILD_VOXL_APP "Build the ModalAI MPA executable" ON)
if(BUILD_VOXL_APP)
    find_library(MODAL_PIPE_LIB modal_pipe PATHS /usr/lib /usr/local/lib)
    find_path(MODAL_PIPE_INC modal_pipe.h PATHS /usr/include /usr/local/include)
    if(MODAL_PIPE_LIB AND MODAL_PIPE_INC)
        add_executable(mono_depth_rescaler
            src/main.cpp
            src/vio/mpa_vio_source.cpp
            src/inference/mpa_backend.cpp
            src/tof/tof_source.cpp)
        target_include_directories(
            mono_depth_rescaler PRIVATE ${MODAL_PIPE_INC})
        target_link_libraries(
            mono_depth_rescaler PRIVATE rescaler_core ${MODAL_PIPE_LIB})
        target_compile_options(
            mono_depth_rescaler PRIVATE -Wall -Wextra -Wpedantic)
        install(TARGETS mono_depth_rescaler DESTINATION /usr/bin)
    else()
        message(STATUS "ModalAI SDK not found; skipping VOXL executable")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(rescaler_core_tests tests/cpp/test_core.cpp)
    target_link_libraries(rescaler_core_tests PRIVATE rescaler_core)
    target_compile_definitions(
        rescaler_core_tests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME rescaler_core_tests COMMAND rescaler_core_tests)
endif()

install(DIRECTORY config/ DESTINATION /etc/mono_depth_rescaler)
