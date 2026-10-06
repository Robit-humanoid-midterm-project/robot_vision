#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "robot_vision/vision_frame_result.hpp"

namespace robot_vision {

// Builds debug images even with the local viewer disabled, preserving ROS outputs.
class VisionViewer {
  public:
    VisionViewer(DetectorConfig config, bool enabled, bool show_preprocess, bool fullscreen);
    ~VisionViewer();
    void set_processing_fps(double fps) { processing_fps_ = fps; }
    VisionViewer(const VisionViewer &) = delete;
    VisionViewer &operator=(const VisionViewer &) = delete;
    cv::Mat annotate(const cv::Mat &frame, const VisionFrameResult &result) const;
    cv::Mat preprocess_view(const DetectResult &result) const;
    cv::Mat no_image_view(const std::string &image_topic) const;
    cv::Mat no_image_preprocess() const;
    void show(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
              const cv::Mat &annotated, const cv::Mat &preprocess);

  private:
    void show_dashboard(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                        const cv::Mat &annotated);
    DetectorConfig debug_config_;
    bool viewer_{false}, show_preprocess_{true}, viewer_fullscreen_{true};
    double processing_fps_{0.0};
    bool window_initialized_{false}, preprocess_initialized_{false};
    const std::string window_name_ = "Robot vision - RAW / OPENCV / RESULT";
    const std::string preprocess_window_name_ = "Obstacle preprocess - RED / BLUE";
};

} // namespace robot_vision
