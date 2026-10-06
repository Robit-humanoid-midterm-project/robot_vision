#pragma once

#include "robot_vision/vision_frame_result.hpp"

namespace robot_vision {

struct RowTrackerConfig {
    int history_frames{5};
    double match_y_px{35.0};
    double match_slope{0.18};
    int max_missing_frames{3};
};

// Calculates results only; ROS scheduling, publishing and rendering stay outside.
class VisionPipeline {
  public:
    VisionPipeline(DetectorConfig detector, LaneConfig lane, double camera_height_m,
                   RowTrackerConfig rows = {});
    VisionFrameResult detect_obstacles(const cv::Mat &frame) const;
    void complete(const cv::Mat &frame, VisionFrameResult &result);
    void reset_tracking();

  private:
    DetectorConfig detector_config_;
    LaneConfig lane_config_;
    double camera_height_m_;
    DistanceEstimator estimator_;
    RowLineTracker row_tracker_;
};

} // namespace robot_vision
