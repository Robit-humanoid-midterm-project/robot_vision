#pragma once

#include "robot_vision/field_geometry.hpp"

namespace robot_vision {
struct GroundFieldConfig {
    // Points belong to the undistorted image using the original K, without cropping.
    std::vector<double> source_points;
    // Calibrated rectangle extends 0.23m outside each side of the 1.4m field.
    double x_min_m{-0.23}, width_m{1.86}, near_m{1.5}, far_m{3.75};
    double field_width_m{1.4}, start_from_left_m{0.7};
    int reference_frames{5};
};

// Metric ground plane is fixed to the camera mounting, not to the field.
// Lock the robot origin at the known start, then measure each new field line.
class GroundFieldEstimator {
 public:
    GroundFieldEstimator(const DetectorConfig &camera, GroundFieldConfig config);
    FieldGeometryResult estimate(const LaneResult &lanes, cv::Size size);
    void clear_pending_reference(); // A camera timeout must not re-anchor a moved robot.
    cv::Mat preview(const cv::Mat &raw, const LaneResult &lanes) const;
    bool ready() const { return ready_; }
 private:
    struct Line { double slope, intercept, norm; };
    std::optional<Line> project(const LaneLine &line) const;
    GroundFieldConfig config_;
    cv::Size size_;
    cv::Mat k_, distortion_, homography_, map_x_, map_y_;
    double robot_x_{0};
    int samples_{0};
    bool ready_{false};
};
} // namespace robot_vision
