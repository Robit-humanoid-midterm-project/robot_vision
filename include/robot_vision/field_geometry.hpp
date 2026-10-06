#pragma once

#include <optional>
#include <vector>
#include <opencv2/core.hpp>

#include "robot_vision/distance_estimator.hpp"
#include "robot_vision/lane_line_estimator.hpp"
#include "robot_vision/row_line_estimator.hpp"

namespace robot_vision {

enum class CrossingStatus { no_lane_or_row, no_intersection, no_row_depth, measured };

struct FieldGeometryResult {
    CrossingStatus status{CrossingStatus::no_lane_or_row};
    std::optional<cv::Point2d> crossing;
    std::optional<cv::Point> principal;
    double lateral_m{0.0};
    double left_distance_m{-1000.0};
    double right_distance_m{-1000.0};
};

// Existing assumption: the camera is inside a 1.5 m wide field and faces its length.
FieldGeometryResult estimate_field_geometry(
    const DetectResult &obstacles, const LaneResult &lane,
    const std::vector<RowLine> &rows, const cv::Size &size, const DetectorConfig &config);

} // namespace robot_vision
