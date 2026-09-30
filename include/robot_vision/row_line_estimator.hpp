#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "robot_vision/distance_estimator.hpp"

namespace robot_vision {

struct BottomSegment {
  cv::Point2f first;
  cv::Point2f last;
  bool from_square{false};
};

struct RowLine {
  cv::Point first;
  cv::Point last;
  std::vector<BottomSegment> observed;
  int square_support{0};
};

// Debug overlay only: extend visible square bases or lower color-mask edges.
std::vector<RowLine> estimate_row_lines(
    const std::array<ColorDebug, 2> &colors,
    const std::vector<Detection> &detections,
    const cv::Size &image_size);

}  // namespace robot_vision
