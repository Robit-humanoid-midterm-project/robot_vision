#pragma once

#include <array>
#include <deque>
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

// A track is matched by image height and slope, never by obstacle color or slot.
// Only rows visible in the current frame are returned; old rows are not drawn.
class RowLineTracker {
 public:
  RowLineTracker(int history_frames = 5, double match_y_px = 35.0,
                 double match_slope = 0.18, int max_missing_frames = 3);
  std::vector<RowLine> smooth(const std::vector<RowLine> &rows,
                              const cv::Size &image_size);
  void reset();

 private:
  struct Track {
    std::deque<cv::Vec2d> samples;  // center y, slope
    int missing_frames{0};
  };
  int history_frames_, max_missing_frames_;
  double match_y_px_, match_slope_;
  cv::Size previous_size_{};
  std::vector<Track> tracks_;
};

}  // namespace robot_vision
