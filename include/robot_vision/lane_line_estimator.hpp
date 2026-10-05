#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace robot_vision {

struct LaneConfig {
  int white_max_saturation{85};
  int white_min_value{175};
  int grass_hue_min{30};
  int grass_hue_max{95};
  int grass_min_saturation{45};
  double roi_top_fraction{0.18};
  double candidate_min_bottom_y_fraction{0.70};
  double reference_y_fraction{0.85};
  double min_abs_dx_per_dy{0.35};
  double max_abs_dx_per_dy{2.3};
  double min_observed_height_fraction{0.18};
  double max_fit_error_px{8.0};
  double group_tolerance_px{18.0};
  double min_grass_support{0.60};
  // Allow a boundary near image center; reject lines crossing into the opposite half.
  double bottom_outer_fraction{0.49};
};

struct LaneLine {
  bool valid{false};
  std::string side;  // left or right, classified by image slope
  cv::Point2f top, bottom;
  int observed_y_min{0}, observed_y_max{0};
  float reference_y_px{0};
  float line_x_at_reference_px{0};
  float pixel_separation_px{0};
  double grass_support{0};
  double fit_error_px{0};
  std::vector<cv::Vec4i> observed_segments;
};

struct LaneResult {
  LaneLine best;
  cv::Mat mask;
};

LaneResult estimate_lane_line(const cv::Mat &bgr, const LaneConfig &config,
                              const cv::Mat &exclude_mask = {});

}  // namespace robot_vision
