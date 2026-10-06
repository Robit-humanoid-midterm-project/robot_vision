#pragma once

#include <opencv2/core.hpp>

namespace robot_vision {

struct BevLine {
  bool valid{false};
  cv::Point2f top{}, bottom{};
  double grass_support{0};
  double fit_error_px{0};
  bool held{false};  // Last reliable observation briefly retained through a missed frame.
};

struct BevLineResult {
  cv::Mat white_mask;
  BevLine left, right;
};

struct BevLineConfig {
  int white_min_value{200};
  int white_max_saturation{85};
  int grass_hue_min{43};
  int grass_hue_max{95};
  int grass_min_saturation{60};
  double min_grass_support{0.75};
  double min_vertical_span_fraction{0.20};
  double max_abs_dx_per_dy{0.25};
  double max_fit_error_px{7.0};
};

BevLineResult detect_bev_lines(const cv::Mat &bev_bgr, const BevLineConfig &config);

// Confirm a new line over two frames, smooth small movements, and bridge brief misses.
// Call once for each newly processed frame, not for repeated redraws of the same frame.
class BevLineTemporalFilter {
 public:
  BevLineResult update(BevLineResult current);
  void reset();

 private:
  struct Track {
    BevLine line, pending;
    int confirmations{0};
    int misses{0};
  };
  static BevLine update_track(Track &track, const BevLine &observation);
  Track left_, right_;
};

}  // namespace robot_vision
