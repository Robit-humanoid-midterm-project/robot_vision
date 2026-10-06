#include "robot_vision/bev_line_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace robot_vision {
namespace {
struct Segment {
  cv::Vec4i ends;
  double reference_x;
  double slope;
  double grass;
};

double grass_support(const cv::Mat &grass, const cv::Vec4i &line, int direction) {
  int checked = 0, matched = 0;
  for (int sample = 1; sample <= 9; ++sample) {
    const double t = sample / 10.0;
    const int x = cvRound(line[0] + t * (line[2] - line[0]));
    const int y = cvRound(line[1] + t * (line[3] - line[1]));
    if (y < 0 || y >= grass.rows) continue;
    for (int offset : {12, 24, 36}) {
      const int inside_x = x + direction * offset;
      if (inside_x < 0 || inside_x >= grass.cols) continue;
      ++checked;
      matched += grass.at<unsigned char>(y, inside_x) != 0;
    }
  }
  return checked ? double(matched) / checked : 0.0;
}

BevLine choose_line(const std::vector<Segment> &segments, int width, int height,
                    const BevLineConfig &config) {
  BevLine best;
  double best_score = -1e9;
  for (const auto &seed : segments) {
    std::vector<cv::Point2f> points;
    int min_y = height, max_y = 0;
    double grass_sum = 0;
    int count = 0;
    for (const auto &candidate : segments) {
      if (std::abs(candidate.reference_x - seed.reference_x) > 15 ||
          std::abs(candidate.slope - seed.slope) > 0.12) continue;
      const auto &s = candidate.ends;
      points.emplace_back(s[0], s[1]);
      points.emplace_back(s[2], s[3]);
      min_y = std::min(min_y, std::min(s[1], s[3]));
      max_y = std::max(max_y, std::max(s[1], s[3]));
      grass_sum += candidate.grass;
      ++count;
    }
    if (points.size() < 2 || max_y - min_y < height * config.min_vertical_span_fraction) continue;
    cv::Vec4f fit;
    cv::fitLine(points, fit, cv::DIST_HUBER, 0, 0.01, 0.01);
    if (std::abs(fit[1]) < 1e-5) continue;
    const double slope = fit[0] / fit[1];
    if (std::abs(slope) > config.max_abs_dx_per_dy) continue;
    const auto x_at = [&](double y) { return fit[2] + slope * (y - fit[3]); };
    double sum_squared = 0;
    for (const auto &point : points) sum_squared += std::pow(point.x - x_at(point.y), 2);
    const double fit_error = std::sqrt(sum_squared / points.size());
    if (fit_error > config.max_fit_error_px) continue;
    const double reference_x = x_at(height * 0.5);
    if (reference_x < 0 || reference_x >= width) continue;
    const double score = max_y - min_y + 80 * grass_sum / count - 8 * fit_error;
    if (score <= best_score) continue;
    best_score = score;
    best = {true, {float(x_at(min_y)), float(min_y)},
            {float(x_at(max_y)), float(max_y)}, grass_sum / count, fit_error};
  }
  return best;
}
}  // namespace

BevLineResult detect_bev_lines(const cv::Mat &bev_bgr, const BevLineConfig &config) {
  if (bev_bgr.empty() || bev_bgr.type() != CV_8UC3)
    throw std::invalid_argument("BEV line detector expects BGR image");
  if (config.white_min_value < 0 || config.white_min_value > 255 ||
      config.white_max_saturation < 0 || config.white_max_saturation > 255 ||
      config.grass_hue_min < 0 || config.grass_hue_max > 179 ||
      config.grass_hue_min > config.grass_hue_max ||
      config.grass_min_saturation < 0 || config.grass_min_saturation > 255 ||
      config.min_grass_support < 0 || config.min_grass_support > 1 ||
      config.min_vertical_span_fraction <= 0 || config.min_vertical_span_fraction > 1 ||
      config.max_abs_dx_per_dy <= 0 || config.max_fit_error_px <= 0)
    throw std::invalid_argument("Invalid BEV line detector settings");

  cv::Mat hsv, grass, edges;
  cv::cvtColor(bev_bgr, hsv, cv::COLOR_BGR2HSV);
  BevLineResult result;
  cv::inRange(hsv, cv::Scalar(0, 0, config.white_min_value),
              cv::Scalar(179, config.white_max_saturation, 255), result.white_mask);
  cv::inRange(hsv, cv::Scalar(config.grass_hue_min, config.grass_min_saturation, 30),
              cv::Scalar(config.grass_hue_max, 255, 255), grass);
  cv::morphologyEx(result.white_mask, result.white_mask, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
  cv::Canny(result.white_mask, edges, 60, 140);
  std::vector<cv::Vec4i> hough;
  cv::HoughLinesP(edges, hough, 1, CV_PI / 180, 30,
                  std::max(60.0, bev_bgr.rows * 0.16), 22);
  std::vector<Segment> left, right;
  for (const auto &s : hough) {
    const double dy = s[3] - s[1];
    if (std::abs(dy) < bev_bgr.rows * 0.15) continue;
    const double slope = (s[2] - s[0]) / dy;
    if (std::abs(slope) > config.max_abs_dx_per_dy) continue;
    const double reference_x = s[0] + slope * (bev_bgr.rows * 0.5 - s[1]);
    if (reference_x >= 0 && reference_x < bev_bgr.cols * 0.5) {
      const double support = grass_support(grass, s, +1);
      if (support >= config.min_grass_support)
        left.push_back({s, reference_x, slope, support});
    } else if (reference_x >= bev_bgr.cols * 0.5 && reference_x < bev_bgr.cols) {
      const double support = grass_support(grass, s, -1);
      if (support >= config.min_grass_support)
        right.push_back({s, reference_x, slope, support});
    }
  }
  result.left = choose_line(left, bev_bgr.cols, bev_bgr.rows, config);
  result.right = choose_line(right, bev_bgr.cols, bev_bgr.rows, config);
  return result;
}

BevLine BevLineTemporalFilter::update_track(Track &track, const BevLine &observation) {
  const auto separation = [](const BevLine &a, const BevLine &b) {
    return std::abs((a.top.x + a.bottom.x - b.top.x - b.bottom.x) * 0.5);
  };
  if (!observation.valid) {
    track.pending = {};
    track.confirmations = 0;
    if (track.line.valid && ++track.misses <= 4) {
      auto held = track.line;
      held.held = true;
      return held;
    }
    track.line = {};
    return {};
  }
  if (track.line.valid && separation(track.line, observation) <= 35) {
    track.misses = 0;
    constexpr float alpha = 0.35f;
    track.line.top = track.line.top * (1 - alpha) + observation.top * alpha;
    track.line.bottom = track.line.bottom * (1 - alpha) + observation.bottom * alpha;
    track.line.grass_support = observation.grass_support;
    track.line.fit_error_px = observation.fit_error_px;
    track.line.held = false;
    track.pending = {};
    track.confirmations = 0;
    return track.line;
  }
  if (track.pending.valid && separation(track.pending, observation) <= 25) {
    ++track.confirmations;
  } else {
    track.pending = observation;
    track.confirmations = 1;
  }
  // A jump needs one extra confirmation to avoid switching to a passing object.
  const int required = track.line.valid ? 3 : 2;
  if (track.confirmations >= required) {
    track.line = observation;
    track.misses = 0;
    track.pending = {};
    track.confirmations = 0;
    return track.line;
  }
  if (track.line.valid) {
    if (++track.misses > 4) {
      track.line = {};
      return {};
    }
    auto held = track.line;
    held.held = true;
    return held;
  }
  return {};
}

BevLineResult BevLineTemporalFilter::update(BevLineResult current) {
  current.left = update_track(left_, current.left);
  current.right = update_track(right_, current.right);
  return current;
}

void BevLineTemporalFilter::reset() {
  left_ = {};
  right_ = {};
}
}  // namespace robot_vision
