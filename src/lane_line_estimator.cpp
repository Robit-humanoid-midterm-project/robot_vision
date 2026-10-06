// 파일 역할: 잔디 주변의 흰색 직선에서 왼쪽 또는 오른쪽 경계선 하나를 선택한다.
// 픽셀 기준 선 위치와 품질을 반환하며, m 단위 경계 거리는 field_geometry.cpp가 계산한다.

#include "robot_vision/lane_line_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace robot_vision {
namespace {
struct Candidate {
  cv::Vec4i segment;
  double slope;
  double reference_x;
  int side;
};

// 지정한 HSV 범위를 만족하는 잔디색 픽셀인지 확인한다. 영상 밖 좌표는 false로 처리한다.
bool is_grass(const cv::Mat &hsv, int x, int y, const LaneConfig &config) {
  if (x < 0 || x >= hsv.cols || y < 0 || y >= hsv.rows) return false;
  const auto pixel = hsv.at<cv::Vec3b>(y, x);
  return pixel[0] >= config.grass_hue_min && pixel[0] <= config.grass_hue_max &&
         pixel[1] >= config.grass_min_saturation && pixel[2] >= 30;
}

// 선분의 경기장 안쪽에서 여러 픽셀을 검사해 잔디색 비율을 반환한다.
// 선반·물체의 흰색 선을 바닥 경계선으로 잘못 고르는 일을 줄이기 위한 조건이다.
double grass_fraction(const cv::Mat &hsv, const cv::Vec4i &segment, int side,
                      const LaneConfig &config) {
  int total = 0, green = 0;
  const int inside = side == 0 ? 1 : -1;
  for (int i = 1; i <= 9; ++i) {
    const double t = i / 10.0;
    const int x = cvRound(segment[0] + t * (segment[2] - segment[0]));
    const int y = cvRound(segment[1] + t * (segment[3] - segment[1]));
    for (int offset : {8, 14, 20}) {
      const int sample_x = x + inside * offset;
      if (sample_x < 0 || sample_x >= hsv.cols || y < 0 || y >= hsv.rows) continue;
      ++total;
      if (is_grass(hsv, sample_x, y, config)) ++green;
    }
  }
  return total ? static_cast<double>(green) / total : 0.0;
}

}  // namespace

// 흰색 마스크 → 선분 추출 → 위치·기울기·잔디 검사 → 직선 맞춤 순으로 경계선을 선택한다.
// 반환 마스크는 디버그 표시용이고 best.valid가 실제 선 검출 성공 여부다.
LaneResult estimate_lane_line(const cv::Mat &bgr, const LaneConfig &config,
                              const cv::Mat &exclude_mask) {
  if (bgr.empty() || bgr.type() != CV_8UC3) throw std::invalid_argument("Expected BGR image");
  if (config.white_max_saturation < 0 || config.white_max_saturation > 255 ||
      config.white_min_value < 0 || config.white_min_value > 255 ||
      config.grass_hue_min < 0 || config.grass_hue_max > 179 ||
      config.grass_hue_min > config.grass_hue_max ||
      config.grass_min_saturation < 0 || config.grass_min_saturation > 255 ||
      config.roi_top_fraction < 0 || config.roi_top_fraction >= 1 ||
      config.candidate_min_bottom_y_fraction <= config.roi_top_fraction ||
      config.candidate_min_bottom_y_fraction >= 1 ||
      config.reference_y_fraction <= config.roi_top_fraction ||
      config.reference_y_fraction >= 1 ||
      config.min_abs_dx_per_dy <= 0 ||
      config.max_abs_dx_per_dy <= config.min_abs_dx_per_dy ||
      config.min_observed_height_fraction <= 0 ||
      config.max_fit_error_px <= 0 || config.group_tolerance_px <= 0 ||
      config.min_grass_support < 0 || config.min_grass_support > 1 ||
      config.bottom_outer_fraction <= 0 || config.bottom_outer_fraction >= 0.5) {
    throw std::invalid_argument("Invalid lane parameters");
  }
  const int w = bgr.cols, h = bgr.rows;
  const float reference_y = static_cast<float>((h - 1) * config.reference_y_fraction);
  LaneResult result;
  cv::Mat hsv;
  // 채도가 낮고 밝은 픽셀을 흰색 후보로 추출한다.
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(0, 0, config.white_min_value),
              cv::Scalar(179, config.white_max_saturation, 255), result.mask);
  // 영상 위쪽은 검출에서 제외한다. 판 색상 제외 마스크도 적용해 바닥 주변을 중심으로 검색한다.
  result.mask(cv::Rect(0, 0, w, std::clamp(cvRound(h * config.roi_top_fraction), 0, h))).setTo(0);
  if (!exclude_mask.empty()) {
    if (exclude_mask.size() != bgr.size() || exclude_mask.type() != CV_8UC1)
      throw std::invalid_argument("Invalid lane exclusion mask");
    result.mask.setTo(0, exclude_mask);
  }
  cv::morphologyEx(result.mask, result.mask, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
  cv::Mat edges;
  // 흰색 영역의 경계 픽셀을 구한 뒤 Hough 변환으로 직선 조각을 찾는다.
  cv::Canny(result.mask, edges, 60, 140);
  std::vector<cv::Vec4i> segments;
  cv::HoughLinesP(edges, segments, 1, CV_PI / 180, 25,
                  std::max(25.0, h * 0.07), 18);
  std::vector<Candidate> candidates;
  for (const auto &segment : segments) {
    // Accept a long boundary leaving through a side before reaching the lower ROI.
    // Short upper-image shelf edges still cannot seed a lane.
    const bool reaches_lower_roi = std::max(segment[1], segment[3]) >=
        h * config.candidate_min_bottom_y_fraction;
    const double dy = segment[3] - segment[1];
    if (std::abs(dy) < h * 0.07) continue;
    // 기울기는 dx/dy로 계산한다. 너무 수평이거나 지정 범위를 벗어난 선분은 제외한다.
    const double slope = (segment[2] - segment[0]) / dy;
    if (std::abs(slope) < config.min_abs_dx_per_dy ||
        std::abs(slope) > config.max_abs_dx_per_dy) continue;
    const double exit_x = slope < 0 ? 0.0 : w - 1.0;
    const double exit_y = segment[1] + (exit_x - segment[0]) / slope;
    const int lower_y = std::max(segment[1], segment[3]);
    // Hough may split a boundary into short pieces; admit lower pieces whose
    // continuation leaves the side, then enforce total observed span in the fit.
    const bool exits_side = exit_y >= h * 0.45 && exit_y < h &&
        lower_y >= h * 0.40 && exit_y >= lower_y - h * 0.05 &&
        exit_y - lower_y <= h * 0.25;
    if (!reaches_lower_roi && !exits_side) continue;
    const int side = slope < 0 ? 0 : 1;
    const double reference_x = segment[0] + slope * (reference_y - segment[1]);
    const double bottom_x = segment[0] + slope * (h - 1 - segment[1]);
    if ((side == 0 && bottom_x > w * config.bottom_outer_fraction) ||
        (side == 1 && bottom_x < w * (1.0 - config.bottom_outer_fraction)) ||
        bottom_x < -w || bottom_x > 2.0 * w) continue;
    if (grass_fraction(hsv, segment, side, config) < config.min_grass_support)
      continue;
    candidates.push_back({segment, slope, reference_x, side});
  }
  // 서로 비슷한 위치·기울기의 선분을 묶어 하나의 경계선 후보로 평가한다.
  double best_score = -1;
  for (const auto &seed : candidates) {
    std::vector<cv::Point2f> points;
    std::vector<cv::Vec4i> support;
    int min_y = h, max_y = 0;
    for (const auto &item : candidates) {
      if (item.side != seed.side ||
          std::abs(item.reference_x - seed.reference_x) > config.group_tolerance_px ||
          std::abs(item.slope - seed.slope) > 0.25) continue;
      const double mid_y = 0.5 * (item.segment[1] + item.segment[3]);
      const double mid_x = 0.5 * (item.segment[0] + item.segment[2]);
      const double seed_x = seed.segment[0] + seed.slope * (mid_y - seed.segment[1]);
      if (std::abs(mid_x - seed_x) > 2.0 * config.max_fit_error_px) continue;
      points.emplace_back(item.segment[0], item.segment[1]);
      points.emplace_back(item.segment[2], item.segment[3]);
      support.push_back(item.segment);
      min_y = std::min(min_y, std::min(item.segment[1], item.segment[3]));
      max_y = std::max(max_y, std::max(item.segment[1], item.segment[3]));
    }
    const int span = max_y - min_y;
    if (span < h * config.min_observed_height_fraction || points.size() < 4)
      continue;
    cv::Vec4f fit;
    // 여러 선분의 끝점에 강건한 직선을 맞추고, 실제 점들이 그 직선에서 얼마나 벗어났는지 검사한다.
    cv::fitLine(points, fit, cv::DIST_HUBER, 0, 0.01, 0.01);
    if (std::abs(fit[1]) < 1e-4) continue;
    const double slope = fit[0] / fit[1];
    if ((seed.side == 0 && slope > -config.min_abs_dx_per_dy) ||
        (seed.side == 1 && slope < config.min_abs_dx_per_dy) ||
        std::abs(slope) > config.max_abs_dx_per_dy) continue;
    double squared_error = 0;
    for (const auto &point : points) {
      const double expected_x = fit[2] + slope * (point.y - fit[3]);
      squared_error += (point.x - expected_x) * (point.x - expected_x);
    }
    const double fit_error = std::sqrt(squared_error / points.size());
    if (fit_error > config.max_fit_error_px) continue;
    const double bottom_x = fit[2] + slope * (h - 1 - fit[3]);
    if ((seed.side == 0 && bottom_x > w * config.bottom_outer_fraction) ||
        (seed.side == 1 && bottom_x < w * (1.0 - config.bottom_outer_fraction)))
      continue;
    double green = 0;
    for (const auto &segment : support)
      green += grass_fraction(hsv, segment, seed.side, config);
    green /= support.size();
    // 길게 관측되고 잔디 지지가 높으며 맞춤 오차가 작은 후보를 우선 선택한다.
    const double score = span + h * green - h * fit_error / config.max_fit_error_px;
    if (score <= best_score) continue;
    best_score = score;
    auto &out = result.best;
    out.valid = true;
    out.side = seed.side == 0 ? "left" : "right";
    out.top = {static_cast<float>(fit[2] + slope *
               (h * config.roi_top_fraction - fit[3])),
               static_cast<float>(h * config.roi_top_fraction)};
    out.bottom = {static_cast<float>(bottom_x), static_cast<float>(h - 1)};
    out.observed_y_min = min_y;
    out.observed_y_max = max_y;
    out.reference_y_px = reference_y;
    out.line_x_at_reference_px = static_cast<float>(fit[2] +
                                     slope * (reference_y - fit[3]));
    // 영상 중앙과 선택한 선의 기준 높이 위치 사이 간격이다. 아직 m 단위 거리가 아니다.
    out.pixel_separation_px = std::abs(out.line_x_at_reference_px - (w - 1) * 0.5f);
    out.grass_support = green;
    out.fit_error_px = fit_error;
    out.observed_segments = std::move(support);
  }
  return result;
}
}  // namespace robot_vision
