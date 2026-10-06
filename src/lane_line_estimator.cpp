// 파일 역할: 잔디 주변의 흰색 직선에서 왼쪽 또는 오른쪽 경계선 하나를 선택한다.
// 픽셀 기준 선 위치와 품질을 반환하며, m 단위 경계 거리는 field_geometry.cpp가 계산한다.

#include "robot_vision/lane_line_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

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

// Hough 선분의 끝점만 곧아 보여도 실제 흰 띠는 휘어 있을 수 있다.
// 관측 구간의 각 행에서 가장 가까운 흰 띠 중앙을 모아 직선·이차곡선으로 비교한다.
// 일정 방향의 휘어짐이 허용치를 넘고 곡선 모델로 더 잘 설명되면 경계선 후보에서 제외한다.
bool follows_straight_white_band(const cv::Mat &mask, const cv::Vec4f &fit,
                                int min_y, int max_y, const LaneConfig &config, double *white_band_error = nullptr, const cv::Mat &camera_matrix = {}, const cv::Mat &distortion = {}) {
  if (config.max_curve_deviation_px == 0) return true;
  if (max_y <= min_y || std::abs(fit[1]) < 1e-4) return false;
  const double slope = fit[0] / fit[1];
  // Hough 끝점은 흰 띠의 가장자리일 수 있으므로 반대쪽 가장자리까지 포함할 여유를 둔다.
  const int radius = std::max(12, cvRound(2.0 * config.group_tolerance_px));
  std::vector<cv::Point2d> centers;
  for (int y = std::max(0, min_y); y <= std::min(mask.rows - 1, max_y); y += 2) {
    const double predicted_x = fit[2] + slope * (y - fit[3]);
    const int start = std::max(0, cvRound(predicted_x) - radius);
    const int stop = std::min(mask.cols - 1, cvRound(predicted_x) + radius);
    const auto *row = mask.ptr<unsigned char>(y);
    double best_distance = radius + 1.0;
    double best_center = 0;
    bool found = false;
    for (int x = start; x <= stop;) {
      if (!row[x]) { ++x; continue; }
      const int first = x;
      while (x <= stop && row[x]) ++x;
      const int last = x - 1;
      // 가로로 이어지는 다른 선이나 교차부의 넓은 흰 띠는 중앙점으로 사용하지 않는다.
      // 영상 경계에서도 흰 띠가 잘리면 실제 중앙을 알 수 없다. 그 점은 곡선 검사에 사용하지 않는다.
      if (first == start || last == stop) continue;
      const double center = (first + last) * 0.5;
      const double distance = std::abs(center - predicted_x);
      if (distance < best_distance) {
        best_distance = distance;
        best_center = center;
        found = true;
      }
    }
    if (found) centers.emplace_back(best_center, y);
  }
  if (centers.size() < 12) return false;
  double low_y = min_y, high_y = max_y;
  // 렌즈 왜곡이 남은 원본에서는 실제 직선도 휘어 보인다.
  // 곡선 판정용 점만 같은 크기의 보정 픽셀 좌표로 바꾼다. 반환하는 선 좌표는 원본 기준으로 유지한다.
  if (!camera_matrix.empty()) {
    std::vector<cv::Point2d> rectified;
    cv::undistortPoints(centers, rectified, camera_matrix, distortion, cv::noArray(), camera_matrix);
    centers = std::move(rectified);
    // 관측 구간의 양 끝도 보정해 곡률 기준의 세로 길이를 같은 좌표계로 맞춘다.
    std::vector<cv::Point2d> anchors{{fit[2] + slope * (min_y - fit[3]), double(min_y)},
                                     {fit[2] + slope * (max_y - fit[3]), double(max_y)}};
    cv::undistortPoints(anchors, rectified, camera_matrix, distortion, cv::noArray(), camera_matrix);
    low_y = rectified[0].y;
    high_y = rectified[1].y;
  }
  const double middle_y = (low_y + high_y) * 0.5;
  const double half_span = (high_y - low_y) * 0.5;
  if (!std::isfinite(half_span) || half_span <= 1e-6) return false;
  cv::Mat design(static_cast<int>(centers.size()), 3, CV_64F);
  cv::Mat x_values(static_cast<int>(centers.size()), 1, CV_64F);
  for (size_t index = 0; index < centers.size(); ++index) {
    // y를 -1~1로 정규화하면 이차항 계수가 중앙과 양 끝을 잇는 직선의 차이(px)가 된다.
    const double t = (centers[index].y - middle_y) / half_span;
    design.at<double>(index, 0) = 1;
    design.at<double>(index, 1) = t;
    design.at<double>(index, 2) = t * t;
    x_values.at<double>(index) = centers[index].x;
  }
  cv::Mat linear, quadratic;
  if (!cv::solve(design.colRange(0, 2), x_values, linear, cv::DECOMP_SVD) ||
      !cv::solve(design, x_values, quadratic, cv::DECOMP_SVD)) return false;
  const double linear_error = cv::norm(design.colRange(0, 2) * linear - x_values) /
                              std::sqrt(static_cast<double>(centers.size()));
  const double quadratic_error = cv::norm(design * quadratic - x_values) /
                                 std::sqrt(static_cast<double>(centers.size()));
  if (white_band_error) *white_band_error = linear_error;
  if (linear_error > config.max_fit_error_px) return false;
  const double deviation = std::abs(quadratic.at<double>(2));
  // 픽셀 잡음을 곡선으로 오인하지 않도록 곡선 맞춤이 직선보다 충분히 좋아지는지도 확인한다.
  return !(deviation > config.max_curve_deviation_px && quadratic_error < 0.8 * linear_error);
}

}  // namespace

// 흰색 마스크 → 선분 추출 → 위치·기울기·잔디 검사 → 직선 맞춤 순으로 경계선을 선택한다.
// 반환 마스크는 디버그 표시용이고 best.valid가 실제 선 검출 성공 여부다.
LaneResult estimate_lane_line(const cv::Mat &bgr, const LaneConfig &config,
                              const cv::Mat &exclude_mask, const cv::Mat &camera_matrix, const cv::Mat &distortion) {
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
      !std::isfinite(config.max_curve_deviation_px) || config.max_curve_deviation_px < 0 ||
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
    // 한 선분만 남았어도 아래의 실제 흰 띠 검사를 통과하면 사용할 수 있다.
    // 곡선 검사를 끈 경우에는 기존처럼 최소 두 선분의 끝점 네 개를 요구한다.
    if (span < h * config.min_observed_height_fraction ||
        points.size() < 2 || (points.size() < 4 && config.max_curve_deviation_px == 0))
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
    double fit_error = std::sqrt(squared_error / points.size());
    if (fit_error > config.max_fit_error_px) continue;
    double white_band_error = 0;
    if (!follows_straight_white_band(result.mask, fit, min_y, max_y, config, &white_band_error, camera_matrix, distortion)) continue;
    // 두 끝점만 있으면 끝점 맞춤 오차는 거의 0이다. 이 경우 실제 흰 띠의 오차를 품질 값으로 사용한다.
    if (points.size() < 4) fit_error = white_band_error;
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
