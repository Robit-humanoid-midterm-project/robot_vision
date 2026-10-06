#include "robot_vision/distance_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace robot_vision {
namespace {

// 네 꼭짓점을 중심 주위의 각도순으로 정렬한다.
// x+y가 가장 작은 점부터 시작하도록 순서를 회전해 PnP 입력 순서를 맞춘다.
std::array<cv::Point2d, 4> order_quad(std::array<cv::Point2d, 4> points) {
  cv::Point2d center;
  for (const auto &p : points)
    center += p;
  center *= 0.25;
  std::sort(points.begin(), points.end(), [&](const auto &a, const auto &b) {
    return std::atan2(a.y - center.y, a.x - center.x) <
           std::atan2(b.y - center.y, b.x - center.x);
  });
  const auto first = std::min_element(
      points.begin(), points.end(), [](const auto &a, const auto &b) {
        return a.x + a.y < b.x + b.y;
      });
  std::rotate(points.begin(), first, points.end());
  return points;
}

// 색상 검출에 사용하는 HSV 상·하한을 검사한다.
// OpenCV 범위(H: 0~179, S/V: 0~255)를 벗어나거나 상·하한이 뒤집히면 예외를 낸다.
void validate_hsv(const std::array<int, 3> &lower, const std::array<int, 3> &upper) {
  for (int i = 0; i < 3; ++i) {
    const int maximum = i == 0 ? 179 : 255;
    if (lower[i] < 0 || lower[i] > upper[i] || upper[i] > maximum) {
      throw std::invalid_argument("Invalid HSV threshold");
    }
  }
}

// 3차원 좌표에 NaN이나 무한대가 없는지 확인한다.
bool finite_vec(const cv::Vec3d &value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

}  // namespace

// 한 변 기반 거리 추정: 선언은 distance_estimator.hpp에서 관리한다.
namespace {

// 좌표가 마스크 안에 있고, 해당 픽셀이 검출 색상 영역인지 확인한다.
bool colored(const cv::Mat &mask, int x, int y) {
  return x >= 0 && y >= 0 && x < mask.cols && y < mask.rows &&
         mask.at<uchar>(y, x) != 0;
}

}  // namespace

// 합쳐진 윤곽에서 한 판의 온전한 변으로 볼 수 있는 후보를 찾는다.
// 양 끝의 볼록한 직각, 수평·수직 방향, 안팎의 색상을 검사한다.
// 이 조건만으로 두 끝점이 같은 판에 속한다고 보장되지는 않는다.
std::vector<VisibleEdge> find_complete_visible_edges(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    double min_edge_px, int border_margin_px) {
  std::vector<VisibleEdge> edges;
  if (contour.size() < 4 || mask.empty())
    return edges;
  const auto bounds = cv::boundingRect(contour);
  if (bounds.x <= border_margin_px || bounds.y <= border_margin_px ||
      bounds.br().x >= mask.cols - border_margin_px ||
      bounds.br().y >= mask.rows - border_margin_px)
    return edges;
  std::vector<cv::Point> polygon;
  // 작은 색상 잡음을 정리하되, 겹친 판 사이의 오목한 꺾임은 보존한다.
  cv::approxPolyDP(contour, polygon, 4.0, true);
  if (polygon.size() < 4)
    return edges;
  const double orientation = cv::contourArea(polygon, true) > 0 ? 1.0 : -1.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const cv::Point2d a = polygon[i], b = polygon[(i + 1) % polygon.size()];
    const cv::Point2d before = a - cv::Point2d(polygon[(i + polygon.size() - 1) % polygon.size()]);
    const cv::Point2d edge = b - a;
    const cv::Point2d after = cv::Point2d(polygon[(i + 2) % polygon.size()]) - b;
    const double length = cv::norm(edge), l0 = cv::norm(before), l1 = cv::norm(after);
    if (length < min_edge_px || l0 < 6 || l1 < 6 ||
        l0 > 1.2 * length || l1 > 1.2 * length)
      continue;
    // 양 끝 모두 바깥쪽 직각 모서리여야 한다. 겹침 지점의 오목한 끝은 제외한다.
    if (orientation * before.cross(edge) <= 0 || orientation * edge.cross(after) <= 0 ||
        std::abs(before.dot(edge)) > 0.25 * l0 * length ||
        std::abs(edge.dot(after)) > 0.25 * length * l1)
      continue;
    const bool horizontal = std::abs(edge.y) < 0.12 * length;
    if (!horizontal && std::abs(edge.x) >= 0.12 * length)
      continue;
    const auto on_border = [&](const cv::Point2d &point) {
      return point.x <= border_margin_px || point.y <= border_margin_px ||
             point.x >= mask.cols - 1 - border_margin_px ||
             point.y >= mask.rows - 1 - border_margin_px;
    };
    if (on_border(a) || on_border(b))
      continue;
    const cv::Point2d inward = orientation * cv::Point2d(-edge.y, edge.x) / length;
    int support = 0;
    for (int sample = 1; sample < 10; ++sample) {
      const auto inside = a + edge * (sample / 10.0) + inward * 3.0;
      const auto outside = a + edge * (sample / 10.0) - inward * 3.0;
      if (colored(mask, cvRound(inside.x), cvRound(inside.y)) &&
          !colored(mask, cvRound(outside.x), cvRound(outside.y)))
        ++support;
    }
    if (support < 8)
      continue;
    cv::Point2d first = a, last = b;
    if ((horizontal && first.x > last.x) || (!horizontal && first.y > last.y))
      std::swap(first, last);
    edges.push_back({first, last, horizontal, horizontal ? inward.y > 0 : inward.x > 0});
  }
  // 긴 변을 우선 사용한다. 얇은 띠에서는 짧은 가림 경계를 40cm로 취급하지 않는다.
  std::stable_sort(edges.begin(), edges.end(), [](const VisibleEdge &a, const VisibleEdge &b) {
    const double la = cv::norm(a.last - a.first), lb = cv::norm(b.last - b.first);
    const auto group_a = std::lround(la / 3.0), group_b = std::lround(lb / 3.0);
    if (group_a != group_b)
      return group_a > group_b;
    return a.upper_or_left && !b.upper_or_left;
  });
  return edges;
}

// 판이 카메라를 정면으로 향한다는 가정으로, 온전한 한 변에서 위치를 추정한다.
// 왜곡 보정한 정규화 좌표의 변 길이로 z를 구하고 판 아래쪽 중심으로 이동한다.
// 반환값은 카메라 기준 (x, y, z), 단위는 m이며 계산 실패 시 nullopt를 반환한다.
std::optional<cv::Vec3d> position_from_visible_edge(
    const VisibleEdge &edge, double side_m,
    const cv::Mat &camera_matrix, const cv::Mat &distortion) {
  if (!(side_m > 0) || !std::isfinite(side_m))
    return std::nullopt;
  std::vector<cv::Point2d> normalized;
  try {
    cv::undistortPoints(std::vector<cv::Point2d>{edge.first, edge.last},
                        normalized, camera_matrix, distortion);
  } catch (const cv::Exception &) {
    return std::nullopt;
  }
  const double projected = edge.horizontal ?
      std::abs(normalized[1].x - normalized[0].x) :
      std::abs(normalized[1].y - normalized[0].y);
  if (!std::isfinite(projected) || projected <= 1e-6)
    return std::nullopt;
  // 정규화 영상 좌표에서는 초점거리가 1이므로 z = 실제 변 길이 / 투영 길이.
  const double z = side_m / projected;
  const cv::Point2d middle = (normalized[0] + normalized[1]) * 0.5;
  // 영상 좌표 y는 아래가 양수. 위쪽 변에서는 한 변, 세로 변에서는 반 변 내려 잡는다.
  const double offset_y = edge.horizontal ?
      (edge.upper_or_left ? side_m : 0.0) : side_m * 0.5;
  // 세로 변을 사용하면 변에서 판 중심까지 가로로 반 변만큼 이동한다.
  const double offset_x = edge.horizontal ? 0.0 :
      (edge.upper_or_left ? side_m * 0.5 : -side_m * 0.5);
  const cv::Vec3d position{middle.x * z + offset_x, middle.y * z + offset_y, z};
  if (!std::isfinite(position[0]) || !std::isfinite(position[1]) ||
      !std::isfinite(position[2]))
    return std::nullopt;
  return position;
}

// 판 아래쪽 중심이 바닥에 있다는 가정으로 직선거리에서 바닥거리를 구한다.
// Ground² = Range² - 카메라 높이², Forward² = Ground² - Lateral².
// 카메라 x축을 좌우 축으로 사용하며, 성립하지 않는 입력은 nullopt로 처리한다.
std::optional<GroundProjection> project_to_ground(
    const Detection &detection, double camera_height_m) {
  const double range = detection.distance_m;
  const double lateral = detection.position[0];
  if (!std::isfinite(range) || !std::isfinite(lateral) ||
      !std::isfinite(camera_height_m) || camera_height_m <= 0 ||
      range < camera_height_m)
    return std::nullopt;
  const double radial_squared = (range - camera_height_m) *
                                (range + camera_height_m);
  const double forward_squared = radial_squared - lateral * lateral;
  if (forward_squared < 0)
    return std::nullopt;
  return GroundProjection{std::sqrt(forward_squared),
                          std::sqrt(radial_squared), lateral};
}

// 카메라 보정값, 판 크기, 검출 기준과 HSV 범위를 검사하고 내부 행렬을 준비한다.
// 잘못된 설정은 생성 시 예외로 알린다.
DistanceEstimator::DistanceEstimator(DetectorConfig config) : config_(std::move(config)) {
  if (config_.camera_matrix.size() != 9 ||
      std::find_if(config_.camera_matrix.begin(), config_.camera_matrix.end(),
                   [](double x) { return !std::isfinite(x); }) != config_.camera_matrix.end() ||
      config_.camera_matrix[0] <= 0 || config_.camera_matrix[4] <= 0 ||
      std::abs(config_.camera_matrix[6]) > 1e-9 ||
      std::abs(config_.camera_matrix[7]) > 1e-9 ||
      std::abs(config_.camera_matrix[8] - 1.0) > 1e-9) {
    throw std::invalid_argument("Invalid camera matrix");
  }
  const auto n = config_.distortion_coefficients.size();
  if ((n != 4 && n != 5 && n != 8 && n != 12 && n != 14) ||
      std::find_if(config_.distortion_coefficients.begin(),
          config_.distortion_coefficients.end(),
                   [](double x) { return !std::isfinite(x); }) != config_.distortion_coefficients.end()) {
    throw std::invalid_argument("Invalid distortion coefficients");
  }
  if (config_.obstacle_size_m <= 0 || config_.calibration_width <= 0 ||
      config_.calibration_height <= 0 || config_.min_area_px <= 0 ||
      config_.min_edge_px <= 0 || config_.min_fill_ratio <= 0 ||
      config_.min_fill_ratio > 1 || config_.border_margin_px < 0 ||
      config_.near_min_edge_px <= config_.min_edge_px ||
      config_.near_min_fill_ratio <= 0 ||
      config_.near_min_fill_ratio > config_.min_fill_ratio ||
      config_.near_max_reprojection_error_px < config_.max_reprojection_error_px ||
      config_.near_max_relative_reprojection_error <
          config_.max_relative_reprojection_error ||
      config_.max_reprojection_error_px <= 0 ||
      config_.max_relative_reprojection_error <= 0 || config_.min_distance_m <= 0 ||
      config_.max_distance_m < config_.min_distance_m) {
    throw std::invalid_argument("Invalid obstacle detector geometry thresholds");
  }
  validate_hsv(config_.red_lower_1, config_.red_upper_1);
  validate_hsv(config_.red_lower_2, config_.red_upper_2);
  validate_hsv(config_.blue_lower, config_.blue_upper);
  camera_matrix_ = cv::Mat(3, 3, CV_64F, config_.camera_matrix.data()).clone();
  distortion_ = cv::Mat(1, static_cast<int>(n), CV_64F,
                        config_.distortion_coefficients.data()).clone();
}

// 영상의 네 꼭짓점과 실제 정사각형 크기로 판의 3차원 자세를 계산한다.
// PnP 후보 중 카메라 앞에 있고 재투영 오차가 가장 작은 결과를 선택한다.
// 거리 기준점은 판 아래쪽 변의 중심이며, distance_m은 그 점까지의 직선거리다.
std::optional<Detection> DistanceEstimator::square_pose(
    const std::array<cv::Point2d, 4> &unordered_corners) const {
  const auto corners = order_quad(unordered_corners);
  const double s = config_.obstacle_size_m / 2.0;
  const std::vector<cv::Point3d> object{{-s, s, 0}, {s, s, 0}, {s, -s, 0}, {-s, -s, 0}};
  const std::vector<cv::Point2d> image(corners.begin(), corners.end());
  std::vector<cv::Mat> rotations, translations;

  // 정사각형 전용 해법과 반복 해법의 후보를 함께 평가한다.
  try {
    cv::solvePnPGeneric(object, image, camera_matrix_, distortion_, rotations,
                        translations, false, cv::SOLVEPNP_IPPE_SQUARE);
    cv::Mat rotation, translation;
    if (cv::solvePnP(object, image, camera_matrix_, distortion_, rotation,
                     translation, false, cv::SOLVEPNP_ITERATIVE)) {
      rotations.push_back(rotation);
      translations.push_back(translation);
    }
  } catch (const cv::Exception &) {
    return std::nullopt;
  }

  std::optional<Detection> best;
  for (size_t candidate = 0; candidate < rotations.size(); ++candidate) {
    try {
      const cv::Mat &rvec = rotations[candidate];
      const cv::Mat &tvec = translations[candidate];
      cv::Vec3d t{tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
      if (!finite_vec(t))
        continue;
      cv::Mat matrix;
      cv::Rodrigues(rvec, matrix);
      bool in_front = true;
      for (const auto &point : object) {
        const double depth = matrix.at<double>(2, 0) * point.x +
                             matrix.at<double>(2, 1) * point.y +
                             matrix.at<double>(2, 2) * point.z + t[2];
        if (depth <= 0 || !std::isfinite(depth))
          in_front = false;
      }
      if (!in_front)
        continue;

      // 추정한 자세를 영상에 다시 투영해 원래 꼭짓점과의 오차를 측정한다.
      std::vector<cv::Point2d> projected;
      cv::projectPoints(object, rvec, tvec, camera_matrix_, distortion_, projected);
      double squared_error = 0;
      for (size_t i = 0; i < 4; ++i) {
        const auto delta = projected[i] - corners[i];
        squared_error += delta.x * delta.x + delta.y * delta.y;
      }
      const double error = std::sqrt(squared_error / 4.0);
      if (!std::isfinite(error))
        continue;
      if (!best || error < best->reprojection_error_px) {
        Detection detection;
        // 판의 로컬 좌표 (0, -s, 0)를 카메라 좌표로 변환한다.
        const cv::Vec3d bottom{
            t[0] - s * matrix.at<double>(0, 1),
            t[1] - s * matrix.at<double>(1, 1),
            t[2] - s * matrix.at<double>(2, 1)};
        if (!finite_vec(bottom) || bottom[2] <= 0)
          continue;
        detection.position = bottom;
        detection.distance_m = cv::norm(bottom);
        detection.reprojection_error_px = error;
        detection.corners = corners;
        best = detection;
      }
    } catch (const cv::Exception &) {
      continue;
    }
  }
  return best;
}

// 윤곽을 사각형 후보로 바꾼 뒤 면적, 형태 일치도, PnP 오차와 거리를 검사한다.
// box_fallback=false: 볼록 껍질을 네 꼭짓점으로 근사한다.
// box_fallback=true: 색상 분리 결과를 감싸는 최소 면적 사각형을 사용한다.
std::optional<Detection> DistanceEstimator::pose_for_contour(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    const cv::Size &image_size, bool box_fallback) const {
  const double area = cv::contourArea(contour);
  if (area < config_.min_area_px)
    return std::nullopt;
  const cv::Rect bounds = cv::boundingRect(contour);
  const int margin = config_.border_margin_px;
  if (bounds.x <= margin || bounds.y <= margin ||
      bounds.x + bounds.width >= image_size.width - margin ||
      bounds.y + bounds.height >= image_size.height - margin) {
    return std::nullopt;
  }

  cv::Mat outline = cv::Mat::zeros(mask.size(), CV_8UC1);
  cv::drawContours(outline, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255),
      cv::FILLED);
  const int outline_pixels = cv::countNonZero(outline);
  std::vector<std::array<cv::Point2d, 4>> candidates;
  if (box_fallback) {
    const cv::RotatedRect rectangle = cv::minAreaRect(contour);
    const double short_side = std::min(rectangle.size.width, rectangle.size.height);
    const double long_side = std::max(rectangle.size.width, rectangle.size.height);
    if (short_side < config_.min_edge_px || long_side / short_side > 2.5) {
      return std::nullopt;
    }
    // 뒤 판의 왼쪽 경계와 앞 판의 오른쪽 경계를 한 사각형으로 묶는 오류를 줄인다.
    // 검사에 포함한 행 사이에서 좌우 경계가 크게 바뀌면 합쳐진 윤곽으로 보고 제외한다.
    int previous_left = -1, previous_right = -1;
    const int jump_limit = std::max(8, cvRound(0.12 * short_side));
    for (int y = bounds.y + 2; y < bounds.br().y - 2; ++y) {
      const uchar *row = outline.ptr<uchar>(y);
      int left = bounds.br().x, right = bounds.x - 1;
      for (int x = bounds.x; x < bounds.br().x; ++x) {
        if (row[x]) {
          left = std::min(left, x);
          right = x;
        }
      }
      if (right - left < 0.4 * short_side)
        continue;
      if (previous_left >= 0 &&
          (std::abs(left - previous_left) > jump_limit ||
           std::abs(right - previous_right) > jump_limit)) {
        return std::nullopt;
      }
      previous_left = left;
      previous_right = right;
    }
    cv::Point2f box[4];
    rectangle.points(box);
    candidates.push_back({cv::Point2d(box[0]), cv::Point2d(box[1]),
                          cv::Point2d(box[2]), cv::Point2d(box[3])});
  } else {
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    const double perimeter = cv::arcLength(hull, true);
    for (double fraction : {0.01, 0.015, 0.02, 0.03, 0.04}) {
      std::vector<cv::Point> quad;
      cv::approxPolyDP(hull, quad, fraction * perimeter, true);
      if (quad.size() == 4 && cv::isContourConvex(quad)) {
        candidates.push_back({cv::Point2d(quad[0]), cv::Point2d(quad[1]),
                              cv::Point2d(quad[2]), cv::Point2d(quad[3])});
      }
    }
  }

  std::optional<Detection> best;
  for (const auto &candidate : candidates) {
    cv::Mat region = cv::Mat::zeros(mask.size(), CV_8UC1);
    std::vector<cv::Point> polygon;
    for (const auto &point : candidate)
      polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    cv::fillConvexPoly(region, polygon, cv::Scalar(255));
    cv::Mat intersection;
    cv::bitwise_and(outline, region, intersection);

    // 윤곽과 사각형의 교집합/합집합 비율로 형태가 얼마나 일치하는지 평가한다.
    const int overlap = cv::countNonZero(intersection);
    const int union_pixels = outline_pixels + cv::countNonZero(region) - overlap;
    const double fit = static_cast<double>(overlap) / std::max(1, union_pixels);
    const auto corners = order_quad(candidate);
    double min_edge = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < 4; ++i) {
      min_edge = std::min(min_edge, cv::norm(corners[i] - corners[(i + 1) % 4]));
    }
    if (min_edge < config_.min_edge_px)
      continue;
    const bool near_face = min_edge >= config_.near_min_edge_px;
    const double min_fill = near_face ? config_.near_min_fill_ratio :
                                        config_.min_fill_ratio;
    if (fit < min_fill)
      continue;
    auto pose = square_pose(corners);
    if (!pose)
      continue;
    const double allowed_absolute_error = near_face ?
        config_.near_max_reprojection_error_px : config_.max_reprojection_error_px;
    const double allowed_relative_error = near_face ?
        config_.near_max_relative_reprojection_error :
        config_.max_relative_reprojection_error;
    const double max_error = std::min(allowed_absolute_error +
                                          (box_fallback ? 1.0 : 0.0),
                                      allowed_relative_error * min_edge);
    if (pose->reprojection_error_px > max_error ||
        pose->distance_m < config_.min_distance_m ||
        pose->distance_m > config_.max_distance_m)
      continue;
    pose->shape_fit = fit;
    if (!best || fit > best->shape_fit)
      best = pose;
  }
  return best;
}

// 같은 색으로 붙은 윤곽을 채도 차이로 분리해 본다.
// 채도 분포에 차이가 있을 때 높은 채도 영역을 추출하고 충분히 큰 윤곽을 반환한다.
// saturated는 분리 결과를 담는 출력 마스크이며, 분리 불가 시 빈 목록을 반환한다.
std::vector<std::vector<cv::Point>> DistanceEstimator::split_touching_color(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    const cv::Mat &hsv, int min_saturation, cv::Mat &saturated) const {
  cv::Mat component = cv::Mat::zeros(mask.size(), CV_8UC1);
  cv::drawContours(component, std::vector<std::vector<cv::Point>>{contour}, 0,
                   cv::Scalar(255), cv::FILLED);
  std::array<int, 256> histogram{};
  std::vector<uint8_t> samples;
  samples.reserve(static_cast<size_t>(cv::contourArea(contour)));
  for (int y = 0; y < mask.rows; ++y) {
    const auto *m = mask.ptr<uint8_t>(y);
    const auto *c = component.ptr<uint8_t>(y);
    const auto *h = hsv.ptr<cv::Vec3b>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (m[x] && c[x]) {
        const uint8_t value = h[x][1];
        samples.push_back(value);
        ++histogram[value];
      }
    }
  }
  if (samples.size() < static_cast<size_t>(config_.min_area_px))
    return {};

  // 채도 분포의 하위 5%와 상위 5%가 비슷하면 분리 근거가 부족하다.
  auto percentile = [&](double fraction) {
    const int target = static_cast<int>(std::ceil(fraction * samples.size()));
    int cumulative = 0;
    for (int value = 0; value < 256; ++value) {
      cumulative += histogram[value];
      if (cumulative >= target)
        return value;
    }
    return 255;
  };
  if (percentile(0.95) - percentile(0.05) < 8)
    return {};

  // Otsu 임계값보다 조금 높은 채도만 남겨 분리 마스크를 만든다.
  cv::Mat sample_mat(static_cast<int>(samples.size()), 1, CV_8UC1, samples.data());
  cv::Mat ignored;
  const double otsu = cv::threshold(sample_mat, ignored, 0, 255,
                                    cv::THRESH_BINARY | cv::THRESH_OTSU);
  const int cutoff = std::min(255, std::max(static_cast<int>(otsu) + 3,
                                            min_saturation + 3));
  saturated = cv::Mat::zeros(mask.size(), CV_8UC1);
  for (int y = 0; y < mask.rows; ++y) {
    const auto *m = mask.ptr<uint8_t>(y);
    const auto *c = component.ptr<uint8_t>(y);
    const auto *h = hsv.ptr<cv::Vec3b>(y);
    auto *out = saturated.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (m[x] && c[x] && h[x][1] >= cutoff)
        out[x] = 255;
    }
  }
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
  cv::morphologyEx(saturated, saturated, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(saturated, saturated, cv::MORPH_CLOSE, kernel);
  std::vector<std::vector<cv::Point>> parts, accepted;
  cv::findContours(saturated, parts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  const double minimum = 0.4 * cv::contourArea(contour);
  for (auto &part : parts) {
    if (cv::contourArea(part) >= minimum)
      accepted.push_back(std::move(part));
  }
  return accepted;
}

// 합쳐진 색상 영역의 넓은 내부를 기준으로 위·아래·왼쪽·오른쪽 영상 경계를 찾는다.
// 정면에 가까운 판을 축에 맞춘 사각형으로 복원하고 PnP로 거리를 계산한다.
// 네 경계나 내부 색상 비율이 충분하지 않으면 nullopt를 반환한다.
std::optional<Detection> DistanceEstimator::line_pose_for_component(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    const cv::Mat &bgr) const {
  const cv::Rect bounds = cv::boundingRect(contour);
  if (bounds.x <= config_.border_margin_px || bounds.y <= config_.border_margin_px ||
      bounds.x + bounds.width >= bgr.cols - config_.border_margin_px ||
      bounds.y + bounds.height >= bgr.rows - config_.border_margin_px) {
    return std::nullopt;
  }
  cv::Mat component = cv::Mat::zeros(mask.size(), CV_8UC1);
  cv::drawContours(component, std::vector<std::vector<cv::Point>>{contour}, 0,
                   cv::Scalar(255), cv::FILLED);

  // 경계에서 가장 멀리 떨어진 내부 픽셀을 판 중심 후보로 사용한다.
  cv::Mat distances;
  cv::distanceTransform(component, distances, cv::DIST_L2, 5);
  cv::Point center;
  double peak = 0;
  cv::minMaxLoc(distances, nullptr, &peak, nullptr, &center);
  if (peak < config_.min_edge_px / 2.0)
    return std::nullopt;

  const cv::Rect search = (cv::Rect(bounds.x - 12, bounds.y - 12,
                                    bounds.width + 24, bounds.height + 24) &
                           cv::Rect(0, 0, bgr.cols, bgr.rows));
  cv::Mat gray, blurred, edges;
  cv::cvtColor(bgr(search), gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
  cv::Canny(blurred, edges, 50, 120);
  std::vector<cv::Vec4i> lines;
  const int minimum_line = std::max(30, static_cast<int>(0.20 * std::min(bounds.width,
      bounds.height)));
  cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 30, minimum_line, 8);


  // 중심 주변의 네 방향에서 가장 가까운 유효 선분을 하나씩 유지한다.
  struct Side {
    bool found{false};
    double coordinate{0};
    double distance{std::numeric_limits<double>::infinity()};
    double length{0};
  } top, bottom, left, right;
  for (const auto &line : lines) {
    const double x1 = line[0] + search.x, y1 = line[1] + search.y;
    const double x2 = line[2] + search.x, y2 = line[3] + search.y;
    const double length = std::hypot(x2 - x1, y2 - y1);
    if (length < minimum_line)
      continue;
    const int mx = cvRound((x1 + x2) / 2.0), my = cvRound((y1 + y2) / 2.0);
    const cv::Rect around(std::max(0, mx - 3), std::max(0, my - 3), 7, 7);
    if ((around & cv::Rect(0, 0, component.cols, component.rows)).area() == 0 ||
        cv::countNonZero(component(around & cv::Rect(0, 0, component.cols,
            component.rows))) == 0) {
      continue;
    }
    if (std::abs(y2 - y1) < 0.20 * length &&
        std::min(x1, x2) - 20 <= center.x && center.x <= std::max(x1, x2) + 20) {
      const double coordinate = (y1 + y2) / 2.0;
      Side &side = coordinate < center.y ? top : bottom;
      const double distance = std::abs(center.y - coordinate);
      if (distance > 0.15 * bounds.height && distance < side.distance) {
        side = {true, coordinate, distance, length};
      }
    }
    if (std::abs(x2 - x1) < 0.20 * length &&
        std::min(y1, y2) - 20 <= center.y && center.y <= std::max(y1, y2) + 20) {
      const double coordinate = (x1 + x2) / 2.0;
      Side &side = coordinate < center.x ? left : right;
      const double distance = std::abs(center.x - coordinate);
      if (distance > 0.15 * bounds.width && distance < side.distance) {
        side = {true, coordinate, distance, length};
      }
    }
  }
  if (!top.found || !bottom.found || !left.found || !right.found)
    return std::nullopt;
  const double width = right.coordinate - left.coordinate;
  const double height = bottom.coordinate - top.coordinate;
  if (width < config_.min_edge_px || height < config_.min_edge_px ||
      width / height < 0.65 || width / height > 1.5 ||
      top.length < 0.25 * width || bottom.length < 0.25 * width ||
      left.length < 0.25 * height || right.length < 0.25 * height)
    return std::nullopt;

  const std::array<cv::Point2d, 4> corners{{
      {left.coordinate, top.coordinate}, {right.coordinate, top.coordinate},
      {right.coordinate, bottom.coordinate}, {left.coordinate, bottom.coordinate}}};
  cv::Mat rectangle = cv::Mat::zeros(mask.size(), CV_8UC1);
  std::vector<cv::Point> polygon;
  for (const auto &corner : corners)
    polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
  cv::fillConvexPoly(rectangle, polygon, cv::Scalar(255));
  cv::Mat color_inside;
  cv::bitwise_and(mask, rectangle, color_inside);
  const double occupancy = static_cast<double>(cv::countNonZero(color_inside)) /
                           std::max(1, cv::countNonZero(rectangle));
  if (occupancy < 0.85)
    return std::nullopt;
  auto pose = square_pose(corners);
  if (!pose)
    return std::nullopt;
  const double max_error = std::min(
      config_.max_reprojection_error_px + 1.0,
      config_.max_relative_reprojection_error * std::min(width, height));
  if (pose->reprojection_error_px > max_error ||
      pose->distance_m < config_.min_distance_m ||
      pose->distance_m > config_.max_distance_m)
    return std::nullopt;
  pose->shape_fit = occupancy;
  pose->color_split_estimate = true;
  return pose;
}

// 온전한 한 변의 위치 추정 결과를 Detection으로 구성한다.
// 거리를 먼저 계산한 뒤 표시·메시지·중복 판정용 정사각형 꼭짓점을 만든다.
// 복원한 네 꼭짓점을 다시 PnP에 넣어 거리를 계산하지는 않는다.
std::optional<Detection> DistanceEstimator::pose_for_visible_edge(
    const VisibleEdge &visible_edge) const {
  // 전체 윤곽 대신 판 하나의 온전한 변으로 거리와 추정 모서리를 만든다.
  const auto *edge = &visible_edge;
  const auto position = position_from_visible_edge(
      *edge, config_.obstacle_size_m, camera_matrix_, distortion_);
  if (!position)
    return std::nullopt;
  const double range = cv::norm(*position);
  if (range < config_.min_distance_m || range > config_.max_distance_m)
    return std::nullopt;
  Detection one_edge;
  one_edge.position = *position;
  one_edge.distance_m = range;
  one_edge.color_split_estimate = true;
  // 메시지에는 추정한 전체 정사각형의 네 모서리를 넣는다.
  const double pixels = cv::norm(edge->last - edge->first);
  if (edge->horizontal) {
    const double dy = edge->upper_or_left ? pixels : -pixels;
    const cv::Point2d shift(0.0, dy);
    one_edge.corners = edge->upper_or_left ?
        std::array<cv::Point2d, 4>{edge->first, edge->last,
                                    edge->last + shift, edge->first + shift} :
        std::array<cv::Point2d, 4>{edge->first + shift, edge->last + shift,
                                    edge->last, edge->first};
  } else {
    const double dx = edge->upper_or_left ? pixels : -pixels;
    const cv::Point2d shift(dx, 0.0);
    one_edge.corners = edge->upper_or_left ?
        std::array<cv::Point2d, 4>{edge->first, edge->first + shift,
                                    edge->last + shift, edge->last} :
        std::array<cv::Point2d, 4>{edge->first + shift, edge->first,
                                    edge->last, edge->last + shift};
  }
  return one_edge;
}

// 한 프레임에서 색상별 판의 위치와 거리를 검출한다.
// 전체 윤곽의 네 꼭짓점 → 영상 경계 → 채도 분리 순으로 복구를 시도한다.
// 복구 단계에서는 온전한 한 변도 추가 검사해 뒤 판을 찾고, 중복을 제거한다.
DetectResult DistanceEstimator::detect(const cv::Mat &bgr) const {
  DetectResult result;
  if (bgr.empty()) {
    result.status = "no_image";
    return result;
  }
  auto color_result = detect_color_regions(bgr, config_);
  result.colors = std::move(color_result.colors);
  result.mask_preview = std::move(color_result.mask_preview);
  result.image_candidates = std::move(color_result.candidates);
  if (bgr.cols != config_.calibration_width || bgr.rows != config_.calibration_height) {
    result.status = "image_size_mismatch";
    // 픽셀 영역은 유지하되, 보정 해상도가 다르면 실제 거리 계산은 생략한다.
    return result;
  }
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  int color_index = 0;
  for (auto &entry : std::vector<std::pair<std::string, cv::Mat>>{
         {"red", result.colors[0].cleaned_mask}, {"blue", result.colors[1].cleaned_mask}}) {
    auto &debug = result.colors[color_index++];
    cv::Mat &mask = entry.second;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
      // 1. 전체 윤곽이 한 판으로 설명되면 네 꼭짓점 계산 결과를 사용한다.
      auto pose = pose_for_contour(contour, mask, bgr.size(), false);
      if (pose) {
        pose->color = entry.first;
        result.detections.push_back(*pose);
        ++debug.detections;
        continue;
      }
      if (cv::contourArea(contour) < 3.0 * config_.min_area_px)
        continue;
      const int min_saturation = entry.first == "red" ?
          std::min(config_.red_lower_1[1], config_.red_lower_2[1]) : config_.blue_lower[1];
      cv::Mat saturated;
      bool recovered = false;

      // 2. 겹침 복구: 넓은 색상 영역 주변의 실제 영상 경계를 먼저 찾는다.
      pose = line_pose_for_component(contour, mask, bgr);
      if (pose) {
        pose->color = entry.first;
        result.detections.push_back(*pose);
        ++debug.detections;
        recovered = true;
      }

      // 3. 영상 경계로 복구하지 못하면 채도 차이로 윤곽을 분리해 본다.
      if (!recovered) {
        for (const auto &part : split_touching_color(
                 contour, mask, hsv, min_saturation, saturated)) {
          pose = pose_for_contour(part, saturated, bgr.size(), true);
          if (pose) {
            pose->color = entry.first;
            pose->color_split_estimate = true;
            result.detections.push_back(*pose);
            ++debug.detections;
            recovered = true;
          }
        }
      }

      // 4. 앞 판을 찾았어도 같은 윤곽에 붙은 뒤 판의 온전한 변을 계속 검사한다.
      for (const auto &edge : find_complete_visible_edges(
          contour, mask, config_.min_edge_px, config_.border_margin_px)) {
        pose = pose_for_visible_edge(edge);
        if (!pose)
          continue;
        pose->color = entry.first;
        const auto box = cv::boundingRect(std::vector<cv::Point2f>(
            pose->corners.begin(), pose->corners.end()));

        // 깊이와 영상 중심이 비슷하거나 상자가 크게 겹치면 이미 찾은 판으로 본다.
        const bool duplicate = std::any_of(result.detections.begin(), result.detections.end(),
            [&](const Detection &known) {
              if (known.color != entry.first)
                return false;
              const auto other = cv::boundingRect(std::vector<cv::Point2f>(
                  known.corners.begin(), known.corners.end()));
              const cv::Point2d center(box.x + box.width * 0.5, box.y + box.height * 0.5);
              const cv::Point2d other_center(other.x + other.width * 0.5,
                  other.y + other.height * 0.5);
              const bool same_depth = std::abs(pose->position[2] - known.position[2]) <
                  0.15 * std::min(pose->position[2], known.position[2]);
              if (same_depth && cv::norm(center - other_center) <
                  0.85 * std::min(box.width, other.width))
                return true;
              const double overlap = (box & other).area();
              return overlap / std::max(1.0, double(box.area() + other.area()) - overlap) > 0.5;
            });
        // 한 판에서 여러 변을 찾더라도 거리 표시는 하나만 남긴다.
        if (duplicate)
          continue;
        result.detections.push_back(*pose);
        ++debug.detections;
      }
    }
  }
  std::sort(result.detections.begin(), result.detections.end(),
            [](const auto &a, const auto &b) { return a.distance_m < b.distance_m; });
  result.status = config_.calibration_verified ? "ok" : "unverified_calibration";
  return result;
}

}  // namespace robot_vision
