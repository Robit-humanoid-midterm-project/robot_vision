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

std::array<cv::Point2d, 4> order_quad(std::array<cv::Point2d, 4> points) {
  cv::Point2d center;
  for (const auto &p : points) center += p;
  center *= 0.25;
  std::sort(points.begin(), points.end(), [&](const auto &a, const auto &b) {
    return std::atan2(a.y - center.y, a.x - center.x) <
           std::atan2(b.y - center.y, b.x - center.x);
  });
  const auto first = std::min_element(points.begin(), points.end(), [](const auto &a, const auto &b) {
    return a.x + a.y < b.x + b.y;
  });
  std::rotate(points.begin(), first, points.end());
  return points;
}

cv::Scalar hsv_scalar(const std::array<int, 3> &value) {
  return cv::Scalar(value[0], value[1], value[2]);
}

void validate_hsv(const std::array<int, 3> &lower, const std::array<int, 3> &upper) {
  for (int i = 0; i < 3; ++i) {
    const int maximum = i == 0 ? 179 : 255;
    if (lower[i] < 0 || lower[i] > upper[i] || upper[i] > maximum) {
      throw std::invalid_argument("Invalid HSV threshold");
    }
  }
}

bool finite_vec(const cv::Vec3d &value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

}  // namespace

std::optional<GroundProjection> project_to_ground(
    const Detection &detection, double camera_height_m) {
  const double range = detection.distance_m;
  const double lateral = detection.position[0];
  if (!std::isfinite(range) || !std::isfinite(lateral) ||
      !std::isfinite(camera_height_m) || camera_height_m <= 0 ||
      range < camera_height_m) return std::nullopt;
  const double radial_squared = (range - camera_height_m) *
                                (range + camera_height_m);
  const double forward_squared = radial_squared - lateral * lateral;
  if (forward_squared < 0) return std::nullopt;
  return GroundProjection{std::sqrt(forward_squared),
                          std::sqrt(radial_squared), lateral};
}

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
      std::find_if(config_.distortion_coefficients.begin(), config_.distortion_coefficients.end(),
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

std::optional<Detection> DistanceEstimator::square_pose(
    const std::array<cv::Point2d, 4> &unordered_corners) const {
  const auto corners = order_quad(unordered_corners);
  const double s = config_.obstacle_size_m / 2.0;
  const std::vector<cv::Point3d> object{{-s, s, 0}, {s, s, 0}, {s, -s, 0}, {-s, -s, 0}};
  const std::vector<cv::Point2d> image(corners.begin(), corners.end());
  std::vector<cv::Mat> rotations, translations;
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
      if (!finite_vec(t)) continue;
      cv::Mat matrix;
      cv::Rodrigues(rvec, matrix);
      bool in_front = true;
      for (const auto &point : object) {
        const double depth = matrix.at<double>(2, 0) * point.x +
                             matrix.at<double>(2, 1) * point.y +
                             matrix.at<double>(2, 2) * point.z + t[2];
        if (depth <= 0 || !std::isfinite(depth)) in_front = false;
      }
      if (!in_front) continue;
      std::vector<cv::Point2d> projected;
      cv::projectPoints(object, rvec, tvec, camera_matrix_, distortion_, projected);
      double squared_error = 0;
      for (size_t i = 0; i < 4; ++i) {
        const auto delta = projected[i] - corners[i];
        squared_error += delta.x * delta.x + delta.y * delta.y;
      }
      const double error = std::sqrt(squared_error / 4.0);
      if (!std::isfinite(error)) continue;
      if (!best || error < best->reprojection_error_px) {
        Detection detection;
        // The square's bottom-edge midpoint is local (0, -s, 0).
        const cv::Vec3d bottom{
            t[0] - s * matrix.at<double>(0, 1),
            t[1] - s * matrix.at<double>(1, 1),
            t[2] - s * matrix.at<double>(2, 1)};
        if (!finite_vec(bottom) || bottom[2] <= 0) continue;
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

std::optional<Detection> DistanceEstimator::pose_for_contour(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    const cv::Size &image_size, bool box_fallback) const {
  const double area = cv::contourArea(contour);
  if (area < config_.min_area_px) return std::nullopt;
  const cv::Rect bounds = cv::boundingRect(contour);
  const int margin = config_.border_margin_px;
  if (bounds.x <= margin || bounds.y <= margin ||
      bounds.x + bounds.width >= image_size.width - margin ||
      bounds.y + bounds.height >= image_size.height - margin) {
    return std::nullopt;
  }

  cv::Mat outline = cv::Mat::zeros(mask.size(), CV_8UC1);
  cv::drawContours(outline, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);
  const int outline_pixels = cv::countNonZero(outline);
  std::vector<std::array<cv::Point2d, 4>> candidates;
  if (box_fallback) {
    const cv::RotatedRect rectangle = cv::minAreaRect(contour);
    const double short_side = std::min(rectangle.size.width, rectangle.size.height);
    const double long_side = std::max(rectangle.size.width, rectangle.size.height);
    if (short_side < config_.min_edge_px || long_side / short_side > 2.5) {
      return std::nullopt;
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
    for (const auto &point : candidate) polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    cv::fillConvexPoly(region, polygon, cv::Scalar(255));
    cv::Mat intersection;
    cv::bitwise_and(outline, region, intersection);
    const int overlap = cv::countNonZero(intersection);
    const int union_pixels = outline_pixels + cv::countNonZero(region) - overlap;
    const double fit = static_cast<double>(overlap) / std::max(1, union_pixels);
    const auto corners = order_quad(candidate);
    double min_edge = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < 4; ++i) {
      min_edge = std::min(min_edge, cv::norm(corners[i] - corners[(i + 1) % 4]));
    }
    if (min_edge < config_.min_edge_px) continue;
    const bool near_face = min_edge >= config_.near_min_edge_px;
    const double min_fill = near_face ? config_.near_min_fill_ratio :
                                        config_.min_fill_ratio;
    if (fit < min_fill) continue;
    auto pose = square_pose(corners);
    if (!pose) continue;
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
        pose->distance_m > config_.max_distance_m) continue;
    pose->shape_fit = fit;
    if (!best || fit > best->shape_fit) best = pose;
  }
  return best;
}

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
  if (samples.size() < static_cast<size_t>(config_.min_area_px)) return {};
  auto percentile = [&](double fraction) {
    const int target = static_cast<int>(std::ceil(fraction * samples.size()));
    int cumulative = 0;
    for (int value = 0; value < 256; ++value) {
      cumulative += histogram[value];
      if (cumulative >= target) return value;
    }
    return 255;
  };
  if (percentile(0.95) - percentile(0.05) < 8) return {};
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
      if (m[x] && c[x] && h[x][1] >= cutoff) out[x] = 255;
    }
  }
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
  cv::morphologyEx(saturated, saturated, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(saturated, saturated, cv::MORPH_CLOSE, kernel);
  std::vector<std::vector<cv::Point>> parts, accepted;
  cv::findContours(saturated, parts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  const double minimum = 0.4 * cv::contourArea(contour);
  for (auto &part : parts) {
    if (cv::contourArea(part) >= minimum) accepted.push_back(std::move(part));
  }
  return accepted;
}

std::optional<Detection> DistanceEstimator::line_pose_for_component(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    const cv::Mat &bgr) const {
  // A near-frontal square can be recovered when same-colored objects merge:
  // use four visible image edges surrounding the largest colored interior.
  const cv::Rect bounds = cv::boundingRect(contour);
  if (bounds.x <= config_.border_margin_px || bounds.y <= config_.border_margin_px ||
      bounds.x + bounds.width >= bgr.cols - config_.border_margin_px ||
      bounds.y + bounds.height >= bgr.rows - config_.border_margin_px) {
    return std::nullopt;
  }
  cv::Mat component = cv::Mat::zeros(mask.size(), CV_8UC1);
  cv::drawContours(component, std::vector<std::vector<cv::Point>>{contour}, 0,
                   cv::Scalar(255), cv::FILLED);
  cv::Mat distances;
  cv::distanceTransform(component, distances, cv::DIST_L2, 5);
  cv::Point center;
  double peak = 0;
  cv::minMaxLoc(distances, nullptr, &peak, nullptr, &center);
  if (peak < config_.min_edge_px / 2.0) return std::nullopt;

  const cv::Rect search = (cv::Rect(bounds.x - 12, bounds.y - 12,
                                    bounds.width + 24, bounds.height + 24) &
                           cv::Rect(0, 0, bgr.cols, bgr.rows));
  cv::Mat gray, blurred, edges;
  cv::cvtColor(bgr(search), gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
  cv::Canny(blurred, edges, 50, 120);
  std::vector<cv::Vec4i> lines;
  const int minimum_line = std::max(30, static_cast<int>(0.20 * std::min(bounds.width, bounds.height)));
  cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 30, minimum_line, 8);

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
    if (length < minimum_line) continue;
    const int mx = cvRound((x1 + x2) / 2.0), my = cvRound((y1 + y2) / 2.0);
    const cv::Rect around(std::max(0, mx - 3), std::max(0, my - 3), 7, 7);
    if ((around & cv::Rect(0, 0, component.cols, component.rows)).area() == 0 ||
        cv::countNonZero(component(around & cv::Rect(0, 0, component.cols, component.rows))) == 0) {
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
  if (!top.found || !bottom.found || !left.found || !right.found) return std::nullopt;
  const double width = right.coordinate - left.coordinate;
  const double height = bottom.coordinate - top.coordinate;
  if (width < config_.min_edge_px || height < config_.min_edge_px ||
      width / height < 0.65 || width / height > 1.5 ||
      top.length < 0.25 * width || bottom.length < 0.25 * width ||
      left.length < 0.25 * height || right.length < 0.25 * height) return std::nullopt;

  const std::array<cv::Point2d, 4> corners{{
      {left.coordinate, top.coordinate}, {right.coordinate, top.coordinate},
      {right.coordinate, bottom.coordinate}, {left.coordinate, bottom.coordinate}}};
  cv::Mat rectangle = cv::Mat::zeros(mask.size(), CV_8UC1);
  std::vector<cv::Point> polygon;
  for (const auto &corner : corners) polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
  cv::fillConvexPoly(rectangle, polygon, cv::Scalar(255));
  cv::Mat color_inside;
  cv::bitwise_and(mask, rectangle, color_inside);
  const double occupancy = static_cast<double>(cv::countNonZero(color_inside)) /
                           std::max(1, cv::countNonZero(rectangle));
  if (occupancy < 0.85) return std::nullopt;
  auto pose = square_pose(corners);
  if (!pose) return std::nullopt;
  const double max_error = std::min(config_.max_reprojection_error_px + 1.0,
                                    config_.max_relative_reprojection_error * std::min(width, height));
  if (pose->reprojection_error_px > max_error ||
      pose->distance_m < config_.min_distance_m ||
      pose->distance_m > config_.max_distance_m) return std::nullopt;
  pose->shape_fit = occupancy;
  pose->color_split_estimate = true;
  return pose;
}

DetectResult DistanceEstimator::detect(const cv::Mat &bgr) const {
  DetectResult result;
  if (bgr.empty()) {
    result.status = "no_image";
    return result;
  }
  result.mask_preview = cv::Mat::zeros(bgr.size(), CV_8UC3);
  if (bgr.cols != config_.calibration_width || bgr.rows != config_.calibration_height) {
    result.status = "image_size_mismatch";
    return result;
  }
  if (bgr.type() != CV_8UC3) throw std::invalid_argument("Expected BGR8 image");
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Mat red1, red2, red, blue;
  cv::inRange(hsv, hsv_scalar(config_.red_lower_1), hsv_scalar(config_.red_upper_1), red1);
  cv::inRange(hsv, hsv_scalar(config_.red_lower_2), hsv_scalar(config_.red_upper_2), red2);
  cv::bitwise_or(red1, red2, red);
  cv::inRange(hsv, hsv_scalar(config_.blue_lower), hsv_scalar(config_.blue_upper), blue);
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
  int color_index = 0;
  for (auto &entry : std::vector<std::pair<std::string, cv::Mat>>{{"red", red}, {"blue", blue}}) {
    auto &debug = result.colors[color_index++];
    cv::Mat &mask = entry.second;
    debug.raw_mask = mask.clone();
    debug.raw_pixels = cv::countNonZero(mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    debug.cleaned_mask = mask.clone();
    debug.cleaned_pixels = cv::countNonZero(mask);
    result.mask_preview.setTo(entry.first == "red" ? cv::Scalar(0, 0, 255) :
                                                     cv::Scalar(255, 0, 0), mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    debug.candidate_contours = static_cast<int>(contours.size());
    for (const auto &contour : contours) {
      auto pose = pose_for_contour(contour, mask, bgr.size(), false);
      if (pose) {
        pose->color = entry.first;
        result.detections.push_back(*pose);
        ++debug.detections;
        continue;
      }
      if (cv::contourArea(contour) < 3.0 * config_.min_area_px) continue;
      const int min_saturation = entry.first == "red" ?
          std::min(config_.red_lower_1[1], config_.red_lower_2[1]) : config_.blue_lower[1];
      cv::Mat saturated;
      bool recovered = false;
      for (const auto &part : split_touching_color(contour, mask, hsv,
                                                    min_saturation, saturated)) {
        pose = pose_for_contour(part, saturated, bgr.size(), true);
        if (pose) {
          pose->color = entry.first;
          pose->color_split_estimate = true;
          result.detections.push_back(*pose);
          ++debug.detections;
          recovered = true;
        }
      }
      if (!recovered) {
        pose = line_pose_for_component(contour, mask, bgr);
        if (pose) {
          pose->color = entry.first;
          result.detections.push_back(*pose);
          ++debug.detections;
        }
      }
    }
  }
  std::sort(result.detections.begin(), result.detections.end(),
            [](const auto &a, const auto &b) { return a.distance_m < b.distance_m; });
  result.status = config_.calibration_verified ? "ok" : "unverified_calibration";
  return result;
}

}  // namespace robot_vision
