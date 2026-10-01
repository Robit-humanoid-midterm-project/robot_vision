#include "robot_vision/distance_estimator.hpp"

#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace robot_vision {
ColorResult detect_color_regions(const cv::Mat &bgr, const DetectorConfig &config) {
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("Expected nonempty BGR8 image");
  ColorResult result;
  result.mask_preview = cv::Mat::zeros(bgr.size(), CV_8UC3);
  cv::Mat hsv, red1, red2;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  auto scalar = [](const std::array<int, 3> &v) { return cv::Scalar(v[0], v[1], v[2]); };
  cv::inRange(hsv, scalar(config.red_lower_1), scalar(config.red_upper_1), red1);
  cv::inRange(hsv, scalar(config.red_lower_2), scalar(config.red_upper_2), red2);
  cv::bitwise_or(red1, red2, result.colors[0].raw_mask);
  cv::inRange(hsv, scalar(config.blue_lower), scalar(config.blue_upper), result.colors[1].raw_mask);
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
  for (size_t i = 0; i < result.colors.size(); ++i) {
    auto &debug = result.colors[i];
    debug.raw_pixels = cv::countNonZero(debug.raw_mask);
    cv::morphologyEx(debug.raw_mask, debug.cleaned_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(debug.cleaned_mask, debug.cleaned_mask, cv::MORPH_CLOSE, kernel);
    debug.cleaned_pixels = cv::countNonZero(debug.cleaned_mask);
    result.mask_preview.setTo(i == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0), debug.cleaned_mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(debug.cleaned_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    debug.candidate_contours = static_cast<int>(contours.size());
    for (const auto &contour : contours) {
      const double area = cv::contourArea(contour);
      if (area < config.min_area_px) continue;
      const auto bounds = cv::boundingRect(contour);
      const auto moments = cv::moments(contour);
      if (moments.m00 <= 0) continue;
      result.candidates.push_back({i == 0 ? "red" : "blue", bounds,
          {moments.m10 / moments.m00, moments.m01 / moments.m00}, area,
          bounds.x == 0 || bounds.y == 0 || bounds.br().x >= bgr.cols || bounds.br().y >= bgr.rows, contour});
    }
  }
  return result;
}
}  // namespace robot_vision
