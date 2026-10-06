// 파일 역할: BGR 영상에서 빨강·파랑 픽셀과 색상 영역 후보를 찾는다.
// 거리 계산 없이 마스크, 윤곽, 후보 위치와 전처리 통계를 반환한다.

#include "robot_vision/distance_estimator.hpp"

#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace robot_vision {
// 색상 범위로 마스크를 만들고 잡음을 정리한 뒤 충분히 큰 윤곽을 후보로 반환한다.
// 후보는 보이는 색상 영역이며, 실제 장애물 판 하나로 확정된 결과는 아니다.
ColorResult detect_color_regions(const cv::Mat &bgr, const DetectorConfig &config) {
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("Expected nonempty BGR8 image");
  ColorResult result;
  result.mask_preview = cv::Mat::zeros(bgr.size(), CV_8UC3);
  cv::Mat hsv, red1, red2;
  // 조명 밝기와 색상 조건을 나눠 지정할 수 있도록 BGR을 HSV로 바꾼다.
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  auto scalar = [](const std::array<int, 3> &v) { return cv::Scalar(v[0], v[1], v[2]); };
  // 빨강은 H 범위의 양 끝에 걸치므로 두 마스크를 만들고 합친다. 파랑은 한 범위를 사용한다.
  cv::inRange(hsv, scalar(config.red_lower_1), scalar(config.red_upper_1), red1);
  cv::inRange(hsv, scalar(config.red_lower_2), scalar(config.red_upper_2), red2);
  cv::bitwise_or(red1, red2, result.colors[0].raw_mask);
  cv::inRange(hsv, scalar(config.blue_lower), scalar(config.blue_upper), result.colors[1].raw_mask);
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
  for (size_t i = 0; i < result.colors.size(); ++i) {
    auto &debug = result.colors[i];
    debug.raw_pixels = cv::countNonZero(debug.raw_mask);
    // 열기는 작은 잡음을 지우고, 닫기는 작은 틈을 메운다. 전후 픽셀 수도 기록한다.
    cv::morphologyEx(debug.raw_mask, debug.cleaned_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(debug.cleaned_mask, debug.cleaned_mask, cv::MORPH_CLOSE, kernel);
    debug.cleaned_pixels = cv::countNonZero(debug.cleaned_mask);
    result.mask_preview.setTo(i == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0), debug.cleaned_mask);
    std::vector<std::vector<cv::Point>> contours;
    // 색상 영역의 바깥 윤곽을 찾는다. 내부 구멍은 별도 후보로 만들지 않는다.
    cv::findContours(debug.cleaned_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    debug.candidate_contours = static_cast<int>(contours.size());
    for (const auto &contour : contours) {
      const double area = cv::contourArea(contour);
      if (area < config.min_area_px) continue;
      const auto bounds = cv::boundingRect(contour);
      // 윤곽의 면적 중심을 구해 후보 위치를 기록한다. 화면 가장자리에 닿았는지도 함께 남긴다.
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
