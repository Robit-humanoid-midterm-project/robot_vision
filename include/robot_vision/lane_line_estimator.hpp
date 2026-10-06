// 파일 역할: 흰색 경계선 검출 설정, 선의 픽셀 좌표와 검출 결과를 선언한다.

#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace robot_vision {

// 흰색·잔디 색상 기준과 검색 영역, 기울기, 선 맞춤 품질 기준을 모은다.
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

// 선택된 선의 방향·영상 좌표·관측 구간·검출 품질. 이 구조체의 선 위치는 픽셀 기준이다.
struct LaneLine {
  // true일 때만 선의 나머지 측정 필드를 검출 성공 결과로 사용한다.
  bool valid{false};
  std::string side;  // left or right, classified by image slope
  cv::Point2f top, bottom;
  int observed_y_min{0}, observed_y_max{0};
  float reference_y_px{0};
  float line_x_at_reference_px{0};
  // 기준 높이에서 영상 중앙과 선 사이의 간격이다. 실제 거리(m)와 구분해야 한다.
  float pixel_separation_px{0};
  double grass_support{0};
  double fit_error_px{0};
  std::vector<cv::Vec4i> observed_segments;
};

// 경계선 하나(best)와 디버그용 흰색 마스크를 함께 반환한다.
struct LaneResult {
  LaneLine best;
  cv::Mat mask;
};

// BGR 영상에서 경계선 하나를 찾는다. exclude_mask의 켜진 픽셀은 검색에서 제외한다.
LaneResult estimate_lane_line(const cv::Mat &bgr, const LaneConfig &config,
                              const cv::Mat &exclude_mask = {});

}  // namespace robot_vision
