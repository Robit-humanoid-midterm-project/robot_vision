#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace robot_vision {

// 정면을 향한 40cm 정사각형 판에서 사용할 변. 연결된 윤곽에서도 여러 후보를 찾는다.
// 양 끝이 모서리처럼 보이는 변만 후보로 삼는다. 끝점까지 가려진 경우는 구분하기 어렵다.
struct VisibleEdge {
  cv::Point2d first;
  cv::Point2d last;
  bool horizontal{true};
  bool upper_or_left{true};
};

std::vector<VisibleEdge> find_complete_visible_edges(
    const std::vector<cv::Point> &contour, const cv::Mat &mask,
    double min_edge_px, int border_margin_px);

// 카메라 보정값으로 두 끝점을 왜곡 보정한 뒤, 40cm 변의 투영 폭에서 깊이를 구한다.
// 반환 위치는 기존 코드와 같이 판의 아래쪽 변 중심이다.
std::optional<cv::Vec3d> position_from_visible_edge(
    const VisibleEdge &edge, double side_m,
    const cv::Mat &camera_matrix, const cv::Mat &distortion);

struct Detection {
  std::string color;
  cv::Vec3d position{0.0, 0.0, 0.0};
  double distance_m{0.0};
  double reprojection_error_px{0.0};
  double shape_fit{0.0};
  bool color_split_estimate{false};
  std::array<cv::Point2d, 4> corners{};
};

struct GroundProjection {
  double forward_m{0.0};
  double radial_m{0.0};
  double lateral_m{0.0};
};

std::optional<GroundProjection> project_to_ground(
    const Detection &detection, double camera_height_m);

struct DetectorConfig {
  double obstacle_size_m{0.4};
  int calibration_width{640};
  int calibration_height{480};
  bool calibration_verified{false};
  std::vector<double> camera_matrix{471.953641, 0, 309.509126, 0, 476.574144, 228.222101, 0, 0, 1};
  std::vector<double> distortion_coefficients{0.026101, -0.086354, -0.007248, -0.006668, 0};
  std::array<int, 3> red_lower_1{0, 100, 60};
  std::array<int, 3> red_upper_1{12, 255, 255};
  std::array<int, 3> red_lower_2{168, 100, 60};
  std::array<int, 3> red_upper_2{179, 255, 255};
  std::array<int, 3> blue_lower{105, 170, 45};
  std::array<int, 3> blue_upper{125, 255, 255};
  double min_area_px{500};
  double min_edge_px{15};
  int border_margin_px{0};
  double min_fill_ratio{0.8};
  double near_min_edge_px{120.0};
  double near_min_fill_ratio{0.72};
  double near_max_reprojection_error_px{5.5};
  double near_max_relative_reprojection_error{0.045};
  double max_reprojection_error_px{3.5};
  double max_relative_reprojection_error{0.03};
  double min_distance_m{0.2};
  double max_distance_m{10};
};

struct ColorDebug {
  cv::Mat raw_mask;
  cv::Mat cleaned_mask;
  int raw_pixels{0};
  int cleaned_pixels{0};
  int candidate_contours{0};
  int detections{0};
};

// Visible color component, not necessarily one complete physical obstacle.
struct ImageCandidate {
  std::string color;
  cv::Rect bounds;
  cv::Point2d center;
  double area_px{0};
  bool touches_border{false};
  std::vector<cv::Point> contour;
};

struct ColorResult {
  std::array<ColorDebug, 2> colors;
  cv::Mat mask_preview;
  std::vector<ImageCandidate> candidates;
};

ColorResult detect_color_regions(const cv::Mat &bgr, const DetectorConfig &config);

struct DetectResult {
  std::vector<ImageCandidate> image_candidates;
  std::vector<Detection> detections;
  cv::Mat mask_preview;
  std::array<ColorDebug, 2> colors;  // red, blue
  std::string status;
};

class DistanceEstimator {
 public:
  explicit DistanceEstimator(DetectorConfig config);
  DetectResult detect(const cv::Mat &bgr) const;
  std::optional<Detection> square_pose(const std::array<cv::Point2d, 4> &corners) const;

 private:
  // 한 변 검출, 거리 계산, 추정 모서리 구성을 한 곳에서 처리한다.
  std::optional<Detection> pose_for_visible_edge(const VisibleEdge &edge) const;
  std::optional<Detection> pose_for_contour(const std::vector<cv::Point> &contour,
                                            const cv::Mat &mask,
                                            const cv::Size &image_size,
                                            bool box_fallback) const;
  std::optional<Detection> line_pose_for_component(
      const std::vector<cv::Point> &contour, const cv::Mat &mask,
      const cv::Mat &bgr) const;
  std::vector<std::vector<cv::Point>> split_touching_color(
      const std::vector<cv::Point> &contour, const cv::Mat &mask,
      const cv::Mat &hsv, int min_saturation, cv::Mat &saturated) const;

  DetectorConfig config_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_;
};

}  // namespace robot_vision
