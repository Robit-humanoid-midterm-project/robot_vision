// 파일 역할: 색상 후보, 장애물 위치·거리, 보정·검출 설정과 거리 추정기 인터페이스를 정의한다.

#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace robot_vision {

// 정면을 향한 40cm 정사각형 판에서 사용할 변. 연결된 윤곽에서도 여러 후보를 찾는다.
// 양 끝이 모서리처럼 보이는 변만 후보로 삼는다. 끝점까지 가려진 경우는 구분하기 어렵다.
// 가려진 판에서도 온전히 보이는 한 변의 양 끝과 방향을 저장한다.
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

// 거리 계산을 통과한 판의 결과. position은 카메라 기준 판 아래쪽 중심이며 단위는 m이다.
struct Detection {
  std::string color;
  // 카메라 광학 좌표: x 오른쪽, y 아래, z 앞쪽. 카메라 기울기를 포함한 로봇 몸체 좌표는 아니다.
  cv::Vec3d position{0.0, 0.0, 0.0};
  // 카메라에서 판 아래쪽 중심까지의 직선거리(Range). 전방거리와 같은 값이 아니다.
  double distance_m{0.0};
  double reprojection_error_px{0.0};
  double shape_fit{0.0};
  bool color_split_estimate{false};
  std::array<cv::Point2d, 4> corners{};
};

// 판까지의 바닥거리와 전방·좌우 성분을 저장한다. 판 밑면이 카메라 아래와 같은 바닥에 있다는 가정이 필요하다.
struct GroundProjection {
  double forward_m{0.0};
  double radial_m{0.0};
  double lateral_m{0.0};
};

std::optional<GroundProjection> project_to_ground(
    const Detection &detection, double camera_height_m);

// 실측 판 크기, 보정 해상도·행렬, HSV 범위와 윤곽·거리 판정 기준을 모은다.
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

// 색상별 정리 전후 마스크와 픽셀·윤곽·검출 개수. 전처리 화면에서 사용한다.
struct ColorDebug {
  cv::Mat raw_mask;
  cv::Mat cleaned_mask;
  int raw_pixels{0};
  int cleaned_pixels{0};
  int candidate_contours{0};
  int detections{0};
};

// Visible color component, not necessarily one complete physical obstacle.
// 영상에 보이는 색상 영역 후보. 물리적인 판 하나라는 보장이나 유효한 거리 값은 없다.
struct ImageCandidate {
  std::string color;
  cv::Rect bounds;
  cv::Point2d center;
  double area_px{0};
  bool touches_border{false};
  std::vector<cv::Point> contour;
};

// 색상 검출 단계의 후보·마스크·통계를 묶는다.
struct ColorResult {
  std::array<ColorDebug, 2> colors;
  cv::Mat mask_preview;
  std::vector<ImageCandidate> candidates;
};

// BGR 영상과 색상 설정을 받아 거리 계산 전의 영역 후보를 만든다.
ColorResult detect_color_regions(const cv::Mat &bgr, const DetectorConfig &config);

// 색상 후보와 거리 계산 성공 결과를 구분해 담고, 처리 상태 문자열도 반환한다.
struct DetectResult {
  std::vector<ImageCandidate> image_candidates;
  std::vector<Detection> detections;
  cv::Mat mask_preview;
  std::array<ColorDebug, 2> colors;  // red, blue
  std::string status;
};

// 보정값을 가진 거리 계산 객체. 사각형과 가림 복구 결과를 검사해 판의 위치를 추정한다.
class DistanceEstimator {
 public:
  // 설정을 검사하고 보정 행렬을 내부 복사본으로 보관한다.
  explicit DistanceEstimator(DetectorConfig config);
  // 한 프레임의 색상 후보와 거리 검출 결과를 반환한다.
  DetectResult detect(const cv::Mat &bgr) const;
  // 네 영상 꼭짓점의 정사각형 자세를 계산한다. 계산 불가면 nullopt다.
  std::optional<Detection> square_pose(const std::array<cv::Point2d, 4> &corners) const;

 private:
  // 한 변 검출, 거리 계산, 추정 모서리 구성을 한 곳에서 처리한다.
  std::optional<Detection> pose_for_visible_edge(const VisibleEdge &edge) const;
  // 윤곽을 사각형으로 근사하고 형태·오차·거리 조건을 검사한다.
  std::optional<Detection> pose_for_contour(const std::vector<cv::Point> &contour,
                                            const cv::Mat &mask,
                                            const cv::Size &image_size,
                                            bool box_fallback) const;
  // 합쳐진 색상 윤곽 안팎의 영상 경계선으로 판 사각형을 복구한다.
  std::optional<Detection> line_pose_for_component(
      const std::vector<cv::Point> &contour, const cv::Mat &mask,
      const cv::Mat &bgr) const;
  // 채도 차이를 이용해 붙은 영역을 분리하며 출력 마스크에도 결과를 채운다.
  std::vector<std::vector<cv::Point>> split_touching_color(
      const std::vector<cv::Point> &contour, const cv::Mat &mask,
      const cv::Mat &hsv, int min_saturation, cv::Mat &saturated) const;

  DetectorConfig config_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_;
};

}  // namespace robot_vision
