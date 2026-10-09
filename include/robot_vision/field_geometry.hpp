// 파일 역할: 차선·행 관계 계산의 반환 형식과 함수 인터페이스를 선언한다.

#pragma once

#include <optional>
#include <vector>
#include <opencv2/core.hpp>

#include "robot_vision/distance_estimator.hpp"
#include "robot_vision/lane_line_estimator.hpp"
#include "robot_vision/row_line_estimator.hpp"

namespace robot_vision {

// 교차점 계산의 진행 상태: 입력 부족, 교차 불가, 깊이 부족, 거리 계산 완료.
enum class CrossingStatus { no_lane_or_row, no_intersection, no_row_depth, measured, waiting_reference, invalid_ground_line };

// 표시용 교차점과 카메라 중앙점, 좌우 거리 값을 저장한다. 미측정 경계 거리는 -1000이다.
struct FieldGeometryResult {
    CrossingStatus status{CrossingStatus::no_lane_or_row};
    std::optional<cv::Point2d> crossing;
    std::optional<cv::Point2d> left_crossing, right_crossing;
    std::optional<cv::Point> principal;
    // Ground line equation X = a*Y + b; diagnostic values before distance smoothing.
    std::optional<cv::Point2d> ground_left_ab, ground_right_ab;
    std::optional<double> ground_robot_x_m;
    bool ground_based{false};
    double lateral_m{0.0};
    double left_distance_m{-1000.0};
    double right_distance_m{-1000.0};
};

// Existing assumption: the camera is inside a 1.5 m wide field and faces its length.
// 영상 픽셀 위치와 같은 행의 장애물 깊이, 보정값으로 경계 거리를 추정한다.
FieldGeometryResult estimate_field_geometry(
    const DetectResult &obstacles, const LaneResult &lane,
    const std::vector<RowLine> &rows, const cv::Size &size, const DetectorConfig &config);

} // namespace robot_vision
