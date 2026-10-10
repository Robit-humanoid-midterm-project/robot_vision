// 파일 역할: 한 프레임의 계산 결과를 화면과 메시지 생성이 공유할 수 있게 정의한다.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "robot_vision/boundary_distance_selector.hpp"

namespace robot_vision {

// One frame's calculations shared by the viewer and ROS message builders.
// No ROS messages or window state belong in this result.
struct VisionFrameResult {
    // 영상 후보와 실제 거리 검출 결과. 후보가 있다고 반드시 판의 거리가 계산된 것은 아니다.
    DetectResult obstacles;
    // Raw red OR blue occupancy in the bottom five rows, left/centre/right.
    std::array<double, 3> obstacle_ratio{{-1000.0, -1000.0, -1000.0}};
    // YOLO 좌우 경계선과 원본 영상 크기의 분할 마스크.
    LaneResult lane;
    // 현재 프레임에서 관측되어 평활화된 행 기준선.
    std::vector<RowLine> rows;
    // 차선과 행의 교차점 및 좌우 경계 거리.
    FieldGeometryResult geometry;
    // Same-frame raw measurements and each source's filtered left coordinate for comparison.
    FieldGeometryResult ground_geometry, crossing_geometry;
    BoundarySource boundary_source{BoundarySource::none};
    std::optional<double> ground_left_median_m, crossing_left_median_m;
    // obstacles.detections와 같은 순서의 바닥 투영 결과. nullopt는 투영 불가를 뜻한다.
    std::vector<std::optional<GroundProjection>> ground_projections;
    // 차선 처리 실패 원인. 중심 노드가 로그로 표시한다.
    std::string lane_error;
};

} // namespace robot_vision
