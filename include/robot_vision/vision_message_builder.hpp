// 파일 역할: 계산 결과를 ROS 메시지로 바꾸는 함수의 인터페이스를 선언한다.

#pragma once

#include <std_msgs/msg/header.hpp>
#include "humanoid_interfaces/msg/vision_data.hpp"
#include "robot_vision/msg/lane_line.hpp"
#include "robot_vision/msg/obstacle_array.hpp"
#include "robot_vision/vision_frame_result.hpp"

namespace robot_vision {

// 영상 시각·좌표계·보정 확인 상태·처리 상태를 담은 메시지 틀을 만든다.
msg::ObstacleArray make_obstacle_array(
    const builtin_interfaces::msg::Time &stamp, const std::string &optical_frame_id,
    bool calibration_verified, const std::string &status);
// 각 장애물의 카메라 좌표, 거리와 투영 유효 여부를 배열에 추가한다.
void append_obstacle_detections(msg::ObstacleArray &array, const VisionFrameResult &result);
// 경계선 검출 결과를 영상 헤더와 함께 ROS 차선 메시지로 변환한다.
msg::LaneLine make_lane_message(const std_msgs::msg::Header &header, const LaneResult &lane);
// Unknown fields use -1000; at most three obstacles with forward distance < 1.5 m.
// 좌우 경계 거리와 유효한 가까운 장애물 최대 세 개를 로봇 제어 메시지로 구성한다.
humanoid_interfaces::msg::VisionData make_master_message(
    const msg::ObstacleArray &array, bool frame_drop,
    double left_distance = -1000.0, double right_distance = -1000.0,
    std::array<double, 3> obstacle_ratio = {{-1000.0, -1000.0, -1000.0}});

} // namespace robot_vision
