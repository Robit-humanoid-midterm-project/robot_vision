#pragma once

#include <std_msgs/msg/header.hpp>
#include "humanoid_interfaces/msg/vision_data.hpp"
#include "robot_vision/msg/lane_line.hpp"
#include "robot_vision/msg/obstacle_array.hpp"
#include "robot_vision/vision_frame_result.hpp"

namespace robot_vision {

msg::ObstacleArray make_obstacle_array(
    const builtin_interfaces::msg::Time &stamp, const std::string &optical_frame_id,
    bool calibration_verified, const std::string &status);
void append_obstacle_detections(msg::ObstacleArray &array, const VisionFrameResult &result);
msg::LaneLine make_lane_message(const std_msgs::msg::Header &header, const LaneResult &lane);
// Unknown fields use -1000; at most three obstacles with forward distance < 1.5 m.
humanoid_interfaces::msg::VisionData make_master_message(
    const msg::ObstacleArray &array, bool frame_drop,
    double left_distance = -1000.0, double right_distance = -1000.0);

} // namespace robot_vision
