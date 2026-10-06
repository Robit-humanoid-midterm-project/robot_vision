#include "robot_vision/vision_message_builder.hpp"

#include <algorithm>
#include <cmath>
#include <rclcpp/time.hpp>

namespace robot_vision {

msg::ObstacleArray make_obstacle_array(
    const builtin_interfaces::msg::Time &stamp, const std::string &optical_frame_id,
    bool calibration_verified, const std::string &status)
{
    msg::ObstacleArray array;
    array.header.stamp = stamp;
    array.header.frame_id = optical_frame_id;
    array.calibration_verified = calibration_verified;
    array.status = status;
    return array;
}

void append_obstacle_detections(msg::ObstacleArray &array, const VisionFrameResult &result)
{
    for (size_t index = 0; index < result.obstacles.detections.size(); ++index) {
        const auto &detection = result.obstacles.detections[index];
        const auto &ground = result.ground_projections.at(index);
        msg::ObstacleDetection item;
        item.color = detection.color;
        item.position.x = detection.position[0];
        item.position.y = detection.position[1];
        item.position.z = detection.position[2];
        item.distance_m = detection.distance_m;
        if (ground)
        {
            item.ground_distance_m = ground->radial_m;
            item.ground_distance_valid = true;
            item.forward_distance_m = ground->forward_m;
            item.forward_distance_valid = true;
        }
        item.color_split_estimate = detection.color_split_estimate;
        item.reprojection_error_px = detection.reprojection_error_px;
        for (size_t i = 0; i < 4; ++i)
        {
            item.corners[i].x = static_cast<float>(detection.corners[i].x);
            item.corners[i].y = static_cast<float>(detection.corners[i].y);
            item.corners[i].z = 0;
        }
        array.detections.push_back(item);
    }
}

msg::LaneLine make_lane_message(const std_msgs::msg::Header &header, const LaneResult &lane)
{
    msg::LaneLine lane_message;
    lane_message.header = header;
    if (lane.best.valid)
    {
        const auto &line = lane.best;
        lane_message.valid = true;
        lane_message.side = line.side;
        lane_message.top.x = line.top.x;
        lane_message.top.y = line.top.y;
        lane_message.bottom.x = line.bottom.x;
        lane_message.bottom.y = line.bottom.y;
        lane_message.reference_y_px = line.reference_y_px;
        lane_message.line_x_at_reference_px = line.line_x_at_reference_px;
        lane_message.pixel_separation_px = line.pixel_separation_px;
        lane_message.observed_y_min = line.observed_y_min;
        lane_message.observed_y_max = line.observed_y_max;
        lane_message.grass_support = static_cast<float>(line.grass_support);
        lane_message.fit_error_px = static_cast<float>(line.fit_error_px);
    }
    return lane_message;
}

humanoid_interfaces::msg::VisionData make_master_message(
    const msg::ObstacleArray &array, bool frame_drop, double left_distance, double right_distance)
{
    humanoid_interfaces::msg::VisionData message;
    message.timestamp = rclcpp::Time(array.header.stamp).seconds();
    message.left_x_1_dist = message.right_x_2_dist = -1000.0;
    message.obstacle_1.fill(-1000.0);
    message.obstacle_2.fill(-1000.0);
    message.obstacle_3.fill(-1000.0);
    if (!frame_drop)
    {
        message.left_x_1_dist = left_distance;
        message.right_x_2_dist = right_distance;
        std::vector<const msg::ObstacleDetection *> nearby;
        for (const auto &obstacle : array.detections)
        {
            if (!obstacle.forward_distance_valid ||
                !std::isfinite(obstacle.forward_distance_m) ||
                obstacle.forward_distance_m < 0.0 || obstacle.forward_distance_m >= 1.5 ||
                !std::isfinite(obstacle.position.x))
                continue;
            // No additional overlap filtering in the outgoing message.
            nearby.push_back(&obstacle);
        }
        std::stable_sort(nearby.begin(), nearby.end(), [](const auto *a, const auto *b) {
            return std::hypot(a->forward_distance_m, a->position.x) <
                   std::hypot(b->forward_distance_m, b->position.x);
        });
        std::array<double, 2> *slots[] = {
            &message.obstacle_1, &message.obstacle_2, &message.obstacle_3};
        for (size_t i = 0; i < std::min(size_t(3), nearby.size()); ++i)
            *slots[i] = {nearby[i]->forward_distance_m, nearby[i]->position.x};
    }
    return message;
}

} // namespace robot_vision
