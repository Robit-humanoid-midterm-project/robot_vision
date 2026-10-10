// 파일 역할: 계산 결과를 ROS 장애물·차선·로봇 제어 메시지로 변환한다.
// 값을 구성해 반환하며, 실제 토픽 발행은 중심 노드가 수행한다.

#include "robot_vision/vision_message_builder.hpp"

#include <algorithm>
#include <cmath>
#include <rclcpp/time.hpp>

namespace robot_vision {

// 영상 시각, 카메라 좌표계 이름, 보정 확인 상태와 처리 상태를 담은 빈 장애물 메시지를 만든다.
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

// 프레임의 장애물 결과를 배열 메시지에 추가한다.
// 카메라 좌표, 거리, 바닥 투영의 유효 여부와 영상 꼭짓점을 복사한다.
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
        // 바닥 투영이 가능한 경우에만 바닥·전방 거리의 valid를 true로 설정한다.
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

// 차선 검출 결과를 ROS 메시지로 바꾼다.
// 유효한 차선이 없으면 헤더만 설정하고 valid는 기본값 false로 남긴다.
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
        lane_message.confidence = static_cast<float>(line.confidence);
    }
    return lane_message;
}

// 로봇에 보낼 좌우 경계 거리와 가까운 장애물 최대 세 개를 구성한다.
// 장애물 좌표는 (전방 거리, 좌우 위치)이며 단위는 m, 미측정 값은 -1000이다.
humanoid_interfaces::msg::VisionData make_master_message(
    const msg::ObstacleArray &array, bool frame_drop, double left_distance, double right_distance,
    std::array<double, 3> obstacle_ratio)
{
    humanoid_interfaces::msg::VisionData message;
    message.frame_drop = frame_drop ? 1.0 : 0.0;
    message.obstacle_ratio.fill(-1000.0);
    message.timestamp = rclcpp::Time(array.header.stamp).seconds();
    // 먼저 모든 측정 필드를 -1000으로 채운다. 영상 끊김이면 이 상태를 그대로 반환한다.
    message.left_x_1_dist = message.right_x_2_dist = -1000.0;
    message.obstacle_1.fill(-1000.0);
    message.obstacle_2.fill(-1000.0);
    message.obstacle_3.fill(-1000.0);
    if (!frame_drop)
    {
        for (size_t i=0; i<obstacle_ratio.size(); ++i)
            if (std::isfinite(obstacle_ratio[i]) && obstacle_ratio[i]>=0.0 && obstacle_ratio[i]<=1.0)
                message.obstacle_ratio[i] = obstacle_ratio[i];
        message.left_x_1_dist = left_distance;
        message.right_x_2_dist = right_distance;
        std::vector<const msg::ObstacleDetection *> nearby;
        for (const auto &obstacle : array.detections)
        {
            // 전방 거리가 유효하고 0 이상 1.5m 미만인 장애물만 제어 메시지의 후보로 남긴다.
            if (!obstacle.forward_distance_valid ||
                !std::isfinite(obstacle.forward_distance_m) ||
                obstacle.forward_distance_m < 0.0 || obstacle.forward_distance_m >= 1.5 ||
                !std::isfinite(obstacle.position.x))
                continue;
            // No additional overlap filtering in the outgoing message.
            nearby.push_back(&obstacle);
        }
        // 전방 거리와 좌우 위치로 바닥상의 상대 거리를 구해 가까운 장애물부터 정렬한다.
        std::stable_sort(nearby.begin(), nearby.end(), [](const auto *a, const auto *b) {
            return std::hypot(a->forward_distance_m, a->position.x) <
                   std::hypot(b->forward_distance_m, b->position.x);
        });
        // 여기의 slots는 obstacle_1~3 메시지 필드의 저장 위치다. 삭제한 3×3 배치 기능과는 관계없다.
        std::array<double, 2> *slots[] = {
            &message.obstacle_1, &message.obstacle_2, &message.obstacle_3};
        for (size_t i = 0; i < std::min(size_t(3), nearby.size()); ++i)
            *slots[i] = {nearby[i]->forward_distance_m, nearby[i]->position.x};
    }
    return message;
}

} // namespace robot_vision
