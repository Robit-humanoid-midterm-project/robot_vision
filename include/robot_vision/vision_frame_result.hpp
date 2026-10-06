#pragma once

#include <optional>
#include <string>
#include <vector>

#include "robot_vision/field_geometry.hpp"

namespace robot_vision {

// One frame's calculations shared by the viewer and ROS message builders.
// No ROS messages or window state belong in this result.
struct VisionFrameResult {
    DetectResult obstacles;
    LaneResult lane;
    std::vector<RowLine> rows;
    FieldGeometryResult geometry;
    std::vector<std::optional<GroundProjection>> ground_projections;
    std::string lane_error;
};

} // namespace robot_vision
