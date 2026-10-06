#include "robot_vision/field_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>

namespace robot_vision {

FieldGeometryResult estimate_field_geometry(
    const DetectResult &obstacles, const LaneResult &lane,
    const std::vector<RowLine> &rows, const cv::Size &size, const DetectorConfig &config)
{
    FieldGeometryResult result;
    if (!lane.best.valid || rows.empty())
        return result;
    result.status = CrossingStatus::no_intersection;
    const double center_x = (size.width - 1) * 0.5;
    const auto row_y = [](const RowLine &row, double x) {
        return row.first.y + (x - row.first.x) *
            (row.last.y - row.first.y) / double(row.last.x - row.first.x);
    };
    const RowLine *front = nullptr;
    for (const auto &row : rows)
        if (row.last.x != row.first.x &&
            (!front || row_y(row, center_x) > row_y(*front, center_x)))
            front = &row;
    if (!front)
        return result;
    const cv::Point2d a = lane.best.top;
    const cv::Point2d d = cv::Point2d(lane.best.bottom) - a;
    const cv::Point2d b = front->first;
    const cv::Point2d e = cv::Point2d(front->last) - b;
    const double denominator = d.cross(e);
    if (std::abs(denominator) <= 1e-6)
        return result;
    const cv::Point2d crossing = a + d * ((b - a).cross(e) / denominator);
    if (!(crossing.x >= 0 && crossing.x < size.width &&
          crossing.y >= 0 && crossing.y < size.height))
        return result;
    result.crossing = crossing;
    result.status = CrossingStatus::no_row_depth;
    std::vector<double> depths;
    for (const auto &detection : obstacles.detections) {
        const cv::Point2d base = (detection.corners[2] + detection.corners[3]) * 0.5;
        if (std::abs(base.y - row_y(*front, base.x)) <= 25.0 &&
            std::isfinite(detection.position[2]) && detection.position[2] > 0)
            depths.push_back(detection.position[2]);
    }
    if (depths.empty())
        return result;
    std::sort(depths.begin(), depths.end());
    const size_t middle = depths.size() / 2;
    const double depth = depths.size() % 2 ? depths[middle] :
        (depths[middle - 1] + depths[middle]) * 0.5;
    const cv::Mat k(3, 3, CV_64F, const_cast<double *>(config.camera_matrix.data()));
    const cv::Mat distortion(1, config.distortion_coefficients.size(), CV_64F,
                             const_cast<double *>(config.distortion_coefficients.data()));
    std::vector<cv::Point2d> normalized;
    cv::undistortPoints(std::vector<cv::Point2d>{crossing}, normalized, k, distortion);
    const double lateral = normalized.front().x * depth;
    if (!std::isfinite(lateral))
        return result;
    result.status = CrossingStatus::measured;
    result.lateral_m = lateral;
    const double boundary_distance = std::abs(lateral);
    if (boundary_distance <= 1.5) {
        if (lane.best.side == "left") {
            result.left_distance_m = boundary_distance;
            result.right_distance_m = 1.5 - boundary_distance;
        } else if (lane.best.side == "right") {
            result.right_distance_m = boundary_distance;
            result.left_distance_m = 1.5 - boundary_distance;
        }
    }
    std::vector<cv::Point2d> principal;
    cv::projectPoints(std::vector<cv::Point3d>{{0, normalized.front().y, 1}},
        cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), k, distortion, principal);
    result.principal = cv::Point(cvRound(principal.front().x), cvRound(crossing.y));
    return result;
}

} // namespace robot_vision
