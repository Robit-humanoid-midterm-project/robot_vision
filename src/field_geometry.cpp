// 파일 역할: 검출한 차선과 장애물 행의 관계로 좌우 경계 거리를 추정한다.
// 기존 가정은 경기장 폭 1.5m, 경기장 안의 카메라, 경기장 길이 방향으로의 정렬이다.

#include "robot_vision/field_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>

namespace robot_vision {

// 장애물·차선·행과 보정값을 받아 교차점, 좌우 위치와 경계 거리를 반환한다.
// 교차점이나 같은 행의 깊이를 못 구하면 거리를 미측정(-1000)으로 유지한다.
FieldGeometryResult estimate_field_geometry(
    const DetectResult &obstacles, const LaneResult &lane,
    const std::vector<RowLine> &rows, const cv::Size &size, const DetectorConfig &config)
{
    FieldGeometryResult result;
    if (!lane.best.valid || rows.empty())
        return result;
    result.status = CrossingStatus::no_intersection;
    const double center_x = (size.width - 1) * 0.5;
    // 행의 직선을 특정 x 위치의 y 값으로 계산한다. 영상 중앙에서 더 아래인 행을 앞쪽 행으로 본다.
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
    // 차선과 앞쪽 행을 두 직선으로 표현한다. 외적으로 교차점을 구하며 평행한 직선은 제외한다.
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
    // 교차한 행 가까이에 밑변이 있는 판들의 카메라 전방 깊이(z)를 모은다.
    std::vector<double> depths;
    for (const auto &detection : obstacles.detections) {
        const cv::Point2d base = (detection.corners[2] + detection.corners[3]) * 0.5;
        if (std::abs(base.y - row_y(*front, base.x)) <= 25.0 &&
            std::isfinite(detection.position[2]) && detection.position[2] > 0)
            depths.push_back(detection.position[2]);
    }
    if (depths.empty())
        return result;
    // 깊이의 중앙값을 사용해 한 장애물의 오차가 경계 거리 전체에 미치는 영향을 줄인다.
    std::sort(depths.begin(), depths.end());
    const size_t middle = depths.size() / 2;
    const double depth = depths.size() % 2 ? depths[middle] :
        (depths[middle - 1] + depths[middle]) * 0.5;
    const cv::Mat k(3, 3, CV_64F, const_cast<double *>(config.camera_matrix.data()));
    const cv::Mat distortion(1, config.distortion_coefficients.size(), CV_64F,
                             const_cast<double *>(config.distortion_coefficients.data()));
    std::vector<cv::Point2d> normalized;
    // 교차점 픽셀을 렌즈 왜곡이 보정된 정규화 좌표로 바꾸고, 행의 깊이를 곱해 좌우 거리(m)를 구한다.
    cv::undistortPoints(std::vector<cv::Point2d>{crossing}, normalized, k, distortion);
    const double lateral = normalized.front().x * depth;
    if (!std::isfinite(lateral))
        return result;
    result.status = CrossingStatus::measured;
    result.lateral_m = lateral;
    const double boundary_distance = std::abs(lateral);
    // 경계까지 거리가 경기장 폭 안에 있을 때만 반대쪽 거리도 1.4m에서 빼서 계산한다.
    if (boundary_distance <= 1.4) {
        if (lane.best.side == "left") {
            result.left_distance_m = boundary_distance;
            result.right_distance_m = 1.4 - boundary_distance;
        } else if (lane.best.side == "right") {
            result.right_distance_m = boundary_distance;
            result.left_distance_m = 1.4 - boundary_distance;
        }
    }
    std::vector<cv::Point2d> principal;
    // 카메라 중앙 방향의 영상 위치를 다시 구한다. 교차점까지 연결선을 그릴 때 사용하는 표시용 점이다.
    cv::projectPoints(std::vector<cv::Point3d>{{0, normalized.front().y, 1}},
        cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), k, distortion, principal);
    result.principal = cv::Point(cvRound(principal.front().x), cvRound(crossing.y));
    return result;
}

} // namespace robot_vision
