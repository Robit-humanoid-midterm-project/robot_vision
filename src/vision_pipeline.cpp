#include "robot_vision/vision_pipeline.hpp"

#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>

namespace robot_vision {

VisionPipeline::VisionPipeline(DetectorConfig detector, LaneConfig lane, double camera_height_m,
                               RowTrackerConfig rows)
    : detector_config_(std::move(detector)), lane_config_(std::move(lane)),
      camera_height_m_(camera_height_m), estimator_(detector_config_),
      row_tracker_(rows.history_frames, rows.match_y_px, rows.match_slope, rows.max_missing_frames)
{
}

VisionFrameResult VisionPipeline::detect_obstacles(const cv::Mat &frame) const
{
    VisionFrameResult result;
    result.obstacles = estimator_.detect(frame);
    return result;
}

void VisionPipeline::complete(const cv::Mat &frame, VisionFrameResult &result)
{
    try {
        cv::Mat color_exclusion;
        cv::cvtColor(result.obstacles.mask_preview, color_exclusion, cv::COLOR_BGR2GRAY);
        cv::dilate(color_exclusion, color_exclusion, cv::getStructuringElement(cv::MORPH_RECT, {7, 7}));
        result.lane = estimate_lane_line(frame, lane_config_, color_exclusion);
    } catch (const std::exception &error) {
        result.lane_error = error.what();
        result.lane.mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    }
    result.rows = row_tracker_.smooth(
        estimate_row_lines(result.obstacles.colors, result.obstacles.detections, frame.size()), frame.size());
    result.geometry = estimate_field_geometry(
        result.obstacles, result.lane, result.rows, frame.size(), detector_config_);
    result.ground_projections.clear();
    for (const auto &detection : result.obstacles.detections)
        result.ground_projections.push_back(project_to_ground(detection, camera_height_m_));
}

void VisionPipeline::reset_tracking()
{
    row_tracker_.reset();
}

} // namespace robot_vision
