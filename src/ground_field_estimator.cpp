#include "robot_vision/ground_field_estimator.hpp"

#include <cmath>
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace robot_vision {
GroundFieldEstimator::GroundFieldEstimator(const DetectorConfig &camera, GroundFieldConfig config)
    : config_(std::move(config)), size_(camera.calibration_width, camera.calibration_height)
{
    if (config_.source_points.size() != 8 || size_.width < 2 || size_.height < 2 ||
        !std::isfinite(config_.width_m) || config_.width_m <= 0 ||
        !std::isfinite(config_.near_m) || config_.near_m < 0 ||
        !std::isfinite(config_.far_m) || config_.far_m <= config_.near_m ||
        !std::isfinite(config_.field_width_m) || config_.field_width_m <= 0 ||
        !std::isfinite(config_.start_from_left_m) || config_.start_from_left_m < 0 ||
        config_.start_from_left_m > config_.field_width_m || config_.reference_frames < 1 ||
        camera.camera_matrix.size() != 9 || camera.distortion_coefficients.empty())
        throw std::invalid_argument("Invalid ground field calibration");
    std::vector<cv::Point2f> source;
    for (int i = 0; i < 4; ++i) {
        const double x = config_.source_points[2*i], y = config_.source_points[2*i+1];
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || x >= size_.width || y < 0 || y >= size_.height)
            throw std::invalid_argument("Ground source point outside calibration image");
        source.emplace_back(x, y);
    }
    // Require the documented far-left, far-right, near-right, near-left winding.
    if (!cv::isContourConvex(source) || cv::contourArea(source, true) < 25)
        throw std::invalid_argument("Ground source points must form a clockwise image rectangle");
    const std::vector<cv::Point2f> target{{0, float(config_.far_m)},
        {float(config_.width_m), float(config_.far_m)},
        {float(config_.width_m), float(config_.near_m)}, {0, float(config_.near_m)}};
    homography_ = cv::getPerspectiveTransform(source, target);
    if (!cv::checkRange(homography_) || std::abs(cv::determinant(homography_)) < 1e-12)
        throw std::invalid_argument("Degenerate ground homography");
    k_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(camera.camera_matrix.data())).clone();
    distortion_ = cv::Mat(1, camera.distortion_coefficients.size(), CV_64F,
                         const_cast<double *>(camera.distortion_coefficients.data())).clone();
    cv::initUndistortRectifyMap(k_, distortion_, cv::Mat(), k_, size_, CV_32FC1, map_x_, map_y_);
}

std::optional<GroundFieldEstimator::Line> GroundFieldEstimator::project(const LaneLine &line) const
{
    if (!line.valid || !std::isfinite(line.top.x) || !std::isfinite(line.top.y) ||
        !std::isfinite(line.bottom.x) || !std::isfinite(line.bottom.y) ||
        line.bottom.y <= line.top.y) return std::nullopt;
    std::vector<cv::Point2f> raw;
    for (int i = 0; i <= 32; ++i) {
        const auto p = line.top + (line.bottom - line.top) * (i / 32.0f);
        if (p.x >= 0 && p.x < size_.width && p.y >= 0 && p.y < size_.height)
            raw.push_back(p);
    }
    if (raw.size() < 3) return std::nullopt;
    std::vector<cv::Point2f> corrected, ground, usable;
    // YOLO coordinates are raw; source_points are already undistorted.
    cv::undistortPoints(raw, corrected, k_, distortion_, cv::noArray(), k_);
    cv::perspectiveTransform(corrected, ground, homography_);
    double min_y = config_.far_m, max_y = config_.near_m;
    for (const auto &p : ground) {
        // Restrict fitting to observed points inside the calibrated floor rectangle.
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.x > config_.width_m ||
            p.y < config_.near_m || p.y > config_.far_m) continue;
        usable.push_back(p);
        min_y = std::min(min_y, double(p.y)); max_y = std::max(max_y, double(p.y));
    }
    if (usable.size() < 3 || max_y - min_y < 0.25) return std::nullopt;
    cv::Vec4f fit;
    cv::fitLine(usable, fit, cv::DIST_HUBER, 0, 0.001, 0.001);
    if (std::abs(fit[1]) < 1e-4) return std::nullopt;
    const double slope = fit[0] / fit[1], intercept = fit[2] - slope * fit[3];
    const double norm = std::hypot(1.0, slope);
    // Side lines should stay near the forward direction; reject bad extrapolations.
    if (!std::isfinite(intercept) || std::abs(slope) > 1.0) return std::nullopt;
    for (const auto &p : usable)
        if (std::abs(p.x - slope*p.y - intercept) / norm > 0.04) return std::nullopt;
    return Line{slope, intercept, norm};
}

cv::Mat GroundFieldEstimator::preview(const cv::Mat &raw, const LaneResult &lanes) const
{
    if (raw.empty() || raw.size() != size_ || raw.type() != CV_8UC3) return {};
    // Equal metres per pixel on both axes so angles and proportions stay readable.
    const int height = 480;
    const double scale = (height-1) / (config_.far_m-config_.near_m);
    const int width = std::max(2, cvRound(config_.width_m*scale)+1);
    const cv::Mat to_pixel = (cv::Mat_<double>(3,3) <<
        scale,0,0, 0,-scale,config_.far_m*scale, 0,0,1);
    const cv::Mat display_h = to_pixel*homography_;
    cv::Mat corrected, bird;
    cv::remap(raw, corrected, map_x_, map_y_, cv::INTER_LINEAR);
    cv::warpPerspective(corrected, bird, display_h, {width,height});
    for (const auto *line : {&lanes.left, &lanes.right}) {
        if (!line->valid) continue;
        std::vector<cv::Point2f> points, undistorted, projected;
        for (int i=0;i<=32;++i) points.push_back(line->top+(line->bottom-line->top)*(i/32.0f));
        cv::undistortPoints(points, undistorted, k_, distortion_, cv::noArray(), k_);
        cv::perspectiveTransform(undistorted, projected, display_h);
        const cv::Scalar color = line->side == "left" ? cv::Scalar(255,180,0) : cv::Scalar(0,220,255);
        for (size_t i=1;i<projected.size();++i) {
            const auto a=projected[i-1], b=projected[i];
            if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(b.x) || !std::isfinite(b.y) ||
                a.x<0 || a.x>=width || a.y<0 || a.y>=height ||
                b.x<0 || b.x>=width || b.y<0 || b.y>=height) continue;
            cv::line(bird,a,b,color,3,cv::LINE_AA);
        }
    }
    for (double y=config_.near_m+0.5; y<config_.far_m; y+=0.5) {
        const int row=cvRound((config_.far_m-y)*scale);
        cv::line(bird,{0,row},{width-1,row},{120,120,120},1);
    }
    return bird;
}

void GroundFieldEstimator::clear_pending_reference()
{
    if (!ready_) { samples_ = 0; robot_x_ = 0; }
}

FieldGeometryResult GroundFieldEstimator::estimate(const LaneResult &lanes, cv::Size size)
{
    FieldGeometryResult result;
    result.ground_based = true;
    result.status = CrossingStatus::invalid_ground_line;
    if (size != size_) { clear_pending_reference(); return result; }
    const auto left = project(lanes.left.valid ? lanes.left :
        lanes.best.side == "left" ? lanes.best : LaneLine{});
    const auto right = project(lanes.right.valid ? lanes.right :
        lanes.best.side == "right" ? lanes.best : LaneLine{});
    if (!left && !right) { clear_pending_reference(); return result; }
    if (!ready_) {
        result.status = CrossingStatus::waiting_reference;
        const double l = left ? left->intercept + config_.start_from_left_m * left->norm : 0;
        const double r = right ? right->intercept -
            (config_.field_width_m - config_.start_from_left_m) * right->norm : 0;
        if (left && right && (std::abs(l-r) > 0.10 || std::abs(left->slope-right->slope) > 0.12)) {
            clear_pending_reference(); return result;
        }
        const double candidate = left && right ? (l+r)/2 : left ? l : r;
        if (samples_ > 0 && std::abs(candidate-robot_x_) > 0.05) clear_pending_reference();
        robot_x_ = (robot_x_*samples_ + candidate) / (samples_+1);
        if (++samples_ < config_.reference_frames) return result;
        ready_ = true;
    }
    const double dl = left ? (robot_x_ - left->intercept) / left->norm : -1000;
    const double dr = right ? (right->intercept - robot_x_) / right->norm : -1000;
    const auto valid = [&](double d) { return std::isfinite(d) && d >= 0 && d <= config_.field_width_m; };
    // Conflicting boundaries are not averaged into a plausible but false coordinate.
    if ((left && !valid(dl)) || (right && !valid(dr)) ||
        (left && right && (std::abs(dl+dr-config_.field_width_m) > 0.12 ||
                           std::abs(left->slope-right->slope) > 0.12))) return result;
    result.left_distance_m = left ? dl : config_.field_width_m-dr;
    result.right_distance_m = right ? dr : config_.field_width_m-dl;
    result.lateral_m = left ? -dl : dr;
    result.status = CrossingStatus::measured;
    return result;
}
} // namespace robot_vision
