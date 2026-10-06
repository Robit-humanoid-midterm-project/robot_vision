#include <limits>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "robot_vision/frame_rate_limiter.hpp"
#include "robot_vision/vision_message_builder.hpp"
#include "robot_vision/vision_pipeline.hpp"
#include "robot_vision/vision_viewer.hpp"

namespace {
using namespace robot_vision;

DetectorConfig simple_camera()
{
    DetectorConfig config;
    config.camera_matrix = {500, 0, 320, 0, 500, 240, 0, 0, 1};
    config.distortion_coefficients = {0, 0, 0, 0, 0};
    return config;
}

Detection obstacle_at_row(double row_y, double depth)
{
    Detection obstacle;
    obstacle.position[2] = depth;
    obstacle.corners[2] = {300, row_y};
    obstacle.corners[3] = {340, row_y};
    return obstacle;
}

LaneResult left_lane()
{
    LaneResult lane;
    lane.best.valid = true;
    lane.best.side = "left";
    lane.best.top = {100, 100};
    lane.best.bottom = {200, 400};
    return lane;
}

TEST(FrameRateLimiter, KeepsRateWhenInputIsCloseToProcessingPeriod)
{
    FrameRateLimiter limiter(30);
    int accepted = 0;
    for (int index = 0; index < 300; ++index)
        accepted += limiter.should_process(FrameRateLimiter::Clock::time_point{} + std::chrono::milliseconds(33 * index));
    EXPECT_GE(accepted, 296);
    EXPECT_LE(accepted, 298);
    FrameRateLimiter half_rate(15);
    accepted = 0;
    for (int index = 0; index < 300; ++index)
        accepted += half_rate.should_process(FrameRateLimiter::Clock::time_point{} + std::chrono::milliseconds(33 * index));
    EXPECT_GE(accepted, 148);
    EXPECT_LE(accepted, 150);
}

TEST(FrameRateLimiter, DoesNotBurstAfterPauseAndResetsCleanly)
{
    FrameRateLimiter limiter(30);
    const auto start = FrameRateLimiter::Clock::time_point{};
    EXPECT_TRUE(limiter.should_process(start));
    EXPECT_FALSE(limiter.should_process(start + std::chrono::milliseconds(10)));
    EXPECT_TRUE(limiter.should_process(start + std::chrono::seconds(3)));
    EXPECT_FALSE(limiter.should_process(start + std::chrono::seconds(3)));
    limiter.reset();
    EXPECT_TRUE(limiter.should_process(start));
    EXPECT_THROW(FrameRateLimiter(0), std::invalid_argument);
    EXPECT_THROW(FrameRateLimiter(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(FieldGeometry, UsesNearestRowAndMedianDepthForBoundaryDistance)
{
    DetectResult obstacles;
    obstacles.detections = {obstacle_at_row(250, 10), obstacle_at_row(350, 1.5),
                           obstacle_at_row(350, 2.5)};
    const std::vector<RowLine> rows{{{0, 250}, {639, 250}}, {{0, 350}, {639, 350}}};
    const auto result = estimate_field_geometry(obstacles, left_lane(), rows, {640, 480}, simple_camera());
    ASSERT_EQ(result.status, CrossingStatus::measured);
    ASSERT_TRUE(result.crossing);
    EXPECT_NEAR(result.crossing->x, 183.333333333, 1e-6);
    EXPECT_NEAR(result.lateral_m, -0.5466666667, 1e-6);
    EXPECT_NEAR(result.left_distance_m, 0.5466666667, 1e-6);
    EXPECT_NEAR(result.left_distance_m + result.right_distance_m, 1.5, 1e-9);
}

TEST(FieldGeometry, KeepsUnknownDistancesWithoutDepthOrOutsideWidth)
{
    const std::vector<RowLine> rows{{{0, 350}, {639, 350}}};
    const auto missing = estimate_field_geometry({}, left_lane(), rows, {640, 480}, simple_camera());
    EXPECT_EQ(missing.status, CrossingStatus::no_row_depth);
    EXPECT_EQ(missing.left_distance_m, -1000);
    DetectResult far;
    far.detections = {obstacle_at_row(350, 10)};
    const auto outside = estimate_field_geometry(far, left_lane(), rows, {640, 480}, simple_camera());
    EXPECT_EQ(outside.status, CrossingStatus::measured);
    EXPECT_EQ(outside.left_distance_m, -1000);
    EXPECT_EQ(outside.right_distance_m, -1000);
}

TEST(VisionMessages, SelectsThreeNearestValidObstaclesAndUsesUnknownOnTimeout)
{
    msg::ObstacleArray array;
    array.header.stamp.sec = 12;
    for (double distance : {1.5, -0.1, 1.2, 0.4, 0.8, 0.2}) {
        msg::ObstacleDetection obstacle;
        obstacle.forward_distance_valid = true;
        obstacle.forward_distance_m = distance;
        array.detections.push_back(obstacle);
    }
    msg::ObstacleDetection invalid;
    invalid.forward_distance_valid = true;
    invalid.forward_distance_m = 0.1;
    invalid.position.x = std::numeric_limits<double>::quiet_NaN();
    array.detections.push_back(invalid);
    const auto normal = make_master_message(array, false, 0.5, 1.0);
    EXPECT_EQ(normal.timestamp, 12);
    EXPECT_EQ(normal.left_x_1_dist, 0.5);
    EXPECT_EQ(normal.right_x_2_dist, 1.0);
    EXPECT_EQ(normal.obstacle_1[0], 0.2);
    EXPECT_EQ(normal.obstacle_2[0], 0.4);
    EXPECT_EQ(normal.obstacle_3[0], 0.8);
    const auto timeout = make_master_message(array, true, 0.5, 1.0);
    EXPECT_EQ(timeout.left_x_1_dist, -1000);
    EXPECT_EQ(timeout.right_x_2_dist, -1000);
    EXPECT_EQ(timeout.obstacle_1, (std::array<double, 2>{-1000, -1000}));
    EXPECT_EQ(timeout.obstacle_2, timeout.obstacle_1);
    EXPECT_EQ(timeout.obstacle_3, timeout.obstacle_1);
}

TEST(VisionPipeline, SizeMismatchStillProducesCandidatesAndDebugImages)
{
    VisionPipeline pipeline(DetectorConfig{}, LaneConfig{}, 0.61);
    cv::Mat image(240, 320, CV_8UC3, cv::Scalar::all(0));
    cv::rectangle(image, {100, 80}, {180, 180}, cv::Scalar(255, 0, 0), cv::FILLED);
    auto result = pipeline.detect_obstacles(image);
    pipeline.complete(image, result);
    EXPECT_EQ(result.obstacles.status, "image_size_mismatch");
    ASSERT_EQ(result.obstacles.image_candidates.size(), 1u);
    EXPECT_TRUE(result.obstacles.detections.empty());
    EXPECT_TRUE(result.ground_projections.empty());
    VisionViewer viewer(DetectorConfig{}, false, false, false);
    EXPECT_EQ(viewer.annotate(image, result).size(), image.size());
    EXPECT_EQ(viewer.preprocess_view(result.obstacles).size(), cv::Size(640, 610));
}

TEST(VisionPipeline, ResetDropsPreviousRowSmoothingHistory)
{
    VisionPipeline pipeline(DetectorConfig{}, LaneConfig{}, 0.61);
    auto process = [&](int bottom) {
        cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));
        cv::rectangle(image, {200, bottom - 50}, {440, bottom}, cv::Scalar(255, 0, 0), cv::FILLED);
        auto result = pipeline.detect_obstacles(image);
        pipeline.complete(image, result);
        return result;
    };
    const auto first = process(300);
    ASSERT_FALSE(first.rows.empty());
    const auto smoothed = process(320);
    ASSERT_FALSE(smoothed.rows.empty());
    pipeline.reset_tracking();
    const auto fresh = process(320);
    ASSERT_FALSE(fresh.rows.empty());
    EXPECT_GT(fresh.rows.front().first.y, smoothed.rows.front().first.y);
}

} // namespace
