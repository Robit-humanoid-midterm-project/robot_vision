// 파일 역할: 합성 영상·지정 결과·선택적 참조 이미지로 예상 동작을 검사한다.
// 실제 로봇을 움직이는 코드가 아니며, 일반 카메라 실행 중에는 이 테스트가 동작하지 않는다.

#include <limits>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "robot_vision/ground_field_estimator.hpp"

#include "robot_vision/frame_rate_limiter.hpp"
#include "robot_vision/vision_message_builder.hpp"
#include "robot_vision/vision_pipeline.hpp"
#include "robot_vision/vision_viewer.hpp"

namespace {
using namespace robot_vision;

// 왜곡이 없는 간단한 카메라 행렬을 만들어 예상 거리 값을 손으로 계산할 수 있게 한다.
DetectorConfig simple_camera()
{
    DetectorConfig config;
    config.camera_matrix = {500, 0, 320, 0, 500, 240, 0, 0, 1};
    config.distortion_coefficients = {0, 0, 0, 0, 0};
    return config;
}

// 지정된 행 높이와 깊이를 가진 가상의 장애물을 만들어 관계 계산을 검사한다.
Detection obstacle_at_row(double row_y, double depth)
{
    Detection obstacle;
    obstacle.position[2] = depth;
    obstacle.corners[2] = {300, row_y};
    obstacle.corners[3] = {340, row_y};
    return obstacle;
}

// 고정된 두 영상 점을 잇는 왼쪽 경계선 검출 결과를 만든다.
LaneResult left_lane()
{
    LaneResult lane;
    lane.best.valid = true;
    lane.best.side = "left";
    lane.best.top = {100, 100};
    lane.best.bottom = {200, 400};
    return lane;
}

// 33ms 입력 간격에서 30/15FPS 제한이 불필요하게 절반으로 떨어지지 않는지 확인한다.
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

// 긴 입력 공백 뒤 연속 따라잡기 처리가 없고 초기화·잘못된 FPS 검사가 동작하는지 확인한다.
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

// 가장 앞쪽 행과 그 행에 속한 깊이 중앙값으로 좌우 경계 거리를 계산하는지 확인한다.
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
    // 현재 경기장 폭의 사용자 근사값 1.4m를 유지한다.
    EXPECT_NEAR(result.left_distance_m + result.right_distance_m, 1.4, 1e-9);
}

// 행 깊이가 없거나 경기장 폭 가정을 벗어나면 경계 거리를 미측정으로 남기는지 확인한다.
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

TEST(FieldGeometry, MeasuresBothModelBoundariesIndependently)
{
    auto lanes = left_lane();
    lanes.left = lanes.best;
    lanes.left.confidence = 0.9;
    lanes.left.observed_y_min = 100;
    lanes.left.observed_y_max = 400;
    lanes.right = lanes.left;
    lanes.right.side = "right";
    lanes.right.top = {450, 100};
    lanes.right.bottom = {450, 400};
    DetectResult obstacles;
    obstacles.detections = {obstacle_at_row(350, 2)};
    const std::vector<RowLine> rows{{{0, 350}, {639, 350}}};
    const auto result = estimate_field_geometry(obstacles, lanes, rows, {640, 480}, simple_camera());
    EXPECT_NEAR(result.left_distance_m, 0.5466666667, 1e-6);
    EXPECT_NEAR(result.right_distance_m, 0.52, 1e-6);
    ASSERT_TRUE(result.left_crossing);
    ASSERT_TRUE(result.right_crossing);
    // A detected right stripe without a measurable crossing uses the left measurement.
    lanes.right.observed_y_max = 300;
    const auto hidden = estimate_field_geometry(obstacles, lanes, rows, {640, 480}, simple_camera());
    EXPECT_GT(hidden.left_distance_m, 0);
    EXPECT_NEAR(hidden.right_distance_m, 1.4 - hidden.left_distance_m, 1e-9);
    const auto message = make_master_message(msg::ObstacleArray{}, false, hidden.left_distance_m, hidden.right_distance_m);
    EXPECT_NEAR(message.right_x_2_dist, 1.4 - hidden.left_distance_m, 1e-9);
    // The same fallback applies with left/right reversed.
    lanes.right.observed_y_max = 400;
    lanes.left.observed_y_max = 300;
    const auto left_hidden = estimate_field_geometry(obstacles, lanes, rows, {640, 480}, simple_camera());
    EXPECT_NEAR(left_hidden.right_distance_m, 0.52, 1e-6);
    EXPECT_NEAR(left_hidden.left_distance_m, 0.88, 1e-6);
    lanes.right.observed_y_max = 300;
    const auto both_hidden = estimate_field_geometry(obstacles, lanes, rows, {640, 480}, simple_camera());
    EXPECT_EQ(both_hidden.left_distance_m, -1000);
    EXPECT_EQ(both_hidden.right_distance_m, -1000);
    const auto mismatch = estimate_field_geometry(obstacles, lanes, rows, {320, 240}, simple_camera());
    EXPECT_EQ(mismatch.left_distance_m, -1000);
    EXPECT_EQ(mismatch.right_distance_m, -1000);
}

TEST(VisionPipeline, UsesInjectedModelLanesAndClearsMissingDetections)
{
    VisionPipeline pipeline(DetectorConfig{}, LaneConfig{}, 0.61);
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));
    auto lanes = left_lane();
    lanes.left = lanes.best;
    lanes.left.confidence = 0.91;
    auto result = pipeline.detect_obstacles(frame);
    pipeline.complete(frame, result, lanes);
    EXPECT_TRUE(result.lane.left.valid);
    const std_msgs::msg::Header header;
    EXPECT_FLOAT_EQ(make_lane_message(header, lanes).confidence, 0);
    lanes.best = lanes.left;
    EXPECT_FLOAT_EQ(make_lane_message(header, lanes).confidence, 0.91);
    pipeline.complete(frame, result);
    EXPECT_FALSE(result.lane.left.valid);
    EXPECT_FALSE(result.lane.right.valid);
    EXPECT_FALSE(result.lane.best.valid);
    EXPECT_EQ(cv::countNonZero(result.lane.mask), 0);
}

// 전방 1.5m 미만의 유효한 장애물 중 가까운 세 개를 선택하고 끊김이면 -1000을 보내는지 확인한다.
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

// 보정 해상도가 달라도 색상 후보·디버그 화면은 유지하고 실제 거리 검출만 생략하는지 확인한다.
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

// 끊김 초기화 후 새 행 위치에 이전 평활화 이력이 섞이지 않는지 확인한다.
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

TEST(VisionPipeline, RunsCrossingFallbackWhileGroundReferenceIsLocking) {
    GroundFieldConfig ground;
    ground.source_points={0,0,639,0,639,479,0,479};
    ground.x_min_m=0;
    VisionPipeline pipeline(simple_camera(),LaneConfig{},0.61,{},ground);
    cv::Mat image(480,640,CV_8UC3,cv::Scalar::all(0));
    cv::rectangle(image,{260,200},{380,320},cv::Scalar(255,0,0),cv::FILLED);
    auto lanes=left_lane();
    lanes.left=lanes.best;
    lanes.left.confidence=0.95;
    lanes.left.observed_y_min=100;
    lanes.left.observed_y_max=400;
    auto result=pipeline.detect_obstacles(image);
    ASSERT_FALSE(result.obstacles.detections.empty());
    pipeline.complete(image,result,lanes);
    EXPECT_EQ(result.ground_geometry.status,CrossingStatus::waiting_reference);
    EXPECT_EQ(result.crossing_geometry.status,CrossingStatus::measured);
    EXPECT_EQ(result.boundary_source,BoundarySource::crossing);
    EXPECT_GE(result.geometry.left_distance_m,0);
    EXPECT_NEAR(result.geometry.left_distance_m+result.geometry.right_distance_m,1.4,1e-9);
    EXPECT_TRUE(result.crossing_geometry.crossing);
    // Missing current lines must not retain either filtered distance.
    pipeline.complete(image,result,{});
    EXPECT_EQ(result.boundary_source,BoundarySource::none);
    EXPECT_EQ(result.geometry.left_distance_m,-1000);
    EXPECT_EQ(result.geometry.right_distance_m,-1000);
}
} // namespace

TEST(BottomRatio, IncludesClippedColoursAndUsesExactThreeRegions) {
    robot_vision::DetectorConfig config;
    robot_vision::VisionPipeline pipeline(config,robot_vision::LaneConfig{},0.61);
    cv::Mat frame(480,640,CV_8UC3,cv::Scalar::all(0));
    frame(cv::Rect(0,475,213,5)).setTo(cv::Scalar(0,0,255));
    frame(cv::Rect(213,475,107,5)).setTo(cv::Scalar(255,0,0));
    const auto result=pipeline.detect_obstacles(frame);
    EXPECT_DOUBLE_EQ(result.obstacle_ratio[0],1.0);
    EXPECT_DOUBLE_EQ(result.obstacle_ratio[1],0.5);
    EXPECT_DOUBLE_EQ(result.obstacle_ratio[2],0.0);
    ASSERT_TRUE(result.obstacles.detections.empty());
    const auto message=robot_vision::make_master_message(robot_vision::msg::ObstacleArray{},false,
        -1000,-1000,result.obstacle_ratio);
    EXPECT_EQ(message.obstacle_ratio,result.obstacle_ratio);
    const auto timeout=robot_vision::make_master_message(robot_vision::msg::ObstacleArray{},true,
        -1000,-1000,result.obstacle_ratio);
    EXPECT_DOUBLE_EQ(timeout.frame_drop,1.0);
    for(double value:timeout.obstacle_ratio) EXPECT_DOUBLE_EQ(value,-1000.0);
}
