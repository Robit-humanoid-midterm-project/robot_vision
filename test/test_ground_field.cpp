#include <gtest/gtest.h>
#include "robot_vision/vision_pipeline.hpp"

namespace {
using namespace robot_vision;
DetectorConfig camera() {
    DetectorConfig c;
    c.camera_matrix = {500,0,320,0,500,240,0,0,1};
    c.distortion_coefficients = {0,0,0,0,0};
    return c;
}
GroundFieldConfig calibration() {
    GroundFieldConfig c;
    c.source_points = {0,0,639,0,639,479,0,479};
    return c;
}
LaneLine line(double intercept, const std::string &side, double slope = 0) {
    LaneLine l;
    l.valid = true; l.side = side; l.confidence = 0.95;
    const auto pixel_x = [&](double y) { return float((intercept+slope*y)*639/1.86); };
    l.top = {pixel_x(3.5), float((3.75-3.5)*479/2.25)};
    l.bottom = {pixel_x(1.7), float((3.75-1.7)*479/2.25)};
    return l;
}
LaneResult lanes(double left = 0.23, double right = 1.63) {
    LaneResult l; l.left = line(left,"left"); l.right = line(right,"right"); l.best = l.left;
    return l;
}
void lock(GroundFieldEstimator &e) {
    for (int i=0;i<4;++i)
        EXPECT_EQ(e.estimate(lanes(),{640,480}).status,CrossingStatus::waiting_reference);
    EXPECT_EQ(e.estimate(lanes(),{640,480}).status,CrossingStatus::measured);
}
TEST(GroundField, MeasuresWithoutObstaclesAndUpdatesAfterRobotMoves) {
    VisionPipeline pipeline(camera(), LaneConfig{}, 0.61, {}, calibration());
    cv::Mat image(480,640,CV_8UC3,cv::Scalar::all(0));
    auto result = pipeline.detect_obstacles(image);
    ASSERT_TRUE(result.obstacles.detections.empty());
    for(int i=0;i<5;++i) pipeline.complete(image,result,lanes());
    ASSERT_EQ(result.geometry.status,CrossingStatus::measured);
    EXPECT_NEAR(result.geometry.left_distance_m,0.7,1e-5);
    pipeline.complete(image,result,lanes(0.03,1.43));
    EXPECT_NEAR(result.geometry.left_distance_m,0.9,1e-5);
    EXPECT_NEAR(result.geometry.right_distance_m,0.5,1e-5);
    pipeline.complete(image,result);
    EXPECT_EQ(result.geometry.left_distance_m,-1000);
}
TEST(GroundField, OneBoundaryAndTimeoutKeepOriginalReference) {
    GroundFieldEstimator e(camera(),calibration()); lock(e);
    e.clear_pending_reference();
    auto l=lanes(0.03,1.43); l.right={};
    auto r=e.estimate(l,{640,480});
    EXPECT_NEAR(r.left_distance_m,0.9,1e-5);
    EXPECT_NEAR(r.right_distance_m,0.5,1e-5);
    l.left={}; l.best={}; l.right=line(1.43,"right");
    r=e.estimate(l,{640,480});
    EXPECT_NEAR(r.left_distance_m,0.9,1e-5);
}
TEST(GroundField, RejectsConflictAndBadInputWithoutStaleOutput) {
    GroundFieldEstimator e(camera(),calibration()); lock(e);
    auto r=e.estimate(lanes(0.23,1.23),{640,480});
    EXPECT_EQ(r.left_distance_m,-1000);
    EXPECT_EQ(e.estimate(lanes(),{320,240}).left_distance_m,-1000);
    auto l=lanes(); l.left.top.x=std::numeric_limits<float>::quiet_NaN(); l.right={}; l.best={};
    EXPECT_EQ(e.estimate(l,{640,480}).left_distance_m,-1000);
    auto c=calibration(); c.source_points={0,0,1,1,2,2,3,3};
    EXPECT_THROW(GroundFieldEstimator(camera(),c),std::invalid_argument);
}
TEST(GroundField, ReferenceMustBeConsecutiveAndStable) {
    GroundFieldEstimator e(camera(),calibration());
    for(int i=0;i<3;++i) e.estimate(lanes(),{640,480});
    e.estimate({}, {640,480});
    EXPECT_FALSE(e.ready()); lock(e);
}
TEST(GroundField, HandlesParallelAngledBoundariesInMetricCoordinates) {
    GroundFieldEstimator e(camera(),calibration());
    LaneResult l; const double slope=0.05;
    l.left=line(0.15,"left",slope);
    l.right=line(0.15+1.4*std::hypot(1.,slope),"right",slope);
    FieldGeometryResult r;
    for(int i=0;i<5;++i) r=e.estimate(l,{640,480});
    EXPECT_NEAR(r.left_distance_m,0.7,1e-4);
    EXPECT_NEAR(r.right_distance_m,0.7,1e-4);
}
} // namespace
