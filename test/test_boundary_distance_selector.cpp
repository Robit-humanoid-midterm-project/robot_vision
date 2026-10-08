#include <gtest/gtest.h>
#include <limits>
#include "robot_vision/boundary_distance_selector.hpp"
#include "robot_vision/vision_message_builder.hpp"

namespace {
using namespace robot_vision;
FieldGeometryResult measured(double left, bool ground = false) {
    FieldGeometryResult result;
    result.status = CrossingStatus::measured;
    result.ground_based = ground;
    result.left_distance_m = left;
    result.right_distance_m = 1.4-left;
    return result;
}
TEST(BoundarySelector, FiltersEachSourceAndRejectsAnIsolatedSpike) {
    BoundaryDistanceSelector ground, crossing;
    for(double x : {0.50,0.95,0.52}) {
        ground.select(measured(x,true),{});
        crossing.select({},measured(x));
    }
    auto g=ground.select(measured(0.51,true),{});
    auto c=crossing.select({},measured(0.51));
    EXPECT_EQ(g.source,BoundarySource::ground);
    EXPECT_EQ(c.source,BoundarySource::crossing);
    EXPECT_DOUBLE_EQ(g.geometry.left_distance_m,0.52);
    EXPECT_DOUBLE_EQ(c.geometry.left_distance_m,0.52);
    EXPECT_NEAR(c.geometry.right_distance_m,0.88,1e-12);
}
TEST(BoundarySelector, GroundWinsAndCurrentGroundFailureUsesCrossingHistory) {
    BoundaryDistanceSelector selector;
    auto output=selector.select(measured(0.7,true),measured(0.5));
    EXPECT_EQ(output.source,BoundarySource::ground);
    EXPECT_DOUBLE_EQ(*output.ground_median_m,0.7);
    EXPECT_DOUBLE_EQ(*output.crossing_median_m,0.5);
    selector.select(measured(0.7,true),measured(0.95));
    output=selector.select({},measured(0.52));
    EXPECT_EQ(output.source,BoundarySource::crossing);
    EXPECT_FALSE(output.ground_median_m);
    EXPECT_DOUBLE_EQ(output.geometry.left_distance_m,0.52);
    const auto message=make_master_message(msg::ObstacleArray{},false,
        output.geometry.left_distance_m,output.geometry.right_distance_m);
    EXPECT_NEAR(message.left_x_1_dist+message.right_x_2_dist,1.4,1e-12);
}
TEST(BoundarySelector, CurrentFailureAndTimeoutDoNotReplayOldMeasurements) {
    BoundaryDistanceSelector selector;
    selector.select(measured(0.4,true),measured(0.4));
    selector.select(measured(0.4,true),measured(0.4));
    auto output=selector.select({},{});
    EXPECT_EQ(output.source,BoundarySource::none);
    EXPECT_EQ(output.geometry.left_distance_m,-1000);
    EXPECT_EQ(output.geometry.right_distance_m,-1000);
    output=selector.select({},measured(0.9));
    EXPECT_DOUBLE_EQ(output.geometry.left_distance_m,0.9);
    selector.reset();
    output=selector.select(measured(0.8,true),{});
    EXPECT_DOUBLE_EQ(output.geometry.left_distance_m,0.8);
}
TEST(BoundarySelector, ConvertsRightOnlyAndNeverClampsInvalidDistances) {
    BoundaryDistanceSelector selector;
    auto right=measured(-1000);right.right_distance_m=0.3;
    const auto output=selector.select({},right);
    EXPECT_NEAR(output.geometry.left_distance_m,1.1,1e-12);
    EXPECT_NEAR(output.geometry.right_distance_m,0.3,1e-12);
    auto bad=measured(2.0,true);
    bad.right_distance_m=std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(selector.select(bad,{}).source,BoundarySource::none);
    bad=measured(0.5,true);bad.status=CrossingStatus::waiting_reference;
    EXPECT_EQ(selector.select(bad,{}).source,BoundarySource::none);
}
} // namespace
