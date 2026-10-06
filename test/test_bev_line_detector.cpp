#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "robot_vision/bev_line_detector.hpp"
#include "robot_vision/bev_metric.hpp"

TEST(BevLineDetector, FindsBothBoundariesAndRejectsHorizontalMarking) {
  cv::Mat field(480, 640, CV_8UC3, cv::Scalar(55, 145, 65));
  cv::line(field, {115, 10}, {100, 465}, {245, 245, 245}, 8);
  cv::line(field, {515, 10}, {530, 465}, {245, 245, 245}, 8);
  cv::line(field, {180, 230}, {465, 230}, {245, 245, 245}, 8);
  const auto result = robot_vision::detect_bev_lines(field, {});
  ASSERT_TRUE(result.left.valid);
  ASSERT_TRUE(result.right.valid);
  EXPECT_NEAR(result.left.bottom.x, 100, 12);
  EXPECT_NEAR(result.right.bottom.x, 530, 12);
  EXPECT_GT(result.left.bottom.y - result.left.top.y, 350);
  EXPECT_GT(result.right.bottom.y - result.right.top.y, 350);
}

TEST(BevLineDetector, MissingRightBoundaryIsNotInvented) {
  cv::Mat field(480, 640, CV_8UC3, cv::Scalar(55, 145, 65));
  cv::line(field, {115, 10}, {100, 465}, {245, 245, 245}, 8);
  const auto result = robot_vision::detect_bev_lines(field, {});
  EXPECT_TRUE(result.left.valid);
  EXPECT_FALSE(result.right.valid);
}

TEST(BevLineDetector, RejectsGrayFloorBoundaryAndCurvedMarking) {
  cv::Mat field(480, 640, CV_8UC3, cv::Scalar(55, 145, 65));
  // A bright gray floor touches grass on the right, but is not a painted line.
  cv::rectangle(field, {510, 0}, {639, 479}, cv::Scalar(180, 180, 180), cv::FILLED);
  // A curved white marking on the left must not become a straight side line.
  cv::ellipse(field, {160, 350}, {90, 140}, 0, 90, 270,
              cv::Scalar(245, 245, 245), 6);
  cv::line(field, {335, 10}, {405, 465}, {245, 245, 245}, 8);
  const auto result = robot_vision::detect_bev_lines(field, {});
  EXPECT_FALSE(result.left.valid);
  ASSERT_TRUE(result.right.valid);
  EXPECT_NEAR(result.right.bottom.x, 405, 15);
}

TEST(BevLineDetector, TemporalFilterBridgesShortMissesButClearsLostLine) {
  robot_vision::BevLineTemporalFilter filter;
  robot_vision::BevLine observation{true, {110, 10}, {100, 460}, 0.9, 2.0};
  robot_vision::BevLineResult sample;
  sample.left = observation;
  EXPECT_FALSE(filter.update(sample).left.valid);  // One frame is not enough to confirm.
  EXPECT_TRUE(filter.update(sample).left.valid);
  sample.left = {};
  for (int i = 0; i < 4; ++i) {
    const auto held = filter.update(sample).left;
    EXPECT_TRUE(held.valid);
    EXPECT_TRUE(held.held);
  }
  EXPECT_FALSE(filter.update(sample).left.valid);
}

TEST(BevLineDetector, TemporalFilterRejectsOneFrameJump) {
  robot_vision::BevLineTemporalFilter filter;
  robot_vision::BevLineResult sample;
  sample.left = {true, {110, 10}, {100, 460}, 0.9, 2.0};
  filter.update(sample);
  filter.update(sample);
  sample.left = {true, {240, 10}, {230, 460}, 0.9, 2.0};
  auto stable = filter.update(sample).left;
  EXPECT_TRUE(stable.valid);
  EXPECT_TRUE(stable.held);
  EXPECT_LT(stable.bottom.x, 140);
  sample.left = {true, {112, 10}, {102, 460}, 0.9, 2.0};
  stable = filter.update(sample).left;
  EXPECT_TRUE(stable.valid);
  EXPECT_FALSE(stable.held);
  EXPECT_LT(stable.bottom.x, 140);
}

TEST(BevMetric, UsesMeasuredBevWidthForFixedHorizontalScale) {
  robot_vision::BevMetricScale scale({640, 480}, 1.86, 1.5, 3.75);
  const auto far_left = scale.at({0, 0}, 319.5);
  const auto near_right = scale.at({639, 479}, 319.5);
  EXPECT_DOUBLE_EQ(far_left.x_from_left_m, 0.0);
  EXPECT_NEAR(far_left.lateral_from_robot_m, -0.93, 1e-12);
  EXPECT_DOUBLE_EQ(far_left.forward_from_robot_m, 3.75);
  EXPECT_NEAR(near_right.x_from_left_m, 1.86, 1e-12);
  EXPECT_NEAR(near_right.lateral_from_robot_m, 0.93, 1e-12);
  EXPECT_NEAR(near_right.forward_from_robot_m, 1.5, 1e-12);
  EXPECT_NEAR(scale.at({319.5f, 239.5f}, 319.5).lateral_from_robot_m, 0.0, 1e-12);
  EXPECT_NEAR(scale.meters_per_horizontal_pixel(), 1.86 / 639.0, 1e-12);
  EXPECT_NEAR(scale.meters_per_vertical_pixel(), 2.25 / 479.0, 1e-12);
  EXPECT_THROW(scale.at({320, 239.5f}, -1), std::out_of_range);
}

TEST(BevMetric, LocatesRobotOriginFromOneLineAndKeepsItFixed) {
  robot_vision::BevRobotReference reference({640, 480}, 1.86, 1.4, 0.7);
  const double mpp = 1.86 / 639.0;
  const double left_x = 320.0 - 0.7 / mpp;
  for (int i = 0; i < 5; ++i) reference.observe(left_x + i * 0.2, std::nullopt);
  ASSERT_TRUE(reference.ready());
  EXPECT_NEAR(reference.robot_x(), 320.4, 1e-12);
  robot_vision::BevMetricScale scale({640, 480}, 1.86, 1.5, 3.75);
  const double before = scale.at({100, 239.5f}, reference.robot_x()).lateral_from_robot_m;
  reference.observe(left_x + 50, std::nullopt);
  EXPECT_NEAR(scale.at({100, 239.5f}, reference.robot_x()).lateral_from_robot_m,
              before, 1e-12);
  EXPECT_LT(before, 0.0);
  reference.reset();
  EXPECT_FALSE(reference.ready());
}

TEST(BevMetric, CanLocateRobotOriginFromRightLineAlone) {
  robot_vision::BevRobotReference reference({640, 480}, 1.86, 1.4, 0.7);
  const double right_x = 320.0 + 0.7 / (1.86 / 639.0);
  for (int i = 0; i < 5; ++i) reference.observe(std::nullopt, right_x);
  ASSERT_TRUE(reference.ready());
  EXPECT_NEAR(reference.robot_x(), 320.0, 1e-12);
}
