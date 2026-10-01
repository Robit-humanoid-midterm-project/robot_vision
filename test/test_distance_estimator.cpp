#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "robot_vision/distance_estimator.hpp"
#include "robot_vision/row_line_estimator.hpp"
#include "robot_vision/lane_line_estimator.hpp"

namespace {

using robot_vision::DetectorConfig;
using robot_vision::DistanceEstimator;

std::vector<cv::Point> projected_square(const DetectorConfig &config, const cv::Vec3d &xyz) {
  const double s = config.obstacle_size_m / 2.0;
  std::vector<cv::Point3d> object{{-s, s, 0}, {s, s, 0}, {s, -s, 0}, {-s, -s, 0}};
  cv::Mat camera_matrix(3, 3, CV_64F, const_cast<double *>(config.camera_matrix.data()));
  cv::Mat distortion(1, static_cast<int>(config.distortion_coefficients.size()), CV_64F,
                     const_cast<double *>(config.distortion_coefficients.data()));
  std::vector<cv::Point2d> image;
  cv::projectPoints(object, cv::Vec3d(CV_PI, 0, 0), xyz,
                    camera_matrix, distortion, image);
  std::vector<cv::Point> pixels;
  for (const auto &point : image) pixels.emplace_back(cvRound(point.x), cvRound(point.y));
  return pixels;
}

cv::Scalar hsv_to_bgr(int saturation) {
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(109, saturation, 210));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const auto pixel = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

TEST(LaneLineCpp, SelectsOneStraightBoundaryAndPixelGap) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(35, 90, 35));
  cv::line(image, {300, 110}, {100, 479}, cv::Scalar::all(255), 7);
  cv::rectangle(image, {0, 270}, {639, 330}, cv::Scalar(35, 90, 35), cv::FILLED);
  cv::ellipse(image, {420, 350}, {100, 90}, 0, 0, 180, cv::Scalar::all(255), 7);
  const auto result = robot_vision::estimate_lane_line(image, {});
  ASSERT_TRUE(result.best.valid);
  EXPECT_EQ(result.best.side, "left");
  EXPECT_NEAR(result.best.bottom.x, 100, 20);
  EXPECT_GT(result.best.pixel_separation_px, 100);
}

TEST(LaneLineCpp, IgnoresUpperShelfLineAndKeepsLowerBoundary) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(35, 90, 35));
  cv::line(image, {100, 90}, {350, 290}, cv::Scalar::all(255), 8);
  EXPECT_FALSE(robot_vision::estimate_lane_line(image, {}).best.valid);
  cv::line(image, {185, 340}, {15, 479}, cv::Scalar::all(255), 8);
  const auto result = robot_vision::estimate_lane_line(image, {});
  ASSERT_TRUE(result.best.valid);
  EXPECT_EQ(result.best.side, "left");
  EXPECT_LT(result.best.bottom.x, 75);
}

TEST(LaneLineCpp, RejectsSemicircleWithoutStraightBoundary) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(35, 90, 35));
  cv::ellipse(image, {320, 310}, {140, 130}, 0, 0, 180,
              cv::Scalar::all(255), 7);
  EXPECT_FALSE(robot_vision::estimate_lane_line(image, {}).best.valid);
}

TEST(LaneLineCpp, RejectsWhiteBackgroundWithoutGrass) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(90));
  cv::line(image, {300, 110}, {100, 479}, cv::Scalar::all(255), 7);
  EXPECT_FALSE(robot_vision::estimate_lane_line(image, {}).best.valid);
}

TEST(LaneLineCpp, ReferenceImageWhenAvailable) {
  const char *path = std::getenv("ROBOT_VISION_SINGLE_LANE_IMAGE");
  if (!path) GTEST_SKIP() << "No reference image supplied";
  const cv::Mat screenshot = cv::imread(path);
  ASSERT_GE(screenshot.cols, 640);
  ASSERT_GE(screenshot.rows, 480);
  const auto result = robot_vision::estimate_lane_line(
      screenshot(cv::Rect(0, 0, 640, 480)), {});
  const char *expected = std::getenv("ROBOT_VISION_SINGLE_LANE_EXPECT_SIDE");
  if (expected && std::string(expected) == "none") {
    EXPECT_FALSE(result.best.valid);
  } else {
    EXPECT_TRUE(result.best.valid);
    if (result.best.valid) {
      EXPECT_TRUE(result.best.side == "left" || result.best.side == "right");
      EXPECT_GE(result.best.grass_support, 0.60);
      if (expected) EXPECT_EQ(result.best.side, expected) << "x=" << result.best.bottom.x;
    }
  }
}

TEST(DistanceEstimatorCpp, ExtendsExposedLowerEdgeThroughPartialOcclusion) {
  std::array<robot_vision::ColorDebug, 2> colors;
  colors[0].cleaned_mask = cv::Mat::zeros(480, 640, CV_8UC1);
  cv::rectangle(colors[0].cleaned_mask, {160, 150}, {420, 330}, 255, cv::FILLED);
  cv::rectangle(colors[0].cleaned_mask, {270, 285}, {330, 360}, 0, cv::FILLED);
  const auto rows = robot_vision::estimate_row_lines(colors, {}, {640, 480});
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_FALSE(rows[0].observed.empty());
  EXPECT_NEAR(rows[0].first.y, 330, 5);
  EXPECT_NEAR(rows[0].last.y, 330, 5);
  EXPECT_EQ(rows[0].first.x, 0);
  EXPECT_EQ(rows[0].last.x, 639);
}

TEST(DistanceEstimatorCpp, MergesTwoVisibleBasesOnSameRow) {
  std::array<robot_vision::ColorDebug, 2> colors;
  colors[0].cleaned_mask = cv::Mat::zeros(480, 640, CV_8UC1);
  colors[1].cleaned_mask = cv::Mat::zeros(480, 640, CV_8UC1);
  cv::rectangle(colors[0].cleaned_mask, {80, 170}, {200, 330}, 255, cv::FILLED);
  cv::rectangle(colors[1].cleaned_mask, {370, 170}, {490, 330}, 255, cv::FILLED);
  const auto rows = robot_vision::estimate_row_lines(colors, {}, {640, 480});
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_GE(rows[0].observed.size(), 2u);
  EXPECT_NEAR(rows[0].first.y, 330, 5);
}

TEST(DistanceEstimatorCpp, LimitsRandomColorRowsToThree) {
  std::array<robot_vision::ColorDebug, 2> colors;
  for (auto &color : colors)
    color.cleaned_mask = cv::Mat::zeros(480, 640, CV_8UC1);
  const std::array<cv::Rect, 4> bases{{{40, 40, 120, 55}, {280, 140, 115, 55},
                                       {100, 245, 110, 55}, {430, 350, 120, 55}}};
  for (size_t i = 0; i < bases.size(); ++i)
    cv::rectangle(colors[i % 2].cleaned_mask, bases[i], 255, cv::FILLED);
  const auto rows = robot_vision::estimate_row_lines(colors, {}, {640, 480});
  ASSERT_EQ(rows.size(), 3u);
  for (const auto &row : rows) EXPECT_FALSE(row.observed.empty());
}

TEST(DistanceEstimatorCpp, AveragesMatchingRowButDoesNotDrawMissingRow) {
  robot_vision::RowLineTracker tracker(3, 35.0, 0.18, 3);
  auto make_row = [](int y) {
    return robot_vision::RowLine{{0, y}, {639, y}, {}, 1};
  };
  auto first = tracker.smooth({make_row(200)}, {640, 480});
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first[0].first.y, 200);
  auto second = tracker.smooth({make_row(212)}, {640, 480});
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(second[0].first.y, 206);
  EXPECT_TRUE(tracker.smooth({}, {640, 480}).empty());
  auto third = tracker.smooth({make_row(218)}, {640, 480});
  ASSERT_EQ(third.size(), 1u);
  EXPECT_EQ(third[0].first.y, 210);
  auto unrelated = tracker.smooth({make_row(350)}, {640, 480});
  ASSERT_EQ(unrelated.size(), 1u);
  EXPECT_EQ(unrelated[0].first.y, 350);
}

TEST(DistanceEstimatorCpp, GroundProjectionSeparatesForwardAndLateral) {
  robot_vision::Detection detection;
  detection.position = {0.4, 0.75, 2.0};
  detection.distance_m = cv::norm(detection.position);
  const auto projected = robot_vision::project_to_ground(detection, 0.75);
  ASSERT_TRUE(projected.has_value());
  EXPECT_NEAR(projected->forward_m, 2.0, 1e-9);
  EXPECT_NEAR(projected->lateral_m, 0.4, 1e-9);
  EXPECT_NEAR(projected->radial_m, std::hypot(2.0, 0.4), 1e-9);
  EXPECT_FALSE(robot_vision::project_to_ground(detection, 3.0).has_value());
}

TEST(DistanceEstimatorCpp, BottomEdgeMidpointIsRangeTarget) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  const auto pixels = projected_square(config, {0, 0, 2});
  std::array<cv::Point2d, 4> corners{};
  for (size_t i = 0; i < corners.size(); ++i) corners[i] = pixels[i];
  const auto detection = estimator.square_pose(corners);
  ASSERT_TRUE(detection.has_value());
  EXPECT_NEAR(detection->position[0], 0.0, 0.02);
  EXPECT_NEAR(detection->position[1], 0.2, 0.02);
  EXPECT_NEAR(detection->position[2], 2.0, 0.02);
  EXPECT_NEAR(detection->distance_m, std::hypot(2.0, 0.2), 0.02);
}

TEST(DistanceEstimatorCpp, DetectsBothColorsAndKnownRange) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));
  cv::fillConvexPoly(image, projected_square(config, {-0.4, 0, 2}), cv::Scalar(0, 0, 255));
  cv::fillConvexPoly(image, projected_square(config, {0.4, 0, 2}), cv::Scalar(255, 0, 0));
  const auto result = estimator.detect(image);
  ASSERT_EQ(result.detections.size(), 2u);
  for (const auto &color : result.colors) {
    EXPECT_FALSE(color.raw_mask.empty());
    EXPECT_FALSE(color.cleaned_mask.empty());
    EXPECT_GT(color.raw_pixels, 0);
    EXPECT_GT(color.cleaned_pixels, 0);
    EXPECT_EQ(color.detections, 1);
  }
  EXPECT_EQ(result.status, "unverified_calibration");
  for (const auto &detection : result.detections) {
    EXPECT_TRUE(detection.color == "red" || detection.color == "blue");
    EXPECT_NEAR(detection.distance_m, std::sqrt(4.16), 0.1);
    EXPECT_FALSE(detection.color_split_estimate);
  }
}

TEST(DistanceEstimatorCpp, SeparatesTouchingBlueSquares) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(30, 105, 25));
  cv::fillConvexPoly(image, projected_square(config, {-0.55, -0.1, 2.5}), hsv_to_bgr(210));
  cv::fillConvexPoly(image, projected_square(config, {-0.1, 0.1, 1.2}), hsv_to_bgr(230));
  const auto result = estimator.detect(image);
  auto front = std::find_if(result.detections.begin(), result.detections.end(),
                            [](const auto &item) { return item.color == "blue" && item.color_split_estimate; });
  ASSERT_NE(front, result.detections.end());
  EXPECT_NEAR(front->distance_m, std::sqrt(1.46), 0.1);
}

TEST(DistanceEstimatorCpp, SeparatesTouchingRedSquares) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(30, 105, 25));
  cv::fillConvexPoly(image, projected_square(config, {-0.55, -0.1, 2.5}),
                     cv::Scalar(35, 50, 180));
  cv::fillConvexPoly(image, projected_square(config, {-0.1, 0.1, 1.2}),
                     cv::Scalar(0, 0, 230));
  const auto result = estimator.detect(image);
  auto front = std::find_if(result.detections.begin(), result.detections.end(),
                            [](const auto &item) { return item.color == "red" && item.color_split_estimate; });
  ASSERT_NE(front, result.detections.end());
  EXPECT_NEAR(front->distance_m, std::sqrt(1.46), 0.2);
}

TEST(DistanceEstimatorCpp, RejectsIncompleteShapesAndWrongResolution) {
  DistanceEstimator estimator(DetectorConfig{});
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));
  cv::rectangle(image, {0, 100}, {100, 200}, cv::Scalar(0, 0, 255), cv::FILLED);
  cv::circle(image, {300, 220}, 50, cv::Scalar(255, 0, 0), cv::FILLED);
  cv::fillConvexPoly(image, std::vector<cv::Point>{{450, 120}, {550, 220}, {450, 220}},
                     cv::Scalar(0, 0, 255));
  EXPECT_TRUE(estimator.detect(image).detections.empty());
  EXPECT_EQ(estimator.detect(cv::Mat(720, 1280, CV_8UC3, cv::Scalar::all(0))).status,
            "image_size_mismatch");
}

TEST(DistanceEstimatorCpp, ReferenceScreenshotWhenAvailable) {
  const char *path = std::getenv("ROBOT_VISION_REFERENCE_IMAGE");
  if (path == nullptr) GTEST_SKIP() << "No local screenshot supplied";
  cv::Mat screenshot = cv::imread(path);
  ASSERT_FALSE(screenshot.empty());
  ASSERT_GE(screenshot.rows, 537);
  ASSERT_GE(screenshot.cols, 631);
  cv::Mat frame = screenshot(cv::Rect(0, 57, 631, 480)).clone();
  cv::copyMakeBorder(frame, frame, 0, 0, 0, 9, cv::BORDER_CONSTANT);
  const auto result = DistanceEstimator(DetectorConfig{}).detect(frame);
  const auto blue = std::find_if(result.detections.begin(), result.detections.end(),
                                 [](const auto &item) { return item.color == "blue" && item.color_split_estimate; });
  const auto red = std::find_if(result.detections.begin(), result.detections.end(),
                                [](const auto &item) { return item.color == "red"; });
  EXPECT_NE(blue, result.detections.end());
  EXPECT_NE(red, result.detections.end());
  if (blue != result.detections.end()) EXPECT_LT(blue->distance_m, 2.5);
}

TEST(DistanceEstimatorCpp, NearRedReferenceScreenshotWhenAvailable) {
  const char *path = std::getenv("ROBOT_VISION_NEAR_RED_REFERENCE_IMAGE");
  if (path == nullptr) GTEST_SKIP() << "No local screenshot supplied";
  cv::Mat screenshot = cv::imread(path);
  ASSERT_FALSE(screenshot.empty());
  ASSERT_GE(screenshot.rows, 480);
  ASSERT_GE(screenshot.cols, 640);
  const auto result = DistanceEstimator(DetectorConfig{}).detect(
      screenshot(cv::Rect(0, 0, 640, 480)));
  const auto near_red = std::find_if(result.detections.begin(), result.detections.end(),
                                     [](const auto &item) {
                                       return item.color == "red" && item.distance_m < 1.5;
                                     });
  EXPECT_NE(near_red, result.detections.end());
}

}  // namespace
