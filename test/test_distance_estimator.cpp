// 파일 역할: 합성 영상·지정 결과·선택적 참조 이미지로 예상 동작을 검사한다.
// 실제 로봇을 움직이는 코드가 아니며, 일반 카메라 실행 중에는 이 테스트가 동작하지 않는다.

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

// 실측 크기의 판을 지정한 3차원 위치에서 카메라 영상으로 투영해 검사용 윤곽을 만든다.
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

// 검사용 HSV 색상을 OpenCV BGR 색으로 변환한다. 채도만 바꾼 판의 분리 검사를 준비한다.
cv::Scalar hsv_to_bgr(int saturation) {
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(109, saturation, 210));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const auto pixel = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

// 합성 잔디 영상의 흰 직선에서 경계선 하나와 중앙 픽셀 간격을 얻는지 확인한다.
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

// 영상 위쪽의 선반 같은 선을 제외하고 아래쪽 바닥 경계선을 고르는지 확인한다.
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

// 반원 모양의 흰 선만 있는 경우 유효한 직선 경계선으로 인정하지 않는지 확인한다.
TEST(LaneLineCpp, RejectsSemicircleWithoutStraightBoundary) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(35, 90, 35));
  cv::ellipse(image, {320, 310}, {140, 130}, 0, 0, 180,
              cv::Scalar::all(255), 7);
  EXPECT_FALSE(robot_vision::estimate_lane_line(image, {}).best.valid);
}

// 주변 잔디색의 지지가 없는 흰 배경을 바닥 경계선으로 오인하지 않는지 확인한다.
TEST(LaneLineCpp, RejectsWhiteBackgroundWithoutGrass) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(90));
  cv::line(image, {300, 110}, {100, 479}, cv::Scalar::all(255), 7);
  EXPECT_FALSE(robot_vision::estimate_lane_line(image, {}).best.valid);
}

// 참조 이미지가 준비된 경우 실제 장면의 경계선 결과를 확인한다. 파일이 없으면 건너뛴다.
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

// 판 하단의 일부 선분이 보이면 그 기준선을 영상 폭으로 연장할 수 있는지 확인한다.
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

// 같은 높이의 두 판 밑변을 하나의 행으로 묶는지 확인한다.
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

// 여러 색상 경계가 있어도 출력 행을 최대 세 개로 제한하는지 확인한다.
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

// 같은 행 위치는 평균내고 현재 보이지 않는 행은 출력하지 않는지 확인한다.
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

// 직선거리와 카메라 높이로 바닥거리·전방거리·좌우 성분을 구분하는지 확인한다.
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

// 판 중심이 아닌 판 아래쪽 중심을 위치·직선거리의 기준점으로 사용하는지 확인한다.
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

// 알려진 위치에 투영한 빨강·파랑 판을 검출하고 예상 거리에 맞는지 확인한다.
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

// 가까워서 영상에 크게 보이는 온전한 정사각형 판도 검출하는지 확인한다.
TEST(DistanceEstimatorCpp, DetectsLargeFullyVisibleSquare) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  for (const cv::Scalar color : {cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0)}) {
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));
    const auto corners = projected_square(config, {0, 0, 0.55});
    for (const auto &corner : corners) {
      ASSERT_GT(corner.x, 3);
      ASSERT_LT(corner.x, 636);
      ASSERT_GT(corner.y, 3);
      ASSERT_LT(corner.y, 476);
    }
    cv::fillConvexPoly(image, corners, color);
    const auto result = estimator.detect(image);
    ASSERT_EQ(result.detections.size(), 1u);
    EXPECT_LT(result.detections[0].distance_m, 0.9);
  }
}

// 서로 붙은 파란 판들을 복구할 수 있는지 확인한다.
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

// 서로 붙은 빨간 판들을 복구할 수 있는지 확인한다.
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

// 깊이가 다른 빨간 판들을 하나의 큰 사각형으로 잘못 묶지 않는지 확인한다.
TEST(DistanceEstimatorCpp, RejectsBoxSpanningNearAndFarRedPlates) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(30, 105, 25));
  auto red_with_saturation = [](int saturation) {
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(0, saturation, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const auto pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
  };
  // The rear plate extends left of the front plate. A bounding box around
  // both would borrow its left side from the rear and its right side from the front.
  cv::rectangle(image, {194, 278}, {259, 342}, red_with_saturation(230), cv::FILLED);
  cv::rectangle(image, {222, 314}, {372, 453}, red_with_saturation(230), cv::FILLED);
  cv::rectangle(image, {280, 360}, {320, 390}, red_with_saturation(90), cv::FILLED);
  const auto result = estimator.detect(image);
  for (const auto &detection : result.detections) {
    if (detection.color != "red" || detection.distance_m >= 2.0) continue;
    double left = 640, right = 0;
    for (const auto &corner : detection.corners) {
      left = std::min(left, corner.x);
      right = std::max(right, corner.x);
    }
    EXPECT_FALSE(left < 210 && right > 360)
        << "Near detection combines the rear plate's left edge with the front plate's right edge";
  }
}

// 판이 가려져도 온전한 위쪽 변으로 거리를 추정하는지 확인한다.
TEST(DistanceEstimatorCpp, EstimatesRangeFromOccludedSquareTopEdge) {
  DetectorConfig config;
  DistanceEstimator estimator(config);
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));
  const auto square = projected_square(config, {0, 0, 2});
  cv::fillConvexPoly(image, square, cv::Scalar(0, 0, 255));
  // 아래 대부분을 가려 위쪽 전체 변과 양 끝 모서리만 남긴다.
  const auto box = cv::boundingRect(square);
  cv::rectangle(image, {box.x, box.y + 25}, {box.br().x, box.br().y},
                cv::Scalar::all(0), cv::FILLED);
  const auto result = estimator.detect(image);
  ASSERT_EQ(result.detections.size(), 1u);
  EXPECT_NEAR(result.detections[0].position[2], 2.0, 0.15);
  EXPECT_TRUE(result.detections[0].color_split_estimate);
}

// 불완전한 형태와 보정 해상도 불일치에서 부적절한 거리 검출을 제외하는지 확인한다.
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

// 참조 화면 이미지가 있을 때 판 검출 결과를 확인한다. 파일이 없으면 건너뛴다.
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

// 가까운 빨간 판의 참조 화면이 있을 때 검출을 확인한다. 파일이 없으면 건너뛴다.
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

// 화면에 잘리거나 사각형이 아닌 색상 영역도 후보로 유지하되 거리 검출과 구분하는지 확인한다.
TEST(ColorRegions, KeepsBorderClippedAndNonSquareRegionsWithoutMetricPose) {
  cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));
  cv::rectangle(image, {0, 100}, {70, 200}, cv::Scalar(0, 0, 255), cv::FILLED);
  cv::fillConvexPoly(image, std::vector<cv::Point>{{300, 100}, {400, 200}, {300, 200}},
                    cv::Scalar(255, 0, 0));
  const auto result = robot_vision::DistanceEstimator(robot_vision::DetectorConfig{}).detect(image);
  EXPECT_EQ(result.image_candidates.size(), 2u);
  EXPECT_TRUE(result.detections.empty());
  EXPECT_TRUE(result.image_candidates[0].touches_border);
}

// 다른 해상도에서도 색상 위치 후보를 남기고 작은 잡음은 제거하는지 확인한다.
TEST(ColorRegions, RetainsImagePositionsAtDifferentResolutionAndRejectsNoise) {
  cv::Mat image(720, 1280, CV_8UC3, cv::Scalar::all(0));
  cv::rectangle(image, {500, 200}, {600, 300}, cv::Scalar(0, 0, 255), cv::FILLED);
  cv::rectangle(image, {20, 20}, {25, 25}, cv::Scalar(255, 0, 0), cv::FILLED);
  const auto result = robot_vision::DistanceEstimator(robot_vision::DetectorConfig{}).detect(image);
  ASSERT_EQ(result.image_candidates.size(), 1u);
  EXPECT_NEAR(result.image_candidates[0].center.x, 550, 1);
  EXPECT_NEAR(result.image_candidates[0].center.y, 250, 1);
  EXPECT_EQ(result.status, "image_size_mismatch");
  EXPECT_TRUE(result.detections.empty());
}

// 같은 색 판이 하나의 윤곽이 되어도 뒤 판의 온전한 왼쪽 변을 찾아야 한다.
// 합쳐진 파란 윤곽에서 뒤 판의 온전한 변을 찾아 거리 후보를 복구하는지 확인한다.
TEST(VisibleEdges, RecoversRearPlateFromMergedBlueContour) {
  robot_vision::DetectorConfig config;
  config.distortion_coefficients = {0, 0, 0, 0, 0};
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));
  cv::rectangle(frame, {110, 150}, {190, 230}, cv::Scalar(255, 0, 0), cv::FILLED);
  cv::rectangle(frame, {150, 190}, {310, 350}, cv::Scalar(255, 0, 0), cv::FILLED);
  const auto result = robot_vision::DistanceEstimator(config).detect(frame);
  ASSERT_EQ(result.detections.size(), 2u);
  EXPECT_NEAR(result.detections[0].position[2], config.camera_matrix[0] * 0.4 / 160, 0.1);
  EXPECT_NEAR(result.detections[1].position[2], config.camera_matrix[4] * 0.4 / 80, 0.1);
}

// 판이 겹친 실제 참조 화면이 있을 때 복구 결과를 확인한다. 파일이 없으면 건너뛴다.
TEST(VisibleEdges, UserOverlapScreenshotWhenAvailable) {
  const char *path = std::getenv("ROBOT_VISION_OVERLAP_CAMERA_IMAGE");
  if (!path) GTEST_SKIP() << "No overlap camera image supplied";
  robot_vision::DetectorConfig config;
  config.red_lower_1 = {0, 60, 60};
  const cv::Mat frame = cv::imread(path);
  ASSERT_FALSE(frame.empty());
  const auto result = robot_vision::DistanceEstimator(config).detect(frame);
  int blue = 0, red = 0;
  for (const auto &d : result.detections) {
    blue += d.color == "blue";
    red += d.color == "red";
    std::cout << d.color << " z=" << d.position[2] << " range=" << d.distance_m << std::endl;
  }
  EXPECT_EQ(blue, 2);
  EXPECT_EQ(red, 0);
}
