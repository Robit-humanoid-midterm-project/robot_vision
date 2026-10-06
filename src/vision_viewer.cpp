#include "robot_vision/vision_viewer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace robot_vision {
namespace {

std::string two_decimals(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

std::string signed_two_decimals(double value)
{
    std::ostringstream stream;
    stream << std::showpos << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

// 판별로 네 값을 묶어 표시한다. 바닥/전방 값은 기존 바닥 투영 결과를 사용한다.
// OpenCV 기본 글꼴은 한글을 지원하지 않아 화면에는 영문 항목명을 사용한다.
void draw_distance_label(cv::Mat &canvas, const Detection &detection, const std::optional<GroundProjection> &ground,
                         std::vector<cv::Rect> &occupied)
{
    if (canvas.empty() || !std::isfinite(detection.distance_m))
        return;
    cv::Point2d anchor = detection.corners[0];
    for (const auto &corner : detection.corners)
    {
        anchor.x = std::min(anchor.x, corner.x);
        anchor.y = std::min(anchor.y, corner.y);
    }
    if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y))
        return;
    const std::vector<std::string> lines{detection.color + (detection.color_split_estimate ? " (est.)" : ""),
                                         "Range: " + two_decimals(detection.distance_m) + " m",
                                         "Ground: " + (ground ? two_decimals(ground->radial_m) + " m" : "N/A"),
                                         "Forward: " + (ground ? two_decimals(ground->forward_m) + " m" : "N/A"),
                                         "Lateral: " + signed_two_decimals(detection.position[0]) + " m"};
    constexpr double scale = 0.43;
    constexpr int line_height = 18, padding = 5;
    int width = 0, baseline = 0;
    for (const auto &line : lines)
        width = std::max(width, cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline).width);
    width = std::min(width + 2 * padding, canvas.cols);
    const int height = std::min(int(lines.size()) * line_height + 2 * padding, canvas.rows);
    const int min_y = std::min(70, canvas.rows - height);
    cv::Rect box(std::clamp(cvRound(anchor.x), 0, canvas.cols - width),
                 std::clamp(cvRound(anchor.y) - height - 5, min_y, canvas.rows - height), width, height);
    // 붙어 있는 판들의 정보창이 서로 덮이지 않도록 빈 자리를 찾는다.
    const auto overlaps = [&](const cv::Rect &candidate) {
        return std::any_of(occupied.begin(), occupied.end(),
                           [&](const cv::Rect &other) { return (candidate & other).area() > 0; });
    };
    if (overlaps(box))
    {
        bool placed = false;
        for (int y = min_y; y <= canvas.rows - height && !placed; y += line_height)
        {
            for (int x = 0; x <= canvas.cols - width; x += 20)
            {
                const cv::Rect candidate(x, y, width, height);
                if (!overlaps(candidate))
                {
                    box = candidate;
                    placed = true;
                    break;
                }
            }
        }
    }
    occupied.push_back(box);
    const cv::Scalar color = detection.color == "red" ? cv::Scalar(80, 110, 255) : cv::Scalar(255, 210, 80);
    // 정보창을 옮겨도 어느 판의 값인지 연결선으로 확인할 수 있다.
    cv::line(canvas, {box.x + box.width / 2, box.y + box.height / 2}, {cvRound(anchor.x), cvRound(anchor.y)}, color, 1,
             cv::LINE_AA);
    cv::rectangle(canvas, box, cv::Scalar(20, 20, 20), cv::FILLED);
    for (size_t i = 0; i < lines.size(); ++i)
        cv::putText(canvas, lines[i], {box.x + padding, box.y + padding + int(i + 1) * line_height - 4},
                    cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

std::string crossing_note(const FieldGeometryResult &geometry, const std::string &side)
{
    switch (geometry.status) {
    case CrossingStatus::no_lane_or_row: return "CROSS: no lane/row";
    case CrossingStatus::no_intersection: return "CROSS: no valid intersection";
    case CrossingStatus::no_row_depth: return "CROSS " + side + ": no row depth";
    case CrossingStatus::measured: break;
    }
    std::string note = "CROSS " + side + " | lateral ~" +
        two_decimals(std::abs(geometry.lateral_m)) + " m | x " +
        signed_two_decimals(geometry.lateral_m) + " m";
    if (std::abs(geometry.lateral_m) <= 1.5)
        note += " | L " + two_decimals(geometry.left_distance_m) +
                " R " + two_decimals(geometry.right_distance_m);
    else
        note += " | outside width";
    return note;
}

} // namespace

VisionViewer::VisionViewer(DetectorConfig config, bool enabled, bool show_preprocess, bool fullscreen)
    : debug_config_(std::move(config)), viewer_(enabled), show_preprocess_(show_preprocess),
      viewer_fullscreen_(fullscreen)
{
}

VisionViewer::~VisionViewer()
{
    if (window_initialized_)
        cv::destroyWindow(window_name_);
    if (preprocess_initialized_)
        cv::destroyWindow(preprocess_window_name_);
}

cv::Mat VisionViewer::annotate(const cv::Mat &frame, const VisionFrameResult &frame_result) const
{
    const auto &result = frame_result.obstacles;
    const auto &lane = frame_result.lane;
    const auto &row_lines = frame_result.rows;
    cv::Mat canvas = frame.clone();
    if (lane.best.valid) {
        const auto &line = lane.best;
        const cv::Scalar color = line.side == "left" ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
        const cv::Point2f delta = line.bottom - line.top;
        const double length = cv::norm(delta);
        if (length > 0)
        {
            for (double offset = 0; offset < length; offset += 18.0)
            {
                const double end = std::min(offset + 10.0, length);
                cv::line(canvas, line.top + delta * (offset / length), line.top + delta * (end / length), color, 2,
                         cv::LINE_AA);
            }
        }
        for (const auto &segment : line.observed_segments)
            cv::line(canvas, {segment[0], segment[1]}, {segment[2], segment[3]}, color, 3, cv::LINE_AA);
        const int ref_y = cvRound(line.reference_y_px);
        cv::circle(canvas, {cvRound(line.line_x_at_reference_px), ref_y}, 6, color, cv::FILLED, cv::LINE_AA);
    }
    for (const auto &row : row_lines)
    {
        const cv::Point2d span = row.last - row.first;
        const double length = cv::norm(span);
        if (length > 0)
        {
            for (double offset = 0; offset < length; offset += 20.0)
            {
                const double end = std::min(offset + 12.0, length);
                cv::line(canvas,
                         row.first +
                             cv::Point(cvRound(span.x * offset / length), cvRound(span.y * offset / length)),
                         row.first + cv::Point(cvRound(span.x * end / length), cvRound(span.y * end / length)),
                         cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
            }
        }
        for (const auto &segment : row.observed)
        {
            cv::line(canvas, segment.first, segment.last, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }
    }
    const auto &geometry = frame_result.geometry;
    if (geometry.crossing) {
        cv::drawMarker(canvas, *geometry.crossing, cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        if (geometry.principal)
            cv::line(canvas, *geometry.principal, *geometry.crossing,
                     cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    // Draw observed color contours independently of metric pose acceptance.
    for (const auto &candidate : result.image_candidates)
    {
        const cv::Scalar color = candidate.color == "red" ? cv::Scalar(0, 80, 255) : cv::Scalar(255, 180, 0);
        const std::vector<std::vector<cv::Point>> contours{candidate.contour};
        cv::drawContours(canvas, contours, -1, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::drawContours(canvas, contours, -1, color, 2, cv::LINE_AA);
    }
    std::vector<cv::Rect> distance_labels;
    for (size_t index = 0; index < result.detections.size(); ++index)
        draw_distance_label(canvas, result.detections[index],
                            frame_result.ground_projections.at(index), distance_labels);
    const std::string crossing_note = robot_vision::crossing_note(geometry, lane.best.side);
    cv::rectangle(canvas, {0, 0}, {canvas.cols, 46}, cv::Scalar(20, 20, 20), cv::FILLED);
    cv::putText(canvas, "COLOR CONTOURS", {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 220, 255), 1,
                cv::LINE_AA);
    cv::putText(canvas,
                "Visible regions: " + std::to_string(result.image_candidates.size()) +
                    " | row lines: " + std::to_string(row_lines.size()),
                {8, 37}, cv::FONT_HERSHEY_SIMPLEX, 0.39, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    const std::string lane_note = lane.best.valid ? ("LANE " + lane.best.side) : "LANE not detected";
    cv::putText(canvas, lane_note + " | GREEN/YELLOW = base/row", {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, lane_note + " | GREEN/YELLOW = base/row", {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(canvas, crossing_note, {8, 81}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, crossing_note, {8, 81}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return canvas;
}

void VisionViewer::show_dashboard(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                    const cv::Mat &annotated)
{
    if (!viewer_)
        return;
    if (!window_initialized_)
    {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, 1200, 626);
        cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
                              viewer_fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        window_initialized_ = true;
    }
    // Keep the annotated image large so contours stay readable.
    // The header sits outside the image and does not cover its status banner.
    cv::Mat dashboard(626, 1200, CV_8UC3, cv::Scalar(18, 18, 18));
    cv::Mat large_annotated, small_raw, combined_mask;
    cv::resize(annotated, large_annotated, {800, 600}, 0, 0, cv::INTER_CUBIC);
    cv::resize(raw, small_raw, {400, 300}, 0, 0, cv::INTER_AREA);
    if (!obstacle_mask.empty() && obstacle_mask.type() == CV_8UC3 && obstacle_mask.size() == raw.size())
    {
        combined_mask = obstacle_mask.clone();
    }
    else
    {
        combined_mask = cv::Mat::zeros(raw.size(), CV_8UC3);
    }
    if (!white_mask.empty() && white_mask.type() == CV_8UC1 && white_mask.size() == raw.size())
    {
        combined_mask.setTo(cv::Scalar::all(255), white_mask);
    }
    cv::resize(combined_mask, combined_mask, {400, 300}, 0, 0, cv::INTER_NEAREST);
    large_annotated.copyTo(dashboard(cv::Rect(0, 26, 800, 600)));
    combined_mask.copyTo(dashboard(cv::Rect(800, 26, 400, 300)));
    small_raw.copyTo(dashboard(cv::Rect(800, 326, 400, 300)));
    auto label = [&](const std::string &name, int x, int y) {
        cv::putText(dashboard, name, {x + 6, y + 19}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
    };
    label(processing_fps_ > 0 ? "ANNOTATED RESULT | " + two_decimals(processing_fps_) + " FPS" :
                              "ANNOTATED RESULT", 0, 0);
    label("OPENCV MASK", 800, 0);
    cv::rectangle(dashboard, {800, 326}, {990, 352}, cv::Scalar(20, 20, 20), cv::FILLED);
    label("RAW CAMERA", 800, 326);
    cv::imshow(window_name_, dashboard);
    const int key = cv::waitKey(1) & 0xff;
    if (key == 'f' || key == 'F' || key == 27)
    {
        viewer_fullscreen_ = key == 27 ? false : !viewer_fullscreen_;
        cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
                              viewer_fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        if (!viewer_fullscreen_)
            cv::resizeWindow(window_name_, 1200, 626);
    }
}

cv::Mat VisionViewer::preprocess_view(const DetectResult &result) const
{
    cv::Mat canvas(610, 640, CV_8UC3, cv::Scalar(22, 22, 22));
    auto put = [&](const std::string &label, int x, int y, double scale = 0.46) {
        cv::putText(canvas, label, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(235, 235, 235), 1,
                    cv::LINE_AA);
    };
    const auto &r = debug_config_;
    put("RED H " + std::to_string(r.red_lower_1[0]) + "-" + std::to_string(r.red_upper_1[0]) + ", " +
            std::to_string(r.red_lower_2[0]) + "-" + std::to_string(r.red_upper_2[0]) +
            "  S>= " + std::to_string(std::min(r.red_lower_1[1], r.red_lower_2[1])) +
            "  V>= " + std::to_string(std::min(r.red_lower_1[2], r.red_lower_2[2])),
        8, 19);
    put("BLUE H " + std::to_string(r.blue_lower[0]) + "-" + std::to_string(r.blue_upper[0]) +
            "  S>= " + std::to_string(r.blue_lower[1]) + "  V>= " + std::to_string(r.blue_lower[2]),
        8, 39);
    put("White = selected pixels | RAW = HSV | CLEAN = open + close", 8, 62, 0.42);
    for (int i = 0; i < 2; ++i)
    {
        const auto &color = result.colors[i];
        for (int stage = 0; stage < 2; ++stage)
        {
            const int x = stage * 320, y = 80 + i * 230;
            const cv::Mat &mask = stage == 0 ? color.raw_mask : color.cleaned_mask;
            put(std::string(i == 0 ? "RED " : "BLUE ") + (stage == 0 ? "RAW " : "CLEAN ") +
                    std::to_string(stage == 0 ? color.raw_pixels : color.cleaned_pixels) + " px",
                x + 8, y + 16, 0.42);
            if (!mask.empty())
            {
                cv::Mat scaled, rgb;
                cv::resize(mask, scaled, {320, 210}, 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(scaled, rgb, cv::COLOR_GRAY2BGR);
                rgb.copyTo(canvas(cv::Rect(x, y + 20, 320, 210)));
            }
        }
    }
    put("RED components " + std::to_string(result.colors[0].candidate_contours) + " detected " +
            std::to_string(result.colors[0].detections) + " | BLUE components " +
            std::to_string(result.colors[1].candidate_contours) + " detected " +
            std::to_string(result.colors[1].detections),
        8, 561, 0.42);
    int y = 584;
    for (const auto &detection : result.detections)
    {
        if (y > 604)
            break;
        put((detection.color == "red" ? "R" : "B") + std::string(" fit ") + two_decimals(detection.shape_fit) +
                " reproj " + two_decimals(detection.reprojection_error_px) + " px" +
                (detection.color_split_estimate ? " approx" : ""),
            8, y, 0.42);
        y += 18;
    }
    return canvas;
}

cv::Mat VisionViewer::no_image_view(const std::string &image_topic) const
{
    cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(0));
    cv::putText(canvas, "NO CAMERA IMAGE", {30, 200}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2,
                cv::LINE_AA);
    cv::putText(canvas, image_topic, {15, 245}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);
    return canvas;
}

cv::Mat VisionViewer::no_image_preprocess() const
{
    cv::Mat preprocess(610, 640, CV_8UC3, cv::Scalar::all(0));
    cv::putText(preprocess, "NO CAMERA IMAGE", {40, 300}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2,
                cv::LINE_AA);
    return preprocess;
}

void VisionViewer::show(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                        const cv::Mat &annotated, const cv::Mat &preprocess)
{
    if (viewer_ && show_preprocess_) {
        cv::imshow(preprocess_window_name_, preprocess);
        preprocess_initialized_ = true;
    }
    show_dashboard(raw, white_mask, obstacle_mask, annotated);
}

} // namespace robot_vision
