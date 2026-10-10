// 파일 역할: 계산 결과를 사람이 확인할 수 있는 영상으로 그리고 OpenCV 창에 표시한다.
// 왼쪽은 검출 결과, 오른쪽 위는 마스크, 오른쪽 아래는 원본 영상이다.

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

// 거리·품질 값을 소수 둘째 자리까지 표시할 문자열로 바꾼다.
std::string two_decimals(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

// 좌우 부호를 명확히 보여 주도록 + 또는 -가 붙은 소수 둘째 자리 문자열을 만든다.
std::string signed_two_decimals(double value)
{
    std::ostringstream stream;
    stream << std::showpos << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

// 판별로 네 값을 묶어 표시한다. 바닥/전방 값은 기존 바닥 투영 결과를 사용한다.
// OpenCV 기본 글꼴은 한글을 지원하지 않아 화면에는 영문 항목명을 사용한다.
// 판의 거리 라벨을 영상에 그린다. 겹치는 기존 라벨을 피하고 대상 판과 연결선을 만든다.
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
    // Range는 직선거리, Ground는 바닥거리, Forward는 전방 성분, Lateral은 좌우 위치를 표시한다.
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

// 경계 거리 계산의 성공·실패 상태를 화면 설명으로 바꾼다. 계산을 새로 수행하지 않는다.
std::string crossing_note(const FieldGeometryResult &geometry, const std::string &side)
{
    if (geometry.ground_based && geometry.status == CrossingStatus::measured)
        return "FIELD X FROM LEFT: " + two_decimals(geometry.left_distance_m) +
            " m | RIGHT: " + two_decimals(geometry.right_distance_m) + " m";
    switch (geometry.status) {
    case CrossingStatus::no_lane_or_row: return "CROSS: no lane/row";
    case CrossingStatus::no_intersection: return "CROSS: no valid intersection";
    case CrossingStatus::no_row_depth: return "CROSS " + side + ": no row depth";
    case CrossingStatus::waiting_reference: return "FIELD: keep start pose, locking reference";
    case CrossingStatus::invalid_ground_line: return "FIELD: no valid ground line";
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

// 전처리 표시용 검출 설정과 창 사용·전체 화면 여부를 보관한다.
VisionViewer::VisionViewer(DetectorConfig config, bool enabled, bool show_preprocess, bool fullscreen)
    : debug_config_(std::move(config)), viewer_(enabled), show_preprocess_(show_preprocess),
      viewer_fullscreen_(fullscreen)
{
}

// 이 객체가 실제로 만든 창만 닫는다.
VisionViewer::~VisionViewer()
{
    if (window_initialized_)
        cv::destroyWindow(window_name_);
    if (preprocess_initialized_)
        cv::destroyWindow(preprocess_window_name_);
}

// 원본을 복사해 차선·행·교차점·판 윤곽·거리와 상태 문구를 겹쳐 그린다.
// 검출 입력 영상 자체는 수정하지 않고 표시용 복사본을 반환한다.
cv::Mat VisionViewer::annotate(const cv::Mat &frame, const VisionFrameResult &frame_result) const
{
    const auto &result = frame_result.obstacles;
    const auto &lane = frame_result.lane;
    const auto &row_lines = frame_result.rows;
    // 표시 도형이 검출 결과나 원본 영상에 섞이지 않도록 독립된 복사본을 만든다.
    cv::Mat canvas = frame.clone();
    for (const auto *candidate : {&lane.left, &lane.right}) {
        if (!candidate->valid) continue;
        const auto &line = *candidate;
        const cv::Scalar color = line.side == "left" ? cv::Scalar(255, 180, 0) : cv::Scalar(0, 220, 255);
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
    // 평활화한 행은 점선으로, 실제 관측한 밑변 조각은 실선으로 구분해 그린다.
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
    const auto &crossings = frame_result.crossing_geometry;
    if (crossings.crossing) {
        cv::drawMarker(canvas, *crossings.crossing, cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
        if (crossings.principal)
            cv::line(canvas, *crossings.principal, *crossings.crossing,
                     cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    // Draw observed color contours independently of metric pose acceptance.
    // 거리 계산을 통과하지 못한 색상 후보도 윤곽은 표시한다. 거리 라벨은 확정된 검출에만 붙는다.
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
    const char *source = frame_result.boundary_source == BoundarySource::ground ? "GROUND" :
                         frame_result.boundary_source == BoundarySource::crossing ? "CROSS" : "NONE";
    const std::string crossing_note = frame_result.boundary_source == BoundarySource::none ?
        "OUT: N/A | " + robot_vision::crossing_note(geometry, lane.best.side) :
        std::string("OUT ") + source + ": L " + two_decimals(geometry.left_distance_m) +
            " R " + two_decimals(geometry.right_distance_m) + " m";
    const std::string ground_note = frame_result.ground_left_median_m ?
        two_decimals(*frame_result.ground_left_median_m) + " m" :
        frame_result.ground_geometry.status == CrossingStatus::waiting_reference ? "LOCKING" : "N/A";
    const std::string comparison_note = "GROUND med3: " + ground_note + " | CROSS med3: " +
        (frame_result.crossing_left_median_m ? two_decimals(*frame_result.crossing_left_median_m) + " m" : "N/A");
    for (const auto &point : {crossings.left_crossing, crossings.right_crossing})
        if (point) cv::drawMarker(canvas, *point, cv::Scalar(255, 255, 255),
                                  cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    cv::rectangle(canvas, {0, 0}, {canvas.cols, 46}, cv::Scalar(20, 20, 20), cv::FILLED);
    cv::putText(canvas, "COLOR CONTOURS", {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 220, 255), 1,
                cv::LINE_AA);
    cv::putText(canvas,
                "Visible regions: " + std::to_string(result.image_candidates.size()) +
                    " | row lines: " + std::to_string(row_lines.size()),
                {8, 37}, cv::FONT_HERSHEY_SIMPLEX, 0.39, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    const std::string lane_note = "YOLO L " +
        (lane.left.valid ? two_decimals(lane.left.confidence) : "N/A") + " R " +
        (lane.right.valid ? two_decimals(lane.right.confidence) : "N/A");
    cv::putText(canvas, lane_note + " | GREEN/YELLOW = base/row", {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, lane_note + " | GREEN/YELLOW = base/row", {8, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.39,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(canvas, crossing_note, {8, 81}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, crossing_note, {8, 81}, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(canvas, comparison_note, {8, 100}, cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, comparison_note, {8, 100}, cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    // Show the exact ground fit used by the estimator, not a second display-only fit.
    const auto &ground = frame_result.ground_geometry;
    const auto coefficients = [](const std::optional<cv::Point2d> &ab) {
        if (!ab) return std::string("N/A");
        std::ostringstream out;
        out << std::fixed << std::setprecision(4) << "a=" << ab->x << " b=" << ab->y << "m";
        return out.str();
    };
    const std::vector<std::string> fit_notes{
        "GROUND X=aY+b | origin X: " + (ground.ground_robot_x_m ?
            two_decimals(*ground.ground_robot_x_m) + "m" : "LOCKING/N/A"),
        "L: " + coefficients(ground.ground_left_ab) + " | R: " + coefficients(ground.ground_right_ab)};
    for (size_t i = 0; i < fit_notes.size(); ++i) {
        const cv::Point position(8, 119 + int(i)*19);
        cv::putText(canvas, fit_notes[i], position, cv::FONT_HERSHEY_SIMPLEX, 0.37,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(canvas, fit_notes[i], position, cv::FONT_HERSHEY_SIMPLEX, 0.37,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    if (canvas.cols >= 3 && canvas.rows >= 5) {
        const int side = canvas.cols/3;
        const int bounds[] = {0,side,canvas.cols-side,canvas.cols};
        std::string note = "BOTTOM 5px L/C/R:";
        for (int i=0; i<3; ++i) {
            cv::rectangle(canvas, {bounds[i],canvas.rows-5},
                {bounds[i+1]-1,canvas.rows-1}, cv::Scalar(0,255,255), 1);
            note += " " + (frame_result.obstacle_ratio[i]>=0 ?
                two_decimals(frame_result.obstacle_ratio[i]) : "N/A");
        }
        cv::putText(canvas, note, {8,canvas.rows-12}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0,0,0), 3, cv::LINE_AA);
        cv::putText(canvas, note, {8,canvas.rows-12}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(0,255,255), 1, cv::LINE_AA);
    }
    return canvas;
}

// 세 영역의 영상을 하나의 통합 화면으로 구성한다.
// 창 크기와 FPS 제목을 설정하고 F/ESC 키로 전체 화면을 전환한다.
void VisionViewer::show_dashboard(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                    const cv::Mat &annotated, const cv::Mat &bev)
{
    if (!viewer_)
        return;
    if (!window_initialized_)
    {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, 1280, 1012);
        cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
                              viewer_fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        window_initialized_ = true;
    }
    const cv::Mat dashboard = dashboard_view(raw, white_mask, obstacle_mask, annotated, bev);
    cv::imshow(window_name_, dashboard);
    // OpenCV 창의 이벤트를 처리하고 키를 읽는다. 이 대기·표시 시간도 성능 로그에 포함된다.
    const int key = cv::waitKey(1) & 0xff;
    if (key == 'f' || key == 'F' || key == 27)
    {
        viewer_fullscreen_ = key == 27 ? false : !viewer_fullscreen_;
        cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN,
                              viewer_fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        if (!viewer_fullscreen_)
            cv::resizeWindow(window_name_, 1280, 1012);
    }
}

cv::Mat VisionViewer::dashboard_view(const cv::Mat &raw, const cv::Mat &white_mask,
    const cv::Mat &obstacle_mask, const cv::Mat &annotated, const cv::Mat &bev) const
{
    cv::Mat dashboard(1012, 1280, CV_8UC3, cv::Scalar(18,18,18));
    const auto panel = [&](const cv::Mat &image, int x, int y, const std::string &title) {
        cv::putText(dashboard,title,{x+8,y+19},cv::FONT_HERSHEY_SIMPLEX,0.55,{255,255,255},1,cv::LINE_AA);
        if (image.empty()) {
            cv::putText(dashboard,"NO IMAGE / BEV DISABLED",{x+80,y+260},
                        cv::FONT_HERSHEY_SIMPLEX,0.6,{0,200,255},1,cv::LINE_AA);
            return;
        }
        const double ratio=std::min(640.0/image.cols,480.0/image.rows);
        cv::Mat fitted;
        cv::resize(image,fitted,{cvRound(image.cols*ratio),cvRound(image.rows*ratio)},0,0,cv::INTER_AREA);
        fitted.copyTo(dashboard(cv::Rect(x+(640-fitted.cols)/2,y+26+(480-fitted.rows)/2,fitted.cols,fitted.rows)));
    };
    cv::Mat combined_mask;
    if (!obstacle_mask.empty() && obstacle_mask.type()==CV_8UC3 && obstacle_mask.size()==raw.size())
        combined_mask=obstacle_mask.clone();
    else combined_mask=cv::Mat::zeros(raw.size(),CV_8UC3);
    if (!white_mask.empty() && white_mask.type()==CV_8UC1 && white_mask.size()==raw.size())
        combined_mask.setTo(cv::Scalar::all(255),white_mask);
    panel(annotated,0,0,processing_fps_>0 ? "RESULT | "+two_decimals(processing_fps_)+" FPS" : "RESULT");
    panel(combined_mask,640,0,"YOLO + COLOR MASK");
    panel(raw,0,506,"RAW CAMERA");
    panel(bev,640,506,"BEV | YOLO LEFT: BLUE / RIGHT: YELLOW | GRID: 0.5m");
    return dashboard;
}

// 빨강·파랑 마스크의 잡음 제거 전후를 나란히 배치하고 HSV 기준·검출 통계를 표시한다.
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
    // 색상 두 종류 각각에 원본 마스크와 정리된 마스크를 배치한다.
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

// 영상이 없을 때 입력 토픽 이름과 안내 문구를 담은 검출 영역용 화면을 반환한다.
cv::Mat VisionViewer::no_image_view(const std::string &image_topic) const
{
    cv::Mat canvas(480, 640, CV_8UC3, cv::Scalar::all(0));
    cv::putText(canvas, "NO CAMERA IMAGE", {30, 200}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2,
                cv::LINE_AA);
    cv::putText(canvas, image_topic, {15, 245}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);
    return canvas;
}

// 영상이 없을 때 사용할 전처리 안내 화면을 반환한다.
cv::Mat VisionViewer::no_image_preprocess() const
{
    cv::Mat preprocess(610, 640, CV_8UC3, cv::Scalar::all(0));
    cv::putText(preprocess, "NO CAMERA IMAGE", {40, 300}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2,
                cv::LINE_AA);
    return preprocess;
}

// 켜진 전처리 창과 통합 창을 갱신한다. 로컬 viewer가 꺼져 있으면 창을 표시하지 않는다.
void VisionViewer::show(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                        const cv::Mat &annotated, const cv::Mat &preprocess, const cv::Mat &bev)
{
    if (viewer_ && show_preprocess_) {
        cv::imshow(preprocess_window_name_, preprocess);
        preprocess_initialized_ = true;
    }
    show_dashboard(raw, white_mask, obstacle_mask, annotated, bev);
}

} // namespace robot_vision
