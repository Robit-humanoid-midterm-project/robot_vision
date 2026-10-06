// 파일 역할: 디버그 영상 생성과 OpenCV 창 표시의 인터페이스 및 창 상태를 선언한다.

#pragma once

#include <string>
#include <opencv2/core.hpp>
#include "robot_vision/vision_frame_result.hpp"

namespace robot_vision {

// Builds debug images even with the local viewer disabled, preserving ROS outputs.
// 계산 결과를 그리는 객체. 화면을 꺼도 요청된 ROS 디버그 영상을 만드는 함수는 사용할 수 있다.
class VisionViewer {
  public:
    // 검출 설정은 전처리 설명에, 나머지 플래그는 창 사용 방식에 사용한다.
    VisionViewer(DetectorConfig config, bool enabled, bool show_preprocess, bool fullscreen);
    // 실제로 초기화한 창만 닫는다.
    ~VisionViewer();
    // 중심 노드가 측정한 FPS를 화면 제목용으로 저장한다.
    void set_processing_fps(double fps) { processing_fps_ = fps; }
    VisionViewer(const VisionViewer &) = delete;
    VisionViewer &operator=(const VisionViewer &) = delete;
    // 입력 복사본에 검출 도형과 거리를 그려 반환한다.
    cv::Mat annotate(const cv::Mat &frame, const VisionFrameResult &result) const;
    // 색상 마스크의 정리 전후를 비교하는 디버그 영상을 반환한다.
    cv::Mat preprocess_view(const DetectResult &result) const;
    // 입력 토픽과 영상 없음 안내를 담은 검출 화면을 반환한다.
    cv::Mat no_image_view(const std::string &image_topic) const;
    // 전처리 창용 영상 없음 안내 화면을 반환한다.
    cv::Mat no_image_preprocess() const;
    // 필요한 창을 표시하고 키 이벤트를 처리한다.
    void show(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
              const cv::Mat &annotated, const cv::Mat &preprocess);

  private:
    // 세 영역을 배치하는 내부 함수. 로봇 제어 값을 계산하지 않는다.
    void show_dashboard(const cv::Mat &raw, const cv::Mat &white_mask, const cv::Mat &obstacle_mask,
                        const cv::Mat &annotated);
    DetectorConfig debug_config_;
    bool viewer_{false}, show_preprocess_{true}, viewer_fullscreen_{true};
    double processing_fps_{0.0};
    bool window_initialized_{false}, preprocess_initialized_{false};
    const std::string window_name_ = "Robot vision - RAW / OPENCV / RESULT";
    const std::string preprocess_window_name_ = "Obstacle preprocess - RED / BLUE";
};

} // namespace robot_vision
