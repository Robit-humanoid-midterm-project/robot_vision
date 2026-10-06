#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>

int main()
{
    // 저장 폴더 생성
    std::filesystem::create_directories("raw");
    std::filesystem::create_directories("preprocessed");

    // /dev/video4 = Insta360 Link
    cv::VideoCapture cap(4, cv::CAP_V4L2);

    if (!cap.isOpened())
    {
        std::cout << "카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    // MJPG 설정
    cap.set(
        cv::CAP_PROP_FOURCC,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
    );

    // Insta360 Link native 해상도 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
    cap.set(cv::CAP_PROP_FPS, 30);

    // 실제 적용된 설정 확인
    std::cout << "Width  : "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH)
              << std::endl;

    std::cout << "Height : "
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
              << std::endl;

    std::cout << "FPS    : "
              << cap.get(cv::CAP_PROP_FPS)
              << std::endl;

    cv::Mat frame;
    cv::Mat frame640;
    cv::Mat hsv, red_mask_1, red_mask_2, red_mask, blue_mask, obstacle_mask;

    int count = 0;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "프레임을 읽을 수 없습니다." << std::endl;
            break;
        }

        // 1280x960 -> 640x480
        cv::resize(
            frame,
            frame640,
            cv::Size(640, 480),
            0,
            0,
            cv::INTER_AREA
        );

        // obstacle_distance.yaml의 현재 HSV 기준과 동일.
        // 두 색을 합쳐 흰색 장애물 / 검은색 배경으로 표현한다.
        cv::cvtColor(frame640, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 60, 60), cv::Scalar(12, 255, 255), red_mask_1);
        cv::inRange(hsv, cv::Scalar(168, 60, 60), cv::Scalar(179, 255, 255), red_mask_2);
        cv::bitwise_or(red_mask_1, red_mask_2, red_mask);
        cv::inRange(hsv, cv::Scalar(105, 170, 45), cv::Scalar(125, 255, 255), blue_mask);
        cv::bitwise_or(red_mask, blue_mask, obstacle_mask);
        // 뒤쪽 장애물의 작은 가시 영역을 보존하기 위해 면적 필터는 적용하지 않는다.
        cv::imshow("Insta360 Link - 640x480", frame640);
        cv::imshow("Obstacle mask - red + blue", obstacle_mask);

        int key = cv::waitKey(1);

        // SPACE를 누르면 이미지 저장
        if (key == 32)
        {
            std::string stem;
            // 프로그램을 다시 실행해도 기존 사진을 덮어쓰지 않는다.
            do
            {
                std::ostringstream name;
                name << "image_" << std::setfill('0') << std::setw(4) << count++;
                stem = name.str();
            } while (std::filesystem::exists("raw/" + stem + ".png") ||
                     std::filesystem::exists("preprocessed/" + stem + ".png"));

            const std::string raw_path = "raw/" + stem + ".png";
            const std::string mask_path = "preprocessed/" + stem + ".png";
            try
            {
                const bool raw_saved = cv::imwrite(raw_path, frame640);
                const bool mask_saved = cv::imwrite(mask_path, obstacle_mask);
                if (raw_saved && mask_saved)
                    std::cout << "저장: " << raw_path << " / " << mask_path << std::endl;
                else
                    std::cerr << "이미지 저장 실패: " << stem << std::endl;
            }
            catch (const cv::Exception &error)
            {
                std::cerr << "이미지 저장 실패: " << error.what() << std::endl;
            }
        }

        // ESC를 누르면 종료
        if (key == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
