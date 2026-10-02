#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>

int main()
{
    // 저장 폴더 생성
    std::filesystem::create_directories("raw");

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

        cv::imshow("Insta360 Link - 640x480", frame640);

        int key = cv::waitKey(1);

        // SPACE를 누르면 이미지 저장
        if (key == 32)
        {
            std::ostringstream filename;

            filename
                << "raw/image_"
                << std::setfill('0')
                << std::setw(4)
                << count
                << ".png";

            cv::imwrite(filename.str(), frame640);

            std::cout
                << "저장: "
                << filename.str()
                << std::endl;

            count++;
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