// 파일 역할: 카메라 입력 시각을 일정한 처리 주기에 맞춰 선택하는 가벼운 제한기.

#pragma once

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace robot_vision {

// Keep deadlines on one timeline. Resetting the deadline at each received frame
// unnecessarily halves the rate when camera and processing periods almost match.
class FrameRateLimiter {
  public:
    // 시스템 날짜 변경에 영향을 받지 않는 단조 시계를 사용한다.
    using Clock = std::chrono::steady_clock;

    // FPS를 초 단위 주기로 바꾼다. 0·음수·무한대나 시계 해상도보다 짧은 주기는 거부한다.
    explicit FrameRateLimiter(double fps)
    {
        if (!std::isfinite(fps) || fps <= 0)
            throw std::invalid_argument("Processing FPS must be positive and finite");
        period_ = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));
        if (period_ <= Clock::duration::zero())
            throw std::invalid_argument("Processing FPS exceeds clock resolution");
    }

    // 현재 시각에 프레임을 처리해도 되는지 판단한다.
    // 허용하면 다음 예정 시각을 앞으로 옮기며, 처리할 때마다 현재 시각 기준으로 주기를 다시 시작하지 않는다.
    bool should_process(Clock::time_point now)
    {
        // 첫 입력은 바로 처리하고 그 시각을 이후 처리 일정의 기준으로 삼는다.
        if (!initialized_) {
            next_ = now;
            initialized_ = true;
        }
        if (now < next_)
            return false;
        // Advance past all missed slots, so a pause cannot create a catch-up burst.
        // 지난 처리 슬롯은 한 번에 건너뛴다. 긴 멈춤 뒤 빠르게 연속 처리하는 따라잡기 동작을 만들지 않는다.
        next_ += ((now - next_) / period_ + 1) * period_;
        return true;
    }

    // 다음 호출을 첫 입력으로 취급하도록 초기화한다.
    void reset() { initialized_ = false; }

  private:
    Clock::duration period_{};
    Clock::time_point next_{};
    bool initialized_{false};
};

} // namespace robot_vision
