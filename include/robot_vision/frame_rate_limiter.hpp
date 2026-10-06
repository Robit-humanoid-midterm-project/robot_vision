#pragma once

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace robot_vision {

// Keep deadlines on one timeline. Resetting the deadline at each received frame
// unnecessarily halves the rate when camera and processing periods almost match.
class FrameRateLimiter {
  public:
    using Clock = std::chrono::steady_clock;

    explicit FrameRateLimiter(double fps)
    {
        if (!std::isfinite(fps) || fps <= 0)
            throw std::invalid_argument("Processing FPS must be positive and finite");
        period_ = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));
        if (period_ <= Clock::duration::zero())
            throw std::invalid_argument("Processing FPS exceeds clock resolution");
    }

    bool should_process(Clock::time_point now)
    {
        if (!initialized_) {
            next_ = now;
            initialized_ = true;
        }
        if (now < next_)
            return false;
        // Advance past all missed slots, so a pause cannot create a catch-up burst.
        next_ += ((now - next_) / period_ + 1) * period_;
        return true;
    }

    void reset() { initialized_ = false; }

  private:
    Clock::duration period_{};
    Clock::time_point next_{};
    bool initialized_{false};
};

} // namespace robot_vision
