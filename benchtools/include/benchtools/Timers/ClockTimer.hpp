#pragma once
#include <benchtools/Core/Time.hpp>
#include <benchtools/Timers/BaseTimer.hpp>

#include <atomic>
#include <chrono>
#include <ctime>

namespace benchtools {

class CpuTimer : public BaseTimer {
public:
  void start() override;

  void stop() override;

  [[nodiscard]] Duration
  duration(time_unit durationType = time_unit::seconds) override;

  [[nodiscard]] std::chrono::duration<double> currentElapsed() override;

private:
  std::clock_t mStart{0};
  std::clock_t mEnd{0};
  std::atomic<bool> mRunning{false};
};

}  // namespace benchtools