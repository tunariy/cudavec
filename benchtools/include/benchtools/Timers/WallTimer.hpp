#pragma once
#include "benchtools/Core/Time.hpp"
#include <atomic>
#include <benchtools/Timers/BaseTimer.hpp>

namespace benchtools {

class LoggingTimer;

class WallTimer : public BaseTimer {
  using clock = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

public:
  WallTimer() = default;

  virtual ~WallTimer() override;

  virtual void start() override;

  virtual void stop() override;

  virtual void reset(bool reset = 0) override;

  virtual Duration duration(time_unit durationType) override;

public:
  friend class LoggingTimer;

private:
  virtual std::chrono::duration<double> currentElapsed() override;

private:
  time_point mStartPoint;
  std::chrono::duration<double> mElapsedTime{default_duration};
  std::atomic<bool> mIsRunning{0};
};

}  // namespace benchtools