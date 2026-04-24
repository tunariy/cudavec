#pragma once
#include <benchtools/Core/Time.hpp>
#include <benchtools/Timers/BaseTimer.hpp>
#include <benchtools/Timers/WallTimer.hpp>

namespace benchtools {

class LoggingTimer : public BaseTimer {
public:
  explicit LoggingTimer() : mTimer(WallTimer{}) { mTimer.start(); };

  virtual ~LoggingTimer() override;

  virtual void start() override;

  virtual void stop() override;

private:
  virtual std::chrono::duration<double> currentElapsed() override;

private:
  WallTimer mTimer{};
  time_unit mUnit{time_unit::seconds};
};
}  // namespace benchtools