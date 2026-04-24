#pragma once

#include <benchtools/Core/Time.hpp>
#include <benchtools/Timers/BaseTimer.hpp>

namespace benchtools {

class ScopedTimer : BaseTimer {
public:
  ScopedTimer() = default;

  ScopedTimer(BaseTimer& timer);

  virtual ~ScopedTimer() override;

  virtual void start() override;

  virtual void stop() override;

  [[nodiscard]] virtual Duration
  duration(time_unit durationType = time_unit::seconds) override;

private:
  virtual std::chrono::duration<double> currentElapsed() override;

private:
  BaseTimer* mTimer;
};
};  // namespace benchtools