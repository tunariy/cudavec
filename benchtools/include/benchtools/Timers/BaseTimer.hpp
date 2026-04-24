#pragma once
#include <benchtools/Core/Time.hpp>
#include <chrono>

namespace benchtools {

class BaseTimer {
public:
  BaseTimer() = default;

  virtual ~BaseTimer();

  virtual void start() = 0;

  virtual void stop() = 0;

  virtual void reset(bool);

  [[nodiscard]] virtual Duration duration(time_unit durationType = time_unit::seconds) {
    return durationCast(default_duration, time_unit::seconds);
  };

private:
  virtual std::chrono::duration<double> currentElapsed() = 0;

protected:
};

}  // namespace benchtools