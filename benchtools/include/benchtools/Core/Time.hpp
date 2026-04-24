#pragma once
#include <chrono>
#include <cstdint>
#include <ratio>
#include <variant>

namespace benchtools {

constexpr auto default_duration = std::chrono::duration<double>::zero();

enum class time_unit : uint8_t {
  months,
  years,
  weeks,
  days,
  hours,
  minutes,
  seconds,
  milliseconds,
  microseconds,
  nanoseconds
};

using duration_t =
  std::variant<std::chrono::nanoseconds, std::chrono::microseconds,
               std::chrono::milliseconds, std::chrono::seconds, std::chrono::minutes,
               std::chrono::hours, std::chrono::days, std::chrono::weeks,
               std::chrono::months, std::chrono::years>;

std::string format(benchtools::time_unit unit);

std::string format(duration_t dur);

struct Duration {
public:
  Duration(duration_t dur) : dur(dur) {};

public:
  operator std::string() const { return format(dur); }

  [[nodiscard]] std::string str() const { return format(dur); }

private:
  duration_t dur;
};

[[nodiscard]] Duration durationCast(const std::chrono::duration<double>& duration,
                                    time_unit unit);

std::string getCurrentTimeDate();

}  // namespace benchtools
